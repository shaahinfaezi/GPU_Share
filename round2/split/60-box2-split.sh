#!/usr/bin/env bash
# BOX 2. llama.cpp (partial GPU offload) beside the custom transformers runtime.
#
# Four arms, same shape as box 1 so the two boxes can be read side by side:
#
#   B1  llamacpp  dedicated   alone on the card
#   B2  custom    dedicated   alone on the card
#   B3  both      hami        co-resident, HAMi quotas, no MPS
#   B4  both      hami-mps    identical, MPS daemon up, clients at 100%
#
# WHAT THIS PAIR TESTS THAT BOX 1 DOES NOT
# ----------------------------------------
# Box 1 shares a card between two tenants of the SAME runtime. This pair shares
# one between two runtimes that have nothing in common: llama.cpp is C++ with a
# hand-managed partial offload and a tiny VRAM footprint, the custom runtime is
# HuggingFace transformers holding whole weights with no paged KV cache and no
# continuous batching. If HAMi's limiter and MPS behave the same across those,
# the mechanism is a property of the layer and not of vLLM -- which is the only
# reason to believe any of round two's numbers generalise to the fleet's real
# mix of runtimes.
#
# BEFORE RUNNING, three things that will otherwise waste the evening:
#   1. model.py needs __name__/__main__ restored (see manifests/box2-custom.yaml).
#      As pasted it exits without starting the server.
#   2. Check the custom model's real size on the PVC against the 6144 MiB quota
#      in that manifest. Too small and it crashloops on load, not gracefully.
#   3. CUDA_VISIBLE_DEVICES has been removed from the custom manifest on
#      purpose. Do not put it back; HAMi sets it, and overriding it is how a
#      pod ends up outside the isolation it appears to be inside.
#
#   chmod +x 60-box2-split.sh
#   nohup ./60-box2-split.sh > logs/box2.log 2>&1 &
#   tail -f logs/box2.log
set -uo pipefail
cd "$(dirname "$0")"

STAMP="$(date -u +%Y%m%d-%H%M%S)"
RUN="results/B-box2-${STAMP}"
LOG="logs/box2-${STAMP}.log"
NS="${R2_NS:-llm-demo}"
PY=python3
mkdir -p logs "${RUN}"
export PYTHONPATH="$(cd .. && pwd):${PYTHONPATH:-}"

RATE="${RATE:-6}"          # total req/s across both tenants
SECS="${SECS:-240}"
SOLO_SECS="${SOLO_SECS:-180}"

log() { echo "$(date -u +%FT%TZ) $*" | tee -a "${LOG}"; }

mps() {   # $1 = on|off
  ${PY} - "$1" >>"${LOG}" 2>&1 <<'EOF' || true
import sys, os
sys.path.insert(0, os.path.dirname(os.getcwd()))
import harness as H
H.set_mps_daemon(sys.argv[1] == "on")
EOF
}

teardown() {
  kubectl delete inferenceservice -n "${NS}" -l r2-series=split-box2 \
      --ignore-not-found --timeout=180s >>"${LOG}" 2>&1
  sleep 15
}

# Resolve pod IP:port for a role, empty if not ready.
endpoint() {
  local role="$1"
  kubectl get pods -n "${NS}" -l "r2-role=${role}" \
    -o jsonpath='{range .items[?(@.status.podIP!="")]}http://{.status.podIP}:8080{end}' \
    2>/dev/null | head -c 200
}

wait_ready() {   # $@ = roles
  local deadline=$(( $(date +%s) + 900 ))
  for role in "$@"; do
    while :; do
      local st
      st=$(kubectl get pods -n "${NS}" -l "r2-role=${role}" \
             -o jsonpath='{.items[*].status.conditions[?(@.type=="Ready")].status}' 2>/dev/null)
      case "${st}" in *True*) log "  ${role} Ready"; break ;; esac
      if [ "$(date +%s)" -ge "${deadline}" ]; then
        log "  TIMEOUT waiting for ${role}; recent events:"
        kubectl get events -n "${NS}" --sort-by=.lastTimestamp 2>/dev/null \
          | tail -15 | tee -a "${LOG}"
        return 1
      fi
      sleep 10
    done
  done
  return 0
}

trap 'log "INTERRUPTED"; teardown; mps off; exit 130' INT TERM

log "=== box 2 split/share series, run ${RUN}"
teardown
mps off

# ------------------------------------------------------------ solo baselines
for ROLE in llamacpp custom; do
  case "${ROLE}" in
    llamacpp) MAN=manifests/box2-llamacpp.yaml ;;
    custom)   MAN=manifests/box2-custom.yaml ;;
  esac
  OUT="${RUN}/dedicated__${ROLE}"
  mkdir -p "${OUT}"
  log ""
  log "=== ${ROLE} dedicated"
  cp "${MAN}" "${OUT}/"
  kubectl apply -f "${MAN}" >>"${LOG}" 2>&1

  if wait_ready "${ROLE}"; then
    EP=$(endpoint "${ROLE}")
    log "  endpoint ${EP}"
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader \
      | sed 's/^/  vram /' | tee -a "${LOG}"
    ${PY} 61-box2-load.py --arm dedicated --endpoint "${ROLE}=${EP}" \
          --rate "${RATE}" --duration "${SOLO_SECS}" --out-dir "${OUT}" \
          2>&1 | tee -a "${LOG}"
  else
    log "  SKIPPED load: ${ROLE} never became Ready"
  fi
  teardown
done

# --------------------------------------------------------------- co-tenancy
for ARM in hami hami-mps; do
  OUT="${RUN}/${ARM}__llamacpp+custom"
  mkdir -p "${OUT}"
  log ""
  log "=== arm ${ARM}"

  # Daemon before pods: a client that starts without it silently runs plain
  # HAMi under an MPS label, and the arm reports a difference of zero.
  [ "${ARM}" = "hami-mps" ] && mps on || mps off

  cp manifests/box2-llamacpp.yaml manifests/box2-custom.yaml "${OUT}/"
  if [ "${ARM}" = "hami-mps" ]; then
    # MPS reaches these two through the env the daemonset injects on this node;
    # if that is not how box 2 is set up, add the three CUDA_MPS_* vars and the
    # /tmp/nvidia-mps hostPath to both manifests exactly as box 1 does.
    log "  NOTE verify both pods are MPS clients before trusting this arm:"
    log "        nvidia-smi | grep -c 'M+C'   # expect 2"
  fi
  kubectl apply -f manifests/box2-llamacpp.yaml -f manifests/box2-custom.yaml \
      >>"${LOG}" 2>&1

  if wait_ready llamacpp custom; then
    EP1=$(endpoint llamacpp); EP2=$(endpoint custom)
    log "  llamacpp ${EP1}"
    log "  custom   ${EP2}"

    # Both tenants must be on the SAME card or this is not a sharing arm.
    BUSY=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
           | awk '$1>400' | wc -l)
    log "  cards in use: ${BUSY} (expect 1; more means HAMi spread them --"
    log "                run ../10-hami-binpack.sh and redo this arm)"
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader \
      | sed 's/^/  proc /' | tee -a "${LOG}"

    ( while :; do
        nvidia-smi --query-gpu=index,memory.used,utilization.gpu,power.draw,clocks.sm \
          --format=csv,noheader,nounits >> "${OUT}/gpu.csv"; sleep 1
      done ) & SAMPLER=$!

    ${PY} 61-box2-load.py --arm "${ARM}" \
          --endpoint "llamacpp=${EP1}" --endpoint "custom=${EP2}" \
          --rate "${RATE}" --duration "${SECS}" --out-dir "${OUT}" \
          2>&1 | tee -a "${LOG}"
    kill "${SAMPLER}" 2>/dev/null
  else
    log "  SKIPPED load: one or both tenants never became Ready"
  fi
  teardown
done

mps off
log ""
log "=== box 2 done. Send ${RUN} back; collect both boxes with:"
log "===   python3 62-box2-collect.py ${RUN}"
