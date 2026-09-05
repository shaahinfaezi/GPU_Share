#!/usr/bin/env bash
# Where does a vLLM tenant's ~3 GiB of host RAM actually go?
#
#   chmod +x 27-hostmem-attribution.sh
#   ./27-hostmem-attribution.sh              # both passes, ~15 min
#   ./27-hostmem-attribution.sh --skip-b     # baseline pass only
#
# T4 and T7 measured the SIZE precisely: 2,358 to 3,049 MiB per tenant, flat
# across a 4.6x parameter range and two model kinds. The ATTRIBUTION -- that it
# is the Python/torch/CUDA runtime -- was asserted, not measured, and the
# flatness argument does not actually distinguish it from the alternative:
#
#   vLLM's --swap-space defaults to 4 GiB of CPU space per GPU
#   ("CPU swap space size (GiB) per GPU", docs.vllm.ai engine args)
#
# We have never set it. A fixed 4 GiB allocation would ALSO be flat across
# models, so every measurement so far is consistent with both explanations.
# That matters: if it is swap space, the memory is RECOVERABLE by setting one
# flag, and the host-RAM ceiling that killed T4 moves.
#
# Pass A deploys one tenant as the tiers do and records where its memory sits.
# Pass B does the same with --swap-space=0 injected. The difference is the
# answer.
set -uo pipefail
cd "$(dirname "$0")"

SKIP_B=0
[ "${1:-}" = "--skip-b" ] && SKIP_B=1
NS="${R2_NS:-llm-demo}"
ROLE="${ROLE:-llm}"
OUT="$(pwd)/results/hostmem-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUT"

say()  { printf '\n\033[1m=== %s\033[0m\n' "$1"; }
ok()   { printf '  \033[32mOK\033[0m    %s\n' "$1"; }
warn() { printf '  \033[33mWARN\033[0m  %s\n' "$1"; }

avail() { awk '/^MemAvailable:/{print int($2/1024)}' /proc/meminfo; }

# KServe does NOT copy an LLMInferenceService's labels onto its pods --
# spec.template is a bare PodSpec with no metadata -- so a selector on r2-role
# matches NOTHING and a wait built on it returns instantly. The first version of
# this script did exactly that: teardown never waited, pass B measured its
# baseline while pass A's engine was still resident, and the run reported a
# 2,121 MiB saving that was pass A's own memory. Two tells gave it away -- the
# two passes returned byte-identical vLLM logs, and EngineCore RSS differed by
# 0.8% when removing 2 GiB of swap space should have collapsed it.
#
# The pods carry app.kubernetes.io/name=r2-<role>. Wait on that, and then wait
# for the memory to actually come back, which is the thing being measured.
teardown() {
  kubectl -n "$NS" delete llminferenceservice --all --ignore-not-found --wait=false >/dev/null 2>&1
  local i n
  for i in $(seq 1 60); do
    n=$(kubectl -n "$NS" get pods -l "app.kubernetes.io/name=r2-${ROLE}"         -o name 2>/dev/null | wc -l)
    [ "$n" = "0" ] && break
    sleep 3
  done
  [ "${n:-0}" = "0" ] || warn "pod still present after 180s"
}

# Block until MemAvailable stops climbing, so a baseline is never taken while a
# previous tenant is still releasing. Without this the second pass is measured
# against a floor the first pass dug.
settle() {
  local last=0 now i
  for i in $(seq 1 40); do
    now=$(avail)
    if [ "$now" -le "$((last + 40))" ] && [ "$i" -gt 3 ]; then
      echo "  MemAvailable settled at ${now} MiB after $((i * 5))s"
      return 0
    fi
    last=$now
    sleep 5
  done
  warn "MemAvailable never settled; baseline may be low"
}

# $1 = label, $2 = 1 to inject --swap-space=0
measure() {
  local LABEL="$1" ZERO="$2"
  local D="$OUT/$LABEL"
  mkdir -p "$D"

  say "[$LABEL] clean slate"
  teardown
  settle
  local BASE; BASE=$(avail)
  ok "MemAvailable before deploy: ${BASE} MiB"

  say "[$LABEL] render"
  python3 render.py --scenario llm-only --arm dedicated --out "$D/rendered" \
          --only "$ROLE" 2>&1 | tail -2 | sed 's/^/  /'
  if [ "$ZERO" = "1" ]; then
    # args items sit at 10 spaces under an 8-space "args:" key.
    sed -i '/^        args:$/a\          - --swap-space=0' "$D"/rendered/*.yaml
    ok "injected --swap-space=0"
    grep -n "swap-space" "$D"/rendered/*.yaml | sed 's/^/    /'
  else
    ok "stock manifest (vLLM applies its own --swap-space default)"
  fi

  say "[$LABEL] deploy and wait"
  kubectl apply -f "$D/rendered" >/dev/null 2>&1
  local POD=""
  for i in $(seq 1 100); do
    POD=$(kubectl -n "$NS" get pods -l "app.kubernetes.io/name=r2-${ROLE}" \
          -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
    if [ -n "$POD" ]; then
      local RDY
      RDY=$(kubectl -n "$NS" get pod "$POD" \
            -o jsonpath='{.status.containerStatuses[0].ready}' 2>/dev/null)
      [ "$RDY" = "true" ] && break
    fi
    sleep 6
  done
  if [ -z "$POD" ]; then warn "no pod appeared"; return 1; fi
  ok "pod ready: $POD"
  echo "$POD" > "$D/pod_name.txt"
  # If both passes report the same pod name, the second measured the first.
  if [ -f "$OUT/A-stock/pod_name.txt" ] && [ "$LABEL" != "A-stock" ]; then
    if [ "$POD" = "$(cat "$OUT/A-stock/pod_name.txt")" ]; then
      warn "SAME POD as pass A -- this run measures nothing. Aborting."
      return 1
    fi
    ok "distinct pod from pass A"
  fi
  sleep 20   # let it settle past the load transient

  local STEADY; STEADY=$(avail)
  echo "  MemAvailable now: ${STEADY} MiB   (drop $((BASE - STEADY)) MiB)"
  echo "$((BASE - STEADY))" > "$D/drop_mib.txt"

  say "[$LABEL] what vLLM says about swap / CPU cache / offload"
  kubectl -n "$NS" logs "$POD" > "$D/vllm.log" 2>&1
  grep -inE "swap|cpu.*cache|offload|KV cache|memory profil|Available KV" "$D/vllm.log" \
    | head -20 | sed 's/^/  /' || echo "  (no matching lines)"

  say "[$LABEL] per-process resident memory inside the container"
  kubectl -n "$NS" exec "$POD" -- ps -eo pid,rss,comm --sort=-rss 2>/dev/null \
    | head -8 | tee "$D/ps.txt" | sed 's/^/  /'

  say "[$LABEL] anonymous vs file-backed -- the decisive split"
  # Anonymous memory is the process's own allocations: torch tensors, the CUDA
  # host context, pinned buffers, and any CPU swap space. File-backed is the
  # mapped .so files and mmap'd weights, which are reclaimable page cache and
  # are NOT what constrains co-residency.
  for PAT in EngineCore vllm; do
    local PID
    PID=$(kubectl -n "$NS" exec "$POD" -- sh -c "pgrep -f '$PAT' | head -1" 2>/dev/null | tr -d '\r')
    [ -z "$PID" ] && continue
    echo "  --- $PAT (pid $PID)"
    kubectl -n "$NS" exec "$POD" -- sh -c "cat /proc/$PID/smaps_rollup" 2>/dev/null \
      | tee "$D/smaps_${PAT}.txt" \
      | grep -iE "^(Rss|Pss|Anonymous|Shared_Clean|Private_Dirty|Swap|Locked)" \
      | sed 's/^/      /'
  done

  teardown
}

measure "A-stock" 0
if [ "$SKIP_B" = "0" ]; then
  measure "B-swapspace0" 1
fi

say "VERDICT"
A=$(cat "$OUT/A-stock/drop_mib.txt" 2>/dev/null || echo 0)
B=$(cat "$OUT/B-swapspace0/drop_mib.txt" 2>/dev/null || echo "")
echo "  pass A (stock)          MemAvailable drop: ${A} MiB"
if [ -n "$B" ]; then
  echo "  pass B (--swap-space=0) MemAvailable drop: ${B} MiB"
  D=$((A - B))
  echo "  difference:             ${D} MiB"
  echo
  echo
  echo "  Cross-check before believing either number: the two passes must show"
  echo "  DIFFERENT pod names and DIFFERENT EngineCore RSS in ps.txt. If RSS is"
  echo "  within a few percent, no swap space was removed and the drop figures"
  echo "  are measuring baseline contamination, not the flag."
  echo
  if [ "$D" -gt 500 ]; then
    echo "  >>> vLLM's CPU swap space accounts for ${D} MiB of the per-tenant cost."
    echo "      It is RECOVERABLE: add --swap-space=0 (or a small value) to the"
    echo "      manifests and the host-RAM ceiling that blocked T4 moves. Re-run"
    echo "      the host calibration afterwards and update HOST_RSS_MIB."
  else
    echo "  >>> swap space is NOT the explanation (difference ${D} MiB)."
    echo "      The cost is the Python/torch/CUDA runtime as claimed. Check the"
    echo "      Anonymous line in the smaps output above: if anonymous memory is"
    echo "      most of Rss, it is genuinely resident and irreducible; if much of"
    echo "      Rss is file-backed, the additive host-RAM model is too pessimistic"
    echo "      because those pages are shared between co-resident tenants."
  fi
fi
echo
echo "  artifacts: $OUT"
