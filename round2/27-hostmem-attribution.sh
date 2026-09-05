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

teardown() {
  kubectl -n "$NS" delete llminferenceservice --all --ignore-not-found --wait=false >/dev/null 2>&1
  for i in $(seq 1 40); do
    [ "$(kubectl -n "$NS" get pods -l r2-role -o name 2>/dev/null | wc -l)" = "0" ] && return 0
    sleep 3
  done
  warn "pods still present after teardown wait"
}

# $1 = label, $2 = 1 to inject --swap-space=0
measure() {
  local LABEL="$1" ZERO="$2"
  local D="$OUT/$LABEL"
  mkdir -p "$D"

  say "[$LABEL] clean slate"
  teardown
  sleep 5
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
