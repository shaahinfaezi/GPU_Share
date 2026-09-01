#!/usr/bin/env bash
# Measure what a model ACTUALLY needs, instead of inferring it from weights.
#
# render.py sizes every quota from on-disk weight size plus a guess at overhead.
# That guess is wrong in a way that scales badly: the CUDA context is a fixed
# cost, so it is 5% of an 8 GiB quota and 12% of a 3.8 GiB one. llm2 was given
# 5632 MiB at util 0.92, which leaves the allocator roughly nothing, and it
# crashlooped in _dummy_sampler_run -- the last big allocation vLLM makes.
#
# This deploys one role by itself at a chosen quota and reads vLLM's own
# accounting out of the startup log:
#
#     total_gpu_memory (X) x gpu_memory_utilization (Y) = Z
#     model weights take A; non_torch_memory takes B;
#     PyTorch activation peak memory takes C;
#     the rest of the memory reserved for KV Cache is D.
#
# Those five numbers size every quota in the matrix correctly. Run it once per
# generative role at a quota you know is generous, then work backwards.
#
#   ./18-vram-probe.sh llm2            # whole card, 11776 MiB
#   ./18-vram-probe.sh llm2 5632       # reproduce the crash and capture it
set -uo pipefail
cd "$(dirname "$0")"
NS=${NS:-llm-demo}
ROLE=${1:?usage: 18-vram-probe.sh <role> [gpumem-mib] }
QUOTA=${2:-11776}
TIMEOUT=${TIMEOUT:-600}
OUT=$(mktemp -d)
trap 'kubectl -n "$NS" delete -f "$OUT" --ignore-not-found --wait=false >/dev/null 2>&1; rm -rf "$OUT"' EXIT

SCEN="${ROLE}-only"
echo "=== probing ${ROLE} alone at ${QUOTA} MiB (scenario ${SCEN})"
python3 render.py --scenario "$SCEN" --arm dedicated --out "$OUT" \
        --gpumem "${ROLE}=${QUOTA}" || exit 1
rm -f "$OUT"/*.json 2>/dev/null
# render.py writes metadata under meta/; apply -f on a directory would submit it
rm -rf "$OUT/meta" 2>/dev/null

kubectl -n "$NS" apply -f "$OUT" >/dev/null || exit 1

echo "=== waiting up to ${TIMEOUT}s for Ready or CrashLoopBackOff"
POD=""; VERDICT="timeout"
for i in $(seq 1 $((TIMEOUT/10))); do
  sleep 10
  POD=$(kubectl -n "$NS" get pods -l "app.kubernetes.io/name=r2-${ROLE}" \
          -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
  [ -n "$POD" ] || continue
  READY=$(kubectl -n "$NS" get pod "$POD" \
            -o jsonpath='{range .status.containerStatuses[?(@.name=="main")]}{.ready}{end}' 2>/dev/null)
  WAIT=$(kubectl -n "$NS" get pod "$POD" \
            -o jsonpath='{range .status.containerStatuses[?(@.name=="main")]}{.state.waiting.reason}{end}' 2>/dev/null)
  RC=$(kubectl -n "$NS" get pod "$POD" \
            -o jsonpath='{range .status.containerStatuses[?(@.name=="main")]}{.restartCount}{end}' 2>/dev/null)
  printf '  [%3ds] pod=%s ready=%s waiting=%s restarts=%s\n' \
         "$((i*10))" "$POD" "${READY:-?}" "${WAIT:-none}" "${RC:-0}"
  [ "$READY" = "true" ] && { VERDICT="ready"; break; }
  [ "$WAIT" = "CrashLoopBackOff" ] && { VERDICT="crashloop"; break; }
done
echo "=== verdict: ${VERDICT}"
[ -n "$POD" ] || { echo "no pod ever appeared"; exit 1; }

LOG=""
for FLAG in "--previous" ""; do
  L=$(kubectl -n "$NS" logs "$POD" -c main $FLAG --tail=6000 2>/dev/null)
  # prefer whichever instance actually got as far as memory profiling
  if printf '%s' "$L" | grep -q "gpu_memory_utilization\|model weights take"; then LOG="$L"; break; fi
  [ -z "$LOG" ] && LOG="$L"
done

echo
echo "=== vLLM memory arithmetic"
printf '%s\n' "$LOG" | grep -iE \
  "total_gpu_memory|model weights take|non_torch|activation peak|KV cache|Maximum concurrency|Available KV" \
  | sed 's/^/  /' | tail -20

echo
echo "=== the exception"
printf '%s\n' "$LOG" | grep -iE \
  "OutOfMemory|out of memory|CUDA error|RuntimeError|ValueError|Traceback|Killed|assert" \
  | sed 's/^/  /' | tail -25

echo
echo "=== last 30 lines"
printf '%s\n' "$LOG" | tail -30 | sed 's/^/  /'

echo
echo "=== resident VRAM right now (device the pod landed on)"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv 2>/dev/null | sed 's/^/  /'
