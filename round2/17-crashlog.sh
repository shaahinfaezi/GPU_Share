#!/usr/bin/env bash
# Dump everything needed to explain why a role's pod will not start.
#
# A CrashLoopBackOff tail truncated at the wrong line has cost two round trips
# already. This pulls the whole picture in one go: the args the container was
# actually given, vLLM's own memory arithmetic, the terminal exception, and the
# HAMi annotation that says how much VRAM the pod was allowed.
set -uo pipefail
NS=${NS:-llm-demo}
ROLE=${1:?usage: 17-crashlog.sh <role>}

POD=$(kubectl -n "$NS" get pods -l "app.kubernetes.io/name=r2-${ROLE}" \
        -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
[ -n "$POD" ] || { echo "no pod for role ${ROLE} in ${NS}"; exit 1; }
echo "=== pod ${POD}"

echo
echo "=== args as scheduled"
kubectl -n "$NS" get pod "$POD" \
  -o jsonpath='{range .spec.containers[?(@.name=="main")]}{.args}{"\n"}{end}'

echo
echo "=== HAMi allocation + resource limits"
kubectl -n "$NS" get pod "$POD" \
  -o jsonpath='{.metadata.annotations.hami\.io/vgpu-devices-allocated}{"\n"}'
kubectl -n "$NS" get pod "$POD" \
  -o jsonpath='{range .spec.containers[?(@.name=="main")]}{.resources.limits}{"\n"}{end}'

echo
echo "=== restart count / last state"
kubectl -n "$NS" get pod "$POD" -o jsonpath='{range .status.containerStatuses[?(@.name=="main")]}restarts={.restartCount} lastExit={.lastState.terminated.exitCode} reason={.lastState.terminated.reason}{"\n"}{end}'

# --previous is the crashed instance; without it we may catch a container that
# has not failed yet this time round. Try previous first, fall back to current.
for FLAG in --previous ""; do
  LOG=$(kubectl -n "$NS" logs "$POD" -c main $FLAG --tail=4000 2>/dev/null)
  [ -n "$LOG" ] && break
done

echo
echo "=== vLLM memory arithmetic"
printf '%s\n' "$LOG" | grep -iE \
  "memory profiling|model weights take|non_torch|PyTorch activation|KV cache size|GPU KV cache size|maximum concurrency|gpu_memory_utilization|Available KV cache|free memory" \
  | tail -20

echo
echo "=== the exception"
printf '%s\n' "$LOG" | grep -nE \
  "Error|error|OutOfMemory|out of memory|Traceback|Killed|Aborted|assert" \
  | tail -30

echo
echo "=== last 40 lines"
printf '%s\n' "$LOG" | tail -40
