#!/usr/bin/env bash
# Is this node actually able to run a GPU workload right now?
#
#   ./25-node-check.sh          check only
#   ./25-node-check.sh --fix    check, and restart HAMi's device plugin if the
#                               node is unregistered, then re-check
#
# This gate exists because of a wasted hour. After T4's OOM cascade the node
# came back Ready and nvidia-smi was clean, but HAMi's scheduler extender had
# lost its registration:
#
#   0/1 nodes are available: 1 node unregistered.
#   FilteringFailed  no available node, 1 nodes do not meet
#
# Every pod sat Pending. wait_ready has a 900s timeout, run.py continued to the
# next sub-run after each failure, and a four-sub-run calibration was going to
# spend an hour proving the same thing four times. Nothing in the existing
# gates -- validate, sudo, MPS toggle -- looks at whether the scheduler can
# place a GPU pod at all.
set -uo pipefail
cd "$(dirname "$0")"

FIX=0
[ "${1:-}" = "--fix" ] && FIX=1
FAIL=0
note() { printf '  %-22s %s\n' "$1" "$2"; }
bad()  { printf '  %-22s %s\n' "$1" "FAIL: $2"; FAIL=1; }

echo "=== node"
NODE=$(kubectl get nodes -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
if [ -z "$NODE" ]; then
  echo "  FATAL: no nodes returned by kubectl. Is k3s up?"
  exit 1
fi
READY=$(kubectl get node "$NODE" -o jsonpath='{.status.conditions[?(@.type=="Ready")].status}' 2>/dev/null)
[ "$READY" = "True" ] && note "$NODE" "Ready" || bad "$NODE" "not Ready (Ready=$READY)"

for c in MemoryPressure DiskPressure PIDPressure; do
  V=$(kubectl get node "$NODE" -o jsonpath="{.status.conditions[?(@.type==\"$c\")].status}" 2>/dev/null)
  [ "$V" = "False" ] && note "$c" "False" || bad "$c" "$V"
done

echo "=== gpu advertised to the scheduler"
check_alloc() {
  ALLOC=$(kubectl get node "$NODE" -o jsonpath='{.status.allocatable.nvidia\.com/gpu}' 2>/dev/null)
  GMEM=$(kubectl get node "$NODE" -o jsonpath='{.status.allocatable.nvidia\.com/gpumem}' 2>/dev/null)
  REG=$(kubectl get node "$NODE" -o jsonpath='{.metadata.annotations.hami\.io/node-nvidia-register}' 2>/dev/null)
  [ -n "$ALLOC" ] && [ "$ALLOC" != "0" ]
}

if check_alloc; then
  note "nvidia.com/gpu" "$ALLOC"
  note "nvidia.com/gpumem" "${GMEM:-<none>}"
  if [ -n "$REG" ]; then
    note "hami registration" "present (${#REG} bytes)"
  else
    bad "hami registration" "annotation hami.io/node-nvidia-register is MISSING -- this is the 'node unregistered' scheduling failure"
  fi
else
  bad "nvidia.com/gpu" "allocatable is '${ALLOC:-<none>}' -- the device plugin is not advertising"
fi

echo "=== hami pods"
HAMI=$(kubectl get pods -A -o wide 2>/dev/null | grep -Ei 'hami|vgpu' || true)
if [ -z "$HAMI" ]; then
  bad "hami pods" "none found in any namespace"
else
  echo "$HAMI" | while read -r line; do echo "    $line"; done
  NOTRUN=$(echo "$HAMI" | awk '$4!="Running"{print $2}' | tr '\n' ' ')
  [ -z "$NOTRUN" ] && note "hami pods" "all Running" || bad "hami pods" "not Running: $NOTRUN"
fi

if [ "$FAIL" = "1" ] && [ "$FIX" = "1" ]; then
  echo
  echo "=== --fix: restarting HAMi's device plugin"
  # Restarting a device plugin is non-destructive: it re-reads the GPUs and
  # re-writes the node annotation. It does NOT touch running workloads.
  DS=$(kubectl get ds -A -o jsonpath='{range .items[*]}{.metadata.namespace}{" "}{.metadata.name}{"\n"}{end}' 2>/dev/null \
       | grep -Ei 'hami.*device|device.*plugin|vgpu' | head -1)
  if [ -n "$DS" ]; then
    NS=$(echo "$DS" | awk '{print $1}'); NM=$(echo "$DS" | awk '{print $2}')
    echo "  kubectl -n $NS rollout restart ds/$NM"
    kubectl -n "$NS" rollout restart "ds/$NM"
    kubectl -n "$NS" rollout status "ds/$NM" --timeout=180s
    echo "  waiting up to 120s for the node to re-register..."
    for i in $(seq 1 24); do
      sleep 5
      if check_alloc && [ -n "$REG" ]; then
        echo "  re-registered after $((i*5))s: nvidia.com/gpu=$ALLOC"
        FAIL=0
        break
      fi
    done
  else
    echo "  could not find a HAMi device-plugin DaemonSet to restart."
  fi
fi

echo "=== host memory"
if [ -r /proc/meminfo ]; then
  TOT=$(awk '/^MemTotal:/{print int($2/1024)}' /proc/meminfo)
  AVA=$(awk '/^MemAvailable:/{print int($2/1024)}' /proc/meminfo)
  SWP=$(awk '/^SwapTotal:/{print int($2/1024)}' /proc/meminfo)
  note "MemTotal" "${TOT} MiB"
  note "MemAvailable" "${AVA} MiB"
  note "SwapTotal" "${SWP} MiB$([ "$SWP" = "0" ] && echo '  (overshoot goes straight to the OOM killer)')"
  # One vLLM engine is ~3.0-3.4 GiB of anon RSS plus ~0.9 GiB for the API
  # server, near enough independent of model size. See FINDINGS 5.13.
  if [ "$AVA" -lt 4096 ]; then
    bad "MemAvailable" "${AVA} MiB will not hold a single vLLM engine"
  fi
fi

echo "=== leftover workload"
LEFT=$(kubectl get pods -n llm-demo -l r2-role -o name 2>/dev/null | wc -l)
[ "$LEFT" = "0" ] && note "r2-role pods" "none" || bad "r2-role pods" "$LEFT still present -- a previous run did not tear down"

USED=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null | tr '\n' ' ')
note "gpu memory.used" "${USED:-<nvidia-smi failed>}"

echo
if [ "$FAIL" = "0" ]; then
  echo "OK -- the scheduler can place a GPU pod on this node."
  exit 0
fi
echo "NOT OK. Do not launch a series: every pod would sit Pending and each"
echo "sub-run would burn its full 900s ready timeout before failing."
[ "$FIX" = "0" ] && echo "Try: ./25-node-check.sh --fix"
exit 1
