#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NS="llm-demo"

for file in 04-image-pinner.yaml \
            40-qwen-hami-mps.yaml \
            31-qwen-hami-compute-profiles.yaml \
            30-qwen-hami.yaml \
            21-qwen-mps.yaml \
            10-qwen-sequential-01.yaml \
            10-qwen-sequential-02.yaml \
            20-mps-daemon.yaml \
            06-gpu-reserver.yaml \
            05-gpu-probe.yaml; do
  kubectl delete -f "${ROOT_DIR}/${file}" --ignore-not-found=true || true
done

# Wait for the vLLM Pods to actually terminate. Until they do they still hold a
# CUDA context and their VRAM, so a "successful" cleanup that returns early
# leaves the GPU occupied for whatever runs next.
echo "Waiting for benchmark workload Pods to terminate..."
deadline=$(( $(date +%s) + 300 ))
while true; do
  remaining="$(kubectl get pods -n "${NS}" --no-headers 2>/dev/null \
    | grep -cE '^(qwen-bench-01|qwen-bench-02)-kserve' || true)"
  remaining="${remaining:-0}"
  if [[ "${remaining}" -eq 0 ]]; then
    echo "All benchmark workload Pods are gone."
    break
  fi
  if (( $(date +%s) >= deadline )); then
    echo "WARNING: ${remaining} Pod(s) still terminating after 300s." >&2
    break
  fi
  sleep 5
done

echo "Deleted Qwen benchmark resources from ${ROOT_DIR}."
echo "The shared namespace (${NS}) and the qwen-local-pv-claim PVC were left untouched."
echo
echo "If a previous run was interrupted while the GPU was in EXCLUSIVE_PROCESS"
echo "compute mode, reset it on the GPU node with: sudo nvidia-smi -c DEFAULT"
