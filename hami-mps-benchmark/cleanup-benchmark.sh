#!/usr/bin/env bash
set -euo pipefail

NS="gpu-share-bench"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

kubectl delete -f "${ROOT_DIR}/40-hami-plus-mps-job.yaml" --ignore-not-found=true
kubectl delete -f "${ROOT_DIR}/31-hami-oom-job.yaml" --ignore-not-found=true
kubectl delete -f "${ROOT_DIR}/30-hami-parallel-job.yaml" --ignore-not-found=true
kubectl delete -f "${ROOT_DIR}/21-mps-parallel-job.yaml" --ignore-not-found=true
kubectl delete -f "${ROOT_DIR}/20-mps-daemon.yaml" --ignore-not-found=true
kubectl delete -f "${ROOT_DIR}/10-sequential-job.yaml" --ignore-not-found=true
kubectl delete -f "${ROOT_DIR}/01-benchmark-configmap.yaml" --ignore-not-found=true
kubectl delete namespace "${NS}" --ignore-not-found=true

echo "Deleted benchmark resources from namespace ${NS}. HAMi itself was not uninstalled."
