#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

kubectl delete -f "${ROOT_DIR}/40-qwen-hami-mps.yaml" --ignore-not-found=true || true
kubectl delete -f "${ROOT_DIR}/31-qwen-hami-compute-profiles.yaml" --ignore-not-found=true || true
kubectl delete -f "${ROOT_DIR}/30-qwen-hami.yaml" --ignore-not-found=true || true
kubectl delete -f "${ROOT_DIR}/21-qwen-mps.yaml" --ignore-not-found=true || true
kubectl delete -f "${ROOT_DIR}/20-mps-daemon.yaml" --ignore-not-found=true || true
kubectl delete -f "${ROOT_DIR}/10-qwen-sequential.yaml" --ignore-not-found=true || true

echo "Deleted Qwen benchmark resources from ${ROOT_DIR}. The shared namespace and PVC were left untouched."