#!/usr/bin/env bash
set -euo pipefail

NS="gpu-share-bench"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${ROOT_DIR}/results/$(date -u +%Y%m%dT%H%M%SZ)"
WITH_HAMI_MPS="false"

if [[ "${1:-}" == "--with-hami-mps" ]]; then
  WITH_HAMI_MPS="true"
fi

mkdir -p "${OUT_DIR}"

apply_base() {
  kubectl apply -f "${ROOT_DIR}/00-namespace.yaml"
  kubectl apply -f "${ROOT_DIR}/01-benchmark-configmap.yaml"
}

delete_job_if_exists() {
  local job_name="$1"
  kubectl delete job -n "${NS}" "${job_name}" --ignore-not-found=true
}

wait_for_job() {
  local job_name="$1"
  local timeout="${2:-900s}"
  kubectl wait -n "${NS}" --for=condition=complete "job/${job_name}" --timeout="${timeout}"
}

collect_job() {
  local job_name="$1"
  local label="$2"
  kubectl get job -n "${NS}" "${job_name}" -o wide > "${OUT_DIR}/${label}.job.txt" || true
  kubectl get pods -n "${NS}" -l "job-name=${job_name}" -o wide > "${OUT_DIR}/${label}.pods.txt" || true
  kubectl logs -n "${NS}" "job/${job_name}" --all-containers=true > "${OUT_DIR}/${label}.logs.txt" || true
}

collect_cluster_state() {
  kubectl get nodes -o wide > "${OUT_DIR}/nodes.txt" || true
  kubectl describe nodes > "${OUT_DIR}/nodes.describe.txt" || true
  kubectl get pods -n "${NS}" -o wide > "${OUT_DIR}/pods.all.txt" || true
}

run_job() {
  local manifest="$1"
  local job_name="$2"
  local label="$3"

  delete_job_if_exists "${job_name}"
  kubectl apply -f "${ROOT_DIR}/${manifest}"
  wait_for_job "${job_name}"
  collect_job "${job_name}" "${label}"
}

echo "Writing benchmark output to ${OUT_DIR}"
apply_base
collect_cluster_state

echo "Running sequential baseline..."
run_job "10-sequential-job.yaml" "gpu-sequential-benchmark" "10-sequential"

echo "Starting MPS daemon..."
kubectl apply -f "${ROOT_DIR}/20-mps-daemon.yaml"
kubectl rollout status -n "${NS}" daemonset/gpu-bench-mps-daemon --timeout=180s

echo "Running MPS benchmark..."
run_job "21-mps-parallel-job.yaml" "gpu-mps-benchmark" "21-mps"

echo "Running HAMi benchmark..."
run_job "30-hami-parallel-job.yaml" "gpu-hami-benchmark" "30-hami"

echo "Running HAMi OOM isolation test..."
run_job "31-hami-oom-job.yaml" "gpu-hami-oom-test" "31-hami-oom"

if [[ "${WITH_HAMI_MPS}" == "true" ]]; then
  echo "Running optional HAMi + MPS benchmark..."
  run_job "40-hami-plus-mps-job.yaml" "gpu-hami-mps-benchmark" "40-hami-plus-mps"
fi

collect_cluster_state
echo "Benchmark complete. Results: ${OUT_DIR}"
