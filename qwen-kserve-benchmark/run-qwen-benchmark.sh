#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_ROOT="${ROOT_DIR}/results"
NS="llm-demo"
PVC="qwen-local-pv-claim"
GATEWAY_NS="kserve"
GATEWAY_NAME="kserve-ingress-gateway"
MODEL_NAME="Qwen/Qwen2.5-0.5B-Instruct"
SERVICE_1="qwen-bench-01"
SERVICE_2="qwen-bench-02"
REQUESTS_PER_SERVICE=10
WARMUP_REQUESTS=1
TIMEOUT_SECONDS=180
WAIT_TIMEOUT_SECONDS=1800
MODE="all"
HAMI_PROFILE="balanced"
WITH_HAMI_MPS="false"
ENDPOINT_1=""
ENDPOINT_2=""
INSECURE="false"

usage() {
  cat <<'EOF'
Usage: ./run-qwen-benchmark.sh [options]

Options:
  --mode MODE                all, sequential, mps, hami, hami-mps
  --hami-profile PROFILE     balanced, conservative, weighted
  --with-hami-mps            include the HAMi + MPS mode when --mode all
  --endpoint-1 URL           explicit service 1 endpoint
  --endpoint-2 URL           explicit service 2 endpoint
  --requests-per-service N   measured requests per service (default: 10)
  --timeout SECONDS          request timeout passed to client.py (default: 180)
  --wait-timeout SECONDS     readiness timeout for each service (default: 1800)
  --insecure                 disable TLS verification in the client
EOF
}

log() {
  printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*"
}

fail() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || fail "Missing required command: $1"
}

run_python() {
  python3 "$@"
}

resource_type="$(kubectl api-resources --api-group=serving.kserve.io -o name | grep -Ei 'llminferenceservices|inferenceservices' | head -n 1 || true)"

delete_all_resources() {
  kubectl delete -f "${ROOT_DIR}/40-qwen-hami-mps.yaml" --ignore-not-found=true >/dev/null 2>&1 || true
  kubectl delete -f "${ROOT_DIR}/31-qwen-hami-compute-profiles.yaml" --ignore-not-found=true >/dev/null 2>&1 || true
  kubectl delete -f "${ROOT_DIR}/30-qwen-hami.yaml" --ignore-not-found=true >/dev/null 2>&1 || true
  kubectl delete -f "${ROOT_DIR}/21-qwen-mps.yaml" --ignore-not-found=true >/dev/null 2>&1 || true
  kubectl delete -f "${ROOT_DIR}/20-mps-daemon.yaml" --ignore-not-found=true >/dev/null 2>&1 || true
  kubectl delete -f "${ROOT_DIR}/10-qwen-sequential.yaml" --ignore-not-found=true >/dev/null 2>&1 || true
}

validate_common_cluster_state() {
  kubectl get namespace "${NS}" >/dev/null
  kubectl get pvc "${PVC}" -n "${NS}" >/dev/null
  kubectl get gateway "${GATEWAY_NAME}" -n "${GATEWAY_NS}" >/dev/null

  if [[ -z "${resource_type}" ]]; then
    fail "Could not find the KServe LLMInferenceService resource type"
  fi

  local gpu_node_count
  gpu_node_count="$(kubectl get nodes -o json | run_python -c '
import json, sys
data = json.load(sys.stdin)
count = 0
for node in data.get("items", []):
    status = node.get("status", {})
    allocatable = status.get("allocatable", {})
    ready = any(
        cond.get("type") == "Ready" and str(cond.get("status")).lower() == "true"
        for cond in status.get("conditions", [])
    )
    if ready and int(allocatable.get("nvidia.com/gpu", "0") or 0) > 0:
        count += 1
print(count)
')"
  if [[ "${gpu_node_count}" -lt 1 ]]; then
    fail "No Ready GPU node with allocatable nvidia.com/gpu found"
  fi
}

validate_hami_components() {
  local hami_hits
  hami_hits="$(kubectl get all -A -o name 2>/dev/null | grep -Ei 'hami' || true)"
  if [[ -z "${hami_hits}" ]]; then
    fail "HAMi components were not detected in the cluster"
  fi
}

apply_mode_manifests() {
  local mode="$1"

  delete_all_resources

  case "${mode}" in
    sequential)
      kubectl apply -f "${ROOT_DIR}/10-qwen-sequential.yaml"
      ;;
    mps)
      kubectl apply -f "${ROOT_DIR}/20-mps-daemon.yaml"
      kubectl rollout status -n "${NS}" daemonset/qwen-benchmark-mps-daemon --timeout=300s
      kubectl apply -f "${ROOT_DIR}/21-qwen-mps.yaml"
      ;;
    hami)
      kubectl apply -f "${ROOT_DIR}/31-qwen-hami-compute-profiles.yaml"
      kubectl apply -f "${ROOT_DIR}/30-qwen-hami.yaml"
      ;;
    hami-mps)
      kubectl apply -f "${ROOT_DIR}/20-mps-daemon.yaml"
      kubectl rollout status -n "${NS}" daemonset/qwen-benchmark-mps-daemon --timeout=300s
      kubectl apply -f "${ROOT_DIR}/31-qwen-hami-compute-profiles.yaml"
      kubectl apply -f "${ROOT_DIR}/40-qwen-hami-mps.yaml"
      ;;
    *)
      fail "Unsupported mode: ${mode}"
      ;;
  esac

  if [[ "${mode}" == hami || "${mode}" == hami-mps ]]; then
    kubectl label "${resource_type}/${SERVICE_1}" -n "${NS}" qwen-hami-profile="${HAMI_PROFILE}" --overwrite
    kubectl label "${resource_type}/${SERVICE_2}" -n "${NS}" qwen-hami-profile="${HAMI_PROFILE}" --overwrite
  fi
}

service_url() {
  local service="$1"
  kubectl get "${resource_type}/${service}" -n "${NS}" -o jsonpath='{.status.url}' 2>/dev/null || true
}

service_ready() {
  local service="$1"
  local payload
  payload="$(kubectl get "${resource_type}/${service}" -n "${NS}" -o json 2>/dev/null || true)"
  if [[ -z "${payload}" ]]; then
    return 1
  fi

  printf '%s' "${payload}" | run_python -c '
import json, sys
data = json.load(sys.stdin)
ready = any(
    cond.get("type") == "Ready" and str(cond.get("status")).lower() == "true"
    for cond in data.get("status", {}).get("conditions", [])
)
sys.exit(0 if ready else 1)
'
}

wait_for_service() {
  local service="$1"
  local start_epoch="$2"
  local timeout_seconds="$3"
  local deadline=$((start_epoch + timeout_seconds))

  while true; do
    if service_ready "${service}"; then
      return 0
    fi

    if (( $(date +%s) >= deadline )); then
      return 1
    fi

    sleep 5
  done
}

select_pod_name() {
  local service="$1"
  local payload
  payload="$(kubectl get pods -n "${NS}" -o json 2>/dev/null || true)"
  if [[ -z "${payload}" ]]; then
    return 1
  fi

  SERVICE_NAME="${service}" run_python -c '
import json, sys
import os

service = os.environ.get("SERVICE_NAME", "")
data = json.load(sys.stdin)
candidates = []
for item in data.get("items", []):
    metadata = item.get("metadata", {})
    name = metadata.get("name", "")
    labels = metadata.get("labels", {})
    score = 0
    if labels.get("serving.kserve.io/inferenceservice") == service:
        score += 100
    if labels.get("serving.knative.dev/service") == service:
        score += 80
    if service in name:
        score += 20
    if any(service in str(value) for value in labels.values()):
        score += 5
    if score:
        candidates.append((score, name))
if candidates:
    candidates.sort(reverse=True)
    print(candidates[0][1])
' <<< "${payload}"
}

collect_service_artifacts() {
  local service="$1"
  local mode_dir="$2"
  local service_dir="${mode_dir}/${service}"
  mkdir -p "${service_dir}"

  kubectl get "${resource_type}/${service}" -n "${NS}" -o yaml > "${service_dir}/service.yaml" || true
  kubectl get "${resource_type}/${service}" -n "${NS}" -o json > "${service_dir}/service.json" || true
  kubectl get "${resource_type}/${service}" -n "${NS}" -o wide > "${service_dir}/service.wide.txt" || true

  local pod_name
  pod_name="$(select_pod_name "${service}")"
  if [[ -n "${pod_name}" ]]; then
    kubectl get pod "${pod_name}" -n "${NS}" -o yaml > "${service_dir}/pod.yaml" || true
    kubectl get pod "${pod_name}" -n "${NS}" -o json > "${service_dir}/pod.json" || true
    kubectl get pod "${pod_name}" -n "${NS}" -o wide > "${service_dir}/pod.wide.txt" || true
    kubectl describe pod "${pod_name}" -n "${NS}" > "${service_dir}/pod.describe.txt" || true
    kubectl logs "${pod_name}" -n "${NS}" --all-containers=true > "${service_dir}/pod.logs.txt" || true
    kubectl logs "${pod_name}" -n "${NS}" --all-containers=true --previous > "${service_dir}/pod.previous-logs.txt" || true
    kubectl exec -n "${NS}" "${pod_name}" -c main -- nvidia-smi --query-gpu=memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits > "${service_dir}/gpu-smi.txt" 2>"${service_dir}/gpu-smi.err" || true
    kubectl exec -n "${NS}" "${pod_name}" -c main -- nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits > "${service_dir}/gpu-processes.txt" 2>"${service_dir}/gpu-processes.err" || true
  fi

  kubectl get events -n "${NS}" --sort-by=.lastTimestamp > "${service_dir}/events.txt" || true

  run_python -c '
import json, pathlib, sys
service_dir = pathlib.Path(sys.argv[1])
summary_path = service_dir / "pod-summary.json"
pod_path = service_dir / "pod.json"
summary = {
    "pod_name": "",
    "node_name": "",
    "phase": "",
    "start_time": "",
    "restart_count": 0,
    "oom_killed": 0,
    "annotations": {},
    "labels": {},
}
if pod_path.exists():
    pod = json.loads(pod_path.read_text(encoding="utf-8"))
    metadata = pod.get("metadata", {})
    status = pod.get("status", {})
    summary["pod_name"] = metadata.get("name", "")
    summary["node_name"] = pod.get("spec", {}).get("nodeName", "")
    summary["phase"] = status.get("phase", "")
    summary["start_time"] = status.get("startTime", "")
    summary["annotations"] = metadata.get("annotations", {})
    summary["labels"] = metadata.get("labels", {})
    restart_count = 0
    oom_killed = 0
    for container in status.get("containerStatuses", []):
        restart_count += int(container.get("restartCount", 0) or 0)
        last_state = container.get("lastState", {})
        terminated = last_state.get("terminated", {})
        if terminated.get("reason") == "OOMKilled":
            oom_killed += 1
        state = container.get("state", {})
        current = state.get("terminated", {})
        if current.get("reason") == "OOMKilled":
            oom_killed += 1
    summary["restart_count"] = restart_count
    summary["oom_killed"] = oom_killed
summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
' "${service_dir}"
}

collect_mode_snapshot() {
  local mode="$1"
  local mode_dir="$2"
  mkdir -p "${mode_dir}"
  kubectl get nodes -o wide > "${mode_dir}/nodes.txt" || true
  kubectl get pods -n "${NS}" -o wide > "${mode_dir}/pods.txt" || true
  kubectl get daemonset -n "${NS}" -o wide > "${mode_dir}/daemonsets.txt" || true
  kubectl get "${resource_type}" -n "${NS}" -o wide > "${mode_dir}/services.txt" || true
}

client_summary_path=""

run_client() {
  local mode="$1"
  local mode_dir="$2"
  local endpoint_1="$3"
  local endpoint_2="$4"
  local sequential_flag="$5"

  local client_json="${mode_dir}/client.json"
  local client_csv="${mode_dir}/client.csv"
  local stdout_file="${mode_dir}/client.stdout.txt"

  local resolved_endpoint_1 resolved_endpoint_2
  resolved_endpoint_1="${endpoint_1:-$(service_url "${SERVICE_1}")}" 
  resolved_endpoint_2="${endpoint_2:-$(service_url "${SERVICE_2}")}" 
  if [[ -z "${resolved_endpoint_1}" || -z "${resolved_endpoint_2}" ]]; then
    fail "Could not resolve service endpoints for ${mode}"
  fi

  local endpoint_args=()
  if [[ -n "${endpoint_1}" ]]; then
    endpoint_args+=("--endpoint=${SERVICE_1}=${endpoint_1}")
  else
    endpoint_args+=("--endpoint=${SERVICE_1}=${resolved_endpoint_1}")
  fi
  if [[ -n "${endpoint_2}" ]]; then
    endpoint_args+=("--endpoint=${SERVICE_2}=${endpoint_2}")
  else
    endpoint_args+=("--endpoint=${SERVICE_2}=${resolved_endpoint_2}")
  fi

  local client_cmd=(python3 "${ROOT_DIR}/client.py" --mode "${mode}" --output "${client_json}" --requests-per-service "${REQUESTS_PER_SERVICE}" --timeout "${TIMEOUT_SECONDS}")
  if [[ "${sequential_flag}" == "true" ]]; then
    client_cmd+=(--sequential)
  fi
  if [[ "${INSECURE}" == "true" ]]; then
    client_cmd+=(--insecure)
  fi
  client_cmd+=("${endpoint_args[@]}")

  "${client_cmd[@]}" | tee "${stdout_file}"
  cp "${client_json}" "${mode_dir}/client.result.json"
  cp "${client_json%.json}.csv" "${client_csv}"
  client_summary_path="${client_json}"
}

collect_gpu_snapshot() {
  local mode_dir="$1"
  local pod_name="$2"
  local gpu_file="${mode_dir}/gpu-snapshot.txt"
  local gpu_process_file="${mode_dir}/gpu-processes.txt"
  if [[ -n "${pod_name}" ]]; then
    kubectl exec -n "${NS}" "${pod_name}" -c main -- nvidia-smi --query-gpu=memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits > "${gpu_file}" 2>"${mode_dir}/gpu-snapshot.err" || true
    kubectl exec -n "${NS}" "${pod_name}" -c main -- nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits > "${gpu_process_file}" 2>"${mode_dir}/gpu-processes.err" || true
  fi
}

write_mode_summary() {
  local mode="$1"
  local mode_dir="$2"
  local sequential_flag="$3"
  local endpoint_1="$4"
  local endpoint_2="$5"
  local ready_1="$6"
  local ready_2="$7"
  local startup_1="$8"
  local startup_2="$9"
  local requests_per_service="${10}"
  local warmup_requests_per_service="${11}"
  local pod_1="${12}"
  local pod_2="${13}"

  run_python -c '
import json
import pathlib
import sys

mode = sys.argv[1]
mode_dir = pathlib.Path(sys.argv[2])
sequential = sys.argv[3] == "true"
endpoint_1 = sys.argv[4]
endpoint_2 = sys.argv[5]
ready_1 = sys.argv[6] == "true"
ready_2 = sys.argv[7] == "true"
startup_1 = float(sys.argv[8]) if sys.argv[8] else None
startup_2 = float(sys.argv[9]) if sys.argv[9] else None
pod_1 = sys.argv[10]
pod_2 = sys.argv[11]

client = json.loads((mode_dir / "client.result.json").read_text(encoding="utf-8"))
services = {}
for service in ("qwen-bench-01", "qwen-bench-02"):
    service_dir = mode_dir / service
    pod_summary_path = service_dir / "pod-summary.json"
    pod_summary = json.loads(pod_summary_path.read_text(encoding="utf-8")) if pod_summary_path.exists() else {}
    service_status_path = service_dir / "service.json"
    service_status = json.loads(service_status_path.read_text(encoding="utf-8")) if service_status_path.exists() else {}
    services[service] = {
        "endpoint": endpoint_1 if service == "qwen-bench-01" else endpoint_2,
        "ready": ready_1 if service == "qwen-bench-01" else ready_2,
        "startup_seconds": startup_1 if service == "qwen-bench-01" else startup_2,
        "pod": pod_summary,
        "resource_status": service_status.get("status", {}),
        "pod_name": pod_1 if service == "qwen-bench-01" else pod_2,
    }

gpu_snapshot_path = mode_dir / "gpu-snapshot.txt"
gpu_snapshot = None
gpu_memory_mib = None
gpu_utilization = None
if gpu_snapshot_path.exists():
    lines = [line.strip() for line in gpu_snapshot_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    gpu_snapshot = lines[0] if lines else ""
    parts = [part.strip() for part in gpu_snapshot.split(",")]
    if len(parts) >= 3:
        try:
            gpu_memory_mib = int(float(parts[1]))
        except ValueError:
            gpu_memory_mib = None
        try:
            gpu_utilization = int(float(parts[2]))
        except ValueError:
            gpu_utilization = None

oom_events = sum(int(services[name]["pod"].get("oom_killed", 0) or 0) for name in services)
restart_count = sum(int(services[name]["pod"].get("restart_count", 0) or 0) for name in services)
acceptance = {
    "both_ready": ready_1 and ready_2,
    "both_processed_ten_requests": client.get("summary", {}).get("successful_requests", 0) == 20,
    "failed_requests": client.get("summary", {}).get("failed_requests", 0),
    "startup_oom_events": oom_events,
    "pod_restarts": restart_count,
    "gpu_memory_used_mib": gpu_memory_mib,
    "gpu_utilization_percent": gpu_utilization,
    "within_12gib": gpu_memory_mib is not None and gpu_memory_mib <= 12288,
}

summary = {
    "mode": mode,
    "sequential": sequential,
  "requests_per_service": int(sys.argv[10]),
  "warmup_requests_per_service": int(sys.argv[11]),
    "endpoints": {
        "qwen-bench-01": endpoint_1,
        "qwen-bench-02": endpoint_2,
    },
    "services": services,
    "client": client,
    "gpu_snapshot": gpu_snapshot,
    "acceptance": acceptance,
}

(mode_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

print(json.dumps({
    "mode": mode,
    "client_successful_requests": client.get("summary", {}).get("successful_requests", 0),
    "client_failed_requests": client.get("summary", {}).get("failed_requests", 0),
    "both_ready": acceptance["both_ready"],
    "within_12gib": acceptance["within_12gib"],
    "gpu_memory_used_mib": gpu_memory_mib,
}, indent=2))
' "${mode}" "${mode_dir}" "${sequential_flag}" "${endpoint_1}" "${endpoint_2}" "${ready_1}" "${ready_2}" "${startup_1}" "${startup_2}" "${REQUESTS_PER_SERVICE}" "${WARMUP_REQUESTS}" "${pod_1}" "${pod_2}"
}

run_mode() {
  local mode="$1"
  local endpoint_1="$2"
  local endpoint_2="$3"
  local sequential_flag="$4"

  local mode_dir="${RESULTS_ROOT}/${timestamp}/${mode}"
  mkdir -p "${mode_dir}"

  log "Preparing ${mode} mode"
  collect_mode_snapshot "${mode}" "${mode_dir}"
  apply_mode_manifests "${mode}"

  local apply_started ready_1 ready_2 ready_elapsed_1 ready_elapsed_2 pod_1 pod_2
  apply_started="$(run_python -c 'import time; print(time.time())')"

  log "Waiting for ${SERVICE_1}"
  if wait_for_service "${SERVICE_1}" "${apply_started%.*}" "${WAIT_TIMEOUT_SECONDS}"; then
    ready_1="true"
    ready_elapsed_1="$(run_python -c "import time; print(round(time.time() - float(${apply_started}), 3))")"
  else
    ready_1="false"
    ready_elapsed_1=""
  fi

  log "Waiting for ${SERVICE_2}"
  if wait_for_service "${SERVICE_2}" "${apply_started%.*}" "${WAIT_TIMEOUT_SECONDS}"; then
    ready_2="true"
    ready_elapsed_2="$(run_python -c "import time; print(round(time.time() - float(${apply_started}), 3))")"
  else
    ready_2="false"
    ready_elapsed_2=""
  fi

  pod_1="$(select_pod_name "${SERVICE_1}")"
  pod_2="$(select_pod_name "${SERVICE_2}")"

  local resolved_endpoint_1 resolved_endpoint_2
  resolved_endpoint_1="${endpoint_1:-$(service_url "${SERVICE_1}")}" 
  resolved_endpoint_2="${endpoint_2:-$(service_url "${SERVICE_2}")}" 
  if [[ -z "${resolved_endpoint_1}" || -z "${resolved_endpoint_2}" ]]; then
    log "Service URLs were not available from status; using explicit endpoints is recommended"
  fi

  if [[ "${ready_1}" != "true" || "${ready_2}" != "true" ]]; then
    log "One or both services did not become Ready"
    collect_service_artifacts "${SERVICE_1}" "${mode_dir}"
    collect_service_artifacts "${SERVICE_2}" "${mode_dir}"
    write_mode_summary "${mode}" "${mode_dir}" "${sequential_flag}" "${endpoint_1}" "${endpoint_2}" "${ready_1}" "${ready_2}" "${ready_elapsed_1}" "${ready_elapsed_2}" "${REQUESTS_PER_SERVICE}" "${WARMUP_REQUESTS}" "${pod_1}" "${pod_2}" || true
    return 1
  fi

  log "Running client workload for ${mode}"
  local client_status=0
  run_client "${mode}" "${mode_dir}" "${resolved_endpoint_1}" "${resolved_endpoint_2}" "${sequential_flag}" || client_status=$?

  collect_service_artifacts "${SERVICE_1}" "${mode_dir}"
  collect_service_artifacts "${SERVICE_2}" "${mode_dir}"
  collect_gpu_snapshot "${mode_dir}" "${pod_1:-${pod_2}}"
  write_mode_summary "${mode}" "${mode_dir}" "${sequential_flag}" "${resolved_endpoint_1}" "${resolved_endpoint_2}" "${ready_1}" "${ready_2}" "${ready_elapsed_1}" "${ready_elapsed_2}" "${REQUESTS_PER_SERVICE}" "${WARMUP_REQUESTS}" "${pod_1}" "${pod_2}"

  log "Completed ${mode}. Results are in ${mode_dir}"
  if [[ "${client_status}" -ne 0 ]]; then
    return "${client_status}"
  fi
}

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="$2"
      shift 2
      ;;
    --hami-profile)
      HAMI_PROFILE="$2"
      shift 2
      ;;
    --with-hami-mps)
      WITH_HAMI_MPS="true"
      shift
      ;;
    --endpoint-1)
      ENDPOINT_1="$2"
      shift 2
      ;;
    --endpoint-2)
      ENDPOINT_2="$2"
      shift 2
      ;;
    --requests-per-service)
      REQUESTS_PER_SERVICE="$2"
      shift 2
      ;;
    --timeout)
      TIMEOUT_SECONDS="$2"
      shift 2
      ;;
    --wait-timeout)
      WAIT_TIMEOUT_SECONDS="$2"
      shift 2
      ;;
    --insecure)
      INSECURE="true"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      fail "Unknown argument: $1"
      ;;
  esac
done

require_cmd kubectl
require_cmd python3

mkdir -p "${RESULTS_ROOT}/${timestamp}"

validate_common_cluster_state

case "${MODE}" in
  all|sequential|mps|hami|hami-mps)
    ;;
  *)
    fail "Unsupported mode: ${MODE}"
    ;;
esac

if [[ "${MODE}" == hami || "${MODE}" == hami-mps || ( "${MODE}" == all && "${WITH_HAMI_MPS}" == "true" ) ]]; then
  validate_hami_components
fi

log "Writing benchmark results to ${RESULTS_ROOT}/${timestamp}"

overall_status=0

if [[ "${MODE}" == all || "${MODE}" == sequential ]]; then
  if ! run_mode sequential "${ENDPOINT_1}" "${ENDPOINT_2}" true; then
    overall_status=1
  fi
fi

if [[ "${MODE}" == all || "${MODE}" == mps ]]; then
  if ! run_mode mps "${ENDPOINT_1}" "${ENDPOINT_2}" false; then
    overall_status=1
  fi
fi

if [[ "${MODE}" == all || "${MODE}" == hami ]]; then
  if ! run_mode hami "${ENDPOINT_1}" "${ENDPOINT_2}" false; then
    overall_status=1
  fi
fi

if [[ "${MODE}" == hami-mps ]]; then
  if ! run_mode hami-mps "${ENDPOINT_1}" "${ENDPOINT_2}" false; then
    overall_status=1
  fi
fi

if [[ "${MODE}" == all && "${WITH_HAMI_MPS}" == "true" ]]; then
  if ! run_mode hami-mps "${ENDPOINT_1}" "${ENDPOINT_2}" false; then
    overall_status=1
  fi
fi

log "Benchmark run complete"
exit "${overall_status}"