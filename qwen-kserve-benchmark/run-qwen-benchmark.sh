#!/usr/bin/env bash
#
# Two-container Qwen KServe GPU-sharing benchmark.
#
# Modes:
#   sequential  one service deployed at a time, full GPU, no sharing. Baseline.
#   mps         both services concurrent, NVIDIA MPS.
#   hami        both services concurrent, HAMi gpumem/gpucores quotas.
#   hami-mps    both services concurrent, HAMi quotas plus MPS.
#
# pipefail matters here: without it the `client | tee` pipeline reports tee's
# exit status and every failed request is silently counted as a success.
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
DRAIN_TIMEOUT_SECONDS=300
GPU_SAMPLE_INTERVAL=2
MODE="all"
HAMI_PROFILE="balanced"
WITH_HAMI_MPS="false"
ENDPOINT_1=""
ENDPOINT_2=""
INSECURE="false"
resource_type=""
gpu_probe_pod=""
sampler_pid=""
timestamp=""

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
  --warmup N                 warm-up requests per service (default: 1)
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

# ---------------------------------------------------------------------------
# Argument parsing. This has to happen before anything creates directories or
# talks to the cluster: previously the results directory and the tee log were
# set up first, so `--help` left an empty run directory behind and a typo in
# --mode was only caught after the cluster had already been queried.
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)                  MODE="$2"; shift 2 ;;
    --hami-profile)          HAMI_PROFILE="$2"; shift 2 ;;
    --with-hami-mps)         WITH_HAMI_MPS="true"; shift ;;
    --endpoint-1)            ENDPOINT_1="$2"; shift 2 ;;
    --endpoint-2)            ENDPOINT_2="$2"; shift 2 ;;
    --requests-per-service)  REQUESTS_PER_SERVICE="$2"; shift 2 ;;
    --warmup)                WARMUP_REQUESTS="$2"; shift 2 ;;
    --timeout)               TIMEOUT_SECONDS="$2"; shift 2 ;;
    --wait-timeout)          WAIT_TIMEOUT_SECONDS="$2"; shift 2 ;;
    --insecure)              INSECURE="true"; shift ;;
    -h|--help)               usage; exit 0 ;;
    *)                       usage >&2; fail "Unknown argument: $1" ;;
  esac
done

case "${MODE}" in
  all|sequential|mps|hami|hami-mps) ;;
  *) fail "Unsupported mode: ${MODE}" ;;
esac

case "${HAMI_PROFILE}" in
  balanced|conservative|weighted) ;;
  *) fail "Unsupported HAMi profile: ${HAMI_PROFILE}" ;;
esac

require_cmd kubectl
require_cmd python3

# ---------------------------------------------------------------------------
# Cluster discovery and validation
# ---------------------------------------------------------------------------
detect_resource_type() {
  # Match the LLM CRD exactly. The old loose grep also matched plain
  # `inferenceservices`, and `head -n 1` would then pick the wrong CRD, so
  # every later kubectl get/label/patch silently targeted the wrong resource.
  resource_type="$(kubectl api-resources --api-group=serving.kserve.io -o name 2>/dev/null \
    | grep -E '^llminferenceservices(\.|$)' | head -n 1 || true)"
  if [[ -z "${resource_type}" ]]; then
    fail "Could not find the KServe LLMInferenceService CRD (serving.kserve.io)"
  fi
}

validate_common_cluster_state() {
  kubectl get namespace "${NS}" >/dev/null || fail "Namespace ${NS} not found"
  kubectl get pvc "${PVC}" -n "${NS}" >/dev/null || fail "PVC ${PVC} not found in ${NS}"
  kubectl get gateway "${GATEWAY_NAME}" -n "${GATEWAY_NS}" >/dev/null \
    || fail "Gateway ${GATEWAY_NAME} not found in ${GATEWAY_NS}"

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
  hami_hits="$(kubectl get pods -A -o name 2>/dev/null | grep -Ei 'hami' || true)"
  if [[ -z "${hami_hits}" ]]; then
    fail "HAMi components were not detected in the cluster"
  fi
}

# ---------------------------------------------------------------------------
# Teardown.
#
# Deleting an LLMInferenceService returns immediately, but the vLLM Pod holds
# its CUDA context and its VRAM until it actually terminates. The old runner
# applied the next mode's manifests straight after the delete, so the incoming
# Pods raced the outgoing ones for the same GPU. That is the direct cause of
# the "Insufficient memory" / crash-loop runs in the captured logs.
# ---------------------------------------------------------------------------
delete_all_resources() {
  local file
  for file in 40-qwen-hami-mps.yaml 31-qwen-hami-compute-profiles.yaml 30-qwen-hami.yaml \
              21-qwen-mps.yaml 10-qwen-sequential-01.yaml 10-qwen-sequential-02.yaml \
              20-mps-daemon.yaml 05-gpu-probe.yaml; do
    kubectl delete -f "${ROOT_DIR}/${file}" --ignore-not-found=true --wait=false >/dev/null 2>&1 || true
  done
}

wait_for_workload_drain() {
  local deadline=$(( $(date +%s) + DRAIN_TIMEOUT_SECONDS ))
  local announced="false"
  while true; do
    local remaining
    remaining="$(kubectl get pods -n "${NS}" --no-headers 2>/dev/null \
      | grep -cE "^(${SERVICE_1}|${SERVICE_2})-kserve" || true)"
    remaining="${remaining:-0}"
    if [[ "${remaining}" -eq 0 ]]; then
      if [[ "${announced}" == "true" ]]; then
        log "All benchmark workload Pods are gone; GPU released"
      fi
      return 0
    fi
    if [[ "${announced}" == "false" ]]; then
      log "Waiting for ${remaining} workload Pod(s) to terminate and release the GPU"
      announced="true"
    fi
    if (( $(date +%s) >= deadline )); then
      log "WARNING: ${remaining} workload Pod(s) still terminating after ${DRAIN_TIMEOUT_SECONDS}s"
      return 1
    fi
    sleep 5
  done
}

reset_cluster_state() {
  delete_all_resources
  wait_for_workload_drain || true
}

# ---------------------------------------------------------------------------
# HAMi profiles.
#
# The ConfigMap used to be decorative: --hami-profile only stamped a label on
# the service and never changed a quota, so `weighted` produced exactly the
# same 50/50 split as `balanced`. Patch the real resource values instead.
# ---------------------------------------------------------------------------
patch_service_profile() {
  local service="$1" cores="$2" mem="$3" util="$4"

  # Rebuild the container args so --gpu-memory-utilization tracks the quota.
  # Under HAMi that flag is a fraction of the gpumem quota, not of the physical
  # device, so the two values must always change together.
  local args_json
  args_json="$(UTIL="${util}" run_python -c '
import json, os
util = os.environ["UTIL"]
print(json.dumps([
    f"--gpu-memory-utilization={util}",
    "--max-model-len=512",
    "--max-num-seqs=1",
    "--no-enable-prefix-caching",
]))
')"

  local patch
  patch="$(CORES="${cores}" MEM="${mem}" ARGS="${args_json}" run_python -c '
import json, os
cores = os.environ["CORES"]
mem = os.environ["MEM"]
args = json.loads(os.environ["ARGS"])
operations = [{"op": "replace", "path": "/spec/template/containers/0/args", "value": args}]
for section in ("limits", "requests"):
    # ~1 is the JSON Pointer escape for a literal / inside a key, which
    # resource names such as nvidia.com/gpucores contain.
    operations.append({
        "op": "replace",
        "path": f"/spec/template/containers/0/resources/{section}/nvidia.com~1gpucores",
        "value": cores,
    })
    operations.append({
        "op": "replace",
        "path": f"/spec/template/containers/0/resources/{section}/nvidia.com~1gpumem",
        "value": mem,
    })
print(json.dumps(operations))
')"

  kubectl patch "${resource_type}/${service}" -n "${NS}" --type=json -p "${patch}" >/dev/null
}

apply_hami_profile() {
  local profile="$1"
  local raw
  raw="$(kubectl get configmap qwen-hami-compute-profiles -n "${NS}" \
    -o "jsonpath={.data.${profile}}" 2>/dev/null || true)"
  if [[ -z "${raw}" ]]; then
    fail "HAMi profile '${profile}' not found in configmap qwen-hami-compute-profiles"
  fi

  local parsed
  parsed="$(printf '%s' "${raw}" | run_python -c '
import json, sys
profile = json.load(sys.stdin)
cores = profile["cores"]
mem = profile["mem"]
util = profile.get("util", [0.90, 0.90])
print(" ".join(str(value) for value in (cores[0], cores[1], mem[0], mem[1], util[0], util[1])))
')"
  local cores_1 cores_2 mem_1 mem_2 util_1 util_2
  read -r cores_1 cores_2 mem_1 mem_2 util_1 util_2 <<< "${parsed}"

  log "HAMi profile '${profile}':"
  log "  ${SERVICE_1}: gpucores=${cores_1} gpumem=${mem_1}MiB vllm-util=${util_1}"
  log "  ${SERVICE_2}: gpucores=${cores_2} gpumem=${mem_2}MiB vllm-util=${util_2}"

  patch_service_profile "${SERVICE_1}" "${cores_1}" "${mem_1}" "${util_1}"
  patch_service_profile "${SERVICE_2}" "${cores_2}" "${mem_2}" "${util_2}"

  kubectl label "${resource_type}/${SERVICE_1}" -n "${NS}" qwen-hami-profile="${profile}" --overwrite >/dev/null
  kubectl label "${resource_type}/${SERVICE_2}" -n "${NS}" qwen-hami-profile="${profile}" --overwrite >/dev/null
}

# ---------------------------------------------------------------------------
# Manifest application
# ---------------------------------------------------------------------------
apply_gpu_probe() {
  kubectl apply -f "${ROOT_DIR}/05-gpu-probe.yaml" >/dev/null
  kubectl rollout status -n "${NS}" daemonset/qwen-benchmark-gpu-probe --timeout=180s >/dev/null 2>&1 \
    || log "WARNING: GPU probe DaemonSet is not Ready; GPU samples may be missing"
  gpu_probe_pod="$(kubectl get pods -n "${NS}" -l app=qwen-benchmark-gpu-probe \
    -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)"
}

apply_mps_daemon() {
  kubectl apply -f "${ROOT_DIR}/20-mps-daemon.yaml" >/dev/null
  # The DaemonSet's readiness probe now checks for the MPS control socket, so a
  # successful rollout genuinely means MPS is accepting clients. Fail loudly:
  # running an "MPS mode" without MPS produces a plausible but meaningless
  # number that is indistinguishable from a real result.
  kubectl rollout status -n "${NS}" daemonset/qwen-benchmark-mps-daemon --timeout=300s \
    || fail "MPS daemon did not become Ready; refusing to run an MPS mode without MPS"
}

apply_mode_manifests() {
  local mode="$1"

  case "${mode}" in
    mps)
      apply_mps_daemon
      kubectl apply -f "${ROOT_DIR}/21-qwen-mps.yaml" >/dev/null
      ;;
    hami)
      kubectl apply -f "${ROOT_DIR}/31-qwen-hami-compute-profiles.yaml" >/dev/null
      kubectl apply -f "${ROOT_DIR}/30-qwen-hami.yaml" >/dev/null
      apply_hami_profile "${HAMI_PROFILE}"
      ;;
    hami-mps)
      apply_mps_daemon
      kubectl apply -f "${ROOT_DIR}/31-qwen-hami-compute-profiles.yaml" >/dev/null
      kubectl apply -f "${ROOT_DIR}/40-qwen-hami-mps.yaml" >/dev/null
      apply_hami_profile "${HAMI_PROFILE}"
      ;;
    *)
      fail "apply_mode_manifests does not handle mode: ${mode}"
      ;;
  esac
}

# ---------------------------------------------------------------------------
# Readiness
# ---------------------------------------------------------------------------
service_url() {
  kubectl get "${resource_type}/$1" -n "${NS}" -o jsonpath='{.status.url}' 2>/dev/null || true
}

service_ready() {
  local payload
  payload="$(kubectl get "${resource_type}/$1" -n "${NS}" -o json 2>/dev/null || true)"
  [[ -n "${payload}" ]] || return 1
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

# The previous version of this function embedded Python with broken
# indentation (an `if` body at the same level as the `if`), so it raised
# IndentationError on every heartbeat and never printed a diagnosis.
service_ready_condition() {
  local payload
  payload="$(kubectl get "${resource_type}/$1" -n "${NS}" -o json 2>/dev/null || true)"
  if [[ -z "${payload}" ]]; then
    printf 'unavailable'
    return 0
  fi
  printf '%s' "${payload}" | run_python -c '
import json, sys

data = json.load(sys.stdin)
ready_condition = None
for condition in data.get("status", {}).get("conditions", []):
    if condition.get("type") == "Ready":
        ready_condition = condition
        break

if ready_condition is None:
    print("Ready condition missing")
else:
    status = ready_condition.get("status", "")
    reason = ready_condition.get("reason", "")
    message = ready_condition.get("message", "")
    print(f"Ready={status} reason={reason} message={message}")
'
}

service_pod_snapshot() {
  kubectl get pods -n "${NS}" --no-headers 2>/dev/null | grep -E "^$1-kserve" || true
}

wait_for_service() {
  local service="$1"
  local start_epoch="$2"
  local timeout_seconds="$3"
  local deadline=$((start_epoch + timeout_seconds))
  local last_heartbeat=0

  while true; do
    if service_ready "${service}"; then
      log "${service} is Ready"
      return 0
    fi

    local now
    now=$(date +%s)

    if (( now - last_heartbeat >= 30 )); then
      local elapsed=$((now - start_epoch))
      log "Waiting for ${service}: ${elapsed}s elapsed / ${timeout_seconds}s timeout"
      log "  condition: $(service_ready_condition "${service}" || true)"
      local pod_snapshot
      pod_snapshot="$(service_pod_snapshot "${service}")"
      if [[ -n "${pod_snapshot}" ]]; then
        printf '%s\n' "${pod_snapshot}" | sed 's/^/    /'
      else
        log "  no Pods yet"
      fi
      last_heartbeat=${now}
    fi

    if (( now >= deadline )); then
      log "Timed out waiting for ${service} after ${timeout_seconds}s"
      return 1
    fi

    sleep 5
  done
}

# The workload Deployment is always named <service>-kserve. Match that
# directly: the old label-scoring heuristic also matched the
# <service>-kserve-router-scheduler Pod, so `nvidia-smi` was exec'd into a
# container that has no GPU and every GPU reading came back empty.
select_pod_name() {
  kubectl get pods -n "${NS}" \
    -o jsonpath="{range .items[*]}{.metadata.name}{'\n'}{end}" 2>/dev/null \
    | grep -E "^$1-kserve-" \
    | grep -v -- '-router-scheduler-' \
    | head -n 1 || true
}

# ---------------------------------------------------------------------------
# GPU sampling.
#
# Samples come from the privileged probe Pod so every mode is measured from the
# same host-level vantage point. Sampling inside a HAMi workload Pod would
# report the vGPU quota rather than the device, which would make the HAMi modes
# non-comparable with the others.
# ---------------------------------------------------------------------------
start_gpu_sampler() {
  local out_file="$1"
  sampler_pid=""
  if [[ -z "${gpu_probe_pod}" ]]; then
    log "No GPU probe Pod available; skipping GPU sampling"
    return 0
  fi

  printf 'unix_seconds,memory_total_mib,memory_used_mib,utilization_gpu_percent\n' > "${out_file}"
  (
    while true; do
      line="$(kubectl exec -n "${NS}" "${gpu_probe_pod}" -c probe -- \
        nvidia-smi --query-gpu=memory.total,memory.used,utilization.gpu \
        --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -d ' ' || true)"
      if [[ -n "${line}" ]]; then
        printf '%s,%s\n' "$(date -u +%s)" "${line}" >> "${out_file}"
      fi
      sleep "${GPU_SAMPLE_INTERVAL}"
    done
  ) &
  sampler_pid=$!
}

stop_gpu_sampler() {
  if [[ -n "${sampler_pid}" ]]; then
    kill "${sampler_pid}" >/dev/null 2>&1 || true
    wait "${sampler_pid}" 2>/dev/null || true
    sampler_pid=""
  fi
}

# ---------------------------------------------------------------------------
# Artifact collection
# ---------------------------------------------------------------------------
collect_service_artifacts() {
  local service="$1"
  local mode_dir="$2"
  local service_dir="${mode_dir}/${service}"
  mkdir -p "${service_dir}"

  kubectl get "${resource_type}/${service}" -n "${NS}" -o yaml > "${service_dir}/service.yaml" 2>/dev/null || true
  kubectl get "${resource_type}/${service}" -n "${NS}" -o json > "${service_dir}/service.json" 2>/dev/null || true

  local pod_name
  pod_name="$(select_pod_name "${service}")"
  if [[ -n "${pod_name}" ]]; then
    kubectl get pod "${pod_name}" -n "${NS}" -o json > "${service_dir}/pod.json" 2>/dev/null || true
    kubectl describe pod "${pod_name}" -n "${NS}" > "${service_dir}/pod.describe.txt" 2>/dev/null || true
    kubectl logs "${pod_name}" -n "${NS}" --all-containers=true --tail=2000 > "${service_dir}/pod.logs.txt" 2>/dev/null || true
    kubectl logs "${pod_name}" -n "${NS}" --all-containers=true --previous --tail=2000 > "${service_dir}/pod.previous-logs.txt" 2>/dev/null || true
    # Per-Pod view. Under HAMi this deliberately shows the virtualized quota,
    # which is how the quota gets verified; host-level truth is in
    # gpu-timeline.csv from the probe Pod.
    kubectl exec -n "${NS}" "${pod_name}" -c main -- nvidia-smi \
      --query-gpu=memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits \
      > "${service_dir}/gpu-smi.txt" 2>"${service_dir}/gpu-smi.err" || true
  fi

  kubectl get events -n "${NS}" --sort-by=.lastTimestamp > "${service_dir}/events.txt" 2>/dev/null || true

  SERVICE_DIR="${service_dir}" run_python -c '
import json, os, pathlib
service_dir = pathlib.Path(os.environ["SERVICE_DIR"])
pod_path = service_dir / "pod.json"
summary = {
    "pod_name": "", "node_name": "", "phase": "", "start_time": "",
    "restart_count": 0, "oom_killed": 0, "annotations": {}, "labels": {},
}
if pod_path.exists() and pod_path.stat().st_size:
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
        for state in (container.get("lastState", {}), container.get("state", {})):
            if state.get("terminated", {}).get("reason") == "OOMKilled":
                oom_killed += 1
    summary["restart_count"] = restart_count
    summary["oom_killed"] = oom_killed
(service_dir / "pod-summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
'
}

collect_mode_snapshot() {
  local mode_dir="$1"
  kubectl get nodes -o wide > "${mode_dir}/nodes.txt" 2>/dev/null || true
  kubectl get pods -n "${NS}" -o wide > "${mode_dir}/pods.txt" 2>/dev/null || true
  kubectl get daemonset -n "${NS}" -o wide > "${mode_dir}/daemonsets.txt" 2>/dev/null || true
  kubectl get "${resource_type}" -n "${NS}" -o wide > "${mode_dir}/services.txt" 2>/dev/null || true
}

# ---------------------------------------------------------------------------
# Client invocation.
#
# The old version did `cp "${client_json%.json}.csv" "${client_csv}"` where both
# expressions expanded to the same path, so `cp` aborted the whole run under
# `set -e` right after a successful workload.
# ---------------------------------------------------------------------------
run_client() {
  local mode="$1" output_json="$2" sequential_flag="$3"
  shift 3
  local endpoint_args=("$@")

  local stdout_file="${output_json%.json}.stdout.txt"
  local client_cmd=(python3 "${ROOT_DIR}/client.py"
    --mode "${mode}"
    --output "${output_json}"
    --requests-per-service "${REQUESTS_PER_SERVICE}"
    --warmup "${WARMUP_REQUESTS}"
    --timeout "${TIMEOUT_SECONDS}"
    --model "${MODEL_NAME}")
  # Note: `[[ cond ]] && cmd` as a standalone statement returns 1 when the
  # condition is false, which under `set -e` terminates the whole script.
  # Always use if/fi for optional steps here.
  if [[ "${sequential_flag}" == "true" ]]; then
    client_cmd+=(--sequential)
  fi
  if [[ "${INSECURE}" == "true" ]]; then
    client_cmd+=(--insecure)
  fi
  client_cmd+=("${endpoint_args[@]}")

  local status=0
  "${client_cmd[@]}" 2>&1 | tee "${stdout_file}" || status=$?
  return "${status}"
}

# ---------------------------------------------------------------------------
# Summary.
#
# Everything is passed through the environment. The old version passed 13
# positional arguments and then read sys.argv[10] and sys.argv[11] for BOTH the
# request counts and the Pod names, so every summary recorded the Pod names as
# "10" and "1", and the acceptance block hardcoded 20 successful requests
# regardless of --requests-per-service.
# ---------------------------------------------------------------------------
write_mode_summary() {
  MODE_NAME="$1" MODE_DIR="$2" SEQUENTIAL="$3" \
  EP1="$4" EP2="$5" READY1="$6" READY2="$7" STARTUP1="$8" STARTUP2="$9" \
  POD1="${10}" POD2="${11}" \
  REQ_PER_SVC="${REQUESTS_PER_SERVICE}" WARMUP="${WARMUP_REQUESTS}" \
  SVC1="${SERVICE_1}" SVC2="${SERVICE_2}" HAMI_PROFILE_NAME="${HAMI_PROFILE}" \
  run_python -c '
import json, os, pathlib

mode = os.environ["MODE_NAME"]
mode_dir = pathlib.Path(os.environ["MODE_DIR"])
svc1, svc2 = os.environ["SVC1"], os.environ["SVC2"]
req_per_svc = int(os.environ["REQ_PER_SVC"])
expected_total = req_per_svc * 2


def flag(name):
    return os.environ.get(name, "") == "true"


def number(name):
    try:
        return float(os.environ.get(name, ""))
    except ValueError:
        return None


client_path = mode_dir / "client.result.json"
client = {}
if client_path.exists() and client_path.stat().st_size:
    client = json.loads(client_path.read_text(encoding="utf-8"))

services = {}
for service, ep, ready, startup, pod in (
    (svc1, "EP1", "READY1", "STARTUP1", "POD1"),
    (svc2, "EP2", "READY2", "STARTUP2", "POD2"),
):
    service_dir = mode_dir / service
    pod_summary_path = service_dir / "pod-summary.json"
    service_status_path = service_dir / "service.json"
    pod_summary = {}
    if pod_summary_path.exists() and pod_summary_path.stat().st_size:
        pod_summary = json.loads(pod_summary_path.read_text(encoding="utf-8"))
    service_status = {}
    if service_status_path.exists() and service_status_path.stat().st_size:
        service_status = json.loads(service_status_path.read_text(encoding="utf-8")).get("status", {})
    services[service] = {
        "endpoint": os.environ.get(ep, ""),
        "ready": flag(ready),
        "startup_seconds": number(startup),
        "pod_name": os.environ.get(pod, ""),
        "pod": pod_summary,
        "resource_status": service_status,
    }

# Host-level GPU timeline collected during the measured window. Peak memory and
# mean utilization are the numbers that actually answer the project question:
# how much of the shared GPU each sharing mode manages to keep busy.
gpu = {
    "samples": 0, "memory_total_mib": None, "peak_memory_used_mib": None,
    "mean_memory_used_mib": None, "peak_utilization_percent": None,
    "mean_utilization_percent": None,
}
timeline_path = mode_dir / "gpu-timeline.csv"
if timeline_path.exists():
    used, util, total = [], [], None
    for line in timeline_path.read_text(encoding="utf-8").splitlines()[1:]:
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            total = int(float(parts[1]))
            used.append(int(float(parts[2])))
            util.append(int(float(parts[3])))
        except ValueError:
            continue
    if used:
        gpu = {
            "samples": len(used),
            "memory_total_mib": total,
            "peak_memory_used_mib": max(used),
            "mean_memory_used_mib": round(sum(used) / len(used), 1),
            "peak_utilization_percent": max(util) if util else None,
            "mean_utilization_percent": round(sum(util) / len(util), 1) if util else None,
        }

client_summary = client.get("summary", {})
oom_events = sum(int(entry["pod"].get("oom_killed", 0) or 0) for entry in services.values())
restart_count = sum(int(entry["pod"].get("restart_count", 0) or 0) for entry in services.values())
peak = gpu["peak_memory_used_mib"]
capacity = gpu["memory_total_mib"]

acceptance = {
    "both_ready": flag("READY1") and flag("READY2"),
    "expected_successful_requests": expected_total,
    "all_requests_succeeded": client_summary.get("successful_requests", 0) == expected_total,
    "failed_requests": client_summary.get("failed_requests", 0),
    "startup_oom_events": oom_events,
    "pod_restarts": restart_count,
    "peak_gpu_memory_used_mib": peak,
    "gpu_memory_total_mib": capacity,
    "within_gpu_capacity": peak is not None and capacity is not None and peak <= capacity,
}

summary = {
    "mode": mode,
    "sequential": flag("SEQUENTIAL"),
    "hami_profile": os.environ.get("HAMI_PROFILE_NAME", "") if mode.startswith("hami") else None,
    "requests_per_service": req_per_svc,
    "warmup_requests_per_service": int(os.environ["WARMUP"]),
    "endpoints": {svc1: os.environ.get("EP1", ""), svc2: os.environ.get("EP2", "")},
    "services": services,
    "gpu": gpu,
    "client": client,
    "acceptance": acceptance,
}
(mode_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def rounded(key, digits):
    try:
        return round(float(client_summary.get(key, 0) or 0), digits)
    except (TypeError, ValueError):
        return None


print(json.dumps({
    "mode": mode,
    "successful_requests": client_summary.get("successful_requests", 0),
    "failed_requests": client_summary.get("failed_requests", 0),
    "wall_seconds": rounded("wall_seconds", 2),
    "output_tokens_per_second": rounded("output_tokens_per_second", 2),
    "latency_p95_seconds": rounded("latency_p95_seconds", 3),
    "fairness_ratio": client_summary.get("fairness_ratio"),
    "peak_gpu_memory_used_mib": peak,
    "mean_gpu_utilization_percent": gpu["mean_utilization_percent"],
    "both_ready": acceptance["both_ready"],
}, indent=2))
'
}

# ---------------------------------------------------------------------------
# Sequential mode: one service on the GPU at a time.
#
# This was the headline bug. The old 10-qwen-sequential.yaml applied BOTH
# LLMInferenceServices at once and only told the client to send its requests one
# after the other. Two vLLM processes stayed resident on the GPU for the whole
# run, so the "sequential baseline" was really an unmanaged co-tenancy run -
# exactly the configuration that crashes. A sequential baseline has to mean one
# CUDA context on the device at a time, so each service is now deployed, driven,
# and torn down on its own before the next one starts.
# ---------------------------------------------------------------------------
run_mode_sequential() {
  local mode="sequential"
  local mode_dir="${RESULTS_ROOT}/${timestamp}/${mode}"
  mkdir -p "${mode_dir}"

  log "Preparing ${mode} mode (one service deployed at a time)"
  reset_cluster_state
  apply_gpu_probe

  local phase_files=()
  local ready_1="false" ready_2="false"
  local startup_1="" startup_2=""
  local pod_1="" pod_2=""
  local endpoint_used_1="" endpoint_used_2=""
  local status=0
  local index

  for index in 1 2; do
    local service manifest explicit_endpoint
    if [[ "${index}" == "1" ]]; then
      service="${SERVICE_1}"; manifest="10-qwen-sequential-01.yaml"; explicit_endpoint="${ENDPOINT_1}"
    else
      service="${SERVICE_2}"; manifest="10-qwen-sequential-02.yaml"; explicit_endpoint="${ENDPOINT_2}"
    fi

    log "Sequential phase ${index}: deploying ${service} alone"
    kubectl apply -f "${ROOT_DIR}/${manifest}" >/dev/null

    local phase_started
    phase_started="$(date +%s)"

    local ready="false" startup=""
    if wait_for_service "${service}" "${phase_started}" "${WAIT_TIMEOUT_SECONDS}"; then
      ready="true"
      startup="$(( $(date +%s) - phase_started ))"
    fi

    local pod_name endpoint
    pod_name="$(select_pod_name "${service}")"
    endpoint="${explicit_endpoint:-$(service_url "${service}")}"

    if [[ "${ready}" == "true" && -n "${endpoint}" ]]; then
      local phase_json="${mode_dir}/client.phase-${index}.json"
      start_gpu_sampler "${mode_dir}/gpu-timeline.phase-${index}.csv"
      run_client "${mode}" "${phase_json}" true "--endpoint=${service}=${endpoint}" || status=$?
      stop_gpu_sampler
      phase_files+=("${phase_json}")
    else
      log "Sequential phase ${index}: ${service} not usable (ready=${ready} endpoint='${endpoint}')"
      status=1
    fi

    collect_service_artifacts "${service}" "${mode_dir}"

    if [[ "${index}" == "1" ]]; then
      ready_1="${ready}"; startup_1="${startup}"; pod_1="${pod_name}"; endpoint_used_1="${endpoint}"
    else
      ready_2="${ready}"; startup_2="${startup}"; pod_2="${pod_name}"; endpoint_used_2="${endpoint}"
    fi

    log "Sequential phase ${index}: removing ${service} to free the GPU"
    kubectl delete -f "${ROOT_DIR}/${manifest}" --ignore-not-found=true --wait=false >/dev/null 2>&1 || true
    wait_for_workload_drain || true
  done

  # One combined timeline so the summary parser sees the same file in every mode.
  {
    printf 'unix_seconds,memory_total_mib,memory_used_mib,utilization_gpu_percent\n'
    for index in 1 2; do
      phase_csv="${mode_dir}/gpu-timeline.phase-${index}.csv"
      if [[ -f "${phase_csv}" ]]; then
        tail -n +2 "${phase_csv}"
      fi
    done
  } > "${mode_dir}/gpu-timeline.csv"

  if [[ "${#phase_files[@]}" -gt 0 ]]; then
    python3 "${ROOT_DIR}/client.py" --mode "${mode}" \
      --output "${mode_dir}/client.result.json" \
      --merge "${phase_files[@]}" > "${mode_dir}/client.stdout.txt" 2>&1 || true
  fi

  collect_mode_snapshot "${mode_dir}"
  write_mode_summary "${mode}" "${mode_dir}" "true" \
    "${endpoint_used_1}" "${endpoint_used_2}" "${ready_1}" "${ready_2}" \
    "${startup_1}" "${startup_2}" "${pod_1}" "${pod_2}"

  log "Completed ${mode}. Results are in ${mode_dir}"
  return "${status}"
}

# ---------------------------------------------------------------------------
# Concurrent modes: both services resident on the GPU together.
# ---------------------------------------------------------------------------
run_mode_concurrent() {
  local mode="$1"
  local mode_dir="${RESULTS_ROOT}/${timestamp}/${mode}"
  mkdir -p "${mode_dir}"

  log "Preparing ${mode} mode (both services concurrent)"
  reset_cluster_state
  apply_gpu_probe
  apply_mode_manifests "${mode}"

  local start_epoch
  start_epoch="$(date +%s)"

  local ready_1="false" ready_2="false" startup_1="" startup_2=""
  if wait_for_service "${SERVICE_1}" "${start_epoch}" "${WAIT_TIMEOUT_SECONDS}"; then
    ready_1="true"; startup_1="$(( $(date +%s) - start_epoch ))"
  fi
  if wait_for_service "${SERVICE_2}" "${start_epoch}" "${WAIT_TIMEOUT_SECONDS}"; then
    ready_2="true"; startup_2="$(( $(date +%s) - start_epoch ))"
  fi

  local pod_1 pod_2 endpoint_1 endpoint_2
  pod_1="$(select_pod_name "${SERVICE_1}")"
  pod_2="$(select_pod_name "${SERVICE_2}")"
  endpoint_1="${ENDPOINT_1:-$(service_url "${SERVICE_1}")}"
  endpoint_2="${ENDPOINT_2:-$(service_url "${SERVICE_2}")}"

  local status=0
  if [[ "${ready_1}" == "true" && "${ready_2}" == "true" && -n "${endpoint_1}" && -n "${endpoint_2}" ]]; then
    log "Running concurrent client workload for ${mode}"
    start_gpu_sampler "${mode_dir}/gpu-timeline.csv"
    run_client "${mode}" "${mode_dir}/client.result.json" false \
      "--endpoint=${SERVICE_1}=${endpoint_1}" "--endpoint=${SERVICE_2}=${endpoint_2}" || status=$?
    stop_gpu_sampler
  else
    log "Skipping client workload: ready_1=${ready_1} ready_2=${ready_2} endpoint_1='${endpoint_1}' endpoint_2='${endpoint_2}'"
    status=1
  fi

  collect_service_artifacts "${SERVICE_1}" "${mode_dir}"
  collect_service_artifacts "${SERVICE_2}" "${mode_dir}"
  collect_mode_snapshot "${mode_dir}"
  write_mode_summary "${mode}" "${mode_dir}" "false" \
    "${endpoint_1}" "${endpoint_2}" "${ready_1}" "${ready_2}" \
    "${startup_1}" "${startup_2}" "${pod_1}" "${pod_2}"

  log "Completed ${mode}. Results are in ${mode_dir}"
  return "${status}"
}

run_mode() {
  if [[ "$1" == "sequential" ]]; then
    run_mode_sequential
  else
    run_mode_concurrent "$1"
  fi
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
detect_resource_type
validate_common_cluster_state

if [[ "${MODE}" == hami || "${MODE}" == hami-mps || "${MODE}" == all ]]; then
  validate_hami_components
fi

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="${RESULTS_ROOT}/${timestamp}"
mkdir -p "${RUN_DIR}"
exec > >(tee -a "${RUN_DIR}/runner.log") 2>&1

cleanup_on_exit() {
  stop_gpu_sampler
}
trap cleanup_on_exit EXIT INT TERM

log "Writing benchmark results to ${RUN_DIR}"
log "Resource type: ${resource_type}"
log "Mode: ${MODE} | HAMi profile: ${HAMI_PROFILE} | requests/service: ${REQUESTS_PER_SERVICE} | warmup/service: ${WARMUP_REQUESTS}"

modes_to_run=()
case "${MODE}" in
  all)
    modes_to_run=(sequential mps hami)
    if [[ "${WITH_HAMI_MPS}" == "true" ]]; then
      modes_to_run+=(hami-mps)
    fi
    ;;
  *)
    modes_to_run=("${MODE}")
    ;;
esac

overall_status=0
for mode in "${modes_to_run[@]}"; do
  if ! run_mode "${mode}"; then
    log "Mode ${mode} reported a failure"
    overall_status=1
  fi
done

log "Tearing down benchmark workloads"
reset_cluster_state

log "Benchmark run complete (status ${overall_status})"
exit "${overall_status}"
