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
REQUESTS_PER_SERVICE=100
WARMUP_REQUESTS=4
# Tokens generated per request. Exact, via ignore_eos + min_tokens.
MAX_TOKENS=256
# In-flight requests per service. >1 lets one service saturate the GPU,
# which is what makes a 50% core cap observable at all.
CONCURRENCY=4
# Must match --max-num-seqs in the manifests, or vLLM queues requests
# internally and the client-side concurrency never reaches the GPU.
MAX_NUM_SEQS=8
MAX_MODEL_LEN=512
TIMEOUT_SECONDS=180
WAIT_TIMEOUT_SECONDS=1800
DRAIN_TIMEOUT_SECONDS=300
GPU_SAMPLE_INTERVAL_MS=250
MODE="all"
HAMI_PROFILE="balanced"
WITH_HAMI_MPS="false"
ENDPOINT_1=""
ENDPOINT_2=""
INSECURE="false"
RESOURCE_TYPE_OVERRIDE=""
HOST_MPS_UNIT="nvidia-mps"
# auto = pin only when there is disk headroom. See check_disk_headroom.
PIN_IMAGE="auto"
# kubelet evicts unused images above this; pinning above it is dangerous.
IMAGE_GC_HIGH_PERCENT=85
DISK_SAFE_PERCENT=80
disk_use_percent=0
host_mps_was_active="false"
ROUTE_MODE="auto"
GPU_UUID_OVERRIDE=""
resource_type=""
gpu_probe_pod=""
gpu_count=0
gpu_uuids=""
target_gpu_uuid=""
other_gpu_uuids=""
sampler_pid=""
timestamp=""
next_local_port=18000
port_forward_pids=()
RESOLVED_ENDPOINT=""

usage() {
  cat <<'EOF'
Usage: ./run-qwen-benchmark.sh [options]

Options:
  --mode MODE                all, sequential, mps, hami, hami-mps
  --hami-profile PROFILE     balanced, conservative, weighted
  --with-hami-mps            include the HAMi + MPS mode when --mode all
  --endpoint-1 URL           explicit service 1 endpoint
  --endpoint-2 URL           explicit service 2 endpoint
  --requests-per-service N   measured requests per service (default: 100)
  --warmup N                 warm-up requests per service (default: 4)
  --max-tokens N             tokens generated per request (default: 256)
  --concurrency N            in-flight requests per service (default: 4)
  --timeout SECONDS          request timeout passed to client.py (default: 180)
  --wait-timeout SECONDS     readiness timeout for each service (default: 1800)
  --resource-type NAME       override CRD detection, e.g.
                             llminferenceservices.serving.kserve.io
  --route MODE               auto (default), gateway, or port-forward.
                             auto probes the advertised gateway URL and falls
                             back to port-forwarding the workload Service.
  --gpu-uuid UUID            pin every mode to this physical GPU
                             (default: the lowest-index GPU)
  --pin-image MODE           auto (default), yes, or no. Keeps the 9.5 GB vLLM
                             image resident between modes. auto refuses when
                             the node has no disk headroom.
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
    --resource-type)         RESOURCE_TYPE_OVERRIDE="$2"; shift 2 ;;
    --route)                 ROUTE_MODE="$2"; shift 2 ;;
    --gpu-uuid)              GPU_UUID_OVERRIDE="$2"; shift 2 ;;
    --pin-image)             PIN_IMAGE="$2"; shift 2 ;;
    --max-tokens)            MAX_TOKENS="$2"; shift 2 ;;
    --concurrency)           CONCURRENCY="$2"; shift 2 ;;
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

case "${ROUTE_MODE}" in
  auto|gateway|port-forward) ;;
  *) fail "Unsupported --route mode: ${ROUTE_MODE} (auto, gateway, port-forward)" ;;
esac

require_cmd kubectl
require_cmd python3

# ---------------------------------------------------------------------------
# Cluster discovery and validation
# ---------------------------------------------------------------------------
# Confirm kubectl can actually reach an API server before anything else.
#
# Without this, a missing kubeconfig surfaces much later as a confusing
# "CRD not found" error. kubectl's built-in default when no kubeconfig is
# readable is http://localhost:8080, so `dial tcp 127.0.0.1:8080: connection
# refused` almost always means "this user has no kubeconfig", not "the cluster
# is down" - which is why the same script works under sudo (root has one) and
# fails as a normal user.
validate_cluster_access() {
  local err
  if err="$(kubectl get --raw /version 2>&1 >/dev/null)"; then
    return 0
  fi

  printf 'ERROR: kubectl cannot reach a Kubernetes API server.\n\n' >&2
  printf '%s\n\n' "${err}" >&2

  if [[ "${err}" == *"localhost:8080"* || "${err}" == *"127.0.0.1:8080"* ]]; then
    printf 'That address is kubectl'"'"'s fallback when no kubeconfig is readable,\n' >&2
    printf 'so this user almost certainly has no ~/.kube/config.\n\n' >&2
    if [[ -f /etc/rancher/k3s/k3s.yaml ]]; then
      printf 'This node looks like k3s. Give your user a kubeconfig with:\n\n' >&2
      printf '  mkdir -p ~/.kube\n' >&2
      printf '  sudo cp /etc/rancher/k3s/k3s.yaml ~/.kube/config\n' >&2
      printf '  sudo chown "$(id -u):$(id -g)" ~/.kube/config\n' >&2
      printf '  chmod 600 ~/.kube/config\n\n' >&2
    else
      printf 'Copy an admin kubeconfig to ~/.kube/config, or set KUBECONFIG.\n\n' >&2
    fi
    printf 'Running the benchmark under sudo also works, but then every file in\n' >&2
    printf 'results/ is owned by root.\n' >&2
  fi
  exit 1
}

# Resolve the resource name to use for every later get/patch/label.
#
# `kubectl api-resources` is unreliable as a single source of truth: it exits
# non-zero and can print NOTHING to stdout when any single aggregated APIService
# in the cluster is unavailable ("unable to retrieve the complete list of server
# APIs"), even though the CRD we want is registered and perfectly usable. So try
# several independent sources and only give up if all of them fail.
#
# The match stays anchored to llminferenceservice: the original loose grep also
# matched plain `inferenceservices`, and `head -n 1` would then pick the wrong
# CRD so every later command silently targeted the wrong resource.
detect_resource_type() {
  if [[ -n "${RESOURCE_TYPE_OVERRIDE}" ]]; then
    resource_type="${RESOURCE_TYPE_OVERRIDE}"
    log "Using resource type from --resource-type: ${resource_type}"
    return 0
  fi

  local api_resources_out api_resources_err crd_out
  api_resources_err="$(mktemp)"

  # Source 1: api-resources scoped to the group.
  api_resources_out="$(kubectl api-resources --api-group=serving.kserve.io -o name 2>"${api_resources_err}" || true)"
  resource_type="$(printf '%s\n' "${api_resources_out}" | grep -E '^llminferenceservices?(\.|$)' | head -n 1 || true)"

  # Source 2: api-resources unscoped. The --api-group filter returns nothing on
  # some versions when the group is served only under a non-preferred version.
  if [[ -z "${resource_type}" ]]; then
    resource_type="$(kubectl api-resources -o name 2>/dev/null \
      | grep -E '^llminferenceservices?\.serving\.kserve\.io$' | head -n 1 || true)"
  fi

  # Source 3: the CRD object itself. This works even when API discovery is
  # partially broken, because it is a plain GET against a known resource.
  if [[ -z "${resource_type}" ]]; then
    crd_out="$(kubectl get crd -o name 2>/dev/null \
      | sed 's|^customresourcedefinition\.apiextensions\.k8s\.io/||' \
      | grep -E '^llminferenceservices?\.serving\.kserve\.io$' | head -n 1 || true)"
    resource_type="${crd_out}"
  fi

  # Source 4: last resort, just probe the name and see if the server accepts it.
  if [[ -z "${resource_type}" ]]; then
    local candidate
    for candidate in llminferenceservices.serving.kserve.io llminferenceservices llmisvc; do
      if kubectl get "${candidate}" -n "${NS}" >/dev/null 2>&1; then
        resource_type="${candidate}"
        break
      fi
    done
  fi

  if [[ -z "${resource_type}" ]]; then
    printf 'ERROR: Could not find the KServe LLMInferenceService CRD (serving.kserve.io).\n' >&2
    printf '\n--- kubectl api-resources --api-group=serving.kserve.io (stdout) ---\n' >&2
    printf '%s\n' "${api_resources_out:-<empty>}" >&2
    printf '\n--- kubectl api-resources --api-group=serving.kserve.io (stderr) ---\n' >&2
    cat "${api_resources_err}" >&2 || true
    printf '\n--- CRDs matching kserve ---\n' >&2
    kubectl get crd -o name 2>&1 | grep -i kserve >&2 || printf '<none>\n' >&2
    printf '\nIf the CRD is present under another name, re-run with:\n' >&2
    printf '  ./run-qwen-benchmark.sh --resource-type <name> ...\n' >&2
    rm -f "${api_resources_err}"
    exit 1
  fi

  rm -f "${api_resources_err}"
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
              20-mps-daemon.yaml 06-gpu-reserver.yaml 05-gpu-probe.yaml; do
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
# Render the profile INTO the manifest and apply once.
#
# The previous approach applied the manifest and then kubectl-patched the
# quotas. That second write changed the Pod template, so KServe rolled the
# Deployment: a new replica appeared while the old one still held the GPU, the
# new one sat Pending until the old released it, and the mode stalled. It cost
# hami-mps 21 minutes of its 25-minute duration and left a Pending Pod that
# broke GPU-placement detection.
#
# Rendering first means one apply, one ReplicaSet, no rollout.
render_hami_manifest() {
  local manifest="$1" profile="$2" out="$3"
  local raw
  raw="$(kubectl get configmap qwen-hami-compute-profiles -n "${NS}"     -o "jsonpath={.data.${profile}}" 2>/dev/null || true)"
  if [[ -z "${raw}" ]]; then
    fail "HAMi profile '${profile}' not found in configmap qwen-hami-compute-profiles"
  fi

  printf '%s' "${raw}" |     MANIFEST="${ROOT_DIR}/${manifest}" OUT="${out}" PROFILE_NAME="${profile}"     SEQS="${MAX_NUM_SEQS}" MODELLEN="${MAX_MODEL_LEN}" run_python -c '
import json, os, pathlib, re, sys

profile = json.load(sys.stdin)
cores = profile["cores"]
mem = profile["mem"]
util = profile.get("util", [0.90, 0.90])
seqs = os.environ["SEQS"]
model_len = os.environ["MODELLEN"]
name = os.environ["PROFILE_NAME"]

text = pathlib.Path(os.environ["MANIFEST"]).read_text(encoding="utf-8")
# Built with chr(10) rather than a backslash escape: this block is embedded in
# a single-quoted shell string, where a literal backslash-n does not survive.
separator = chr(10) + "---" + chr(10)
# Document 0 is qwen-bench-01, document 1 is qwen-bench-02.
documents = text.split(separator)
rendered = []
for position, document in enumerate(documents):
    which = min(position, 1)
    # Anchored on "=<number>" and \": <number>\" so prose in the header
    # comments (which also mentions these flags) is never rewritten.
    document = re.sub(r"--gpu-memory-utilization=[0-9.]+",
                      f"--gpu-memory-utilization={util[which]}", document)
    document = re.sub(r"--max-num-seqs=[0-9]+", f"--max-num-seqs={seqs}", document)
    document = re.sub(r"--max-model-len=[0-9]+", f"--max-model-len={model_len}", document)
    document = re.sub(r"(nvidia\.com/gpucores:\s*)\"[0-9]+\"",
                      lambda m: m.group(1) + chr(34) + str(cores[which]) + chr(34), document)
    document = re.sub(r"(nvidia\.com/gpumem:\s*)\"[0-9]+\"",
                      lambda m: m.group(1) + chr(34) + str(mem[which]) + chr(34), document)
    document = re.sub(r"(qwen-hami-profile:\s*)\S+", lambda m: m.group(1) + name, document)
    rendered.append(document)

pathlib.Path(os.environ["OUT"]).write_text(separator.join(rendered), encoding="utf-8")

print(" ".join(str(v) for v in (cores[0], cores[1], mem[0], mem[1], util[0], util[1])))
'
}

apply_hami_manifest() {
  local manifest="$1" profile="$2"
  local rendered="${RUN_DIR}/rendered-${profile}-${manifest}"
  local parsed
  parsed="$(render_hami_manifest "${manifest}" "${profile}" "${rendered}")"

  local cores_1 cores_2 mem_1 mem_2 util_1 util_2
  read -r cores_1 cores_2 mem_1 mem_2 util_1 util_2 <<< "${parsed}"

  log "HAMi profile '${profile}' (rendered into ${manifest}, single apply):"
  log "  ${SERVICE_1}: gpucores=${cores_1} gpumem=${mem_1}MiB vllm-util=${util_1}"
  log "  ${SERVICE_2}: gpucores=${cores_2} gpumem=${mem_2}MiB vllm-util=${util_2}"

  kubectl apply -f "${rendered}" >/dev/null
}

# ---------------------------------------------------------------------------
# Manifest application
# ---------------------------------------------------------------------------
# Pin the vLLM image on the node for the whole run. See 04-image-pinner.yaml:
# the between-mode drain leaves the 9.5 GB image unreferenced and kubelet
# garbage-collects it, costing ~4 minutes of re-pull on the next mode.
# Pinning the image is only safe when the node has disk headroom.
#
# Measured on this node: kubelet ran image GC every 5 minutes at 88% disk and
# logged "Attempted to free 14261362688 bytes, but only found 301463 bytes
# eligible to free". The one large eligible item was the 9.5 GB vLLM image,
# unreferenced between modes - so kubelet deleted it and the next mode re-pulled
# it for ~4 minutes.
#
# Pinning it removes kubelet's only real GC candidate. Above the high threshold
# that means GC can free nothing at all, and the node heads for
# DiskPressure=True, which taints it NoSchedule and evicts running Pods. Losing
# Pods mid-benchmark is far worse than a slow image pull, so refuse to pin
# unless there is room.
check_disk_headroom() {
  local path="/var/lib/rancher/k3s"
  [[ -d "${path}" ]] || path="/"
  disk_use_percent="$(df -P "${path}" 2>/dev/null | awk 'NR==2 {gsub("%","",$5); print $5+0}')"
  disk_use_percent="${disk_use_percent:-0}"
  log "Node disk usage at ${path}: ${disk_use_percent}%"

  if (( disk_use_percent >= IMAGE_GC_HIGH_PERCENT )); then
    log "WARNING: disk is at or above kubelet's image-gc-high-threshold (${IMAGE_GC_HIGH_PERCENT}%)."
    log "         kubelet will keep evicting the 9.5 GB vLLM image between modes, so expect"
    log "         roughly 4 extra minutes of image pull per mode, and a risk of DiskPressure"
    log "         evicting Pods mid-run. Free disk space before trusting startup timings."
  fi
}

apply_image_pinner() {
  case "${PIN_IMAGE}" in
    no)
      log "Image pinning disabled (--pin-image no)"
      return 0
      ;;
    auto)
      if (( disk_use_percent >= DISK_SAFE_PERCENT )); then
        log "Not pinning the vLLM image: disk at ${disk_use_percent}% (safe limit ${DISK_SAFE_PERCENT}%)."
        log "  Pinning here would leave kubelet nothing to garbage collect and risk DiskPressure."
        log "  Free disk space, or force with --pin-image yes."
        return 0
      fi
      ;;
  esac

  kubectl apply -f "${ROOT_DIR}/04-image-pinner.yaml" >/dev/null
  # Do NOT block on this.
  #
  # The pinner uses the same 9.5 GB image, so becoming Ready can take minutes on
  # a cold cache. Waiting for it bought nothing and made the runner look hung:
  # the first mode's Pod pulls the image regardless, and the pinner only has to
  # be holding a reference by the time the SECOND mode starts. It pulls in the
  # background while mode 1 deploys.
  log "Image pinner applied (pulling in the background; it only needs to be up"
  log "  before the second mode, so not waiting for it)."
}

# Reported once per mode so a re-pull is attributable rather than mysterious.
log_image_pinner_state() {
  local ready
  ready="$(kubectl get deployment/qwen-benchmark-image-pinner -n "${NS}" \
    -o jsonpath='{.status.readyReplicas}' 2>/dev/null || true)"
  if [[ "${ready:-0}" -ge 1 ]]; then
    log "Image pinner is holding the vLLM image (no re-pull expected)"
  elif [[ "${PIN_IMAGE}" != "no" ]]; then
    log "Image pinner not up yet; this mode may re-pull the 9.5 GB image (~4 min)"
  fi
}

# Removed only at the very end, never by the between-mode teardown.
delete_image_pinner() {
  kubectl delete -f "${ROOT_DIR}/04-image-pinner.yaml" --ignore-not-found=true --wait=false >/dev/null 2>&1 || true
}

apply_gpu_probe() {
  kubectl apply -f "${ROOT_DIR}/05-gpu-probe.yaml" >/dev/null
  log "Waiting up to 180s for the GPU probe DaemonSet..."
  kubectl rollout status -n "${NS}" daemonset/qwen-benchmark-gpu-probe --timeout=180s >/dev/null 2>&1 \
    || log "WARNING: GPU probe DaemonSet is not Ready; GPU samples may be missing"
  gpu_probe_pod="$(kubectl get pods -n "${NS}" -l app=qwen-benchmark-gpu-probe \
    -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)"
}

# Count physical GPUs from the probe Pod, not from allocatable nvidia.com/gpu.
# Under HAMi, allocatable is physical_gpus x deviceSplitCount (e.g. 20 for two
# GPUs at the default split of 10), so it cannot be used as a device count.
detect_gpu_inventory() {
  gpu_count=0
  gpu_uuids=""
  target_gpu_uuid=""
  other_gpu_uuids=""
  if [[ -z "${gpu_probe_pod}" ]]; then
    log "WARNING: no GPU probe Pod; cannot confine the benchmark to one GPU"
    return 0
  fi

  gpu_uuids="$(kubectl exec -n "${NS}" "${gpu_probe_pod}" -c probe -- \
    nvidia-smi --query-gpu=index,uuid --format=csv,noheader 2>/dev/null \
    | tr -d ' \r' | grep . || true)"
  gpu_count="$(printf '%s\n' "${gpu_uuids}" | grep -c . || true)"
  gpu_count="${gpu_count:-0}"

  # Pick ONE target GPU and keep every mode on it.
  #
  # Previously the reserver took whichever device HAMi handed it, so the free
  # GPU differed per mode: MPS measured GPU-fa422f59 while HAMi measured
  # GPU-bc62f652. Even with identical hardware that is not a controlled
  # comparison. Index 0 is chosen deterministically unless --gpu-uuid says
  # otherwise.
  if [[ -n "${GPU_UUID_OVERRIDE}" ]]; then
    target_gpu_uuid="${GPU_UUID_OVERRIDE}"
  else
    target_gpu_uuid="$(printf '%s\n' "${gpu_uuids}" | sort -t, -k1,1n | head -n 1 | cut -d, -f2)"
  fi

  other_gpu_uuids="$(printf '%s\n' "${gpu_uuids}" | cut -d, -f2 \
    | grep -v -x -- "${target_gpu_uuid}" | paste -sd, - || true)"

  log "Physical GPUs visible on the node: ${gpu_count}"
  log "Target GPU for every mode: ${target_gpu_uuid}"
  if [[ -n "${other_gpu_uuids}" ]]; then
    log "GPUs to be reserved away: ${other_gpu_uuids}"
  fi
}

# Reserve every GPU but one, so all benchmark Pods are forced to share a single
# device. Without this, HAMi spreads the two services across separate GPUs and
# nothing is actually shared.
apply_gpu_reserver() {
  local to_reserve=0
  if [[ "${gpu_count}" -gt 1 ]]; then
    to_reserve=$(( gpu_count - 1 ))
  fi

  kubectl apply -f "${ROOT_DIR}/06-gpu-reserver.yaml" >/dev/null

  # Pin the reserver to the NON-target GPUs so the same device is left free for
  # every mode. Unlike LLMInferenceService.spec.template, a Deployment's Pod
  # template does have metadata, so HAMi's device-selection annotation can be
  # set here.
  if [[ -n "${other_gpu_uuids}" ]]; then
    kubectl patch deployment/qwen-benchmark-gpu-reserver -n "${NS}" --type=merge \
      -p "{\"spec\":{\"template\":{\"metadata\":{\"annotations\":{\"nvidia.com/use-gpuuuid\":\"${other_gpu_uuids}\"}}}}}" >/dev/null
  fi

  kubectl scale deployment/qwen-benchmark-gpu-reserver -n "${NS}" --replicas="${to_reserve}" >/dev/null

  if [[ "${to_reserve}" -eq 0 ]]; then
    log "Single GPU node: no reserver replicas needed"
    return 0
  fi

  log "Reserving ${to_reserve} of ${gpu_count} GPUs so the benchmark shares exactly one device"
  log "Waiting up to 180s for the GPU reserver..."
  if ! kubectl rollout status -n "${NS}" deployment/qwen-benchmark-gpu-reserver --timeout=180s >/dev/null 2>&1; then
    log "WARNING: GPU reserver did not become Ready. The benchmark may spread across GPUs."
    log "         Check: kubectl describe deployment/qwen-benchmark-gpu-reserver -n ${NS}"
  fi
}

# Read back which physical GPU each service actually landed on. This is the
# check that would have caught the split-across-two-GPUs run immediately
# instead of after a full benchmark of meaningless numbers.
service_gpu_uuid() {
  local service="$1" pod_name
  pod_name="$(select_pod_name "${service}")"
  [[ -n "${pod_name}" ]] || { printf ''; return 0; }
  kubectl get pod "${pod_name}" -n "${NS}" \
    -o jsonpath='{.metadata.annotations.hami\.io/vgpu-devices-allocated}' 2>/dev/null \
    | sed 's/,.*//' | tr -d ' \r' || true
}

verify_same_gpu() {
  local mode_dir="$1"
  local uuid_1 uuid_2
  uuid_1="$(service_gpu_uuid "${SERVICE_1}")"
  uuid_2="$(service_gpu_uuid "${SERVICE_2}")"

  printf '%s\n' "{\"${SERVICE_1}\": \"${uuid_1}\", \"${SERVICE_2}\": \"${uuid_2}\", \"same_gpu\": $( [[ -n "${uuid_1}" && "${uuid_1}" == "${uuid_2}" ]] && echo true || echo false )}" \
    > "${mode_dir}/gpu-placement.json"

  if [[ -z "${uuid_1}" || -z "${uuid_2}" ]]; then
    log "NOTE: could not read HAMi GPU placement annotations (not a HAMi-scheduled mode?)"
    return 0
  fi
  if [[ "${uuid_1}" == "${uuid_2}" ]]; then
    log "GPU placement OK: both services share ${uuid_1}"
  else
    log "WARNING: services are on DIFFERENT GPUs - this is not GPU sharing"
    log "  ${SERVICE_1}: ${uuid_1}"
    log "  ${SERVICE_2}: ${uuid_2}"
    log "  Results for this mode do not measure sharing. Check the GPU reserver."
  fi
}

# A host-level systemd MPS unit defeats every attempt to stop MPS.
#
# GPU_Share/configs/nvidia-mps.service declares Restart=always and uses the same
# /tmp/nvidia-mps pipe directory as this benchmark's DaemonSet. If it is
# installed, systemd resurrects nvidia-cuda-mps-control seconds after any kill,
# so MPS servers reappear inside the sequential and HAMi baselines. During the
# MPS modes it is worse: two control daemons end up sharing one pipe directory,
# which is why the resident-server count doubled.
#
# The runner already runs as root, so stop the unit for the duration of the run
# and put it back exactly as it was on exit.
stop_host_mps_service() {
  command -v systemctl >/dev/null 2>&1 || return 0
  systemctl list-unit-files "${HOST_MPS_UNIT}.service" >/dev/null 2>&1 || return 0
  if systemctl is-active --quiet "${HOST_MPS_UNIT}" 2>/dev/null; then
    host_mps_was_active="true"
    log "Host systemd unit '${HOST_MPS_UNIT}' is active (Restart=always)."
    log "Stopping it for the duration of the benchmark so MPS state is controlled."
    systemctl stop "${HOST_MPS_UNIT}" >/dev/null 2>&1 \
      || log "WARNING: failed to stop ${HOST_MPS_UNIT}; MPS servers will keep respawning"
    sleep 2
  fi
}

restore_host_mps_service() {
  if [[ "${host_mps_was_active}" == "true" ]]; then
    systemctl start "${HOST_MPS_UNIT}" >/dev/null 2>&1 || true
    host_mps_was_active="false"
  fi
}

# Guarantee MPS is really gone before a non-MPS mode measures anything.
#
# Deleting the DaemonSet is not enough. nvidia-cuda-mps-server processes are
# spawned on the HOST by the control daemon, so they outlive the Pod whenever
# preStop does not complete (interrupted run, killed Pod, OOM). Each surviving
# server holds a CUDA context and ~30 MiB of VRAM and stays visible in
# nvidia-smi, which silently contaminates the sequential and HAMi baselines.
#
# The probe Pod is privileged with hostPID, so it can see and signal them.
ensure_mps_stopped() {
  stop_host_mps_service
  kubectl delete -f "${ROOT_DIR}/20-mps-daemon.yaml" --ignore-not-found=true --wait=false >/dev/null 2>&1 || true

  if [[ -z "${gpu_probe_pod}" ]]; then
    log "No GPU probe Pod; cannot verify MPS is stopped"
    return 0
  fi

  kubectl exec -n "${NS}" "${gpu_probe_pod}" -c probe -- bash -c '
    export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
    # Ask politely first so servers drain their clients, then insist.
    echo quit | timeout 10 nvidia-cuda-mps-control >/dev/null 2>&1 || true
    sleep 2
    # Kill by PID from nvidia-smi rather than pkill: procps is not guaranteed to
    # be present in the CUDA base image, and a silently missing pkill is why MPS
    # servers survived into the sequential baseline on the last run. hostPID is
    # set, so these are real host PIDs.
    for attempt in 1 2; do
      pids="$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null         | grep 'mps-server' | cut -d, -f1 | tr -d ' ')"
      [ -z "$pids" ] && break
      for pid in $pids; do
        kill -TERM "$pid" 2>/dev/null || true
      done
      sleep 3
      for pid in $pids; do
        kill -KILL "$pid" 2>/dev/null || true
      done
      sleep 1
    done
    nvidia-smi -c DEFAULT >/dev/null 2>&1 || true
    exit 0
  ' >/dev/null 2>&1 || true

  local remaining
  remaining="$(kubectl exec -n "${NS}" "${gpu_probe_pod}" -c probe -- \
    nvidia-smi --query-compute-apps=process_name --format=csv,noheader 2>/dev/null \
    | grep -c 'nvidia-cuda-mps-server' || true)"
  remaining="${remaining:-0}"
  if [[ "${remaining}" -gt 0 ]]; then
    log "WARNING: ${remaining} nvidia-cuda-mps-server process(es) still resident."
    log "         This mode is NOT MPS-free. Clear manually: sudo pkill -f nvidia-cuda-mps"
  else
    log "MPS confirmed stopped; no MPS servers resident on the GPU"
  fi
}

apply_mps_daemon() {
  kubectl apply -f "${ROOT_DIR}/20-mps-daemon.yaml" >/dev/null
  log "Waiting up to 300s for the MPS daemon (readiness = MPS control socket present)..."
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
      # HAMi mode must be MPS-free or the gpucores result is unattributable.
      ensure_mps_stopped
      kubectl apply -f "${ROOT_DIR}/31-qwen-hami-compute-profiles.yaml" >/dev/null
      apply_hami_manifest "30-qwen-hami.yaml" "${HAMI_PROFILE}"
      ;;
    hami-mps)
      apply_mps_daemon
      kubectl apply -f "${ROOT_DIR}/31-qwen-hami-compute-profiles.yaml" >/dev/null
      apply_hami_manifest "40-qwen-hami-mps.yaml" "${HAMI_PROFILE}"
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

# ---------------------------------------------------------------------------
# Endpoint resolution.
#
# .status.url reported http://<node-ip>/llm-demo/<service>, but every request
# came back "404 page not found" - the plain-text body k3s's Traefik returns.
# Traefik owns :80 on the node IP, so the URL KServe advertises was answered by
# the wrong ingress entirely and never reached the KServe gateway. vLLM itself
# was healthy and serving /v1/chat/completions the whole time.
#
# So: probe the advertised URL, and if it does not answer, port-forward the
# service's own workload Service and drive vLLM directly. That bypasses the
# gateway and the llm-d scheduler, which for a GPU-sharing benchmark is a
# feature - it removes routing overhead from the latency measurement and
# isolates GPU behaviour, which is what the project is actually about.
# ---------------------------------------------------------------------------
probe_http_endpoint() {
  local base="$1"
  BASE_URL="${base}" MODEL="${MODEL_NAME}" run_python -c '
import json, os, sys
import requests

base = os.environ["BASE_URL"].rstrip("/")
payload = {
    "model": os.environ["MODEL"],
    "messages": [{"role": "user", "content": "ping"}],
    "max_tokens": 1,
    "temperature": 0,
    "stream": False,
}
for suffix in ("/v1/chat/completions", "/openai/v1/chat/completions"):
    try:
        response = requests.post(base + suffix, json=payload, timeout=15)
        if response.ok:
            sys.exit(0)
    except Exception:
        pass
sys.exit(1)
' >/dev/null 2>&1
}

start_port_forward() {
  local service="$1"
  RESOLVED_ENDPOINT=""

  local svc_name port local_port
  svc_name="$(kubectl get svc -n "${NS}" -o name 2>/dev/null | sed 's|^service/||' \
    | grep -E "^${service}-kserve-workload-svc$" | head -n 1 || true)"
  if [[ -z "${svc_name}" ]]; then
    svc_name="$(kubectl get svc -n "${NS}" -o name 2>/dev/null | sed 's|^service/||' \
      | grep -E "^${service}-kserve" | grep -v -- '-epp-' | head -n 1 || true)"
  fi
  if [[ -z "${svc_name}" ]]; then
    log "Could not find a workload Service for ${service}"
    return 1
  fi

  port="$(kubectl get svc "${svc_name}" -n "${NS}" -o jsonpath='{.spec.ports[0].port}' 2>/dev/null || true)"
  port="${port:-8000}"
  local_port="${next_local_port}"
  next_local_port=$(( next_local_port + 1 ))

  kubectl port-forward -n "${NS}" "svc/${svc_name}" "${local_port}:${port}" >/dev/null 2>&1 &
  port_forward_pids+=("$!")
  log "Port-forwarding ${svc_name}:${port} -> 127.0.0.1:${local_port} for ${service}"

  local attempt
  for attempt in $(seq 1 30); do
    if probe_http_endpoint "http://127.0.0.1:${local_port}"; then
      RESOLVED_ENDPOINT="http://127.0.0.1:${local_port}"
      return 0
    fi
    sleep 2
  done

  log "Port-forward for ${service} did not answer after 60s"
  return 1
}

stop_port_forwards() {
  local pid
  for pid in "${port_forward_pids[@]:-}"; do
    [[ -n "${pid}" ]] && kill "${pid}" >/dev/null 2>&1 || true
  done
  port_forward_pids=()
}

# Sets RESOLVED_ENDPOINT.
resolve_endpoint() {
  local service="$1" explicit="$2"
  RESOLVED_ENDPOINT=""

  if [[ -n "${explicit}" ]]; then
    RESOLVED_ENDPOINT="${explicit}"
    return 0
  fi

  case "${ROUTE_MODE}" in
    port-forward)
      start_port_forward "${service}"
      return $?
      ;;
    gateway)
      RESOLVED_ENDPOINT="$(service_url "${service}")"
      [[ -n "${RESOLVED_ENDPOINT}" ]]
      return $?
      ;;
  esac

  # auto
  local advertised
  advertised="$(service_url "${service}")"
  if [[ -n "${advertised}" ]] && probe_http_endpoint "${advertised}"; then
    log "${service}: using advertised gateway URL ${advertised}"
    RESOLVED_ENDPOINT="${advertised}"
    return 0
  fi
  if [[ -n "${advertised}" ]]; then
    log "${service}: advertised URL ${advertised} did not answer; falling back to port-forward"
  fi
  start_port_forward "${service}"
  return $?
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
#
# Prefer a Running Pod. During a rolling update KServe briefly has two
# replicas, and the incoming one can be Pending (it cannot get the GPU until
# the outgoing one releases it). A Pending Pod carries no HAMi placement
# annotation and no vLLM log, so picking it made the HAMi mode report
# same_gpu=false and lose its model-init timings even though the run was fine.
select_pod_name() {
  local pods chosen
  pods="$(kubectl get pods -n "${NS}" \
    -o jsonpath="{range .items[*]}{.metadata.name}{' '}{.status.phase}{'\n'}{end}" 2>/dev/null \
    | grep -E "^$1-kserve-" \
    | grep -v -- '-router-scheduler-' || true)"
  [[ -n "${pods}" ]] || return 0

  chosen="$(printf '%s\n' "${pods}" | awk '$2=="Running"{print $1; exit}')"
  if [[ -z "${chosen}" ]]; then
    chosen="$(printf '%s\n' "${pods}" | awk 'NF{print $1; exit}')"
  fi
  printf '%s' "${chosen}"
}

# ---------------------------------------------------------------------------
# GPU sampling.
#
# Samples come from the privileged probe Pod so every mode is measured from the
# same host-level vantage point. Sampling inside a HAMi workload Pod would
# report the vGPU quota rather than the device, which would make the HAMi modes
# non-comparable with the others.
# ---------------------------------------------------------------------------
# Two defects in the previous sampler made every GPU number unreliable:
#
#  1. It ran `nvidia-smi | head -n 1`, which on a multi-GPU node always reports
#     GPU 0 regardless of where the workload is. The MPS run landed on GPU
#     fa422f59 and was scored against the idle GPU: peak memory 74 MiB.
#  2. One kubectl exec every 2s against a client run lasting under 3s yielded
#     one or two samples, so peak/mean were meaningless.
#
# Now: a single long-lived exec streams every GPU with its UUID at 250 ms via
# nvidia-smi's own --loop-ms, so there is no per-sample exec overhead, and the
# summary filters to the GPU the workload actually ran on.
start_gpu_sampler() {
  local out_file="$1"
  sampler_pid=""
  if [[ -z "${gpu_probe_pod}" ]]; then
    log "No GPU probe Pod available; skipping GPU sampling"
    return 0
  fi

  printf 'timestamp,index,uuid,memory_total_mib,memory_used_mib,utilization_gpu_percent\n' > "${out_file}"
  (
    kubectl exec -n "${NS}" "${gpu_probe_pod}" -c probe -- \
      nvidia-smi --query-gpu=timestamp,index,uuid,memory.total,memory.used,utilization.gpu \
      --format=csv,noheader,nounits --loop-ms="${GPU_SAMPLE_INTERVAL_MS}" 2>/dev/null \
      >> "${out_file}"
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
import datetime, json, os, pathlib, re
service_dir = pathlib.Path(os.environ["SERVICE_DIR"])
pod_path = service_dir / "pod.json"
summary = {
    "pod_name": "", "node_name": "", "phase": "", "start_time": "",
    "restart_count": 0, "oom_killed": 0, "annotations": {}, "labels": {},
    "model_init": {},
}


def parse_ts(value):
    if not value:
        return None
    try:
        return datetime.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


# Model initialization timings, straight out of the vLLM startup log.
# These separate "how long until this model could serve traffic" from the
# steady-state request metrics, and they are the numbers that actually differ
# between sharing modes (contending for the same GPU slows model load).
def parse_model_init(path):
    init = {
        "weights_load_seconds": None,
        "engine_init_seconds": None,
        "chat_template_warmup_seconds": None,
        "cuda_graph_capture_seconds": None,
        "api_server_started": False,
        "total_model_init_seconds": None,
    }
    if not path.exists() or not path.stat().st_size:
        return init
    text = path.read_text(encoding="utf-8", errors="replace")

    match = re.search(r"Model loading took [\d.]+ \w+ and ([\d.]+) seconds", text)
    if match:
        init["weights_load_seconds"] = float(match.group(1))
    match = re.search(r"init engine \([^)]*\) took ([\d.]+) seconds", text)
    if match:
        init["engine_init_seconds"] = float(match.group(1))
    match = re.search(r"Chat template warmup completed in ([\d.]+)\s*ms", text)
    if match:
        init["chat_template_warmup_seconds"] = round(float(match.group(1)) / 1000.0, 3)
    match = re.search(r"Graph capturing finished in ([\d.]+) secs", text)
    if match:
        init["cuda_graph_capture_seconds"] = float(match.group(1))
    init["api_server_started"] = "Starting vLLM API server" in text

    parts = [
        init["weights_load_seconds"],
        init["engine_init_seconds"],
        init["chat_template_warmup_seconds"],
    ]
    if any(part is not None for part in parts):
        init["total_model_init_seconds"] = round(sum(part for part in parts if part), 3)
    return init

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

    # Wall-clock from the container actually starting to the Pod reporting
    # Ready. This brackets the whole model bring-up including image/PVC work
    # that the vLLM log itself cannot see.
    started_at = None
    for container in status.get("containerStatuses", []):
        if container.get("name") == "main":
            started_at = parse_ts(container.get("state", {}).get("running", {}).get("startedAt"))
    ready_at = None
    for condition in status.get("conditions", []):
        if condition.get("type") == "Ready" and str(condition.get("status")).lower() == "true":
            ready_at = parse_ts(condition.get("lastTransitionTime"))
    summary["container_started_at"] = started_at.isoformat() if started_at else ""
    summary["ready_at"] = ready_at.isoformat() if ready_at else ""
    if started_at and ready_at:
        summary["container_start_to_ready_seconds"] = round((ready_at - started_at).total_seconds(), 3)

summary["model_init"] = parse_model_init(service_dir / "pod.logs.txt")
(service_dir / "pod-summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
'
}

collect_mode_snapshot() {
  local mode_dir="$1"
  kubectl get nodes -o wide > "${mode_dir}/nodes.txt" 2>/dev/null || true
  kubectl get pods -n "${NS}" -o wide > "${mode_dir}/pods.txt" 2>/dev/null || true
  kubectl get daemonset -n "${NS}" -o wide > "${mode_dir}/daemonsets.txt" 2>/dev/null || true
  kubectl get "${resource_type}" -n "${NS}" -o wide > "${mode_dir}/services.txt" 2>/dev/null || true
  # Routing artifacts. Without these, a gateway 404 is undiagnosable after the
  # fact - which is exactly what happened on the first full run.
  # Host-level view of who is actually resident on the GPU. This is the direct
  # evidence of sharing: both VLLM::EngineCore processes should appear on ONE
  # GPU, and it also shows how many nvidia-cuda-mps-server processes exist
  # (one per client UID per GPU, so stale ones from earlier runs stand out).
  if [[ -n "${gpu_probe_pod}" ]]; then
    kubectl exec -n "${NS}" "${gpu_probe_pod}" -c probe -- nvidia-smi \
      --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
      --format=csv,noheader 2>/dev/null > "${mode_dir}/gpu-compute-apps.csv" || true
    kubectl exec -n "${NS}" "${gpu_probe_pod}" -c probe -- nvidia-smi \
      > "${mode_dir}/nvidia-smi.txt" 2>/dev/null || true
  fi
  kubectl get httproute -n "${NS}" -o yaml > "${mode_dir}/httproutes.yaml" 2>/dev/null || true
  kubectl get gateway -A -o wide > "${mode_dir}/gateways.txt" 2>/dev/null || true
  kubectl get svc -n "${NS}" -o wide > "${mode_dir}/services-k8s.txt" 2>/dev/null || true
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
    --max-tokens "${MAX_TOKENS}"
    --concurrency "${CONCURRENCY}"
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
  MODE_DURATION="${MODE_DURATION_SECONDS:-}" TARGET_GPU="${target_gpu_uuid}" \
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
        # Deploy -> Ready, i.e. everything including scheduling and image pull.
        "startup_seconds": number(startup),
        # Container start -> Ready, i.e. model bring-up without scheduling.
        "container_start_to_ready_seconds": pod_summary.get("container_start_to_ready_seconds"),
        # Parsed out of the vLLM startup log.
        "model_init": pod_summary.get("model_init", {}),
        "pod_name": os.environ.get(pod, ""),
        "pod": pod_summary,
        "resource_status": service_status,
    }

# Model initialization rollup: the number to compare across sharing modes.
model_init_rollup = {}
for service, entry in services.items():
    init = entry.get("model_init", {}) or {}
    model_init_rollup[service] = {
        "weights_load_seconds": init.get("weights_load_seconds"),
        "engine_init_seconds": init.get("engine_init_seconds"),
        "chat_template_warmup_seconds": init.get("chat_template_warmup_seconds"),
        "cuda_graph_capture_seconds": init.get("cuda_graph_capture_seconds"),
        "total_model_init_seconds": init.get("total_model_init_seconds"),
        "container_start_to_ready_seconds": entry.get("container_start_to_ready_seconds"),
        "deploy_to_ready_seconds": entry.get("startup_seconds"),
    }
_init_totals = [
    value["total_model_init_seconds"]
    for value in model_init_rollup.values()
    if value["total_model_init_seconds"] is not None
]
model_init_rollup["slowest_total_model_init_seconds"] = max(_init_totals) if _init_totals else None

# Host-level GPU timeline collected during the measured window. Peak memory and
# mean utilization are the numbers that actually answer the project question:
# how much of the shared GPU each sharing mode manages to keep busy.
gpu = {
    "samples": 0, "memory_total_mib": None, "peak_memory_used_mib": None,
    "mean_memory_used_mib": None, "peak_utilization_percent": None,
    "mean_utilization_percent": None,
}
gpu["target_gpu_uuid"] = os.environ.get("TARGET_GPU", "")
timeline_path = mode_dir / "gpu-timeline.csv"
if timeline_path.exists():
    # Rows are timestamp,index,uuid,total,used,util - one per GPU per tick.
    # Keep only the GPU the workload actually ran on, otherwise an idle device
    # drags peak/mean toward zero (which is how MPS scored 74 MiB).
    target = os.environ.get("TARGET_GPU", "")
    per_gpu = {}
    for line in timeline_path.read_text(encoding="utf-8", errors="replace").splitlines()[1:]:
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 6:
            continue
        uuid = parts[2]
        try:
            entry = per_gpu.setdefault(uuid, {"total": None, "used": [], "util": []})
            entry["total"] = int(float(parts[3]))
            entry["used"].append(int(float(parts[4])))
            entry["util"].append(int(float(parts[5])))
        except ValueError:
            continue

    chosen = None
    if target and target in per_gpu:
        chosen = target
    elif per_gpu:
        # Fall back to the busiest device rather than the first one.
        chosen = max(per_gpu, key=lambda key: max(per_gpu[key]["used"] or [0]))

    if chosen:
        entry = per_gpu[chosen]
        used, util = entry["used"], entry["util"]
        if used:
            gpu = {
                "samples": len(used),
                "sampled_gpu_uuid": chosen,
                "target_gpu_uuid": target,
                "sampled_target_gpu": bool(target) and chosen == target,
                "memory_total_mib": entry["total"],
                "peak_memory_used_mib": max(used),
                "mean_memory_used_mib": round(sum(used) / len(used), 1),
                "peak_utilization_percent": max(util) if util else None,
                "mean_utilization_percent": round(sum(util) / len(util), 1) if util else None,
                "all_gpus_seen": sorted(per_gpu),
            }

# Who was actually resident on the GPU during the run, from the host-level
# probe. distinct_gpus_with_vllm is the hard evidence of sharing: it must be 1.
# mps_servers is how leftover MPS processes contaminating a non-MPS mode show
# up in the results instead of being noticed by eye in nvidia-smi.
residency = {
    "vllm_processes": 0,
    "mps_servers": 0,
    "distinct_gpus_with_vllm": 0,
    "gpus_with_vllm": [],
}
apps_path = mode_dir / "gpu-compute-apps.csv"
if apps_path.exists():
    vllm_gpus = set()
    for line in apps_path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            continue
        gpu_uuid, name = parts[0], parts[2]
        if "mps-server" in name:
            residency["mps_servers"] += 1
        elif name and name != "process_name":
            residency["vllm_processes"] += 1
            vllm_gpus.add(gpu_uuid)
    residency["gpus_with_vllm"] = sorted(vllm_gpus)
    residency["distinct_gpus_with_vllm"] = len(vllm_gpus)

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
    # A concurrent mode is only valid if every vLLM process sat on one GPU.
    "all_vllm_on_one_gpu": residency["distinct_gpus_with_vllm"] == 1,
    # Non-MPS modes must have zero MPS servers resident.
    "mps_servers_resident": residency["mps_servers"],
    "mps_expected": mode in ("mps", "hami-mps"),
}

summary = {
    "mode": mode,
    "sequential": flag("SEQUENTIAL"),
    "hami_profile": os.environ.get("HAMI_PROFILE_NAME", "") if mode.startswith("hami") else None,
    "requests_per_service": req_per_svc,
    "warmup_requests_per_service": int(os.environ["WARMUP"]),
    "endpoints": {svc1: os.environ.get("EP1", ""), svc2: os.environ.get("EP2", "")},
    "mode_duration_seconds": number("MODE_DURATION"),
    "model_init": model_init_rollup,
    "gpu_residency": residency,
    "services": services,
    "gpu": gpu,
    "gpu_placement": (
        json.loads((mode_dir / "gpu-placement.json").read_text(encoding="utf-8"))
        if (mode_dir / "gpu-placement.json").exists()
        else {}
    ),
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
    "slowest_model_init_seconds": model_init_rollup["slowest_total_model_init_seconds"],
    "mode_duration_seconds": number("MODE_DURATION"),
    "same_gpu": summary["gpu_placement"].get("same_gpu"),
    "distinct_gpus_with_vllm": residency["distinct_gpus_with_vllm"],
    "mps_servers_resident": residency["mps_servers"],
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
  local mode_start_epoch
  mode_start_epoch="$(date +%s)"

  log "Preparing ${mode} mode (one service deployed at a time)"
  reset_cluster_state
  apply_gpu_probe
  log_image_pinner_state
  detect_gpu_inventory
  apply_gpu_reserver
  # The baseline must have no MPS servers resident, or its VRAM and process
  # counts are not comparable with the other modes.
  ensure_mps_stopped

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

    local pod_name endpoint=""
    pod_name="$(select_pod_name "${service}")"
    if [[ "${ready}" == "true" ]]; then
      resolve_endpoint "${service}" "${explicit_endpoint}" || true
      endpoint="${RESOLVED_ENDPOINT}"
    fi

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

    stop_port_forwards
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
  MODE_DURATION_SECONDS="$(( $(date +%s) - mode_start_epoch ))"
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
  local mode_start_epoch
  mode_start_epoch="$(date +%s)"

  log "Preparing ${mode} mode (both services concurrent)"
  reset_cluster_state
  apply_gpu_probe
  log_image_pinner_state
  detect_gpu_inventory
  apply_gpu_reserver
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

  local pod_1 pod_2 endpoint_1="" endpoint_2=""
  pod_1="$(select_pod_name "${SERVICE_1}")"
  pod_2="$(select_pod_name "${SERVICE_2}")"

  # Confirm both services really landed on the same physical GPU before
  # measuring anything. If they did not, this mode is not testing sharing.
  verify_same_gpu "${mode_dir}"

  if [[ "${ready_1}" == "true" ]]; then
    resolve_endpoint "${SERVICE_1}" "${ENDPOINT_1}" || true
    endpoint_1="${RESOLVED_ENDPOINT}"
  fi
  if [[ "${ready_2}" == "true" ]]; then
    resolve_endpoint "${SERVICE_2}" "${ENDPOINT_2}" || true
    endpoint_2="${RESOLVED_ENDPOINT}"
  fi

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

  stop_port_forwards
  collect_service_artifacts "${SERVICE_1}" "${mode_dir}"
  collect_service_artifacts "${SERVICE_2}" "${mode_dir}"
  collect_mode_snapshot "${mode_dir}"
  MODE_DURATION_SECONDS="$(( $(date +%s) - mode_start_epoch ))"
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
validate_cluster_access
detect_resource_type
validate_common_cluster_state

if [[ "${MODE}" == hami || "${MODE}" == hami-mps || "${MODE}" == all ]]; then
  validate_hami_components
fi

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="${RESULTS_ROOT}/${timestamp}"
mkdir -p "${RUN_DIR}"
exec > >(tee -a "${RUN_DIR}/runner.log") 2>&1

# Running under sudo is fine, but root-owned result files cannot then be read,
# edited or deleted by the invoking user, and a later non-root run fails when it
# tries to append to a root-owned runner.log. Hand the results back.
restore_results_ownership() {
  if [[ -n "${SUDO_UID:-}" && -n "${SUDO_GID:-}" && -d "${RESULTS_ROOT}" ]]; then
    chown -R "${SUDO_UID}:${SUDO_GID}" "${RESULTS_ROOT}" 2>/dev/null || true
  fi
}

cleanup_on_exit() {
  stop_gpu_sampler
  restore_host_mps_service
  stop_port_forwards
  restore_results_ownership
}
trap cleanup_on_exit EXIT INT TERM

stop_host_mps_service

log "Writing benchmark results to ${RUN_DIR}"
log "Resource type: ${resource_type}"
log "Mode: ${MODE} | HAMi profile: ${HAMI_PROFILE}"
log "Workload: ${REQUESTS_PER_SERVICE} req/service x ${MAX_TOKENS} tokens, concurrency ${CONCURRENCY}/service, warmup ${WARMUP_REQUESTS}"

check_disk_headroom
apply_image_pinner

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
delete_image_pinner

log "Benchmark run complete (status ${overall_status})"
exit "${overall_status}"
