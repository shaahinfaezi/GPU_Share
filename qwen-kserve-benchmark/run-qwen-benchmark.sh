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
# Measured client passes per deployment. Repeats run IN PLACE against the same
# Pods. Measured on this node, 91% of a mode's wall time is Pod churn and only
# 9% is measurement (4 modes = 26 min total, 143 s of it measuring), so
# redeploying per repeat would triple the run to measure the same thing.
#
# What this gives: run-to-run variance of the MEASUREMENT under a fixed
# deployment, which is what "the effect is outside noise" requires.
# What it does NOT give: deploy-to-deploy variance (scheduling, init, placement).
REPEATS=1
# Idle seconds between repeat passes so a GPU heated by the previous pass does
# not clock-throttle the next one. Sampled clocks/temperature make this checkable.
COOLDOWN_SECONDS=20
# Value stamped into CUDA_MPS_ACTIVE_THREAD_PERCENTAGE in the MPS manifests.
# 100 makes MPS work-conserving (any client may use every SM) and leaves HAMi's
# gpucores as the only compute control, which is what the asym-mps mode needs.
MPS_THREAD_PERCENT=50
# varied (default) gives every request a unique deterministic prompt; fixed
# reproduces the earlier published run, which repeated one identical prompt and
# so let cached prefill work leak into the measurement.
PROMPT_MODE="varied"
ENDPOINT_1=""
ENDPOINT_2=""
INSECURE="false"
RESOURCE_TYPE_OVERRIDE=""
HOST_MPS_UNIT="nvidia-mps"
# Image pinning defaults to OFF.
#
# It was added to avoid a ~4 minute re-pull per mode, but it pulls the same
# 9.5 GB image concurrently with the vLLM Pods. Concurrent pulls of one image
# open separate containerd ingest sessions, so partial blobs pile up before
# commit, and an interrupted run leaves them behind. Observed on this node:
# disk went from 65% to 94% across two interrupted runs. Saving four minutes is
# not worth risking DiskPressure evictions mid-benchmark, so this is opt-in.
PIN_IMAGE="no"
# kubelet evicts unused images above this; pinning above it is dangerous.
IMAGE_GC_HIGH_PERCENT=85
DISK_SAFE_PERCENT=80
# Above this a run is likely to be evicted by DiskPressure part-way through,
# wasting the whole ~25 minutes. Overridable with --allow-full-disk.
DISK_ABORT_PERCENT=90
ALLOW_FULL_DISK="false"
# A cold 9.5 GB pull from the local registry takes minutes; the runner now waits
# for it so the pull happens once rather than racing the workload Pods.
PIN_IMAGE_TIMEOUT_SECONDS=900
# The probe image comes from Docker Hub, so a cold pull can exceed the old 180s.
PROBE_TIMEOUT_SECONDS=600
# Without the probe there is no GPU sampling and no same-GPU verification, so a
# missing probe aborts the run unless this is set.
ALLOW_NO_GPU_PROBE="false"
# Base image for the GPU probe. Interchangeable: nvidia-smi and the driver
# libraries are injected by the NVIDIA container runtime, not baked in, so any
# image with a shell works. Defaults to one confirmed present in the local
# registry, because this node cannot reach Docker Hub (403).
PROBE_IMAGE="127.0.0.1:5000/kserve/huggingfaceserver:v0.17.0-gpu"
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
pmon_pid=""
cpu_sampler_pid=""
timestamp=""
next_local_port=18000
port_forward_pids=()
RESOLVED_ENDPOINT=""
CURRENT_REPEAT=0

usage() {
  cat <<'EOF'
Usage: ./run-qwen-benchmark.sh [options]

Options:
  --mode MODES               comma-separated list, or one of the presets.
                             Presets:  all       sequential,mps,hami
                                       published sequential,mps,hami,hami-mps
                                       new       mem-only,mps-100,hami-mps-100,
                                                 asym-hami,asym-mps
                                       every     all nine modes
                             Modes:
                               sequential    one tenant at a time (baseline)
                               mem-only      2 tenants, gpumem only, NO MPS.
                                             Control that separates "the driver
                                             time-slices contexts" from "HAMi's
                                             launch gate forces the switches".
                               mps           2 tenants, MPS at 50% threads,
                                             no gpucores
                               mps-100       as mps but 100% threads
                                             (work-conserving)
                               hami          2 tenants, gpucores 50/50, no MPS
                               hami-mps      gpucores 50/50 + MPS at 50%
                               hami-mps-100  gpucores 50/50 + MPS at 100%
                               asym-hami     gpucores 75/25, no MPS. Control:
                                             does the limiter bind at all?
                               asym-mps      gpucores 75/25 + MPS at 100%.
                                             Decides whether gpucores binds
                                             UNDER MPS, or is blind because NVML
                                             reports at device level.
  --hami-profile PROFILE     balanced, conservative, weighted, compute-skew.
                             Ignored by the asym-* modes, which force
                             compute-skew.
  --repeats N                measured client passes per deployment (default: 1).
                             Repeats reuse the same Pods and are separated by
                             --cooldown-seconds.
  --cooldown-seconds N       idle time between repeat passes (default: 20)
  --mps-thread-percent N     CUDA_MPS_ACTIVE_THREAD_PERCENTAGE (default: 50).
                             Overridden per mode by the mps-100/*-100 modes.
  --prompt-mode MODE         varied (default) or fixed. fixed repeats one
                             identical prompt, reproducing the earlier run.
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
  --pin-image MODE           no (default), auto, or yes. Keeps the 9.5 GB vLLM
                             image resident between modes. Off by default: it
                             pulls concurrently with the workload Pods and can
                             fill the disk.
  --allow-full-disk          run even above 90% disk (DiskPressure may evict
                             Pods and invalidate the run)
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
    --mode|--modes)          MODE="$2"; shift 2 ;;
    --hami-profile)          HAMI_PROFILE="$2"; shift 2 ;;
    --repeats)               REPEATS="$2"; shift 2 ;;
    --cooldown-seconds)      COOLDOWN_SECONDS="$2"; shift 2 ;;
    --mps-thread-percent)    MPS_THREAD_PERCENT="$2"; shift 2 ;;
    --prompt-mode)           PROMPT_MODE="$2"; shift 2 ;;
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
    --allow-full-disk)       ALLOW_FULL_DISK="true"; shift ;;
    --allow-no-gpu-probe)    ALLOW_NO_GPU_PROBE="true"; shift ;;
    --probe-image)           PROBE_IMAGE="$2"; shift 2 ;;
    --probe-timeout)         PROBE_TIMEOUT_SECONDS="$2"; shift 2 ;;
    --pin-image-timeout)     PIN_IMAGE_TIMEOUT_SECONDS="$2"; shift 2 ;;
    --max-tokens)            MAX_TOKENS="$2"; shift 2 ;;
    --concurrency)           CONCURRENCY="$2"; shift 2 ;;
    --insecure)              INSECURE="true"; shift ;;
    -h|--help)               usage; exit 0 ;;
    *)                       usage >&2; fail "Unknown argument: $1" ;;
  esac
done

ALL_MODES="sequential mem-only mps mps-100 hami hami-mps hami-mps-100 asym-hami asym-mps"

# Expand presets, then validate every requested mode BEFORE touching the
# cluster. A typo in a nine-mode list must not be discovered 40 minutes in.
expand_modes() {
  local raw="$1"
  case "${raw}" in
    all)       printf 'sequential mps hami' ;;
    published) printf 'sequential mps hami hami-mps' ;;
    new)       printf 'mem-only mps-100 hami-mps-100 asym-hami asym-mps' ;;
    every)     printf '%s' "${ALL_MODES}" ;;
    *)         printf '%s' "${raw//,/ }" ;;
  esac
}

modes_to_run=()
read -r -a modes_to_run <<< "$(expand_modes "${MODE}")"
if [[ "${MODE}" == "all" && "${WITH_HAMI_MPS}" == "true" ]]; then
  modes_to_run+=(hami-mps)
fi
if [[ ${#modes_to_run[@]} -eq 0 ]]; then
  fail "No modes selected"
fi
for _mode in "${modes_to_run[@]}"; do
  case " ${ALL_MODES} " in
    *" ${_mode} "*) ;;
    *) fail "Unsupported mode: ${_mode} (valid: ${ALL_MODES// /, })" ;;
  esac
done

case "${HAMI_PROFILE}" in
  balanced|conservative|weighted|compute-skew) ;;
  *) fail "Unsupported HAMi profile: ${HAMI_PROFILE}" ;;
esac

case "${PROMPT_MODE}" in
  varied|fixed) ;;
  *) fail "Unsupported --prompt-mode: ${PROMPT_MODE} (varied, fixed)" ;;
esac

if ! [[ "${REPEATS}" =~ ^[1-9][0-9]*$ ]]; then
  fail "--repeats must be a positive integer, got: ${REPEATS}"
fi
if ! [[ "${MPS_THREAD_PERCENT}" =~ ^[0-9]+$ ]] \
   || (( MPS_THREAD_PERCENT < 1 || MPS_THREAD_PERCENT > 100 )); then
  fail "--mps-thread-percent must be 1-100, got: ${MPS_THREAD_PERCENT}"
fi

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

  printf '%s' "${raw}" |     MANIFEST="${ROOT_DIR}/${manifest}" OUT="${out}" PROFILE_NAME="${profile}"     SEQS="${MAX_NUM_SEQS}" MODELLEN="${MAX_MODEL_LEN}" MPSPCT="${MPS_THREAD_PERCENT}" run_python -c '
import json, os, pathlib, re, sys

profile = json.load(sys.stdin)
cores = profile["cores"]
mem = profile["mem"]
util = profile.get("util", [0.90, 0.90])
seqs = os.environ["SEQS"]
model_len = os.environ["MODELLEN"]
name = os.environ["PROFILE_NAME"]
mps_pct = os.environ.get("MPSPCT", "50")

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
    # MPS thread percentage. Matches the two-line env entry so only the value
    # belonging to this key is touched, never a number in a nearby comment.
    document = re.sub(
        r"(name:\s*CUDA_MPS_ACTIVE_THREAD_PERCENTAGE\s*\n\s*value:\s*)" + chr(34) + r"[0-9]+" + chr(34),
        lambda m: m.group(1) + chr(34) + str(mps_pct) + chr(34), document)
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

  # Only claim a gpucores value if the manifest actually carries the key. The
  # MPS-only manifests deliberately omit it, and logging "gpucores=50" for a
  # mode that sets no compute limit at all would misreport the experiment.
  local cores_note_1="${cores_1}" cores_note_2="${cores_2}"
  if ! grep -q 'nvidia\.com/gpucores' "${rendered}"; then
    cores_note_1="(unset - no HAMi compute limit)"
    cores_note_2="(unset - no HAMi compute limit)"
  fi
  local mps_note=""
  if grep -q 'CUDA_MPS_ACTIVE_THREAD_PERCENTAGE' "${rendered}"; then
    mps_note=" mps-threads=${MPS_THREAD_PERCENT}%"
  fi

  log "HAMi profile '${profile}' (rendered into ${manifest}, single apply):"
  log "  ${SERVICE_1}: gpucores=${cores_note_1} gpumem=${mem_1}MiB vllm-util=${util_1}${mps_note}"
  log "  ${SERVICE_2}: gpucores=${cores_note_2} gpumem=${mem_2}MiB vllm-util=${util_2}${mps_note}"

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

  if (( disk_use_percent >= DISK_ABORT_PERCENT )) && [[ "${ALLOW_FULL_DISK}" != "true" ]]; then
    printf 'ERROR: node disk is %s%%, at or above the %s%% abort threshold.

'       "${disk_use_percent}" "${DISK_ABORT_PERCENT}" >&2
    printf 'A run started here is likely to be cut short by DiskPressure evicting Pods,
' >&2
    printf 'which wastes ~25 minutes and produces results that cannot be trusted.

' >&2
    printf 'Reclaim space, largest wins first:

' >&2
    printf '  # partial blobs left by interrupted image pulls
' >&2
    printf '  sudo du -sh /var/lib/rancher/k3s/agent/containerd/io.containerd.content.v1.content/ingest
' >&2
    printf '  sudo ctr -a /run/k3s/containerd/containerd.sock -n k8s.io content prune references

' >&2
    printf '  # unused images
' >&2
    printf '  sudo k3s crictl rmi --prune

' >&2
    printf '  # journal is often several GB
' >&2
    printf '  sudo journalctl --vacuum-size=200M

' >&2
    printf '  # find anything else large
' >&2
    printf '  sudo du -xh --max-depth=1 / 2>/dev/null | sort -h | tail -15

' >&2
    printf 'Then re-run. To proceed anyway: --allow-full-disk
' >&2
    exit 1
  fi

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

  # BLOCK until the pinner holds the image.
  #
  # An earlier version deliberately did not wait, reasoning that mode 1 pulls the
  # image anyway so waiting bought nothing. That reasoning is wrong on a COLD
  # cache: not waiting means the pinner and mode 1 pull the SAME 9.5 GB image
  # simultaneously, opening separate containerd ingest sessions, while the GPU
  # probe pulls its own base image from Docker Hub at the same moment.
  #
  # Observed on this node with an empty image cache: the probe missed its
  # readiness window (so the runner reported "Physical GPUs visible: 0") and the
  # workload Pod sat in ContainerCreating for 26 minutes before the run was
  # abandoned.
  #
  # Waiting makes the pull SERIAL. It costs a few seconds when the image is
  # already resident, and it is the difference between a working run and a
  # stalled one when it is not.
  log "Waiting up to ${PIN_IMAGE_TIMEOUT_SECONDS}s for the image pinner to hold the vLLM image."
  log "  This serializes the 9.5 GB pull instead of racing the workload Pods for it."
  if kubectl rollout status -n "${NS}" deployment/qwen-benchmark-image-pinner \
       --timeout="${PIN_IMAGE_TIMEOUT_SECONDS}s" >/dev/null 2>&1; then
    log "Image pinner is holding the vLLM image; no mode should need to re-pull it"
  else
    log "WARNING: image pinner did not become Ready in ${PIN_IMAGE_TIMEOUT_SECONDS}s."
    log "  Continuing, but mode 1 will now compete with it for the same pull."
    kubectl get pods -n "${NS}" -l app=qwen-benchmark-image-pinner -o wide 2>/dev/null || true
  fi
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
  # Render the probe image so it can be pointed at whatever this node can
  # actually pull, without editing the manifest.
  local rendered="${RUN_DIR}/rendered-05-gpu-probe.yaml"
  sed "s|127.0.0.1:5000/kserve/huggingfaceserver:v0.17.0-gpu|${PROBE_IMAGE}|" \
    "${ROOT_DIR}/05-gpu-probe.yaml" > "${rendered}"
  log "GPU probe image: ${PROBE_IMAGE}"
  kubectl apply -f "${rendered}" >/dev/null
  log "Waiting up to ${PROBE_TIMEOUT_SECONDS}s for the GPU probe DaemonSet..."
  if ! kubectl rollout status -n "${NS}" daemonset/qwen-benchmark-gpu-probe \
        --timeout="${PROBE_TIMEOUT_SECONDS}s" >/dev/null 2>&1; then
    kubectl get pods -n "${NS}" -l app=qwen-benchmark-gpu-probe -o wide 2>/dev/null || true
    kubectl describe pods -n "${NS}" -l app=qwen-benchmark-gpu-probe 2>/dev/null \
      | sed -n '/Events:/,$p' | tail -15 || true
    # Previously this only warned. Continuing without the probe is worse than
    # stopping: GPU sampling, the GPU inventory and the same-GPU verification
    # all depend on it, so the run proceeds to produce numbers that cannot be
    # shown to describe two tenants on one device. The last run logged
    # "Physical GPUs visible on the node: 0" and kept going.
    if [[ "${ALLOW_NO_GPU_PROBE}" == "true" ]]; then
      log "WARNING: GPU probe not Ready; continuing without GPU sampling (--allow-no-gpu-probe)"
    else
      fail "GPU probe DaemonSet did not become Ready in ${PROBE_TIMEOUT_SECONDS}s.
  Without it there is no GPU sampling, no GPU inventory, and no same-GPU
  verification, so the run cannot demonstrate that the tenants shared one device.
  Its image (nvidia/cuda:12.4.1-base-ubuntu22.04) comes from Docker Hub: check
  connectivity, and check that a large concurrent image pull is not starving it.
  Re-run with --allow-no-gpu-probe to measure anyway."
    fi
  fi
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
    if [[ "${ALLOW_NO_GPU_PROBE}" != "true" ]]; then
      fail "No GPU probe Pod, so the benchmark cannot be confined to one GPU."
    fi
    log "WARNING: no GPU probe Pod; cannot confine the benchmark to one GPU"
    return 0
  fi

  gpu_uuids="$(kubectl exec -n "${NS}" "${gpu_probe_pod}" -c probe -- \
    nvidia-smi --query-gpu=index,uuid --format=csv,noheader 2>/dev/null \
    | tr -d ' \r' | grep . || true)"
  gpu_count="$(printf '%s\n' "${gpu_uuids}" | grep -c . || true)"
  gpu_count="${gpu_count:-0}"

  # Zero GPUs means nvidia-smi inside the probe returned nothing. Every
  # placement guarantee downstream is then vacuous, so stop rather than log
  # "Physical GPUs visible on the node: 0" and carry on measuring.
  if (( gpu_count == 0 )) && [[ "${ALLOW_NO_GPU_PROBE}" != "true" ]]; then
    fail "The GPU probe reported 0 GPUs. Check 'nvidia-smi -L' on the node and
  that the NVIDIA container runtime is injecting the driver into the probe Pod.
  Every same-GPU guarantee depends on this, so the run would be unverifiable."
  fi

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
  local saved_pct="${MPS_THREAD_PERCENT}"
  local profiles="${ROOT_DIR}/31-qwen-hami-compute-profiles.yaml"

  case "${mode}" in
    mem-only)
      # The control that separates driver-level context switching from HAMi's
      # launch gate. A stray MPS server would silently turn this into `mps`.
      ensure_mps_stopped
      kubectl apply -f "${ROOT_DIR}/22-qwen-mem-only.yaml" >/dev/null
      ;;
    mps|mps-100)
      if [[ "${mode}" == "mps-100" ]]; then MPS_THREAD_PERCENT=100; fi
      apply_mps_daemon
      kubectl apply -f "${profiles}" >/dev/null
      # Rendered through the profile machinery only to stamp the MPS thread
      # percentage and keep gpumem/util centralized. 21-qwen-mps.yaml carries no
      # gpucores key, so the profile's cores value has nothing to substitute
      # into and HAMi applies no compute limit - which is the point of the mode.
      apply_hami_manifest "21-qwen-mps.yaml" "balanced"
      ;;
    hami)
      # HAMi mode must be MPS-free or the gpucores result is unattributable.
      ensure_mps_stopped
      kubectl apply -f "${profiles}" >/dev/null
      apply_hami_manifest "30-qwen-hami.yaml" "${HAMI_PROFILE}"
      ;;
    asym-hami)
      # Does nvidia.com/gpucores bind AT ALL on this cluster, with no MPS in the
      # picture? compute-skew varies cores 75/25 and holds memory identical, so
      # a throughput split can only be the compute cap.
      ensure_mps_stopped
      kubectl apply -f "${profiles}" >/dev/null
      apply_hami_manifest "30-qwen-hami.yaml" "compute-skew"
      ;;
    hami-mps|hami-mps-100)
      if [[ "${mode}" == "hami-mps-100" ]]; then MPS_THREAD_PERCENT=100; fi
      apply_mps_daemon
      kubectl apply -f "${profiles}" >/dev/null
      apply_hami_manifest "40-qwen-hami-mps.yaml" "${HAMI_PROFILE}"
      ;;
    asym-mps)
      # The decisive experiment. MPS threads at 100% so MPS imposes NO split of
      # its own, leaving nvidia.com/gpucores 75/25 as the only asymmetry.
      #   throughput splits ~75/25 -> HAMi's limiter binds under MPS
      #   throughput splits ~50/50 -> the limiter is blind under MPS, because
      #                               NVML reports utilization at device level
      #                               once the MPS server owns the context
      MPS_THREAD_PERCENT=100
      apply_mps_daemon
      kubectl apply -f "${profiles}" >/dev/null
      apply_hami_manifest "40-qwen-hami-mps.yaml" "compute-skew"
      ;;
    *)
      fail "apply_mode_manifests does not handle mode: ${mode}"
      ;;
  esac

  MPS_THREAD_PERCENT="${saved_pct}"
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

  # Extra columns beyond memory/util:
  #   clocks.sm + temperature     - modes run back to back and the GPU heats up.
  #                                 Without these, a later mode that was clock
  #                                 throttled looks like a worse sharing mode.
  #   clocks_throttle_reasons     - says so explicitly rather than by inference.
  #   utilization.memory          - memory-controller busy, which separates
  #                                 "SMs are stalled on memory" from "SMs idle".
  printf 'timestamp,index,uuid,memory_total_mib,memory_used_mib,utilization_gpu_percent,utilization_memory_percent,clocks_sm_mhz,temperature_c,power_w,throttle_reasons\n' > "${out_file}"
  (
    kubectl exec -n "${NS}" "${gpu_probe_pod}" -c probe -- \
      nvidia-smi --query-gpu=timestamp,index,uuid,memory.total,memory.used,utilization.gpu,utilization.memory,clocks.sm,temperature.gpu,power.draw,clocks_throttle_reasons.active \
      --format=csv,noheader,nounits --loop-ms="${GPU_SAMPLE_INTERVAL_MS}" 2>/dev/null \
      >> "${out_file}"
  ) &
  sampler_pid=$!
}

# Per-process SM utilization.
#
# DCGM is not installed on this node and its profiling fields (SM_ACTIVE,
# SM_OCCUPANCY) are in any case generally unavailable on GeForce parts, so true
# SM occupancy cannot be measured here. `nvidia-smi pmon` does work on consumer
# cards and gives the next best thing: per-PROCESS sm%, which is what the
# asymmetric experiment actually needs.
#
# Two things to expect when reading the output:
#   * without MPS, each vLLM process appears separately, so a 75/25 gpucores
#     split should be visible directly as a per-process sm% split;
#   * with MPS, the work is attributed to nvidia-cuda-mps-server rather than to
#     the tenants. That is not a limitation of the tool - it is direct evidence
#     for the claim that NVML cannot attribute per process once MPS owns the
#     context, which is exactly why HAMi's limiter may be blind under MPS.
start_pmon_sampler() {
  local out_file="$1"
  pmon_pid=""
  if [[ -z "${gpu_probe_pod}" ]]; then
    return 0
  fi
  : > "${out_file}"
  (
    kubectl exec -n "${NS}" "${gpu_probe_pod}" -c probe -- \
      nvidia-smi pmon -s um -d 1 2>/dev/null >> "${out_file}"
  ) &
  pmon_pid=$!
}

# Node CPU. The user observed ~100% CPU during the HAMi mode, and HAMi-core's
# limiter is documented as a token-bucket rate limiter driven by an NVML-polling
# watcher thread. If that polling saturates the node, vLLM's Python scheduling
# loop is starved and throughput falls for reasons that have nothing to do with
# GPU context switching. Without this series that alternative explanation cannot
# be ruled out, and it predicts the same throughput drop.
start_cpu_sampler() {
  local out_file="$1"
  cpu_sampler_pid=""
  printf 'epoch,cpu_busy_percent,load1,procs_running\n' > "${out_file}"
  (
    OUT="${out_file}" run_python -c '
import os, time

out = open(os.environ["OUT"], "a", buffering=1)


def snapshot():
    with open("/proc/stat") as handle:
        fields = [int(value) for value in handle.readline().split()[1:]]
    idle = fields[3] + (fields[4] if len(fields) > 4 else 0)
    return sum(fields), idle


previous_total, previous_idle = snapshot()
while True:
    time.sleep(1.0)
    total, idle = snapshot()
    delta_total = total - previous_total
    delta_idle = idle - previous_idle
    previous_total, previous_idle = total, idle
    busy = 100.0 * (delta_total - delta_idle) / delta_total if delta_total else 0.0
    with open("/proc/loadavg") as handle:
        load = handle.read().split()
    print("%.3f,%.2f,%s,%s" % (time.time(), busy, load[0], load[3].split("/")[0]),
          file=out, flush=True)
' 2>/dev/null
  ) &
  cpu_sampler_pid=$!
}

# Per-container CPU accounting, including CFS throttling. cgroup v2 on this node
# (confirmed by preflight), so inside the container /sys/fs/cgroup/cpu.stat is
# that container own cgroup and carries nr_throttled and throttled_usec.
#
# Each Pod is limited to cpu: "2". If HAMi's polling thread competes with vLLM
# inside that 2-core budget, the container is CFS throttled and decode slows -
# again with no GPU-side cause. A non-zero throttled_usec delta in a HAMi mode
# and a zero one in the MPS mode would settle it.
snapshot_pod_cpu() {
  local pod="$1" out_file="$2"
  if [[ -z "${pod}" ]]; then
    return 0
  fi
  {
    printf 'epoch %s\n' "$(date +%s.%N)"
    kubectl exec -n "${NS}" "${pod}" -c main -- cat /sys/fs/cgroup/cpu.stat 2>/dev/null \
      || printf 'unavailable 1\n'
  } > "${out_file}" 2>/dev/null || true
}

stop_gpu_sampler() {
  local pid
  for pid in "${sampler_pid}" "${pmon_pid}" "${cpu_sampler_pid}"; do
    if [[ -n "${pid}" ]]; then
      kill "${pid}" >/dev/null 2>&1 || true
      wait "${pid}" 2>/dev/null || true
    fi
  done
  sampler_pid=""; pmon_pid=""; cpu_sampler_pid=""
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
    --prompt-mode "${PROMPT_MODE}"
    # Repeat passes reuse the same Pods, so shift the prompt sequence per pass:
    # otherwise pass 2 replays pass 1's prompts against an already-warm server.
    --prompt-offset "$(( ${CURRENT_REPEAT:-0} * 100000 ))"
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
# Pick the pass that gets published, and record the spread across passes.
#
# The published number is a REAL measured pass, never an average: averaging
# throughput, utilization and latency across passes produces a composite that
# never occurred, and its internal relationships (tok/s vs util vs p95) no
# longer reconcile. The median pass is a genuine observation; the spread beside
# it is what tells the reader whether a difference between modes is meaningful.
select_representative_pass() {
  local mode_dir="$1"
  MODE_DIR="${mode_dir}" run_python -c '
import json, os, pathlib, shutil, statistics

mode_dir = pathlib.Path(os.environ["MODE_DIR"])
scored = []
for pass_dir in sorted(mode_dir.glob("pass-*")):
    result = pass_dir / "client.result.json"
    if not result.exists() or not result.stat().st_size:
        continue
    try:
        summary = json.loads(result.read_text(encoding="utf-8"))["summary"]
    except (ValueError, KeyError):
        continue
    scored.append((summary.get("output_tokens_per_second", 0.0), pass_dir))

if not scored:
    print("no usable passes")
    raise SystemExit(0)

scored.sort(key=lambda item: item[0])
chosen = scored[(len(scored) - 1) // 2][1]
for name in ("client.result.json", "client.result.csv", "client.result.responses.txt",
             "client.result.stdout.txt", "gpu-timeline.csv", "gpu-pmon.txt",
             "node-cpu.csv"):
    source = chosen / name
    if source.exists():
        shutil.copy2(source, mode_dir / name)
# Per-Pod cgroup snapshots carry the service name, so copy them by pattern.
for source in chosen.glob("cpu-*.txt"):
    shutil.copy2(source, mode_dir / source.name)

values = [value for value, _ in scored]
median = statistics.median(values)
report = {
    "passes": len(scored),
    "representative_pass": chosen.name,
    "aggregate_tokens_per_second": {
        "values": values,
        "median": median,
        "min": min(values),
        "max": max(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "spread_percent": ((max(values) - min(values)) / median * 100) if median else 0.0,
    },
}
(mode_dir / "repeats.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print("%d pass(es), published %s, spread %.2f%% (%.1f-%.1f tok/s)" % (
    len(scored), chosen.name, report["aggregate_tokens_per_second"]["spread_percent"],
    min(values), max(values)))
'
}

# Sequential equivalent of select_representative_pass: promote the median pass
# of one phase, plus its matching GPU timeline, to the canonical phase file.
promote_median_phase() {
  local mode_dir="$1" index="$2" out_json="$3"
  MODE_DIR="${mode_dir}" IDX="${index}" OUT="${out_json}" run_python -c '
import json, os, pathlib, shutil, statistics

mode_dir = pathlib.Path(os.environ["MODE_DIR"])
index = os.environ["IDX"]
scored = []
for path in sorted(mode_dir.glob("client.phase-" + index + ".pass-*.json")):
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))["summary"]
    except (ValueError, KeyError):
        continue
    scored.append((summary.get("output_tokens_per_second", 0.0), path))

if not scored:
    print("phase " + index + ": no usable passes")
    raise SystemExit(0)

scored.sort(key=lambda item: item[0])
chosen = scored[(len(scored) - 1) // 2][1]
shutil.copy2(chosen, os.environ["OUT"])
suffix = chosen.name.split(".pass-")[1].split(".json")[0]
timeline = mode_dir / ("gpu-timeline.phase-" + index + ".pass-" + suffix + ".csv")
if timeline.exists():
    shutil.copy2(timeline, mode_dir / ("gpu-timeline.phase-" + index + ".csv"))

values = [value for value, _ in scored]
median = statistics.median(values)
print("phase %s: %d pass(es), published pass-%s, spread %.2f%%" % (
    index, len(scored), suffix,
    ((max(values) - min(values)) / median * 100) if median else 0.0))
'
}

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
def parse_smi_timestamp(text):
    """nvidia-smi stamps local time as YYYY/MM/DD HH:MM:SS.mmm."""
    import datetime
    try:
        return datetime.datetime.strptime(text.strip(), "%Y/%m/%d %H:%M:%S.%f").timestamp()
    except ValueError:
        return None


# Restrict every host-level series to the window the client was actually
# measuring in. The samplers start before the client, which then spends time on
# URL probing and warm-up; averaging those in dilutes utilization by a different
# amount in every mode, because warm-up length is not constant across modes.
client_summary = client.get("summary", {}) if isinstance(client, dict) else {}
window_start = client_summary.get("measure_start_epoch")
window_end = client_summary.get("measure_end_epoch")


def in_window(epoch):
    if epoch is None or window_start is None or window_end is None:
        return True
    return window_start <= epoch <= window_end


timeline_path = mode_dir / "gpu-timeline.csv"
if timeline_path.exists():
    # Rows are timestamp,index,uuid,total,used,util,util_mem,clocks,temp,power,
    # throttle - one per GPU per tick. Keep only the GPU the workload actually
    # ran on, otherwise an idle device drags peak/mean toward zero (which is how
    # MPS once scored 74 MiB).
    target = os.environ.get("TARGET_GPU", "")
    per_gpu = {}
    clipped = 0
    total_rows = 0
    for line in timeline_path.read_text(encoding="utf-8", errors="replace").splitlines()[1:]:
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 6:
            continue
        total_rows += 1
        stamp = parse_smi_timestamp(parts[0])
        if not in_window(stamp):
            clipped += 1
            continue
        uuid = parts[2]
        try:
            entry = per_gpu.setdefault(uuid, {
                "total": None, "used": [], "util": [], "util_mem": [],
                "clocks": [], "temp": [], "power": [], "throttle": set(),
            })
            entry["total"] = int(float(parts[3]))
            entry["used"].append(int(float(parts[4])))
            entry["util"].append(int(float(parts[5])))
        except ValueError:
            continue
        # Extended columns are best effort: an older driver may not report all
        # of them, and a missing clock must not discard an otherwise good row.
        for position, key, cast in ((6, "util_mem", float), (7, "clocks", float),
                                    (8, "temp", float), (9, "power", float)):
            if len(parts) > position:
                try:
                    entry[key].append(cast(parts[position]))
                except ValueError:
                    pass
        if len(parts) > 10 and parts[10] not in ("", "[N/A]"):
            entry["throttle"].add(parts[10])

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
            def average(values):
                return round(sum(values) / len(values), 1) if values else None

            gpu = {
                "samples": len(used),
                "sampled_gpu_uuid": chosen,
                "target_gpu_uuid": target,
                "sampled_target_gpu": bool(target) and chosen == target,
                "memory_total_mib": entry["total"],
                "peak_memory_used_mib": max(used),
                "mean_memory_used_mib": round(sum(used) / len(used), 1),
                "peak_utilization_percent": max(util) if util else None,
                "mean_utilization_percent": average(util),
                "all_gpus_seen": sorted(per_gpu),
                # Clipping bookkeeping, so a reader can tell whether the window
                # actually applied or the whole timeline was used.
                "samples_before_window_clip": total_rows,
                "samples_clipped_outside_window": clipped,
                "window_start_epoch": window_start,
                "window_end_epoch": window_end,
                # SATURATION: what fraction of the measured window was the GPU
                # pinned? mean utilization alone cannot distinguish "busy the
                # whole time at 90%" from "at 100% half the time and idle the
                # rest", and those are different sharing outcomes.
                "saturation_fraction_ge_95": (
                    round(sum(1 for value in util if value >= 95) / len(util), 4)
                    if util else None),
                "saturation_fraction_ge_99": (
                    round(sum(1 for value in util if value >= 99) / len(util), 4)
                    if util else None),
                "idle_fraction_le_5": (
                    round(sum(1 for value in util if value <= 5) / len(util), 4)
                    if util else None),
                # Memory-controller busy. High util.gpu with high util.memory is
                # a memory-bound kernel; high util.gpu with LOW util.memory and
                # low throughput is the signature of time lost between kernels.
                "mean_memory_utilization_percent": average(entry["util_mem"]),
                # Thermals and clocks. Modes run back to back, so without these
                # a later mode that was clock throttled is indistinguishable
                # from a genuinely worse sharing mechanism.
                "mean_sm_clock_mhz": average(entry["clocks"]),
                "min_sm_clock_mhz": min(entry["clocks"]) if entry["clocks"] else None,
                "max_temperature_c": max(entry["temp"]) if entry["temp"] else None,
                "mean_power_w": average(entry["power"]),
                "throttle_reasons_seen": sorted(entry["throttle"]),
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

# ---------------------------------------------------------------------------
# Per-process SM utilization (nvidia-smi pmon).
#
# Columns: gpu pid type sm mem enc dec command; header lines start with "#".
# Without MPS this attributes SM time per vLLM process, so an asymmetric
# gpucores split shows up directly as a per-process split. With MPS the work is
# attributed to nvidia-cuda-mps-server instead - which is not a shortcoming of
# the tool but direct evidence that NVML cannot see per-tenant usage once MPS
# owns the context, the reason the HAMi limiter may be blind under MPS.
# ---------------------------------------------------------------------------
pmon = {"samples": 0, "per_process": {}, "attributed_to_mps_server": False}
pmon_path = mode_dir / "gpu-pmon.txt"
if pmon_path.exists():
    observed = {}
    sample_count = 0
    for line in pmon_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 8:
            continue
        try:
            sm_percent = float(parts[3])
        except ValueError:
            continue  # "-" whenever the process was not resident on that tick
        observed.setdefault(parts[-1] + ":" + parts[1], []).append(sm_percent)
        sample_count += 1
    for key, values in observed.items():
        pmon["per_process"][key] = {
            "samples": len(values),
            "mean_sm_percent": round(sum(values) / len(values), 1),
            "peak_sm_percent": max(values),
        }
    pmon["samples"] = sample_count
    pmon["attributed_to_mps_server"] = any("mps-server" in key for key in pmon["per_process"])
    tenant_means = sorted(
        (entry["mean_sm_percent"] for key, entry in pmon["per_process"].items()
         if "mps-server" not in key), reverse=True)
    if len(tenant_means) >= 2 and sum(tenant_means):
        # The number that answers the asymmetric experiment: with gpucores
        # 75/25 this should read near [0.75, 0.25] if the limiter binds, and
        # near [0.5, 0.5] if it does not.
        pmon["tenant_sm_share_split"] = [
            round(value / sum(tenant_means), 3) for value in tenant_means]

# ---------------------------------------------------------------------------
# CPU. The alternative explanation for the HAMi throughput loss that has
# nothing to do with the GPU: the HAMi-core limiter runs an NVML-polling watcher
# thread, each Pod is capped at cpu: "2", and a CPU-starved vLLM decodes slower.
# ---------------------------------------------------------------------------
cpu = {"samples": 0}
cpu_path = mode_dir / "node-cpu.csv"
if cpu_path.exists():
    busy_values, load_values = [], []
    for line in cpu_path.read_text(encoding="utf-8", errors="replace").splitlines()[1:]:
        parts = line.split(",")
        if len(parts) < 4:
            continue
        try:
            epoch, busy, load1 = float(parts[0]), float(parts[1]), float(parts[2])
        except ValueError:
            continue
        if not in_window(epoch):
            continue
        busy_values.append(busy)
        load_values.append(load1)
    if busy_values:
        cpu = {
            "samples": len(busy_values),
            "mean_busy_percent": round(sum(busy_values) / len(busy_values), 1),
            "peak_busy_percent": round(max(busy_values), 1),
            "saturation_fraction_ge_90": round(
                sum(1 for value in busy_values if value >= 90) / len(busy_values), 4),
            "mean_load1": round(sum(load_values) / len(load_values), 2),
        }


def read_cpu_stat(path):
    values = {}
    if path is None or not path.exists():
        return values
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) == 2:
            try:
                values[parts[0]] = float(parts[1])
            except ValueError:
                pass
    return values


def first_match(pattern):
    matches = sorted(mode_dir.glob(pattern))
    return matches[0] if matches else None


# cgroup v2: throttled_usec accumulating across the run means the container hit
# its CPU quota, and decode slowed for a reason that is not GPU contention.
pod_cpu = {}
for service in (svc1, svc2):
    before = read_cpu_stat(first_match("cpu-" + service + "*before*.txt"))
    after = read_cpu_stat(first_match("cpu-" + service + "*after*.txt"))
    if not before or not after:
        continue
    entry = {}
    for key in ("usage_usec", "user_usec", "system_usec", "nr_periods",
                "nr_throttled", "throttled_usec"):
        if key in before and key in after:
            entry[key + "_delta"] = after[key] - before[key]
    periods = entry.get("nr_periods_delta") or 0
    entry["throttled_period_fraction"] = (
        round(entry.get("nr_throttled_delta", 0) / periods, 4) if periods else None)
    entry["cpu_seconds_used"] = round(entry.get("usage_usec_delta", 0) / 1e6, 2)
    entry["was_cpu_throttled"] = bool(entry.get("nr_throttled_delta", 0) > 0)
    pod_cpu[service] = entry

# ---------------------------------------------------------------------------
# SM occupancy proxy.
#
# True occupancy needs DCGM profiling fields (DCGM_FI_PROF_SM_ACTIVE /
# SM_OCCUPANCY), which NVIDIA does not expose on this GeForce part. What is
# available is the ratio of useful work to busy time: utilization.gpu marks a
# sample busy when ANY kernel is resident, so it measures TIME, not WIDTH. Two
# modes with equal utilization but different token rates differ in how much of
# the SM array did useful work - the quantity occupancy would have reported.
# ---------------------------------------------------------------------------
occupancy_proxy = None
_mean_util = gpu.get("mean_utilization_percent")
_tokens_per_second = client_summary.get("output_tokens_per_second")
if _mean_util and _tokens_per_second:
    occupancy_proxy = {
        "tokens_per_second_per_utilization_point": round(_tokens_per_second / _mean_util, 3),
        "is_proxy": True,
        "note": ("proxy only; true SM occupancy requires DCGM "
                 "DCGM_FI_PROF_SM_ACTIVE / DCGM_FI_PROF_SM_OCCUPANCY, "
                 "unavailable on this GPU"),
    }

repeats = {}
if (mode_dir / "repeats.json").exists():
    repeats = json.loads((mode_dir / "repeats.json").read_text(encoding="utf-8"))

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
    "mps_expected": mode in ("mps", "mps-100", "hami-mps", "hami-mps-100", "asym-mps"),
    # ERROR RATE as an acceptance criterion, not a footnote: a mode that wins on
    # throughput while dropping requests has not won.
    "error_rate": client_summary.get("error_rate", 0),
    "error_breakdown": client_summary.get("errors", {}),
    "error_free": client_summary.get("failed_requests", 0) == 0,
    # If the client burned about one core per in-flight thread, the measurement
    # is of the client, not the GPU, and the mode comparison is void.
    "client_bound_suspected": client_summary.get("client_bound_suspected", False),
    "client_saturation_ratio": client_summary.get("client_saturation_ratio"),
    # Prompts must be unique per request, or cached prefill leaks into results.
    "distinct_prompts": client_summary.get("distinct_prompts"),
    "prompt_mode": client_summary.get("prompt_mode"),
    # Contended-window coverage. If this is far below 1.0 the tenants spent much
    # of the run NOT overlapping, and whole-run throughput overstates sharing.
    "contended_fraction_of_wall": (
        (client_summary.get("contended") or {}).get("fraction_of_wall")),
    # A GPU clock well below its ceiling means a later mode was penalized by
    # heat rather than by its sharing mechanism.
    "min_sm_clock_mhz": gpu.get("min_sm_clock_mhz"),
    "gpu_throttle_reasons_seen": gpu.get("throttle_reasons_seen", []),
    # Either Pod hitting its cgroup CPU quota is an alternative explanation for
    # a throughput difference that has nothing to do with GPU sharing.
    "any_pod_cpu_throttled": any(
        entry.get("was_cpu_throttled") for entry in pod_cpu.values()),
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
    "gpu_per_process": pmon,
    "sm_occupancy_proxy": occupancy_proxy,
    "node_cpu": cpu,
    "pod_cpu": pod_cpu,
    "repeats": repeats,
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
      local repeat pass_json
      for (( repeat = 1; repeat <= REPEATS; repeat++ )); do
        CURRENT_REPEAT=$(( repeat - 1 ))
        pass_json="${mode_dir}/client.phase-${index}.pass-${repeat}.json"
        log "Sequential phase ${index}: pass ${repeat}/${REPEATS}"
        snapshot_pod_cpu "${pod_name}" \
          "${mode_dir}/cpu-${service}-phase${index}-pass${repeat}-before.txt"
        start_gpu_sampler "${mode_dir}/gpu-timeline.phase-${index}.pass-${repeat}.csv"
        start_pmon_sampler "${mode_dir}/gpu-pmon.phase-${index}.pass-${repeat}.txt"
        start_cpu_sampler "${mode_dir}/node-cpu.phase-${index}.pass-${repeat}.csv"
        run_client "${mode}" "${pass_json}" true "--endpoint=${service}=${endpoint}" || status=$?
        stop_gpu_sampler
        snapshot_pod_cpu "${pod_name}" \
          "${mode_dir}/cpu-${service}-phase${index}-pass${repeat}-after.txt"
        if (( repeat < REPEATS )); then
          log "Cooling down ${COOLDOWN_SECONDS}s before the next pass"
          sleep "${COOLDOWN_SECONDS}"
        fi
      done
      CURRENT_REPEAT=0
      log "$(promote_median_phase "${mode_dir}" "${index}" "${phase_json}")"
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
    local repeat pass_dir
    for (( repeat = 1; repeat <= REPEATS; repeat++ )); do
      CURRENT_REPEAT=$(( repeat - 1 ))
      pass_dir="${mode_dir}/pass-${repeat}"
      mkdir -p "${pass_dir}"
      log "Running concurrent client workload for ${mode} (pass ${repeat}/${REPEATS})"
      snapshot_pod_cpu "${pod_1}" "${pass_dir}/cpu-${SERVICE_1}-before.txt"
      snapshot_pod_cpu "${pod_2}" "${pass_dir}/cpu-${SERVICE_2}-before.txt"
      start_gpu_sampler "${pass_dir}/gpu-timeline.csv"
      start_pmon_sampler "${pass_dir}/gpu-pmon.txt"
      start_cpu_sampler "${pass_dir}/node-cpu.csv"
      run_client "${mode}" "${pass_dir}/client.result.json" false \
        "--endpoint=${SERVICE_1}=${endpoint_1}" "--endpoint=${SERVICE_2}=${endpoint_2}" || status=$?
      stop_gpu_sampler
      snapshot_pod_cpu "${pod_1}" "${pass_dir}/cpu-${SERVICE_1}-after.txt"
      snapshot_pod_cpu "${pod_2}" "${pass_dir}/cpu-${SERVICE_2}-after.txt"
      if (( repeat < REPEATS )); then
        log "Cooling down ${COOLDOWN_SECONDS}s before the next pass"
        sleep "${COOLDOWN_SECONDS}"
      fi
    done
    CURRENT_REPEAT=0
    log "Repeats: $(select_representative_pass "${mode_dir}")"
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

# Every mode now runs through HAMi's device plugin (it owns nvidia.com/gpu on
# this cluster), and every mode's manifests carry nvidia.com/gpumem, so validate
# the HAMi components for all of them rather than only the gpucores modes.
validate_hami_components

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
log "Modes: ${modes_to_run[*]}"
log "HAMi profile: ${HAMI_PROFILE} | MPS threads: ${MPS_THREAD_PERCENT}% | prompts: ${PROMPT_MODE}"
log "Repeats: ${REPEATS} in-place pass(es) per deployment, ${COOLDOWN_SECONDS}s cooldown between passes"
log "Workload: ${REQUESTS_PER_SERVICE} req/service x ${MAX_TOKENS} tokens, concurrency ${CONCURRENCY}/service, warmup ${WARMUP_REQUESTS}"

check_disk_headroom
apply_image_pinner

# modes_to_run was resolved and validated during argument parsing, before the
# cluster was touched, so a typo in a nine-mode list fails immediately rather
# than forty minutes into a run.
overall_status=0
mode_index=0
for mode in "${modes_to_run[@]}"; do
  mode_index=$(( mode_index + 1 ))
  log "=== Mode ${mode_index}/${#modes_to_run[@]}: ${mode} ==="
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
