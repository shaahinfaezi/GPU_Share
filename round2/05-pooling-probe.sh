#!/usr/bin/env bash
# Settle the pooling question, and find out what the GPU can actually report.
#
# WHY THIS EXISTS
# Preflight section 4 came back with "routes: none found" and an empty flag
# list. That was my probe failing, not vLLM answering -- see probe_vllm.py for
# both bugs. This runs the corrected probe, which interrogates the argparse
# parser and the FastAPI router as objects.
#
# It also inventories DCGM, because requirement 1 asks for SM OCCUPANCY and
# DCGM_FI_DEV_GPU_UTIL is not that. That metric means "was any kernel
# resident", and round one already found it pinned at 95-99% in nearly every
# configuration, so it cannot separate two tenants. Real occupancy lives in
# DCGM_FI_PROF_*, which is a datacenter-GPU feature that may not exist on a
# GeForce RTX 3080 Ti. Better to learn that now than while staring at an empty
# panel during the video.
#
#   ./05-pooling-probe.sh          # both checks
#   ./05-pooling-probe.sh --dcgm   # metric inventory only (no pod, ~5s)
#   ./05-pooling-probe.sh --vllm   # vLLM capability probe only
set -uo pipefail

NS="${NS:-llm-demo}"
IMG="${IMG:-127.0.0.1:5000/kserve/huggingfaceserver:v0.17.0-gpu}"
HERE="$(cd "$(dirname "$0")" && pwd)"
DO_VLLM=1
DO_DCGM=1

for a in "$@"; do
  case "$a" in
    --dcgm) DO_VLLM=0 ;;
    --vllm) DO_DCGM=0 ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown option: $a" >&2; exit 2 ;;
  esac
done

hr() { printf '\n=== %s\n' "$1"; }

# --------------------------------------------------------------- vLLM probe
if [[ "${DO_VLLM}" -eq 1 ]]; then
  hr "A. what this vLLM build can actually do"
  kubectl delete pod r2-probe -n "${NS}" --ignore-not-found >/dev/null 2>&1
  kubectl run r2-probe -n "${NS}" --restart=Never --image="${IMG}" \
    --command -- sleep 600 >/dev/null 2>&1
  echo "--- waiting for r2-probe (image is resident, so this is seconds)"
  if kubectl wait --for=condition=Ready pod/r2-probe -n "${NS}" --timeout=600s >/dev/null 2>&1; then
    kubectl exec -i -n "${NS}" r2-probe -- python - < "${HERE}/probe_vllm.py" 2>&1
  else
    echo "  probe pod never became Ready:"
    kubectl describe pod r2-probe -n "${NS}" 2>&1 | tail -15
  fi
  kubectl delete pod r2-probe -n "${NS}" --ignore-not-found --wait=false >/dev/null 2>&1
fi

# --------------------------------------------------------------- DCGM probe
if [[ "${DO_DCGM}" -eq 1 ]]; then
  hr "B. what the GPU can report (the SM-occupancy question)"

  IP_HAMI="$(kubectl -n kube-system get svc my-hami-webui-dcgm-exporter \
    -o jsonpath='{.spec.clusterIP}' 2>/dev/null)"
  IP_OPER="$(kubectl -n gpu-operator get svc -l app=nvidia-dcgm-exporter \
    -o jsonpath='{.items[0].spec.clusterIP}' 2>/dev/null)"
  if [[ -z "${IP_OPER}" ]]; then
    IP_OPER="$(kubectl -n gpu-operator get svc -o jsonpath='{range .items[*]}{.metadata.name}{" "}{.spec.clusterIP}{"\n"}{end}' 2>/dev/null \
      | grep -i dcgm | awk '{print $2}' | head -1)"
  fi

  probe_exporter() {
    local label="$1" ip="$2"
    echo
    echo "--- ${label} exporter @ ${ip:-<not found>}:9400"
    if [[ -z "${ip}" ]]; then echo "    not found; skipped"; return; fi
    local body n prof
    body="$(curl -s -m 10 "http://${ip}:9400/metrics" 2>/dev/null)"
    n="$(printf '%s' "${body}" | grep -c '^DCGM_' 2>/dev/null)"
    echo "    DCGM series          : ${n:-0}"
    if [[ "${n:-0}" -eq 0 ]]; then
      echo "    unreachable from the host netns (ClusterIP). Retrying in-cluster:"
      kubectl run r2-curl --rm -i --restart=Never -n "${NS}" \
        --image="${IMG}" --command -- \
        bash -lc "curl -s -m 10 http://${ip}:9400/metrics | grep -c '^DCGM_'" 2>/dev/null \
        | sed 's/^/      /'
      return
    fi
    prof="$(printf '%s' "${body}" | grep -o '^DCGM_FI_PROF_[A-Z_0-9]*' | sort -u)"
    if [[ -n "${prof}" ]]; then
      echo "    PROFILING metrics    : PRESENT"
      printf '%s\n' "${prof}" | sed 's/^/        /'
    else
      echo "    PROFILING metrics    : ABSENT   <== no true SM occupancy here"
    fi
    echo "    device metrics       :"
    printf '%s' "${body}" | grep -o '^DCGM_FI_DEV_[A-Z_0-9]*' | sort -u | head -24 | sed 's/^/        /'
  }

  probe_exporter "hami-webui  (v3.5.0)" "${IP_HAMI}"
  probe_exporter "gpu-operator (v4.1.1)" "${IP_OPER}"

  echo
  echo "--- can NVML report PER-PROCESS SM utilization on this card?"
  echo "    This is the per-tenant number no device-level metric can give."
  echo "    Caveat: under MPS every client's work is submitted by the single"
  echo "    nvidia-cuda-mps-server process, so NVML attributes it all there."
  echo "    Per-tenant attribution therefore works for the dedicated and HAMi"
  echo "    arms and collapses under hami-mps -- which is itself worth stating."
  if command -v nvidia-smi >/dev/null 2>&1; then
    python3 - <<'NVPY' 2>/dev/null || echo "    pynvml not importable on the host (that is fine; see below)"
try:
    import pynvml as N
except Exception:
    raise SystemExit(1)
N.nvmlInit()
for i in range(N.nvmlDeviceGetCount()):
    h = N.nvmlDeviceGetHandleByIndex(i)
    try:
        s = N.nvmlDeviceGetProcessUtilization(h, 0)
        print("    GPU %d: SUPPORTED, %d sample(s)" % (i, len(s)))
        for p in list(s)[:4]:
            print("        pid=%s smUtil=%s%% memUtil=%s%%" % (p.pid, p.smUtil, p.memUtil))
    except Exception as e:
        print("    GPU %d: NOT SUPPORTED (%s)" % (i, type(e).__name__))
NVPY
    echo "    nvidia-smi pmon fallback (same NVML counter, no python needed):"
    timeout 8 nvidia-smi pmon -c 2 2>&1 | head -8 | sed 's/^/      /'
  else
    echo "    nvidia-smi not on PATH here"
  fi

  echo
  echo "--- MPS daemon flags currently in effect"
  ps -eo pid,user,args 2>/dev/null | grep -E "nvidia-cuda-mps-(control|server)" \
    | grep -v grep | cut -c1-120 | sed 's/^/    /'
  if ps -eo args 2>/dev/null | grep -v grep | grep -q -- "-multiuser-server"; then
    echo "    -multiuser-server : PRESENT"
  else
    echo "    -multiuser-server : ABSENT"
    echo "      Harmless while every tenant runs as one uid, which is true today"
    echo "      and was true for all of round one. It stops being harmless the"
    echo "      moment an arm runs two different uids: round one measured the"
    echo "      newcomer hanging in CUDA init with no error, backing off to 51s."
  fi
fi

hr "what this decides"
cat <<'VERDICT'
  POOLING
    routes include /v1/embeddings plus a rerank-or-score path
        -> pooling tenants stay LLMInferenceService; render.py needs no change.
    routes absent, or KServe whitelists rather than forwards args
        -> render.py --pooling-via vllm, the fallback already chosen.
    Either way the decisive test is a real load:
        ./verify-models.sh --serve embed
        ./verify-models.sh --serve rerank

  SM OCCUPANCY
    DCGM_FI_PROF_* present
        -> genuine occupancy; add DCGM_FI_PROF_SM_ACTIVE and SM_OCCUPANCY
           panels to the dashboard.
    DCGM_FI_PROF_* absent (the expected result on GeForce)
        -> SM occupancy is not measurable on this hardware. Report that plainly
           instead of passing DCGM_FI_DEV_GPU_UTIL off as occupancy. The honest
           substitutes are achieved tokens/s per tenant, vllm:num_requests_
           running, and the A:B throughput ratio round one already used.
VERDICT
