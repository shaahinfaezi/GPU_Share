#!/usr/bin/env bash
# Round-2 preflight — run this on the GPU box FIRST.
#
# It answers, from the cluster itself, the questions that decide how the rest of
# round two has to be written. Nothing here changes any state; it only reads.
# Paste the output back and the manifests can be finalised against facts rather
# than assumptions.
#
#   ./00-preflight.sh            # human-readable
#   ./00-preflight.sh > pf.txt   # and send pf.txt
set -uo pipefail
NS="${NS:-llm-demo}"
DATA_DIR="${DATA_DIR:-/data2}"
IMG="${IMG:-127.0.0.1:5000/kserve/huggingfaceserver:v0.17.0-gpu}"

hr() { printf '\n=== %s %s\n' "$1" "$(printf '%.0s-' $(seq 1 $((60 - ${#1})) ) )"; }
have() { command -v "$1" >/dev/null 2>&1; }

hr "1. node and cluster shape"
if have kubectl; then
  kubectl get nodes -o wide 2>&1 | head -10
  echo "--- node count: $(kubectl get nodes --no-headers 2>/dev/null | wc -l)"
  echo "--- GPU nodes:"
  kubectl get nodes -o json 2>/dev/null \
    | grep -o '"nvidia.com/gpu":"[0-9]*"' | sort | uniq -c
else
  echo "kubectl NOT FOUND"
fi

hr "2. ${DATA_DIR} — capacity for the model cache"
if [[ -d "${DATA_DIR}" ]]; then
  df -h "${DATA_DIR}"
  echo "--- writable by this user: $([[ -w ${DATA_DIR} ]] && echo yes || echo NO)"
  echo "--- filesystem: $(stat -f -c %T "${DATA_DIR}" 2>/dev/null)"
  echo "--- existing contents:"; ls -la "${DATA_DIR}" 2>/dev/null | head -10
else
  echo "${DATA_DIR} DOES NOT EXIST on this host"
fi
echo "--- root filesystem for comparison:"; df -h / | tail -1

hr "3. where the CURRENT model PVC actually lives"
if have kubectl; then
  kubectl get pvc -n "${NS}" 2>&1 | head
  pv="$(kubectl get pvc qwen-local-pv-claim -n "${NS}" -o jsonpath='{.spec.volumeName}' 2>/dev/null)"
  if [[ -n "${pv}" ]]; then
    echo "--- backing PV: ${pv}"
    kubectl get pv "${pv}" -o jsonpath='{.spec.hostPath.path}{"\n"}' 2>/dev/null
    kubectl get pv "${pv}" -o jsonpath='{.spec.local.path}{"\n"}' 2>/dev/null
    kubectl get pv "${pv}" -o jsonpath='storageClass={.spec.storageClassName}{"\n"}' 2>/dev/null
  fi
  echo "--- storage classes:"; kubectl get storageclass 2>&1 | head -5
fi

hr "4. does the serving image support embeddings and rerank?"
# This decides whether the embedding tenant can be an LLMInferenceService at all
# or has to be a plain vllm serve Deployment.
if have docker || have ctr || have crictl; then :; fi
if have kubectl; then
  echo "--- probing inside a throwaway pod (30s)"
  kubectl run r2-probe --rm -i --restart=Never -n "${NS}" \
    --image="${IMG}" --command --timeout=120s -- \
    bash -lc '
      echo "python: $(python -c "import sys;print(sys.version.split()[0])" 2>/dev/null)"
      echo "vllm:   $(python -c "import vllm;print(vllm.__version__)" 2>/dev/null || echo ABSENT)"
      echo "--- vllm serve flags of interest:"
      (vllm serve --help 2>/dev/null || python -m vllm.entrypoints.openai.api_server --help 2>/dev/null) \
        | grep -E -- "--task|--runner|--convert|--served-model-name|--kv-cache-dtype|--max-num-batched" \
        | head -20
      echo "--- kserve huggingfaceserver entrypoint:"
      python -c "import kserve, huggingfaceserver, inspect, os; print(os.path.dirname(huggingfaceserver.__file__))" 2>/dev/null || echo "huggingfaceserver module not importable"
    ' 2>&1 | head -40
fi

hr "5. is there already a Prometheus / Grafana?"
if have kubectl; then
  kubectl get pods -A 2>/dev/null | grep -Ei "prometheus|grafana|dcgm|kube-state" | head -12
  echo "--- ServiceMonitor CRD present: $(kubectl get crd servicemonitors.monitoring.coreos.com >/dev/null 2>&1 && echo YES || echo no)"
  echo "--- PodMonitor CRD present:     $(kubectl get crd podmonitors.monitoring.coreos.com   >/dev/null 2>&1 && echo YES || echo no)"
  echo "--- HAMi components:"; kubectl get pods -A 2>/dev/null | grep -i hami | head -6
  echo "--- HAMi scheduler metrics port:"
  kubectl get svc -A 2>/dev/null | grep -i hami | head -6
fi

hr "6. network reach to HuggingFace (download feasibility)"
# 13 GiB at a bad rate is days, not hours. Measure before committing to a tier.
if have curl; then
  echo "--- DNS + TLS to huggingface.co:"
  curl -s -o /dev/null -w '  connect=%{time_connect}s  ttfb=%{time_starttfb}s  http=%{http_code}\n' \
    --max-time 20 https://huggingface.co/api/models/Qwen/Qwen2.5-0.5B-Instruct 2>&1 || echo "  FAILED"
  echo "--- throughput sample (10 MiB from the CDN):"
  curl -s -o /dev/null --max-time 60 -w '  %{size_download} bytes in %{time_total}s = %{speed_download} B/s\n' \
    -r 0-10485759 \
    "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/model.safetensors" 2>&1 || echo "  FAILED"
  echo "  (at 1 MB/s, 13 GiB tier-1 takes ~3.7 hours; at 100 KB/s it is ~37 hours)"
else
  echo "curl NOT FOUND"
fi

hr "7. GPU + MPS current state"
if have nvidia-smi; then
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,compute_mode --format=csv
  echo "--- compute apps:"; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>/dev/null | head
else
  echo "nvidia-smi not on the host (may still be inside pods)"
fi
echo "--- MPS control daemons running:"
ps -eo user,pid,args 2>/dev/null | grep -E "mps-control|mps-server" | grep -v grep | head
echo "--- host nvidia-mps unit: $(systemctl is-active nvidia-mps 2>/dev/null || echo 'n/a')"

hr "8. existing benchmark namespace state"
if have kubectl; then
  kubectl get llminferenceservice -n "${NS}" 2>&1 | head
  kubectl get pods -n "${NS}" -o wide 2>&1 | head -15
fi

hr "done"
echo "Send this whole output back; it settles the storage, image-capability,"
echo "monitoring and download-time questions in one go."
