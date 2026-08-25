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
# Decides whether the pooling tenants can stay LLMInferenceService or have to
# become plain vllm Deployments. No --rm -i: that attaches, and a slow scheduler
# makes it time out and delete the pod before it ever runs.
if have kubectl; then
  kubectl delete pod r2-probe -n "${NS}" --ignore-not-found >/dev/null 2>&1
  kubectl run r2-probe -n "${NS}" --restart=Never --image="${IMG}" \
    --command -- sleep 300 >/dev/null 2>&1
  echo "--- waiting for the probe pod (up to 180s)"
  if kubectl wait --for=condition=Ready pod/r2-probe -n "${NS}" --timeout=180s >/dev/null 2>&1; then
    echo "--- versions"
    kubectl exec -n "${NS}" r2-probe -- python -c 'import sys;print("python:",sys.version.split()[0])' 2>&1 | tail -1
    kubectl exec -n "${NS}" r2-probe -- python -c 'import vllm;print("vllm:  ",vllm.__version__)' 2>&1 | tail -1
    echo "--- the flags that settle the pooling question"
    kubectl exec -n "${NS}" r2-probe -- bash -lc 'vllm serve --help 2>/dev/null | grep -E -- "--task|--runner|--convert|--kv-cache-dtype" | head -12' 2>&1
    echo "--- does this vllm build register pooling endpoints?"
    kubectl exec -n "${NS}" r2-probe -- python -c 'from vllm.entrypoints.openai import api_server as a; import inspect,re; s=inspect.getsource(a); r=sorted(set(re.findall(chr(34)+r"(/v[0-9]/[a-z_]+|/rerank|/score|/health)"+chr(34), s))); print("routes:", " ".join(r) if r else "none found")' 2>&1 | tail -3
    echo "--- kserve huggingfaceserver present?"
    kubectl exec -n "${NS}" r2-probe -- python -c 'import huggingfaceserver,os;print(os.path.dirname(huggingfaceserver.__file__))' 2>&1 | tail -1
  else
    echo "  probe pod never became Ready:"
    kubectl describe pod r2-probe -n "${NS}" 2>&1 | tail -12
  fi
  kubectl delete pod r2-probe -n "${NS}" --ignore-not-found --wait=false >/dev/null 2>&1
fi

hr "5. is there already a Prometheus / Grafana?"
if have kubectl; then
  kubectl get pods -A 2>/dev/null | grep -Ei "prometheus|grafana|dcgm|kube-state" | head -12
  echo "--- ServiceMonitor CRD present: $(kubectl get crd servicemonitors.monitoring.coreos.com >/dev/null 2>&1 && echo YES || echo no)"
  echo "--- PodMonitor CRD present:     $(kubectl get crd podmonitors.monitoring.coreos.com   >/dev/null 2>&1 && echo YES || echo no)"
  echo "--- HAMi components:"; kubectl get pods -A 2>/dev/null | grep -i hami | head -6
  echo "--- HAMi services:"; kubectl get svc -A 2>/dev/null | grep -i hami | head -6
  echo "--- do the HAMi exporters actually answer?"
  for hp in "hami-device-plugin-monitor:31992" "hami-scheduler:31993"; do
    svc="${hp%%:*}"; port="${hp##*:}"
    n=$(kubectl run r2-curl-$$ -n "${NS}" --restart=Never --rm -q -i --image=curlimages/curl:8.10.1 \
          --timeout=60s -- -s -m 8 "http://${svc}.kube-system:${port}/metrics" 2>/dev/null | grep -c '^[a-zA-Z]' )
    echo "    ${svc}:${port} -> ${n:-0} metric lines"
  done
  echo "--- existing ServiceMonitors:"; kubectl get servicemonitor -A 2>/dev/null | head -12
fi

hr "6. network reach to HuggingFace (download feasibility)"
# 13 GiB at a bad rate is days, not hours. Measure before committing to a tier.
if have curl; then
  echo "--- DNS + TLS to huggingface.co:"
  curl -s -o /dev/null --max-time 20 \
    -w '  connect=%{time_connect}s  ttfb=%{time_starttransfer}s  http=%{http_code}\n' \
    https://huggingface.co/api/models/Qwen/Qwen2.5-0.5B-Instruct 2>&1 || echo "  FAILED"
  echo "--- throughput sample (32 MiB of real weights):"
  # -L is essential: huggingface.co answers 302 to cdn-lfs. Without it curl
  # measures the redirect body (~1 KB) and reports a meaningless rate.
  curl -sL -o /dev/null --max-time 120 \
    -H 'Range: bytes=0-33554431' \
    -w '  %{size_download} bytes in %{time_total}s = %{speed_download} B/s  (http %{http_code})\n' \
    'https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/model.safetensors' 2>&1 \
    || echo "  FAILED"
  echo "  rule of thumb: 11 GiB tier-1 takes ~3h at 1 MB/s, ~30h at 100 KB/s"
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
