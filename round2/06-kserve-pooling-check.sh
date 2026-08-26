#!/usr/bin/env bash
# Does the KSERVE path work at all, and does it carry vLLM's pooling flags?
#
# Two questions, in that order, because the first run of this check answered
# neither: it waited on `-l r2-role=embed`, found nothing, and concluded "use
# --pooling-via vllm". That verdict was not earned. No pod carrying that label
# ever appeared, so nothing about pooling was tested.
#
# The likely reason is that KServe does not copy an LLMInferenceService's
# metadata.labels down onto the pods it creates. If so it is a much bigger
# problem than pooling, because the whole harness selects on that label:
#
#   harness.pods_for()          -l r2-role=<role>   -> readiness, endpoints,
#                                                      GPU placement check
#   harness.teardown()          -l r2-role
#   monitoring PodMonitor       r2-role: Exists     -> ALL vLLM metrics
#
# A silent mismatch there means every run times out waiting for a pod that is
# running fine, and the dashboard stays empty. So this version never selects on
# that label. It snapshots the namespace, applies, and reports whatever pod
# actually appears -- with its real labels. That answers the propagation
# question directly instead of inferring it from a timeout.
#
#   ./06-kserve-pooling-check.sh              # embed
#   ./06-kserve-pooling-check.sh --role rerank
#   ./06-kserve-pooling-check.sh --keep       # leave it up to poke at
#   ./06-kserve-pooling-check.sh --diagnose   # report on what is already there
set -uo pipefail

NS="${NS:-llm-demo}"
ROLE="embed"
KEEP=0
DIAG_ONLY=0
WAIT_POD="${WAIT_POD:-180}"     # seconds for a pod to APPEAR
WAIT_READY="${WAIT_READY:-600}" # seconds for it to become Ready
HERE="$(cd "$(dirname "$0")" && pwd)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --role) ROLE="${2:-embed}"; shift 2 ;;
    --keep) KEEP=1; shift ;;
    --diagnose) DIAG_ONLY=1; KEEP=1; shift ;;
    -h|--help) sed -n '2,28p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

case "${ROLE}" in
  embed)       SCEN="llm-embed";  PATHS="/v1/embeddings" ;;
  embed-small) SCEN="dense4";     PATHS="/v1/embeddings" ;;
  rerank)      SCEN="vl-rerank";  PATHS="/v1/rerank /rerank /v1/score /score" ;;
  llm)         SCEN="llm-only";   PATHS="/v1/chat/completions" ;;
  *) echo "role must be llm, embed, embed-small or rerank" >&2; exit 2 ;;
esac

OUT="${HERE}/.kcheck"
CR="r2-${ROLE}"

# Infrastructure that lives in this namespace and must not be mistaken for
# something this check created.
KEEPERS='qwen-benchmark-image-pinner|qwen-benchmark-mps-daemon'

cleanup() {
  if [[ "${KEEP}" -eq 0 ]]; then
    echo
    echo "--- cleaning up"
    # Delete the CR by NAME. Deleting by -l r2-role only works if the label is
    # on the object, which is the very thing in question here.
    kubectl delete llminferenceservice "${CR}" -n "${NS}" \
      --ignore-not-found --wait=false >/dev/null 2>&1
    kubectl delete deployment,service,pod -n "${NS}" -l r2-role \
      --ignore-not-found --wait=false >/dev/null 2>&1
    rm -rf "${OUT}"
  else
    echo
    echo "--- left running. Remove with:"
    echo "      kubectl delete llminferenceservice ${CR} -n ${NS}"
  fi
}
trap cleanup EXIT

pods_now() {
  kubectl get pods -n "${NS}" --no-headers -o custom-columns=N:.metadata.name \
    2>/dev/null | grep -Ev "${KEEPERS}" | sort
}

diagnose() {
  echo
  echo "================= DIAGNOSIS ================="
  echo
  echo "--- the LLMInferenceService and its status conditions"
  kubectl get llminferenceservice "${CR}" -n "${NS}" -o yaml 2>&1 \
    | sed -n '/^status:/,$p' | head -50 | sed 's/^/      /'
  echo
  echo "--- everything in ${NS} right now, WITH LABELS"
  kubectl get pods -n "${NS}" --show-labels 2>&1 | sed 's/^/      /'
  echo
  echo "--- workloads KServe did or did not create"
  kubectl get deploy,rs,svc,httproute -n "${NS}" 2>&1 | head -20 | sed 's/^/      /'
  echo
  echo "--- recent namespace events (newest last)"
  kubectl get events -n "${NS}" --sort-by .lastTimestamp 2>&1 \
    | tail -20 | sed 's/^/      /'
  echo
  echo "--- does spec.template accept pod metadata at all?"
  echo "    (PodSpec has no metadata; PodTemplateSpec does. If it is a bare"
  echo "     PodSpec, labels CANNOT be set on the pod from the CR and the"
  echo "     harness must select on what KServe does set.)"
  kubectl explain llminferenceservice.spec.template --recursive 2>&1 \
    | head -12 | sed 's/^/      /'
  echo
  echo "--- KServe controller complaints about this object"
  kubectl logs -n kserve -l control-plane=kserve-controller-manager \
    --tail=4000 2>/dev/null | grep -i "${CR}" | tail -15 | sed 's/^/      /' \
    || echo "      (controller logs not readable under that selector)"
  echo
  echo "============================================="
}

if [[ "${DIAG_ONLY}" -eq 1 ]]; then
  diagnose
  exit 0
fi

echo "=== rendering ${ROLE} through the KServe path"
python3 "${HERE}/render.py" --scenario "${SCEN}" --arm dedicated --only "${ROLE}" \
  --out "${OUT}" --pooling-via kserve || exit 1
echo
echo "--- flags requested:"
grep -E "^\s+- --" "${OUT}"/r2-dedicated-"${ROLE}".yaml | sed 's/^/      /'

echo
echo "=== applying"
kubectl delete llminferenceservice "${CR}" -n "${NS}" --ignore-not-found >/dev/null 2>&1
BEFORE="$(pods_now)"
kubectl apply -f "${OUT}" || exit 1

echo
echo "=== 1. does a pod appear at all, and what labels does it carry?"
POD=""
t0=$SECONDS
while (( SECONDS - t0 < WAIT_POD )); do
  NEW="$(comm -13 <(printf '%s\n' "${BEFORE}") <(pods_now) | head -1)"
  if [[ -n "${NEW}" ]]; then POD="${NEW}"; break; fi
  sleep 5
done

if [[ -z "${POD}" ]]; then
  echo "  NO POD APPEARED in ${WAIT_POD}s."
  echo "  The CR was accepted but KServe created no workload. This is NOT a"
  echo "  pooling problem -- it would affect the chat models identically."
  diagnose
  exit 1
fi

echo "  pod appeared after $(( SECONDS - t0 ))s: ${POD}"
echo "  labels actually on the pod:"
kubectl get pod "${POD}" -n "${NS}" -o jsonpath='{.metadata.labels}' 2>/dev/null \
  | tr ',' '\n' | sed 's/^/      /'
echo
if kubectl get pod "${POD}" -n "${NS}" -o jsonpath='{.metadata.labels}' 2>/dev/null \
     | grep -q "r2-role"; then
  echo "  r2-role IS propagated to the pod -- the harness selectors are fine."
else
  echo "  *** r2-role is NOT on the pod. ***"
  echo "  harness.pods_for(), wait_ready(), endpoints(), the GPU placement"
  echo "  check and the PodMonitor all select on it, so every run would time"
  echo "  out and the dashboard would stay empty. Fix that before T1."
fi

echo
echo "=== 2. does it become Ready?"
if ! kubectl wait --for=condition=Ready "pod/${POD}" -n "${NS}" \
     --timeout="${WAIT_READY}s" 2>&1 | sed 's/^/      /'; then
  echo "  never became Ready. Container state and logs:"
  kubectl get pod "${POD}" -n "${NS}" -o wide 2>&1 | sed 's/^/      /'
  kubectl describe pod "${POD}" -n "${NS}" 2>&1 \
    | sed -n '/Containers:/,/Conditions:/p' | head -30 | sed 's/^/      /'
  echo "  --- logs (a rejected flag shows up here):"
  kubectl logs "${POD}" -n "${NS}" --all-containers --tail=80 2>&1 | sed 's/^/      /'
  diagnose
  exit 1
fi

IP="$(kubectl get pod "${POD}" -n "${NS}" -o jsonpath='{.status.podIP}')"
echo "  Ready at ${IP}"

echo
echo "=== 3. did the pooling flags survive into the running process?"
echo "--- container args as admitted:"
kubectl get pod "${POD}" -n "${NS}" \
  -o jsonpath='{.spec.containers[*].args}' 2>/dev/null | tr ',' '\n' | sed 's/^/      /'
echo "--- the real command line inside the container:"
kubectl exec -n "${NS}" "${POD}" -c main -- sh -c \
  'tr "\0" "\n" < /proc/1/cmdline' 2>/dev/null | sed 's/^/      /' | head -30 \
  || kubectl exec -n "${NS}" "${POD}" -- sh -c \
       'tr "\0" "\n" < /proc/1/cmdline' 2>/dev/null | sed 's/^/      /' | head -30
echo "--- what vLLM logged about runner/convert:"
kubectl logs "${POD}" -n "${NS}" --all-containers 2>/dev/null \
  | grep -iE "runner|convert|pooling|task=" | head -8 | sed 's/^/      /'

echo
echo "=== 4. does it answer?"
PORT=""
for port in 8080 8000 80; do
  if [[ "$(curl -s -o /dev/null -w '%{http_code}' -m 5 \
           "http://${IP}:${port}/v1/models" 2>/dev/null)" == "200" ]]; then
    PORT="${port}"; break
  fi
done
if [[ -z "${PORT}" ]]; then
  echo "  no port among 8080/8000/80 answered /v1/models"
  kubectl get pod "${POD}" -n "${NS}" \
    -o jsonpath='{.spec.containers[*].ports}' 2>/dev/null | sed 's/^/      ports: /'
  exit 1
fi
echo "  serving on port ${PORT}"
MODEL="$(curl -s -m 5 "http://${IP}:${PORT}/v1/models" \
  | python3 -c 'import json,sys; print((json.load(sys.stdin).get("data") or [{}])[0].get("id",""))' 2>/dev/null)"
echo "  advertised model id: ${MODEL:-<none>}"

ok=0
for p in ${PATHS}; do
  for prefix in "" "/openai"; do
    if [[ "${ROLE}" == "rerank" ]]; then
      body="{\"model\":\"${MODEL}\",\"query\":\"what is MPS\",\"documents\":[\"NVIDIA MPS shares a GPU\",\"unrelated\"]}"
    elif [[ "${ROLE}" == "llm" ]]; then
      body="{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"say ok\"}],\"max_tokens\":8}"
    else
      body="{\"model\":\"${MODEL}\",\"input\":[\"hello world\",\"second document\"]}"
    fi
    code="$(curl -s -o /tmp/r2k.json -w '%{http_code}' -m 30 \
            "http://${IP}:${PORT}${prefix}${p}" \
            -H 'content-type: application/json' -d "${body}" 2>/dev/null)"
    printf '    %-40s %s\n' "${prefix}${p}" "${code}"
    if [[ "${code}" == "200" ]]; then
      ok=1
      python3 - <<'PY' 2>/dev/null || head -c 200 /tmp/r2k.json
import json
d = json.load(open("/tmp/r2k.json"))
rows = d.get("data")
if isinstance(rows, list) and rows and "embedding" in (rows[0] or {}):
    print("      -> %d vector(s), dim %d, usage %s"
          % (len(rows), len(rows[0]["embedding"]), d.get("usage")))
else:
    print("      -> %s" % json.dumps(d)[:220])
PY
      break
    fi
  done
  [[ "${ok}" -eq 1 ]] && break
done

echo
echo "=== 5. did it fit the quota?"
kubectl get pod "${POD}" -n "${NS}" \
  -o jsonpath='{.spec.containers[0].resources.limits}' 2>/dev/null | sed 's/^/      /'
echo
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader 2>/dev/null | sed 's/^/      /'

echo
if [[ "${ok}" -eq 1 ]]; then
  echo "VERDICT: the KServe path serves ${ROLE}. Keep --pooling-via kserve."
else
  echo "VERDICT: the pod runs but no ${ROLE} endpoint answered."
  echo "         Use --pooling-via vllm, already proven by verify-models.sh."
  exit 1
fi
