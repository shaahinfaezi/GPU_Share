#!/usr/bin/env bash
# The last unknown before T1: does the KSERVE path carry vLLM's pooling flags?
#
# What is already settled:
#   ./verify-models.sh --serve embed   -> bge-m3 served real 1024-dim vectors
# but that ran a BARE `vllm serve` pod. render.py defaults to
# --pooling-via kserve, which goes through LLMInferenceService and the
# huggingfaceserver entrypoint instead. Those are different code paths, and
# only one of them has been exercised.
#
# 05-pooling-probe.sh found parse_known_args and maybe_add_vllm in
# huggingfaceserver/__main__.py, which is strong evidence the flags are
# forwarded -- but that is static analysis of source text, which is exactly the
# kind of reasoning that produced the wrong answer about /v1/embeddings the
# first time. This runs it.
#
# It also checks a second thing worth knowing early: whether bge-m3 fits the
# 3072 MiB gpumem quota at --dtype=half. fp32 weights are 2.14 GiB; halved
# they are ~1.14 GiB, leaving roughly 1.6 GiB for activations at
# max_model_len=8192 and max_num_seqs=64. That should fit. "Should" is why
# this runs before T1 rather than during it.
#
#   ./06-kserve-pooling-check.sh              # embed
#   ./06-kserve-pooling-check.sh --role rerank
#   ./06-kserve-pooling-check.sh --keep       # leave it running to poke at
set -uo pipefail

NS="${NS:-llm-demo}"
ROLE="embed"
KEEP=0
HERE="$(cd "$(dirname "$0")" && pwd)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --role) ROLE="${2:-embed}"; shift 2 ;;
    --keep) KEEP=1; shift ;;
    -h|--help) sed -n '2,26p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

case "${ROLE}" in
  embed)       SCEN="llm-embed";  PATHS="/v1/embeddings" ;;
  embed-small) SCEN="dense4";     PATHS="/v1/embeddings" ;;
  rerank)      SCEN="vl-rerank";  PATHS="/v1/rerank /rerank /v1/score /score" ;;
  *) echo "role must be embed, embed-small or rerank" >&2; exit 2 ;;
esac

OUT="${HERE}/.kcheck"
cleanup() {
  if [[ "${KEEP}" -eq 0 ]]; then
    echo
    echo "--- cleaning up"
    kubectl delete -f "${OUT}" --ignore-not-found --wait=false >/dev/null 2>&1
    rm -rf "${OUT}"
  else
    echo
    echo "--- left running (--keep). Remove with:"
    echo "      kubectl delete -f ${OUT}"
  fi
}
trap cleanup EXIT

echo "=== rendering ${ROLE} through the KServe path"
python3 "${HERE}/render.py" --scenario "${SCEN}" --arm dedicated --only "${ROLE}" \
  --out "${OUT}" --pooling-via kserve || exit 1

echo
echo "--- the flags render.py asked for:"
grep -E "^\s+- --" "${OUT}"/r2-dedicated-"${ROLE}".yaml | sed 's/^/      /'

echo
echo "=== applying"
kubectl delete -f "${OUT}" --ignore-not-found --wait=false >/dev/null 2>&1
sleep 3
kubectl apply -f "${OUT}" || exit 1

echo
echo "--- waiting for Ready (bge-m3 took ~110s as a bare pod; KServe adds a"
echo "    router, so allow more)"
if ! kubectl wait --for=condition=Ready pod -l "r2-role=${ROLE}" -n "${NS}" \
     --timeout=600s 2>/dev/null; then
  echo
  echo "NEVER BECAME READY. This is the answer, not a failure of the check."
  POD="$(kubectl get pods -n "${NS}" -l "r2-role=${ROLE}" \
         -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)"
  if [[ -n "${POD}" ]]; then
    echo "--- pod state:"
    kubectl get pod "${POD}" -n "${NS}" -o wide 2>&1 | sed 's/^/      /'
    echo "--- events:"
    kubectl get events -n "${NS}" --field-selector "involvedObject.name=${POD}" \
      --sort-by .lastTimestamp 2>/dev/null | tail -8 | sed 's/^/      /'
    echo "--- logs (the flag error, if there is one, is here):"
    kubectl logs "${POD}" -n "${NS}" --tail=60 2>&1 | sed 's/^/      /'
  fi
  echo
  echo "VERDICT: use --pooling-via vllm. That path is already proven by"
  echo "         ./verify-models.sh --serve ${ROLE}, so nothing is blocked --"
  echo "         run.py just needs --pooling-via vllm."
  exit 1
fi

POD="$(kubectl get pods -n "${NS}" -l "r2-role=${ROLE}" \
       -o jsonpath='{.items[0].metadata.name}')"
IP="$(kubectl get pod "${POD}" -n "${NS}" -o jsonpath='{.status.podIP}')"
echo
echo "--- pod ${POD} at ${IP}"

echo
echo "=== 1. did the pooling flags survive into the actual process?"
# The pod spec is what KServe admitted; /proc/1/cmdline is what actually ran.
# They can differ, and only the second one matters.
echo "--- container args as admitted:"
kubectl get pod "${POD}" -n "${NS}" \
  -o jsonpath='{.spec.containers[0].args}' 2>/dev/null | tr ',' '\n' | sed 's/^/      /'
echo "--- the real command line inside the container:"
kubectl exec -n "${NS}" "${POD}" -- sh -c \
  'tr "\0" "\n" < /proc/1/cmdline' 2>/dev/null | sed 's/^/      /' | head -30
echo "--- what vLLM logged about runner/convert:"
kubectl logs "${POD}" -n "${NS}" 2>/dev/null \
  | grep -iE "runner|convert|pooling|embed|task=" | head -8 | sed 's/^/      /'

echo
echo "=== 2. does it actually answer?"
for port in 8080 8000 80; do
  code="$(curl -s -o /dev/null -w '%{http_code}' -m 5 "http://${IP}:${port}/v1/models" 2>/dev/null)"
  if [[ "${code}" == "200" ]]; then PORT="${port}"; break; fi
done
if [[ -z "${PORT:-}" ]]; then
  echo "  no port among 8080/8000/80 answered /v1/models"
  exit 1
fi
echo "--- serving on port ${PORT}"
MODEL="$(curl -s -m 5 "http://${IP}:${PORT}/v1/models" \
  | python3 -c 'import json,sys; print((json.load(sys.stdin).get("data") or [{}])[0].get("id",""))' 2>/dev/null)"
echo "--- advertised model id: ${MODEL:-<none>}"

ok=0
for p in ${PATHS}; do
  for prefix in "" "/openai"; do
    url="http://${IP}:${PORT}${prefix}${p}"
    if [[ "${ROLE}" == "rerank" ]]; then
      body="{\"model\":\"${MODEL}\",\"query\":\"what is MPS\",\"documents\":[\"NVIDIA MPS shares a GPU\",\"unrelated\"]}"
    else
      body="{\"model\":\"${MODEL}\",\"input\":[\"hello world\",\"second document\"]}"
    fi
    code="$(curl -s -o /tmp/r2k.json -w '%{http_code}' -m 30 "${url}" \
            -H 'content-type: application/json' -d "${body}" 2>/dev/null)"
    printf '    %-42s %s\n' "${prefix}${p}" "${code}"
    if [[ "${code}" == "200" ]]; then
      ok=1
      python3 - <<'PY' 2>/dev/null || head -c 200 /tmp/r2k.json
import json
d = json.load(open("/tmp/r2k.json"))
if "data" in d:
    rows = d["data"]
    dim = len((rows[0] or {}).get("embedding") or [])
    print("      -> %d vector(s), dim %d, usage %s"
          % (len(rows), dim, d.get("usage")))
else:
    print("      -> %s" % json.dumps(d)[:220])
PY
      break
    fi
  done
  [[ "${ok}" -eq 1 ]] && break
done

echo
echo "=== 3. did it fit the quota?"
kubectl get pod "${POD}" -n "${NS}" \
  -o jsonpath='{.spec.containers[0].resources.limits}' 2>/dev/null | sed 's/^/      /'
echo
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader 2>/dev/null | sed 's/^/      /'

echo
if [[ "${ok}" -eq 1 ]]; then
  echo "VERDICT: KServe carries the pooling flags. Keep render.py's default"
  echo "         (--pooling-via kserve) and every model stays on one"
  echo "         deployment path, which is what makes the arms comparable."
else
  echo "VERDICT: the pod runs but no pooling endpoint answered. Use"
  echo "         --pooling-via vllm for run.py; that path is already proven."
  exit 1
fi
