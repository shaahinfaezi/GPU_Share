#!/usr/bin/env bash
set -euo pipefail

NAMESPACE="${NAMESPACE:-llm-demo}"
SERVICE="${SERVICE:-qwen-zero-offload}"

kubectl -n "${NAMESPACE}" wait --for=condition=Ready "inferenceservice/${SERVICE}" --timeout=25m
kubectl -n "${NAMESPACE}" get inferenceservice "${SERVICE}" -o jsonpath='{.status.url}{"\n"}'
kubectl -n "${NAMESPACE}" get pods -o wide

cat <<'EOF'
For a direct test, choose the qwen-zero-offload pod listed above and run:

  kubectl -n llm-demo port-forward pod/<pod-name> 8080:8080

Then, in another terminal:

  curl -sS http://127.0.0.1:8080/v1/models/qwen-zero-offload:predict \
    -H 'Content-Type: application/json' \
    -d '{"instances":[{"prompt":"What is the capital of France?","max_new_tokens":32,"temperature":0}]}'
EOF
