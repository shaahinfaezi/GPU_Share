#!/bin/bash
# triton_warmup.sh <model_name> <port(8000 default)> <warmup_count>
MODEL=${1:-"mymodel"}
TRITON_HOST=${2:-"localhost"}
TRITON_PORT=${3:-8000}
WARMUP_REQS=${4:-10}

echo "Warming up model $MODEL with $WARMUP_REQS requests"
for i in $(seq 1 $WARMUP_REQS); do
  curl -s -X POST "http://${TRITON_HOST}:${TRITON_PORT}/v2/repository/models/${MODEL}/load" || true
  sleep 0.2
done
echo "Warmup requests sent"
