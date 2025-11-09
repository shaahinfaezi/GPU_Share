#!/bin/bash
# collect_metrics.sh <out_csv> <interval_seconds>
OUT=${1:-metrics.csv}
INTERVAL=${2:-1}

echo "timestamp,util,mem_used,mem_free,mem_total" > $OUT
while true; do
  ts=$(date +%s)
  line=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.free,memory.total --format=csv,noheader,nounits)
  echo "${ts},${line}" >> $OUT
  sleep $INTERVAL
done
