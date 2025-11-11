#!/bin/bash
# ---------------------------------------------------------------------------
# monitor_gpu_test.sh
#   Monitors GPU usage, VRAM, and active containers while tests run.
#   Press Ctrl+C to stop monitoring.
# ---------------------------------------------------------------------------

set -e

OUTFILE="nvidia_timeline.csv"
INTERVAL=1

echo "timestamp,gpu_id,utilization,memory_used_mb,memory_total_mb,proc_count" > $OUTFILE

echo "=== Starting GPU monitoring (every $INTERVAL s) ==="
while true; do
  TS=$(date -u +"%Y-%m-%dT%H:%M:%S")
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,processes.count \
             --format=csv,noheader,nounits | \
  awk -v ts="$TS" -F',' '{printf "%s,%s,%s,%s,%s,%s\n", ts, $1, $2, $3, $4, $5}' >> $OUTFILE
  sleep $INTERVAL
done
