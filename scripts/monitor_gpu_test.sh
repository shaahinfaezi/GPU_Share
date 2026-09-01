#!/usr/bin/env bash
set -euo pipefail

INTERVAL="${INTERVAL:-2}"
OUTFILE="${OUTFILE:-nvidia_timeline.csv}"
NUM_CONTAINERS="${NUM_CONTAINERS:-4}"
START_PORT="${START_PORT:-6000}"

command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found"; exit 1; }
command -v curl >/dev/null || { echo "curl not found"; exit 1; }

if [ ! -f "$OUTFILE" ]; then
  echo "timestamp,gpu_id,utilization_pct,memory_used_mb,memory_total_mb" > "$OUTFILE"
fi

echo "Monitoring every ${INTERVAL}s. Writing GPU stats to $OUTFILE. Ctrl+C to stop."
while true; do
  TS=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

  # GPU stats snapshot
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total \
    --format=csv,noheader,nounits 2>/dev/null | \
    awk -v ts="$TS" -F',' '{printf "%s,%s,%s,%s,%s\n", ts,$1,$2,$3,$4}' >> "$OUTFILE"

  # Health probes
  for ((i=0;i<NUM_CONTAINERS;i++)); do
    PORT=$((START_PORT+i))
    STATUS=$(curl -s --max-time 3 "http://localhost:${PORT}/health" || true)
    echo "[$TS] health :$PORT -> ${STATUS:-unreachable}"
  done

  sleep "$INTERVAL"
done