#!/usr/bin/env bash
set -euo pipefail

# Config (override via env)
IMAGE="${IMAGE:-gpu-llm:cu118}"
NUM_CONTAINERS="${NUM_CONTAINERS:-4}"
START_PORT="${START_PORT:-6000}"
VRAM_SLICE_MB="${VRAM_SLICE_MB:-1500}"
MODEL_NAME="${MODEL_NAME:-distilgpt2}"
CONTAINER_PREFIX="${CONTAINER_PREFIX:-llm_tenant}"

# Preflight
command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found"; exit 1; }
command -v docker >/dev/null || { echo "docker not found"; exit 1; }
command -v python3 >/dev/null || { echo "python3 not found"; exit 1; }

# Start MPS if available
if systemctl list-unit-files | grep -q nvidia-mps; then
  echo "Starting NVIDIA MPS..."
  sudo systemctl start nvidia-mps || true
fi

# Pick Dockerfile
DOCKERFILE=""
if [ -f "docker/Dockerfile.scheduler" ]; then
  DOCKERFILE="docker/Dockerfile.scheduler"
elif [ -f "docker/Dockerfile" ]; then
  DOCKERFILE="docker/Dockerfile"
elif [ -f "Dockerfile" ]; then
  DOCKERFILE="Dockerfile"
else
  echo "No Dockerfile found (checked docker/Dockerfile.scheduler, docker/Dockerfile, Dockerfile)."
  exit 1
fi

echo "Building image '$IMAGE' from $DOCKERFILE ..."
docker build -f "$DOCKERFILE" -t "$IMAGE" .

# Launch scheduler
export IMAGE START_PORT NUM_CONTAINERS VRAM_SLICE_MB MODEL_NAME CONTAINER_PREFIX
rm -f scheduler.log scheduler.pid
echo "Starting scheduler..."
nohup python3 src/gpu_scheduler.py \
  --image "$IMAGE" \
  --instances "$NUM_CONTAINERS" \
  --vram_slice_mb "$VRAM_SLICE_MB" \
  --start_port "$START_PORT" \
  > scheduler.log 2>&1 &
echo $! > scheduler.pid
echo "Scheduler PID $(cat scheduler.pid)"

# Give containers time to come up
sleep 6

# Probe endpoints
echo "Probing inference endpoints..."
for ((i=0;i<NUM_CONTAINERS;i++)); do
  PORT=$((START_PORT+i))
  echo -n "POST /infer @ :$PORT -> "
  curl --max-time 15 -s -X POST "http://localhost:${PORT}/infer" \
    -H "Content-Type: application/json" \
    -d "{\"prompt\":\"Hello from port ${PORT}\"}" | head -c 160 || true
  echo
done

echo
echo "Active containers:"
docker ps --filter "name=${CONTAINER_PREFIX}" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

echo
echo "Follow logs: tail -f scheduler.log"
echo "Stop containers: docker stop \$(docker ps -q --filter name=${CONTAINER_PREFIX}) || true"
echo "Stop scheduler:  pkill -F scheduler.pid || true"