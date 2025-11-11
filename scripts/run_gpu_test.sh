#!/usr/bin/env bash
set -euo pipefail

# Config (override via env)
IMAGE="${IMAGE:-gpu-llm:cu118}"
NUM_CONTAINERS="${NUM_CONTAINERS:-4}"
START_PORT="${START_PORT:-6000}"
VRAM_SLICE_MB="${VRAM_SLICE_MB:-1500}"
MODEL_NAME="${MODEL_NAME:-distilgpt2}"
CONTAINER_PREFIX="${CONTAINER_PREFIX:-llm_tenant}"

# --- Log files (now saved to user's home directory) ---
LOG_DIR="$HOME/gpu_share_logs"
mkdir -p "$LOG_DIR" # Ensure the log directory exists
BUILD_LOG="$LOG_DIR/build.log"
SCHEDULER_LOG="$LOG_DIR/scheduler.log"

# Preflight
command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found"; exit 1; }
command -v docker >/dev/null || { echo "docker not found"; exit 1; }
command -v python3 >/dev/null || { echo "python3 not found"; exit 1; }

# Start MPS if available
if sudo systemctl list-unit-files | grep -q nvidia-mps; then
  echo "Starting NVIDIA MPS..."
  sudo systemctl start nvidia-mps || true
fi

# Pick Dockerfile
# IMPORTANT: This part assumes the script is run from /opt/GPU_Share
DOCKERFILE=""
if [ -f "docker/Dockerfile.scheduler" ]; then
  DOCKERFILE="docker/Dockerfile.scheduler"
elif [ -f "docker/Dockerfile" ]; then
  DOCKERFILE="docker/Dockerfile"
elif [ -f "Dockerfile" ]; then
  DOCKERFILE="Dockerfile"
else
  echo "No Dockerfile found (checked docker/Dockerfile.scheduler, docker/Dockerfile, Dockerfile)." >&2
  exit 1
fi

# --- Docker build with logging ---
echo "Building image '$IMAGE' from $DOCKERFILE... Logs are in $BUILD_LOG"
# The script will wait here until the build is done.
if sudo docker build -f "$DOCKERFILE" -t "$IMAGE" . > "$BUILD_LOG" 2>&1; then
  echo "Docker build completed successfully."
else
  echo "Docker build failed. Check '$BUILD_LOG' for details." >&2
  exit 1
fi

# Launch scheduler
export IMAGE START_PORT NUM_CONTAINERS VRAM_SLICE_MB MODEL_NAME CONTAINER_PREFIX
rm -f "$LOG_DIR/scheduler.pid"
echo "Starting scheduler... Logs are in $SCHEDULER_LOG"
nohup python3 src/gpu_scheduler.py \
  --image "$IMAGE" \
  --instances "$NUM_CONTAINERS" \
  --vram_slice_mb "$VRAM_SLICE_MB" \
  --start_port "$START_PORT" \
  > "$SCHEDULER_LOG" 2>&1 &
echo $! > "$LOG_DIR/scheduler.pid"
echo "Scheduler started with PID $(cat "$LOG_DIR/scheduler.pid")"

# ... rest of the script remains the same ...
echo "Waiting for containers to initialize..."
sleep 10
echo
echo "Probing endpoints and listing active containers..."
for ((i=0;i<NUM_CONTAINERS;i++)); do
  PORT=$((START_PORT+i))
  echo -n "POST /infer @ :$PORT -> "
  curl --max-time 15 -s -X POST "http://localhost:${PORT}/infer" \
    -H "Content-Type: application/json" \
    -d "{\"prompt\":\"Hello from port ${PORT}\"}" | head -c 160 || echo "Request failed or timed out."
  echo
done
echo
echo "Active containers:"
sudo docker ps --filter "name=${CONTAINER_PREFIX}" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
echo
echo "Follow scheduler logs: tail -f $SCHEDULER_LOG"
echo "Stop containers: docker stop \$(docker ps -q --filter name=${CONTAINER_PREFIX}) || true"
echo "Stop scheduler: pkill -F $LOG_DIR/scheduler.pid || true"
