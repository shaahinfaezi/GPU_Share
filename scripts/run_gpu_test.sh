#!/usr/bin/env bash
set -euo pipefail

# ==================== CONFIG (override with env vars) ====================
IMAGE="${IMAGE:-gpu-llm:cu118}"
NUM_CONTAINERS="${NUM_CONTAINERS:-4}"
START_PORT="${START_PORT:-6000}"
VRAM_SLICE_MB="${VRAM_SLICE_MB:-1500}"
MODEL_NAME="${MODEL_NAME:-distilgpt2}"
CONTAINER_PREFIX="${CONTAINER_PREFIX:-llm_tenant}"

# Paths (do not change unless needed)
BUILD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="/var/log/gpu-llm"
PID_DIR="/var/run/gpu-llm"
BUILD_LOG="$LOG_DIR/build.log"
BUILD_PID="$PID_DIR/build.pid"
SCHEDULER_LOG="$LOG_DIR/scheduler.log"
SCHEDULER_PID="$PID_DIR/scheduler.pid"

# Create dirs
sudo mkdir -p "$LOG_DIR" "$PID_DIR"
sudo chown $(whoami):$(whoami) "$LOG_DIR" "$PID_DIR" 2>/dev/null || true

# ==================== PREFLIGHT ====================
echo "=== Preflight Checks ==="
command -v nvidia-smi >/dev/null || { echo "ERROR: nvidia-smi not found"; exit 1; }
command -v docker >/dev/null || { echo "ERROR: docker not found"; exit 1; }
command -v python3 >/dev/null || { echo "ERROR: python3 not found"; exit 1; }

# Start MPS if available
if systemctl list-unit-files | grep -q nvidia-mps; then
  echo "Starting NVIDIA MPS..."
  sudo systemctl start nvidia-mps || true
fi

# Pick Dockerfile
DOCKERFILE=""
for file in "docker/Dockerfile.scheduler" "docker/Dockerfile" "Dockerfile"; do
  [ -f "$BUILD_DIR/$file" ] && DOCKERFILE="$BUILD_DIR/$file" && break
done
[ -z "$DOCKERFILE" ] && { echo "ERROR: No Dockerfile found"; exit 1; }

echo "Found Dockerfile: $DOCKERFILE"

# ==================== BUILD IMAGE (DETACHED + LOGGED) ====================
echo "=== Building Docker image '$IMAGE' ==="
echo "Build log → $BUILD_LOG"
echo "Build PID  → $BUILD_PID"

# Kill any old build
[ -f "$BUILD_PID" ] && kill $(cat "$BUILD_PID") 2>/dev/null || true

# Start build in background
docker buildx build \
  -f "$DOCKERFILE" \
  -t "$IMAGE" \
  --load \
  --progress=plain \
  "$BUILD_DIR" \
  > "$BUILD_LOG" 2>&1 &

BUILD_PID_VAL=$!
echo "$BUILD_PID_VAL" > "$BUILD_PID"
echo "Build started in background (PID: $BUILD_PID_VAL)"
echo "Monitor: tail -f $BUILD_LOG"

# ==================== WAIT FOR BUILD TO FINISH (optional) ====================
echo "Waiting for build to complete... (Ctrl+C to detach)"
timeout 3600 tail -f "$BUILD_LOG" 2>/dev/null &
TAIL_PID=$!
wait $BUILD_PID_VAL && BUILD_EXIT=0 || BUILD_EXIT=$?
kill $TAIL_PID 2>/dev/null || true

if [ $BUILD_EXIT -eq 0 ] && docker images --format "{{.Repository}}:{{.Tag}}" | grep -q "^${IMAGE}$"; then
  echo "BUILD SUCCESS"
else
  echo "BUILD FAILED – see $BUILD_LOG"
  tail -50 "$BUILD_LOG"
  exit 1
fi

# ==================== LAUNCH SCHEDULER (DETACHED) ====================
echo "=== Starting GPU Scheduler ==="
rm -f "$SCHEDULER_LOG" "$SCHEDULER_PID"

nohup python3 "$BUILD_DIR/src/gpu_scheduler.py" \
  --image "$IMAGE" \
  --instances "$NUM_CONTAINERS" \
  --vram_slice_mb "$VRAM_SLICE_MB" \
  --start_port "$START_PORT" \
  > "$SCHEDULER_LOG" 2>&1 &

echo $! > "$SCHEDULER_PID"
echo "Scheduler started (PID: $(cat $SCHEDULER_PID))"

# ==================== PROBE ENDPOINTS ====================
sleep 8
echo "=== Probing Inference Endpoints ==="
for ((i=0; i<NUM_CONTAINERS; i++)); do
  PORT=$((START_PORT + i))
  echo -n "POST /infer @ :$PORT -> "
  curl --max-time 12 -s -X POST "http://localhost:${PORT}/infer" \
    -H "Content-Type: application/json" \
    -d "{\"prompt\":\"Hello from port $PORT\"}" | head -c 160 || echo "TIMEOUT"
  echo
done

# ==================== FINAL STATUS ====================
echo
echo "=== System Status ==="
echo "Active containers:"
docker ps --filter "name=${CONTAINER_PREFIX}" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" || echo "None"

echo
echo "=== Useful Commands ==="
echo "View build log:     tail -f $BUILD_LOG"
echo "View scheduler log: tail -f $SCHEDULER_LOG"
echo "Stop containers:    docker stop \$(docker ps -q --filter name=$CONTAINER_PREFIX)"
echo "Stop scheduler:     pkill -F $SCHEDULER_PID || true"
echo "Full status check:  bash $0 --status"
