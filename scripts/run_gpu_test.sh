#!/usr/bin/env bash
set -euo pipefail

# --- Install prerequisites (Ubuntu/Debian) ---
install_prereqs() {
  if [ -f /etc/os-release ]; then
    . /etc/os-release
  else
    echo "Unknown OS (no /etc/os-release). Skipping auto-install."
    return 0
  fi

  if [[ "$ID" != "ubuntu" && "$ID" != "debian" ]]; then
    echo "Auto-install currently supports Ubuntu/Debian. Skipping."
    return 0
  fi

  sudo apt-get update -y

  # Basic tools
  if ! command -v curl >/dev/null 2>&1; then
    sudo apt-get install -y curl ca-certificates
  fi

  # Python and pip (host, for scheduler)
  if ! command -v python3 >/dev/null 2>&1; then
    sudo apt-get install -y python3
  fi
  if ! command -v pip3 >/dev/null 2>&1; then
    sudo apt-get install -y python3-pip
  fi

  # Docker engine
  if ! command -v docker >/dev/null 2>&1; then
    sudo apt-get install -y docker.io
    sudo systemctl enable --now docker
  fi

  # NVIDIA Container Toolkit (for GPU passthrough into containers)
  if command -v nvidia-smi >/dev/null 2>&1; then
    if ! dpkg -s nvidia-container-toolkit >/dev/null 2>&1; then
      distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
      curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
        sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
      curl -fsSL https://nvidia.github.io/libnvidia-container/${distribution}/libnvidia-container.list | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null
      sudo apt-get update -y
      sudo apt-get install -y nvidia-container-toolkit
    fi
    # Configure Docker runtime if needed
    if command -v nvidia-ctk >/dev/null 2>&1; then
      sudo nvidia-ctk runtime configure --runtime=docker || true
      sudo systemctl restart docker || true
    fi
  else
    echo "Warning: NVIDIA driver not detected (nvidia-smi missing). Containers will run on CPU."
  fi

  # Python deps for scheduler (host) - Uninstall old pynvml, install modern nvidia-ml-py
  echo "Checking/installing host Python dependencies (docker, nvidia-ml-py)..."
  python3 -m pip uninstall -y pynvml > /dev/null 2>&1 || true
  python3 -m pip install --user --upgrade pip
  python3 -m pip install --user docker "nvidia-ml-py<12"
  export PATH="$HOME/.local/bin:$PATH"
}

#install_prereqs

# Config (override via env)
IMAGE="${IMAGE:-gpu-llm:cu118}"
NUM_CONTAINERS="${NUM_CONTAINERS:-4}"
START_PORT="${START_PORT:-6000}"
VRAM_SLICE_MB="${VRAM_SLICE_MB:-1500}"
MODEL_NAME="${MODEL_NAME:-distilgpt2}"
CONTAINER_PREFIX="${CONTAINER_PREFIX:-llm_tenant}"

# --- Log files (now saved to user's home directory) ---
LOG_DIR="$HOME/gpu_share_logs"
mkdir -p "$LOG_DIR"
BUILD_LOG="$LOG_DIR/build.log"
SCHEDULER_LOG="$LOG_DIR/scheduler.log"

# --- Force cleanup of previous runs ---
echo "--- Cleaning up leftover processes and containers from previous runs ---"
# Kill any lingering scheduler process using its last known PID
if [ -f "$LOG_DIR/scheduler.pid" ]; then
    echo "Stopping previous scheduler process..."
    sudo pkill -F "$LOG_DIR/scheduler.pid" || true
    rm -f "$LOG_DIR/scheduler.pid"
fi
# Find, stop, and remove any containers with the target prefix
LEFTOVER_CONTAINERS=$(sudo docker ps -aq --filter "name=${CONTAINER_PREFIX}")
if [ -n "$LEFTOVER_CONTAINERS" ]; then
  echo "Stopping and removing leftover containers..."
  sudo docker stop $LEFTOVER_CONTAINERS > /dev/null 2>&1 || true
  sudo docker rm $LEFTOVER_CONTAINERS > /dev/null 2>&1 || true
  echo "Cleanup complete."
else
  echo "No leftover containers found."
fi
echo "--------------------------------------------------------------------"


# Preflight (after install)
command -v docker >/dev/null || { echo "docker not found"; exit 1; }
command -v python3 >/dev/null || { echo "python3 not found"; exit 1; }

# Start MPS if available
if sudo systemctl list-unit-files | grep -q nvidia-mps; then
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
  echo "No Dockerfile found (checked docker/Dockerfile.scheduler, docker/Dockerfile, Dockerfile)." >&2
  exit 1
fi

# --- Docker build with logging ---
echo "Building image '$IMAGE' from $DOCKERFILE... Logs are in $BUILD_LOG"
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

echo
echo "Waiting for scheduler to launch all containers..."
LAUNCH_TIMEOUT=60 # seconds
SECONDS=0
while true; do
  # Count running containers with the correct prefix
  RUNNING_COUNT=$(sudo docker ps --filter "name=${CONTAINER_PREFIX}" --format "{{.Names}}" | wc -l)

  echo -ne "Launched ${RUNNING_COUNT}/${NUM_CONTAINERS} containers... \r"

  if [ "$RUNNING_COUNT" -ge "$NUM_CONTAINERS" ]; then
    echo -e "\nAll containers launched."
    break
  fi

  if [ "$SECONDS" -ge "$LAUNCH_TIMEOUT" ]; then
    echo -e "\nError: Timeout waiting for containers to be launched by the scheduler." >&2
    exit 1
  fi

  sleep 2
  SECONDS=$((SECONDS + 2))
done


echo
echo "Waiting for containers to become healthy (this can take a few minutes on first run)..."
for ((i=0;i<NUM_CONTAINERS;i++)); do
  PORT=$((START_PORT+i))
  echo -n "Waiting for container on port $PORT..."
  # Loop for up to 2 minutes (120 seconds) waiting for the health endpoint
  for j in {1..60}; do
    # Use curl's exit code to check for success (200 OK)
    if curl -fsS --max-time 2 "http://localhost:${PORT}/health" > /dev/null; then
      echo " OK!"
      break
    fi
    echo -n "."
    sleep 2
  done
done


echo
echo "Probing endpoints and listing active containers..."
for ((i=0;i<NUM_CONTAINERS;i++)); do
  PORT=$((START_PORT+i))
  echo -n "POST /infer @ :$PORT -> "
  curl --max-time 30 -s -X POST "http://localhost:${PORT}/infer" \
    -H "Content-Type: application/json" \
    -d "{\"prompt\":\"Hello from port ${PORT}\"}" | head -c 220 || echo "Request failed or timed out."
  echo
done

echo
echo "Active containers:"
sudo docker ps --filter "name=${CONTAINER_PREFIX}" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

echo
echo "Follow scheduler logs: tail -f $SCHEDULER_LOG"
echo "Stop containers: docker stop \$(docker ps -q --filter name=${CONTAINER_PREFIX}) || true"
echo "Stop scheduler: pkill -F $LOG_DIR/scheduler.pid || true"