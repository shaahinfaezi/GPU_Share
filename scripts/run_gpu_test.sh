#!/usr/bin/env bash
set -euo pipefail

# --- Scheduler virtual environment ---
VENV_DIR="/opt/gpu_share_scheduler_venv"

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
  if ! dpkg -s python3-venv >/dev/null 2>&1; then
    sudo apt-get install -y python3-venv
  fi

  # Docker engine
  if ! command -v docker >/dev/null 2>&1; then
    sudo apt-get install -y docker.io
    sudo systemctl enable --now docker
  fi

  # NVIDIA Container Toolkit (for GPU passthrough into containers)
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "NVIDIA driver (nvidia-smi) not found."
    
    # Check for WSL
    if grep -q "Microsoft" /proc/version 2>/dev/null || grep -q "WSL" /proc/version 2>/dev/null; then
      echo "Error: You are running on WSL (Windows Subsystem for Linux)."
      echo "You cannot install NVIDIA drivers inside WSL. You must install them on Windows."
      echo "Please install the latest NVIDIA drivers on your Windows host and restart WSL."
      echo "Note: NVIDIA MPS (Multi-Process Service) is NOT supported on WSL."
      exit 1
    else
      echo "Attempting to install NVIDIA drivers..."
      sudo apt-get install -y nvidia-driver-535 nvidia-utils-535
    fi
  fi

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
    echo "Error: Failed to install/detect NVIDIA driver. Cannot proceed with GPU test."
    exit 1
  fi

  # Python deps for scheduler using venv
  # (Moved to setup_venv function so it runs even if install_prereqs is skipped)
}

setup_venv() {
  echo "Setting up Python virtual environment at $VENV_DIR ..."
  if [ ! -d "$VENV_DIR" ]; then
    sudo python3 -m venv "$VENV_DIR"
  fi

  echo "Installing scheduler Python dependencies in virtual environment..."
  sudo "$VENV_DIR/bin/pip" install --upgrade pip
  sudo "$VENV_DIR/bin/pip" install docker "nvidia-ml-py==11.525.150"
  
  # Verify installation
  echo "Verifying pynvml installation..."
  sudo "$VENV_DIR/bin/python" -c "import pynvml; print('✓ pynvml imported successfully')" || {
    echo "ERROR: pynvml import failed in venv"
    exit 1
  }
}

# Comment out to skip reinstalling dependencies on every run
#install_prereqs
setup_venv

# Config (override via env)
IMAGE="${IMAGE:-gpu-llm:cu118}"
NUM_CONTAINERS="${NUM_CONTAINERS:-3}"
START_PORT="${START_PORT:-6000}"
VRAM_LIMIT_MB="${VRAM_LIMIT_MB:-3000}" # Renamed for clarity with the shim
MODEL_NAME="${MODEL_NAME:-gpt2}"
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

# --- Fix hostname resolution warning ---
fix_hostname() {
  local current_hostname=$(hostname)
  if ! grep -q "127.0.0.1.*$current_hostname" /etc/hosts; then
    echo "Fixing hostname resolution for $current_hostname..."
    echo "127.0.0.1 $current_hostname" | sudo tee -a /etc/hosts > /dev/null
  fi
}
fix_hostname

# --- Check if base image already exists locally ---
IMAGE_NAME="nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04"

echo "--- Checking for NVIDIA CUDA base image ---"

if sudo docker image inspect "$IMAGE_NAME" > /dev/null 2>&1; then
  echo "Base image already exists locally: $IMAGE_NAME"
  BASE_IMAGE="$IMAGE_NAME"
else
  echo "Base image not found locally. You need to pull it manually."
  echo ""
  echo "Try one of these commands (find a mirror that works for you):"
  echo "  sudo docker pull hub.hamdocker.ir/$IMAGE_NAME && sudo docker tag hub.hamdocker.ir/$IMAGE_NAME $IMAGE_NAME"
  echo "  sudo docker pull docker.iranserver.com/$IMAGE_NAME && sudo docker tag docker.iranserver.com/$IMAGE_NAME $IMAGE_NAME"
  echo ""
  echo "Or if you have the image cached elsewhere, load it with:"
  echo "  sudo docker load < image.tar"
  echo ""
  echo "Once you have the image, re-run this script."
  exit 1
fi

echo "Using base image: $BASE_IMAGE"

# --- Build the Docker image ---
echo "--- Building Docker image: $IMAGE (log: $BUILD_LOG) ---"
# Pass MODEL_NAME and BASE_IMAGE as build arguments
sudo docker build \
  --network host \
  --build-arg "MODEL_NAME=${MODEL_NAME}" \
  --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
  -t "$IMAGE" \
  -f docker/Dockerfile.scheduler . 2>&1 | tee "$BUILD_LOG"

BUILD_EXIT_CODE=${PIPESTATUS[0]}
if [ $BUILD_EXIT_CODE -ne 0 ]; then
  echo "Error: Docker build failed with exit code $BUILD_EXIT_CODE" >&2
  echo "Check $BUILD_LOG for details."
  exit 1
fi

echo "Build complete. See $BUILD_LOG for details."

# --- Start NVIDIA MPS for GPU sharing ---
echo "--- Setting up NVIDIA MPS for concurrent GPU access ---"

# Get number of GPUs
NUM_GPUS=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)
echo "Detected $NUM_GPUS GPU(s)"

# Set GPU to DEFAULT mode (not exclusive) for initial setup
# MPS will be started later by the benchmark script if needed
# Ensure TARGET_GPU is defined (defaults to 0)
TARGET_GPU="${TARGET_GPU:-0}"
export TARGET_GPU
echo "Setting GPU ${TARGET_GPU} to DEFAULT compute mode for initial setup..."
sudo nvidia-smi -i ${TARGET_GPU} -c DEFAULT || true

# MPS will be started by benchmark_mps.py before Test 2
echo "MPS daemon will be started later by the benchmark script"

# --- Run the scheduler in the background ---
echo "--- Starting GPU scheduler (log: $SCHEDULER_LOG) ---"
# Export variables for the scheduler process
export IMAGE
export NUM_CONTAINERS
export START_PORT
export VRAM_SLICE_MB="$VRAM_LIMIT_MB" # Pass the limit to the scheduler
export MODEL_NAME
export CONTAINER_PREFIX
# Target GPU for all scripts (set to '0' by default)
export TARGET_GPU="0"

# Run the scheduler as a background process and save its PID
# Use explicit PYTHONPATH to ensure venv packages are found under sudo
# Dynamically find site-packages to avoid hardcoding python version
SITE_PACKAGES=$(find "$VENV_DIR/lib" -maxdepth 2 -type d -name "site-packages" | head -n 1)
if [ -z "$SITE_PACKAGES" ]; then
    echo "Error: Could not find site-packages in $VENV_DIR"
    exit 1
fi

sudo PYTHONPATH="$SITE_PACKAGES" \
  "$VENV_DIR/bin/python" src/gpu_scheduler.py > "$SCHEDULER_LOG" 2>&1 &
SCHEDULER_PID=$!
echo $SCHEDULER_PID > "$LOG_DIR/scheduler.pid"
echo "Scheduler running in background with PID $SCHEDULER_PID."
echo "View logs with: sudo tail -f $SCHEDULER_LOG"

# Give the scheduler a moment to start up or fail
sleep 5
if ! ps -p $SCHEDULER_PID > /dev/null; then
    echo "ERROR: Scheduler failed to start immediately. Checking logs:"
    tail -n 20 "$SCHEDULER_LOG"
    exit 1
fi

# --- Wait for containers to launch and become healthy ---
echo "--- Waiting for containers to be launched by the scheduler... ---"
END_PORT=$((START_PORT + NUM_CONTAINERS - 1))
HEALTH_CHECK_TIMEOUT=300 # 5 minutes total
SLEEP_INTERVAL=10
SECONDS_WAITED=0

# Loop through the expected port range
for PORT in $(seq $START_PORT $END_PORT); do
  echo "Waiting for container on port $PORT to become healthy..."
  while true; do
    # Check if the scheduler process is still alive
    if ! ps -p $SCHEDULER_PID > /dev/null; then
      echo "Error: Scheduler process with PID $SCHEDULER_PID has died."
      echo "--- Last 30 lines of scheduler log ($SCHEDULER_LOG): ---"
      tail -n 30 "$SCHEDULER_LOG"
      echo "--- Current docker containers: ---"
      sudo docker ps -a --filter "name=${CONTAINER_PREFIX}"
      echo "--------------------------------------------------------"
      exit 1
    fi

    # Check if the container for this port exists and is running
    CONTAINER_ID=$(sudo docker ps -q --filter "publish=${PORT}" --filter "name=${CONTAINER_PREFIX}")
    if [ -z "$CONTAINER_ID" ]; then
        echo "Container for port $PORT not found yet. Waiting... (checking docker ps -a)"
        sudo docker ps -a --filter "name=${CONTAINER_PREFIX}" | tail -5
    else
        # Container found, now check its health endpoint
        HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:${PORT}/health" || true)
        if [ "$HTTP_STATUS" = "200" ]; then
          echo "Container on port $PORT is healthy (HTTP 200)."
          break # Exit the inner while loop, move to the next port
        else
          echo "Container on port $PORT found, but not healthy yet (HTTP status: $HTTP_STATUS). Retrying..."
        fi
    fi

    # Timeout check
    SECONDS_WAITED=$((SECONDS_WAITED + SLEEP_INTERVAL))
    if [ $SECONDS_WAITED -ge $HEALTH_CHECK_TIMEOUT ]; then
      echo "Error: Timeout reached waiting for containers to become healthy."
      echo "--- Current Docker container status: ---"
      sudo docker ps -a --filter "name=${CONTAINER_PREFIX}"
      echo ""
      echo "--- Checking logs of failed/exited containers: ---"
      for CONTAINER in $(sudo docker ps -aq --filter "name=${CONTAINER_PREFIX}" --filter "status=exited"); do
        CONTAINER_NAME=$(sudo docker inspect --format='{{.Name}}' $CONTAINER | sed 's/\///')
        echo "=== Logs for $CONTAINER_NAME ==="
        sudo docker logs --tail 50 $CONTAINER
        echo ""
      done
      echo "--- Last 50 lines of scheduler log ($SCHEDULER_LOG): ---"
      tail -n 50 "$SCHEDULER_LOG"
      echo "--------------------------------------------------------"
      exit 1
    fi
    sleep $SLEEP_INTERVAL
  done
done

echo "--- All containers are healthy. Running final probe. ---"
# Final probe to check inference on the first container
PROBE_PORT=$START_PORT
echo "Probing inference endpoint on port $PROBE_PORT..."
curl -X POST http://localhost:$PROBE_PORT/infer \
   -H "Content-Type: application/json" \
   -d '{"prompt": "Hello, world"}'

echo -e "\n\n--- Test finished. Scheduler is still running. ---"
echo "To stop the scheduler and clean up containers, run:"
echo "sudo pkill -F $LOG_DIR/scheduler.pid"
echo "Or use the teardown script: sudo bash ./scripts/teardown.sh"
echo "--------------------------------------------------------------------"
exit 0