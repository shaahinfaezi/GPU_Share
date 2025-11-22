#!/usr/bin/env bash
set -euo pipefail

# --- Configure DNS to bypass Docker Hub 403 Forbidden (Shecan) ---
configure_dns() {
  echo "--- Configuring DNS to bypass Docker Hub 403 Forbidden ---"
  # Backup existing resolv.conf if not already backed up
  if [ ! -f /etc/resolv.conf.bak ]; then
    sudo cp /etc/resolv.conf /etc/resolv.conf.bak
  fi
  
  # Set Shecan DNS servers (178.22.122.100, 185.51.200.2)
  echo "nameserver 178.22.122.100" | sudo tee /etc/resolv.conf > /dev/null
  echo "nameserver 185.51.200.2" | sudo tee -a /etc/resolv.conf > /dev/null
  
  echo "DNS updated. Restarting Docker daemon to pick up changes..."
  sudo systemctl restart docker || true
}
configure_dns

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

# --- Find a working Docker Mirror ---
find_working_mirror() {
  local image_tag="pytorch/pytorch:2.3.1-cuda11.8-cudnn8-devel"
  local mirrors=(
    "docker.ir"
    "docker.arvancloud.ir"
    "huecker.io"
    "docker.io" # Fallback to official
  )

  echo "--- Searching for a working Docker mirror for $image_tag ---"
  
  for mirror in "${mirrors[@]}"; do
    if [ "$mirror" == "docker.io" ]; then
      full_image="$image_tag"
    else
      full_image="$mirror/$image_tag"
    fi
    
    echo "Trying to pull from $mirror ($full_image)..."
    if sudo docker pull "$full_image"; then
      echo "Successfully pulled from $mirror."
      # Tag it as the official name so local builds find it if they don't use ARG
      if [ "$mirror" != "docker.io" ]; then
        sudo docker tag "$full_image" "$image_tag"
      fi
      echo "$full_image"
      return 0
    else
      echo "Failed to pull from $mirror."
    fi
  done
  
  echo "Error: Could not pull base image from any known mirror." >&2
  return 1
}

# --- Pre-pull base image ---
BASE_IMAGE_FULL=$(find_working_mirror)
if [ $? -ne 0 ]; then
  exit 1
fi
echo "Using base image: $BASE_IMAGE_FULL"

# --- Build the Docker image ---
echo "--- Building Docker image: $IMAGE (log: $BUILD_LOG) ---"
# Pass MODEL_NAME and BASE_IMAGE as build arguments
sudo docker build \
  --build-arg "MODEL_NAME=${MODEL_NAME}" \
  --build-arg "BASE_IMAGE=${BASE_IMAGE_FULL}" \
  -t "$IMAGE" \
  -f docker/Dockerfile.scheduler . > "$BUILD_LOG" 2>&1

echo "Build complete. See $BUILD_LOG for details."

# --- Start NVIDIA MPS if available ---
echo "--- Attempting to start NVIDIA MPS for better concurrent performance ---"
sudo nvidia-smi -i 0 -c EXCLUSIVE_PROCESS # Set GPU 0 to exclusive mode if possible
sudo systemctl start nvidia-mps # This may fail if not configured, which is OK
# The scheduler will also try to start it, this is just an early attempt.

# --- Run the scheduler in the background ---
echo "--- Starting GPU scheduler (log: $SCHEDULER_LOG) ---"
# Export variables for the scheduler process
export IMAGE
export NUM_CONTAINERS
export START_PORT
export VRAM_SLICE_MB="$VRAM_LIMIT_MB" # Pass the limit to the scheduler
export MODEL_NAME
export CONTAINER_PREFIX

# Run the scheduler as a background process and save its PID
sudo -E python3 src/gpu_scheduler.py > "$SCHEDULER_LOG" 2>&1 &
SCHEDULER_PID=$!
echo $SCHEDULER_PID > "$LOG_DIR/scheduler.pid"
echo "Scheduler running in background with PID $SCHEDULER_PID."

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
      echo "--- Last 20 lines of scheduler log ($SCHEDULER_LOG): ---"
      tail -n 20 "$SCHEDULER_LOG"
      echo "--------------------------------------------------------"
      exit 1
    fi

    # Check if the container for this port exists and is running
    CONTAINER_ID=$(sudo docker ps -q --filter "publish=${PORT}" --filter "name=${CONTAINER_PREFIX}")
    if [ -z "$CONTAINER_ID" ]; then
        echo "Container for port $PORT not found yet. Waiting..."
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
      sudo docker ps -a
      echo "--- Last 20 lines of scheduler log ($SCHEDULER_LOG): ---"
      tail -n 20 "$SCHEDULER_LOG"
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
# Pass MODEL_NAME as a build argument to the Dockerfile
if sudo docker build \
    --build-arg MODEL_NAME=$MODEL_NAME \
    -f "$DOCKERFILE" -t "$IMAGE" . > "$BUILD_LOG" 2>&1; then
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
HEALTH_TIMEOUT=120 # seconds
ALL_HEALTHY=true

for ((i=0;i<NUM_CONTAINERS;i++)); do
  PORT=$((START_PORT+i))
  CONTAINER_NAME="${CONTAINER_PREFIX}_*_$PORT" # Use glob to find name regardless of GPU ID
  
  echo "Checking container on port $PORT..."
  
  HEALTHY=false
  for j in $(seq 1 $((HEALTH_TIMEOUT / 2))); do
    # Check if container is still running
    if ! sudo docker ps --filter "name=${CONTAINER_NAME}" --format "{{.Names}}" | grep -q .; then
        echo -e "\nError: Container for port $PORT has stopped unexpectedly."
        # Find the *stopped* container to get its logs
        STOPPED_CONTAINER_ID=$(sudo docker ps -aq --filter "name=${CONTAINER_NAME}" --filter "status=exited" | head -n 1)
        if [ -n "$STOPPED_CONTAINER_ID" ]; then
            echo "--- Last 50 lines of logs from stopped container $STOPPED_CONTAINER_ID: ---"
            sudo docker logs --tail 50 "$STOPPED_CONTAINER_ID"
            echo "----------------------------------------------------------------"
        fi
        HEALTHY=false
        break
    fi

    # Check health endpoint
    if curl -fsS --max-time 2 "http://localhost:${PORT}/health" > /dev/null; then
      echo "Container on port $PORT is healthy."
      HEALTHY=true
      break
    fi
    echo -n "."
    sleep 2
  done

  if [ "$HEALTHY" = false ]; then
    echo -e "\nError: Container on port $PORT failed to become healthy within $HEALTH_TIMEOUT seconds."
    ALL_HEALTHY=false
    # Get logs from the (presumably) still-running but unhealthy container
    RUNNING_CONTAINER_ID=$(sudo docker ps -q --filter "name=${CONTAINER_NAME}")
    if [ -n "$RUNNING_CONTAINER_ID" ]; then
        echo "--- Last 50 lines of logs from unhealthy container $RUNNING_CONTAINER_ID: ---"
        sudo docker logs --tail 50 "$RUNNING_CONTAINER_ID"
        echo "--------------------------------------------------------------------"
    fi
  fi
done

if [ "$ALL_HEALTHY" = false ]; then
    echo
    echo "One or more containers failed to start. Please check the logs above." >&2
    exit 1
fi


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