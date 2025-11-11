#!/bin/bash
# ---------------------------------------------------------------------------
# run_gpu_test.sh
#   Builds the ultra-light LLM mock (ctypes-based), starts MPS,
#   launches multiple inference containers across all GPUs, and runs benchmark.
# ---------------------------------------------------------------------------

set -e

IMAGE="llm-ultra-min:cuda115"
CONTAINERS=8               # total containers to create
PER_INSTANCE_MB=150        # simulated model memory per container
START_PORT=6000
RPS=40                     # requests per second
DURATION=120               # benchmark duration seconds

echo "=== Step 1: Enable NVIDIA MPS ==="
sudo systemctl enable --now nvidia-mps
sleep 2
sudo systemctl status nvidia-mps --no-pager

echo "=== Step 2: Build lightweight image ==="
cd /opt/GPU_Share
docker build -t $IMAGE .

echo "=== Step 3: Run GPU Scheduler ==="
nohup python3 gpu_scheduler.py \
  --image $IMAGE \
  --instances $CONTAINERS \
  --per_instance_mb $PER_INSTANCE_MB \
  --start_port $START_PORT \
  > scheduler.log 2>&1 &

echo "Scheduler running in background (see scheduler.log)"
sleep 10

echo "=== Step 4: Run VRAM predictor ==="
PORTS=""
for ((i=0; i<$CONTAINERS; i++)); do
  PORTS="$PORTS $((START_PORT+i))"
done

nohup python3 vrampredictor.py \
  --ports $PORTS \
  --out vram_run.csv \
  --interval 1 \
  --duration $DURATION \
  > vrampredictor.log 2>&1 &

sleep 5

echo "=== Step 5: Run benchmark ==="
python3 benchmark.py \
  --targets $CONTAINERS \
  --rps $RPS \
  --duration $DURATION \
  --out bench_run.csv

echo "=== Step 6: Wait for background logs ==="
wait

echo "=== Step 7: Cleanup ==="
pkill -f gpu_scheduler.py || true
docker ps -a | awk '/vm_/{print $1}' | xargs -r docker rm -f
echo "All test containers removed."

echo "=== DONE ==="
echo "Results:"
echo "  - VRAM log: vram_run.csv"
echo "  - Benchmark results: bench_run.csv"
echo "  - Scheduler log: scheduler.log"
