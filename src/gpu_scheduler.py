import docker
import warnings
# Suppress the pynvml deprecation warning that nvidia-ml-py still shows
warnings.filterwarnings("ignore", category=FutureWarning, module="pynvml")
try:
    import pynvml
except ImportError:
    # Fallback for when pynvml is not found, try importing nvidia_ml_py directly if possible
    # or just print a clearer error message
    print("Error: 'pynvml' module not found. Please ensure 'nvidia-ml-py' is installed.")
    print("If you installed 'nvidia-ml-py', try running: pip install pynvml")
    # In newer versions of nvidia-ml-py, the module is still called pynvml, 
    # but sometimes there are path issues.
    raise

import subprocess
import time
import json
import threading
import requests
import random
import os
from datetime import datetime
import argparse

# Configurable parameters
IMAGE = os.environ.get("IMAGE", "gpu-llm:cu118")
CONTAINER_PREFIX = os.environ.get("CONTAINER_PREFIX", "llm_tenant")
INFER_PORT_START = int(os.environ.get("START_PORT", 6000))
VRAM_SLICE_MB = int(os.environ.get("VRAM_SLICE_MB", 3000))
CHECK_INTERVAL = 5
GPU_REFRESH_INTERVAL = 2

# Init Docker + NVML
client = docker.from_env()
pynvml.nvmlInit()

def get_gpus():
    g = []
    for i in range(pynvml.nvmlDeviceGetCount()):
        h = pynvml.nvmlDeviceGetHandleByIndex(i)
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        g.append({
            "id": i,
            "free_mb": mem.free // (1024*1024),
            "used_mb": mem.used // (1024*1024),
            "total_mb": mem.total // (1024*1024)
        })
    return g

def select_gpu(required_mb):
    for g in sorted(get_gpus(), key=lambda x: -x["free_mb"]):
        if g["free_mb"] > required_mb:
            return g["id"]
    return None

def launch_container(image, gpu_id, port, vram_slice_mb):
    env = {
        "NVIDIA_VISIBLE_DEVICES": str(gpu_id),
        "MODEL_NAME": os.environ.get("MODEL_NAME", "distilgpt2"),
        "GPU_VRAM_LIMIT_MB": str(vram_slice_mb) # Pass the VRAM limit to the container
    }
    name = f"{CONTAINER_PREFIX}_{gpu_id}_{port}"
    
    # Remove existing container with the same name to prevent conflict
    try:
        existing_container = client.containers.get(name)
        print(f"[scheduler] Removing conflicting container: {name}")
        existing_container.remove(force=True)
    except docker.errors.NotFound:
        pass # This is the expected case, no conflict
    except Exception as e:
        print(f"[scheduler] Error removing container {name}: {e}")

    c = client.containers.run(
        image,
        detach=True,
        name=name,
        environment=env,
        runtime="nvidia",
        ports={"5000/tcp": port},
        mem_limit=f"{vram_slice_mb}m"
    )
    print(f"[scheduler] started {name} gpu={gpu_id} port={port}")
    return {"name": name, "gpu_id": gpu_id, "port": port}

def write_snapshot():
    snap = get_gpus()
    ts = datetime.utcnow().isoformat()
    with open("vram_log.jsonl", "a") as f:
        f.write(json.dumps({"ts": ts, "snapshot": snap}) + "\n")
    print(f"[{ts}] GPU status:")
    for g in snap:
        print(f"  gpu {g['id']}: used={g['used_mb']}MB free={g['free_mb']}MB total={g['total_mb']}MB")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image", default=IMAGE)
    p.add_argument("--instances", type=int, default=int(os.environ.get("NUM_CONTAINERS", "4")))
    p.add_argument("--vram_slice_mb", type=int, default=VRAM_SLICE_MB)
    p.add_argument("--start_port", type=int, default=INFER_PORT_START)
    return p.parse_args()

def main():
    args = parse_args()
    # MPS should already be started by run_gpu_test.sh, but try again just in case
    subprocess.run(["sudo", "systemctl", "start", "nvidia-mps"], check=False)

    containers = []
    port = args.start_port
    print(f"[scheduler] Starting to launch {args.instances} containers...")
    print(f"[scheduler] Image: {args.image}, VRAM per container: {args.vram_slice_mb}MB")
    
    for i in range(args.instances):
        gpu_id = select_gpu(args.vram_slice_mb)
        if gpu_id is None:
            print("[scheduler] insufficient free memory, stop launching.")
            break
        print(f"[scheduler] Launching container {i+1}/{args.instances} on GPU {gpu_id}, port {port}")
        info = launch_container(args.image, gpu_id, port, args.vram_slice_mb)
        containers.append(info)
        port += 1
        
        # Wait longer for the container to fully initialize on the GPU
        # This prevents CUDA "device busy" errors when multiple containers start on same GPU
        wait_time = 15  # seconds
        print(f"[scheduler] Waiting {wait_time}s for container to initialize...")
        time.sleep(wait_time)

    print(f"[scheduler] Successfully launched {len(containers)} containers")
    print("[scheduler] Entering monitoring loop...")
    while True:
        write_snapshot()
        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    main()