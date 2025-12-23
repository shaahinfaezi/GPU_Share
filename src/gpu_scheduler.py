import docker
import warnings
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Suppress the pynvml deprecation warning that nvidia-ml-py still shows
warnings.filterwarnings("ignore", category=FutureWarning, module="pynvml")
try:
    import pynvml
except ImportError:
    # Fallback for when pynvml is not found, try importing nvidia_ml_py directly if possible
    # or just print a clearer error message
    logger.error("'pynvml' module not found. Please ensure 'nvidia-ml-py' is installed.")
    logger.error("If you installed 'nvidia-ml-py', try running: pip install pynvml")
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
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(h)
            gpu_util = util.gpu
            mem_util = util.memory
        except:
            gpu_util = -1
            mem_util = -1
        g.append({
            "id": i,
            "free_mb": mem.free // (1024*1024),
            "used_mb": mem.used // (1024*1024),
            "total_mb": mem.total // (1024*1024),
            "gpu_util": gpu_util,
            "mem_util": mem_util
        })
    return g

def select_gpu(required_mb):
    gpus = get_gpus()
    logger.debug(f"GPU selection - required: {required_mb}MB")
    for g in gpus:
        logger.debug(f"  GPU {g['id']}: free={g['free_mb']}MB, used={g['used_mb']}MB, util={g['gpu_util']}%")
    
    # For single GPU testing, force GPU 0
    gpu_0 = next((g for g in gpus if g['id'] == 0), None)
    if gpu_0 and gpu_0["free_mb"] > required_mb:
        logger.info(f"Forced selection of GPU 0 (free: {gpu_0['free_mb']}MB)")
        return gpu_0["id"]
    
    logger.warning(f"GPU 0 not available or insufficient memory ({gpu_0['free_mb'] if gpu_0 else 'N/A'}MB free)")
    return None

def check_mps_status():
    """Check if MPS daemon is running and accessible"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        logger.info(f"Current GPU compute apps: {result.stdout.strip() or 'None'}")
    except Exception as e:
        logger.warning(f"Could not query compute apps: {e}")
    
    mps_pipe = os.environ.get("CUDA_MPS_PIPE_DIRECTORY", "/tmp/nvidia-mps")
    if os.path.exists(mps_pipe):
        contents = os.listdir(mps_pipe)
        logger.info(f"MPS pipe directory ({mps_pipe}) contents: {contents}")
    else:
        logger.warning(f"MPS pipe directory ({mps_pipe}) does not exist!")

def launch_container(image, gpu_id, port, vram_slice_mb):
    logger.info(f"Launching container: image={image}, gpu={gpu_id}, port={port}, vram_limit={vram_slice_mb}MB")
    
    # Do NOT force MPS environment for every container.
    # Containers should work without MPS; when an MPS daemon is running
    # with the default pipe dir (/tmp/nvidia-mps) and the directory is bind-mounted,
    # CUDA can route contexts through MPS without requiring explicit envs here.
    env = {
        "NVIDIA_VISIBLE_DEVICES": str(gpu_id),
        "MODEL_NAME": os.environ.get("MODEL_NAME", "distilgpt2"),
        "GPU_VRAM_LIMIT_MB": str(vram_slice_mb), # Pass the VRAM limit to the container
        "CUDA_LAUNCH_BLOCKING": "1", # Better error messages
    }
    logger.debug(f"Container environment: {env}")
    
    name = f"{CONTAINER_PREFIX}_{gpu_id}_{port}"
    
    # Remove existing container with the same name to prevent conflict
    try:
        existing_container = client.containers.get(name)
        logger.warning(f"Removing conflicting container: {name}")
        existing_container.remove(force=True)
    except docker.errors.NotFound:
        pass # This is the expected case, no conflict
    except Exception as e:
        logger.error(f"Error removing container {name}: {e}")

    # Mount MPS pipe directory for GPU sharing (bind host -> container)
    # This is harmless if the MPS daemon is not running; when it is,
    # the default /tmp/nvidia-mps path will be used by CUDA.
    mps_pipe_dir = os.environ.get("CUDA_MPS_PIPE_DIRECTORY", "/tmp/nvidia-mps")
    
    try:
        c = client.containers.run(
            image,
            detach=True,
            name=name,
            environment=env,
            runtime="nvidia",
            ports={"5000/tcp": port},
            mem_limit=f"{vram_slice_mb}m",
            volumes={mps_pipe_dir: {"bind": mps_pipe_dir, "mode": "rw"}},
            ipc_mode="host"  # Required for MPS
        )
        logger.info(f"✓ Started container {name} (id={c.short_id}) on GPU {gpu_id}, port {port}")
        return {"name": name, "gpu_id": gpu_id, "port": port, "container_id": c.short_id}
    except Exception as e:
        logger.error(f"✗ Failed to start container {name}: {e}")
        raise

def write_snapshot():
    snap = get_gpus()
    ts = datetime.utcnow().isoformat()
    with open("vram_log.jsonl", "a") as f:
        f.write(json.dumps({"ts": ts, "snapshot": snap}) + "\n")
    logger.info(f"GPU Snapshot:")
    for g in snap:
        logger.info(f"  GPU {g['id']}: used={g['used_mb']}MB free={g['free_mb']}MB total={g['total_mb']}MB | util={g['gpu_util']}% mem_util={g['mem_util']}%")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image", default=IMAGE)
    p.add_argument("--instances", type=int, default=int(os.environ.get("NUM_CONTAINERS", "4")))
    p.add_argument("--vram_slice_mb", type=int, default=VRAM_SLICE_MB)
    p.add_argument("--start_port", type=int, default=INFER_PORT_START)
    return p.parse_args()

def main():
    args = parse_args()
    
    logger.info("=" * 60)
    logger.info("GPU Scheduler Starting")
    logger.info("=" * 60)
    logger.info(f"Configuration:")
    logger.info(f"  Image: {args.image}")
    logger.info(f"  Instances: {args.instances}")
    logger.info(f"  VRAM per container: {args.vram_slice_mb}MB")
    logger.info(f"  Start port: {args.start_port}")
    logger.info(f"  Container prefix: {CONTAINER_PREFIX}")
    logger.info(f"  MPS pipe dir: {os.environ.get('CUDA_MPS_PIPE_DIRECTORY', '/tmp/nvidia-mps')}")
    
    # Check MPS status
    logger.info("-" * 40)
    logger.info("Checking MPS status...")
    check_mps_status()
    
    # Log initial GPU state
    logger.info("-" * 40)
    logger.info("Initial GPU State:")
    write_snapshot()
    
    # Do not auto-start MPS here; tests/scripts control MPS lifecycle explicitly

    containers = []
    port = args.start_port
    logger.info("-" * 40)
    logger.info(f"Starting to launch {args.instances} containers...")
    
    for i in range(args.instances):
        gpu_id = select_gpu(args.vram_slice_mb)
        if gpu_id is None:
            logger.error("Insufficient free memory, stopping container launches.")
            break
        logger.info(f"Launching container {i+1}/{args.instances} on GPU {gpu_id}, port {port}")
        info = launch_container(args.image, gpu_id, port, args.vram_slice_mb)
        containers.append(info)
        port += 1
        
        # Wait longer for the container to fully initialize on the GPU
        # This prevents CUDA "device busy" errors when multiple containers start on same GPU
        wait_time = 15  # seconds
        logger.info(f"Waiting {wait_time}s for container to initialize...")
        time.sleep(wait_time)
        
        # Log GPU state after each container launch
        logger.info("GPU state after container launch:")
        write_snapshot()

    logger.info("=" * 60)
    logger.info(f"Successfully launched {len(containers)} containers")
    logger.info("Container Summary:")
    for c in containers:
        logger.info(f"  {c['name']} -> GPU {c['gpu_id']}, port {c['port']}")
    logger.info("=" * 60)
    logger.info("Entering monitoring loop...")
    while True:
        write_snapshot()
        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    main()