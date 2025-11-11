#!/usr/bin/env python3
"""
gpu_scheduler.py

Simple GPU-aware scheduler that:
- polls each GPU free memory (via pynvml)
- maintains a virtual partition table per GPU (sum of allocated TARGET_MB)
- starts containers assigned to a GPU if that GPU has required free memory (plus safety margin)
- exposes a simple REST admission endpoint (optional) to request starting a container

Usage (simple CLI):
  python3 gpu_scheduler.py --image llm-lite:cuda115 --instances 6 --per_instance_mb 300

Requires:
  pip3 install docker pynvml requests flask
"""
import argparse, time, os, threading, signal
import docker, pynvml, requests
from flask import Flask, request, jsonify
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Configurable safety margin
SAFETY_MARGIN_MB = int(os.environ.get("SAFETY_MARGIN_MB", "200"))

client = docker.from_env()
pynvml.nvmlInit()

app = Flask(__name__)

class GPUState:
    def __init__(self, idx):
        self.idx = idx
        self.virt_allocated_mb = 0  # sum of TARGET_MB for containers assigned
        self.containers = []  # list of container names assigned

def gpu_count():
    return pynvml.nvmlDeviceGetCount()

def gpu_free_mb(idx):
    h = pynvml.nvmlDeviceGetHandleByIndex(idx)
    mem = pynvml.nvmlDeviceGetMemoryInfo(h)
    return int(mem.free // (1024*1024))

def choose_gpu_for_mb(gpus, needed_mb):
    # choose GPU with enough free MB minus current virtual allocations
    best = None
    best_space = -1
    for g in gpus:
        free = gpu_free_mb(g.idx) - g.virt_allocated_mb - SAFETY_MARGIN_MB
        if free >= needed_mb and free > best_space:
            best_space = free
            best = g
    return best

def start_container_on_gpu(image, gpu_idx, target_mb, host_port):
    name = f"vm_{gpu_idx}_{int(time.time()*1000)%100000}"
    logging.info(f"Starting container {name} on GPU {gpu_idx} (alloc {target_mb} MB) port {host_port}")
    # DeviceRequest maps to specific device
    device_request = docker.types.DeviceRequest(count=1, device_ids=[str(gpu_idx)], capabilities=[['gpu']])
    c = client.containers.run(
        image,
        detach=True,
        name=name,
        environment={"TARGET_MB": str(target_mb)},
        device_requests=[device_request],
        ports={'5000/tcp': host_port},
        shm_size='256m'
    )
    return c, name

def stop_container_by_name(name):
    try:
        c = client.containers.get(name)
        logging.info(f"Stopping container {name}")
        try:
            c.exec_run("curl -s -X POST http://localhost:5000/unload", timeout=3)
        except:
            pass
        c.stop(timeout=5)
        c.remove()
    except Exception as e:
        logging.warning(f"stop_container_by_name error: {e}")

def scheduler_loop(image, total_instances, per_instance_mb, start_port, poll_interval=3):
    # initialize GPU states
    num_gpus = gpu_count()
    gpus = [GPUState(i) for i in range(num_gpus)]
    next_port = start_port
    created = 0
    # ensure existing managed containers are known
    while created < total_instances:
        gpu = choose_gpu_for_mb(gpus, per_instance_mb)
        if gpu is None:
            logging.info("No GPU currently has enough free memory for another instance. Waiting...")
            time.sleep(poll_interval)
            # refresh states - check if any containers stopped/removed
            # also recompute virt_allocated_mb by listing running named containers
            # naive approach here: we just loop
            continue
        # start a container on selected GPU
        c, name = start_container_on_gpu(image, gpu.idx, per_instance_mb, next_port)
        # after start, give some time then call /load to allocate memory
        time.sleep(2)
        try:
            requests.post(f"http://localhost:{next_port}/load", json={"mb": per_instance_mb}, timeout=6)
        except Exception as e:
            logging.warning(f"post-load failed for {name}: {e}")
        gpu.virt_allocated_mb += per_instance_mb
        gpu.containers.append(name)
        logging.info(f"Assigned {name} to GPU {gpu.idx}. virtual allocated now: {gpu.virt_allocated_mb} MB")
        next_port += 1
        created += 1

    logging.info("All requested instances created. Scheduler will continue to monitor.")
    # Monitor loop: remove containers if actual free memory low, etc.
    try:
        while True:
            for g in gpus:
                # check real free
                free = gpu_free_mb(g.idx)
                logging.debug(f"GPU {g.idx} real free: {free} MB, virtual sum: {g.virt_allocated_mb} MB")
                # if free < 100MB, try to evict one container from that GPU (LIFO)
                if free < 100 and g.containers:
                    victim = g.containers.pop()
                    logging.warning(f"Free mem low on GPU {g.idx}. Evicting {victim}")
                    stop_container_by_name(victim)
                    # approximate reduce virt allocation by per_instance_mb
                    g.virt_allocated_mb = max(0, g.virt_allocated_mb - per_instance_mb)
            time.sleep(poll_interval)
    except KeyboardInterrupt:
        logging.info("Scheduler exiting due to KeyboardInterrupt")
        # optional cleanup left to user

# Simple REST endpoint to request start (optional)
@app.route("/request_start", methods=["POST"])
def request_start():
    data = request.json or {}
    mb = int(data.get("mb", 250))
    # choose gpu
    num_gpus = gpu_count()
    gpus = [GPUState(i) for i in range(num_gpus)]
    gpu = choose_gpu_for_mb(gpus, mb)
    if gpu is None:
        return jsonify({"status":"no_capacity"}), 503
    # we won't start via REST here in this minimal implementation
    return jsonify({"status":"ok", "gpu": gpu.idx}), 200

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--instances", type=int, default=4)
    parser.add_argument("--per_instance_mb", type=int, default=300)
    parser.add_argument("--start_port", type=int, default=6000)
    args = parser.parse_args()
    scheduler_loop(args.image, args.instances, args.per_instance_mb, args.start_port)

if __name__ == "__main__":
    main()
