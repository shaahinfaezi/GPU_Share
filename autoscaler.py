#!/usr/bin/env python3
"""
autoscaler.py — memory-aware docker autoscaler & admission controller
Maintains between MIN_CONTAINERS and MAX_CONTAINERS instances of llm-mock,
and only launches a new container when GPU free memory > MIN_FREE_MB_TO_LAUNCH.

Requires:
  pip3 install docker pynvml requests
"""
import os
import time
import sys
import signal
import socket
import fcntl
import struct
import docker
import pynvml
import requests

# CONFIG via env or defaults
IMAGE_NAME = os.environ.get("IMAGE_NAME", "llm-mock:cuda13")
CONTAINER_BASE_NAME = os.environ.get("CONTAINER_BASE_NAME", "llm-instance")
MIN_FREE_MB_TO_LAUNCH = int(os.environ.get("MIN_FREE_MB_TO_LAUNCH", "9000"))  # MB
SAFETY_MARGIN_MB = int(os.environ.get("SAFETY_MARGIN_MB", "600"))           # MB safety guard
MIN_CONTAINERS = int(os.environ.get("MIN_CONTAINERS", "1"))
MAX_CONTAINERS = int(os.environ.get("MAX_CONTAINERS", "4"))
CHECK_INTERVAL = int(os.environ.get("CHECK_INTERVAL", "5"))
MODEL_MB = int(os.environ.get("MODEL_MB", "8000"))  # default model allocation
DOCKER_PORT_BASE = int(os.environ.get("DOCKER_PORT_BASE", "5000"))

LOCKFILE = "/tmp/llm_autoscaler.lock"

client = docker.from_env()
pynvml.nvmlInit()
gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)

def acquire_lock():
    # simple lock via file
    fd = os.open(LOCKFILE, os.O_CREAT | os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return fd
    except IOError:
        os.close(fd)
        return None

def release_lock(fd):
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
    except:
        pass

def gpu_free_mb():
    mem = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
    return int(mem.free // (1024 * 1024))

def list_managed_containers():
    allc = client.containers.list(all=True)
    managed = []
    for c in allc:
        if c.name.startswith(CONTAINER_BASE_NAME):
            managed.append(c)
    # sort by name (predictable)
    managed.sort(key=lambda x: x.name)
    return managed

def next_container_idx(managed):
    # find next numeric index not used
    used = set()
    for c in managed:
        parts = c.name.split("-")
        if parts[-1].isdigit():
            used.add(int(parts[-1]))
    i = 1
    while i in used:
        i += 1
    return i

def start_container(idx):
    name = f"{CONTAINER_BASE_NAME}-{idx}"
    port = DOCKER_PORT_BASE + idx
    print(f"[autoscaler] starting {name} on host port {port}")
    env = {"TARGET_MB": str(MODEL_MB)}
    # run container detached
    c = client.containers.run(
        IMAGE_NAME,
        detach=True,
        name=name,
        environment=env,
        runtime="nvidia",
        remove=False,
        ports={'5000/tcp': port},
        shm_size='1g'
    )
    # small delay, then trigger /load to allocate memory
    time.sleep(2.5)
    try:
        requests.post(f"http://localhost:{port}/load", json={"mb": MODEL_MB}, timeout=5)
    except Exception as e:
        print("warning: could not trigger /load:", e)
    return c

def stop_container(container):
    print(f"[autoscaler] stopping {container.name}")
    # try an unload first
    try:
        port = container.attrs['NetworkSettings']['Ports']['5000/tcp'][0]['HostPort']
        requests.post(f"http://localhost:{port}/unload", timeout=2)
    except Exception:
        pass
    try:
        container.stop(timeout=5)
    except Exception:
        pass
    try:
        container.remove()
    except Exception:
        pass

def ensure_min_containers():
    managed = list_managed_containers()
    running = [c for c in managed if c.status == 'running']
    while len(running) < MIN_CONTAINERS:
        idx = next_container_idx(managed)
        c = start_container(idx)
        managed.append(c)
        running.append(c)
        time.sleep(1)

def main_loop():
    print("[autoscaler] starting main loop")
    while True:
        fd = acquire_lock()
        if not fd:
            # another instance is running
            print("[autoscaler] lock busy, sleeping")
            time.sleep(CHECK_INTERVAL)
            continue
        try:
            free_mb = gpu_free_mb()
            print(f"[autoscaler] GPU free MB: {free_mb}")
            managed = list_managed_containers()
            running = [c for c in managed if c.status == 'running']
            # scale up
            if free_mb > (MIN_FREE_MB_TO_LAUNCH + SAFETY_MARGIN_MB) and len(running) < MAX_CONTAINERS:
                idx = next_container_idx(managed)
                try:
                    start_container(idx)
                except Exception as e:
                    print("[autoscaler] failed to start:", e)
            # scale down condition: free memory low and more than min containers
            elif free_mb < (MODEL_MB // 2) and len(running) > MIN_CONTAINERS:
                # pick last started container to stop
                to_stop = running[-1]
                stop_container(to_stop)
        finally:
            release_lock(fd)
        time.sleep(CHECK_INTERVAL)

def graceful_shutdown(signum, frame):
    print("[autoscaler] received shutdown, exiting")
    sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGTERM, graceful_shutdown)
    signal.signal(signal.SIGINT, graceful_shutdown)
    ensure_min_containers()
    main_loop()
