#!/usr/bin/env python3
# server.py — simple LLM-mock web server that allocates GPU tensors to simulate model load
from flask import Flask, request, jsonify
import torch
import os
import time
import threading

app = Flask(__name__)

# Default model allocation size in MB (tune for your experiments)
TARGET_MB = int(os.environ.get("TARGET_MB", "8000"))

# We keep references in this list to prevent garbage collection (simulate loaded model)
_allocated = []
_alloc_lock = threading.Lock()

def allocate_mb(mb):
    # Allocate approx mb MB of float32 (4 bytes) on GPU
    num_elements = (mb * 1024 * 1024) // 4
    t = torch.empty(num_elements, device="cuda", dtype=torch.float32)
    t.zero_()
    return t

@app.route("/load", methods=["POST"])
def load():
    global _allocated
    mb = int(request.json.get("mb", TARGET_MB))
    with _alloc_lock:
        try:
            t = allocate_mb(mb)
            _allocated.append(t)
            return jsonify({"status": "ok", "action": "allocated", "mb": mb, "len_alloc": len(_allocated)}), 200
        except Exception as e:
            return jsonify({"status": "error", "error": str(e)}), 500

@app.route("/unload", methods=["POST"])
def unload():
    global _allocated
    with _alloc_lock:
        _allocated = []
        torch.cuda.empty_cache()
    return jsonify({"status": "freed"}), 200

@app.route("/infer", methods=["POST"])
def infer():
    # Simulate a small inference kernel (burn some compute)
    # Accepts {"work_ms":50} to simulate heavier tasks
    work = int(request.json.get("work_ms", 20))
    # a tiny kernel: elementwise op on a small tensor
    x = torch.rand((1024, 1024), device="cuda")
    start = time.time()
    # loop compute for approximate time (not precise)
    while (time.time() - start) < (work / 1000.0):
        x = x * 0.999999 + 0.000001 * torch.randn_like(x)
    return jsonify({"status": "ok", "work_ms": work, "timestamp": time.time()}), 200

@app.route("/info", methods=["GET"])
def info():
    import pynvml
    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(0)
    mem = pynvml.nvmlDeviceGetMemoryInfo(h)
    return jsonify({
        "total": mem.total,
        "free": mem.free,
        "used": mem.used,
        "allocated_models": len(_allocated)
    })

if __name__ == "__main__":
    # ensure CUDA is visible
    if not torch.cuda.is_available():
        print("CUDA not available inside container")
    app.run(host="0.0.0.0", port=5000)
