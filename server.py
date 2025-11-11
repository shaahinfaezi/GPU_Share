#!/usr/bin/env python3
"""
Ultra-light LLM mock using PyCUDA.
Allocates small GPU memory chunks and performs trivial GPU ops per request.
"""
from flask import Flask, request, jsonify
import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
import pynvml, time, os

app = Flask(__name__)

ALLOC_MB = int(os.environ.get("TARGET_MB", "100"))  # default model VRAM size

# allocate small chunk to simulate model load
buf = drv.mem_alloc(ALLOC_MB * 1024 * 1024)

@app.route("/infer", methods=["POST"])
def infer():
    t0 = time.time()
    # simulate compute by GPU memcpy (uses SMs briefly)
    arr = np.random.rand(512, 512).astype(np.float32)
    gpu_buf = drv.mem_alloc(arr.nbytes)
    drv.memcpy_htod(gpu_buf, arr)
    drv.memcpy_dtoh(arr, gpu_buf)
    drv.Context.synchronize()
    latency = (time.time() - t0) * 1000
    return jsonify({"status": "ok", "latency_ms": round(latency, 2)})

@app.route("/info", methods=["GET"])
def info():
    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(0)
    m = pynvml.nvmlDeviceGetMemoryInfo(h)
    return jsonify({
        "total_mb": m.total // (1024*1024),
        "used_mb": m.used // (1024*1024),
        "free_mb": m.free // (1024*1024),
        "alloc_mb": ALLOC_MB
    })

@app.route("/load", methods=["POST"])
def load():
    global buf
    mb = int(request.json.get("mb", ALLOC_MB))
    buf = drv.mem_alloc(mb * 1024 * 1024)
    return jsonify({"status": "ok", "allocated": mb})

@app.route("/unload", methods=["POST"])
def unload():
    global buf
    buf.free()
    return jsonify({"status": "freed"})

if __name__ == "__main__":
    print(f"Starting lightweight LLM mock (alloc={ALLOC_MB} MB)")
    app.run(host="0.0.0.0", port=5000)
