#!/usr/bin/env python3
# server.py - tiny CUDA runtime mock using ctypes (no pycuda)
from flask import Flask, request, jsonify
import ctypes, os, time, numpy as np, pynvml

app = Flask(__name__)

# load libcudart (the runtime library)
# ctypes will find it if in LD_LIBRARY_PATH (runtime image has it)
_lib = None
for name in ("libcudart.so", "libcudart.so.11.0", "libcudart.so.11.5"):
    try:
        _lib = ctypes.CDLL(name)
        break
    except OSError:
        _lib = None

if _lib is None:
    raise RuntimeError("Could not load libcudart. Ensure you run in an NVIDIA CUDA runtime container.")

# function prototypes
_lib.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
_lib.cudaFree.argtypes = [ctypes.c_void_p]
_lib.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
_lib.cudaDeviceSynchronize.argtypes = []
# copy kinds
cudaMemcpyHostToDevice = 1
cudaMemcpyDeviceToHost = 2

# simple wrapper helpers
def cuda_malloc_bytes(nbytes):
    ptr = ctypes.c_void_p()
    err = _lib.cudaMalloc(ctypes.byref(ptr), ctypes.c_size_t(nbytes))
    if err != 0:
        raise RuntimeError(f"cudaMalloc failed with code {err}")
    return ptr

def cuda_free(ptr):
    if ptr:
        _lib.cudaFree(ptr)

def cuda_memcpy_htod(dev_ptr, src_np):
    # ensure contiguous
    arr = np.ascontiguousarray(src_np)
    src_ptr = arr.ctypes.data_as(ctypes.c_void_p)
    n = arr.nbytes
    err = _lib.cudaMemcpy(dev_ptr, src_ptr, ctypes.c_size_t(n), cudaMemcpyHostToDevice)
    if err != 0:
        raise RuntimeError(f"cudaMemcpy HtoD failed {err}")

def cuda_memcpy_dtoh(dst_np, dev_ptr):
    dst = np.ascontiguousarray(dst_np)
    dst_ptr = dst.ctypes.data_as(ctypes.c_void_p)
    n = dst.nbytes
    err = _lib.cudaMemcpy(dst_ptr, dev_ptr, ctypes.c_size_t(n), cudaMemcpyDeviceToHost)
    if err != 0:
        raise RuntimeError(f"cudaMemcpy DtoH failed {err}")

def cuda_sync():
    _lib.cudaDeviceSynchronize()

# Simulated model allocation holder
_alloc_ptr = None
_alloc_size = 0

@app.route("/load", methods=["POST"])
def load():
    global _alloc_ptr, _alloc_size
    mb = int(request.json.get("mb", os.environ.get("TARGET_MB", "100")))
    nbytes = mb * 1024 * 1024
    try:
        if _alloc_ptr:
            cuda_free(_alloc_ptr)
        _alloc_ptr = cuda_malloc_bytes(nbytes)
        _alloc_size = nbytes
        # touch memory: allocate a small host array and copy to device once
        tmp = np.zeros(1024, dtype=np.float32)
        cuda_memcpy_htod(_alloc_ptr, tmp)
        cuda_sync()
        return jsonify({"status":"ok","allocated_mb":mb}), 200
    except Exception as e:
        return jsonify({"status":"error","error":str(e)}), 500

@app.route("/unload", methods=["POST"])
def unload():
    global _alloc_ptr, _alloc_size
    try:
        cuda_free(_alloc_ptr)
    except Exception:
        pass
    _alloc_ptr = None
    _alloc_size = 0
    return jsonify({"status":"freed"}), 200

@app.route("/infer", methods=["POST"])
def infer():
    work = float(request.json.get("work_ms", 20))
    # do a small host->device->host roundtrip to use GPU briefly
    shape = (256, 256)
    a = np.random.rand(*shape).astype(np.float32)
    b = np.zeros_like(a)
    dev = None
    try:
        dev = cuda_malloc_bytes(a.nbytes)
        cuda_memcpy_htod(dev, a)
        # trivial memcpy back
        cuda_memcpy_dtoh(b, dev)
        cuda_sync()
        cuda_free(dev)
    except Exception as e:
        if dev:
            try: cuda_free(dev)
            except: pass
        return jsonify({"status":"error","error":str(e)}), 500
    # busy-wait approximate compute time (not precise)
    t0 = time.time()
    while (time.time() - t0) < (work/1000.0):
        pass
    return jsonify({"status":"ok","work_ms":work}), 200

@app.route("/info", methods=["GET"])
def info():
    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(0)
    m = pynvml.nvmlDeviceGetMemoryInfo(h)
    return jsonify({
        "total_mb": int(m.total//(1024*1024)),
        "used_mb": int(m.used//(1024*1024)),
        "free_mb": int(m.free//(1024*1024)),
        "alloc_bytes": _alloc_size
    })

if __name__ == "__main__":
    print("Starting ultra-light LLM mock (ctypes -> libcudart).")
    app.run(host="0.0.0.0", port=5000)
