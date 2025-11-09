# Multi-tenant LLM prototype (MPS + memory-aware autoscaler) — CUDA 13, RTX 3080 Ti

This repo contains a minimal prototype to test software-based multi-tenancy for LLM inference on a consumer GPU (RTX 3080 Ti, CUDA 13). It uses:

- NVIDIA MPS (compute concurrency)
- Docker containers for per-LLM servers (mock or real)
- A Python autoscaler/admission-controller that checks GPU free memory (pynvml) and starts/stops containers
- Optional Triton warmup guidance
- Benchmark scripts and metric collection

## Preconditions (on host / rented machine)
- CUDA 13 driver installed and working for RTX 3080 Ti (verify `nvidia-smi`)
- `nvidia-container-toolkit` installed and configured for Docker (or containerd)
- Docker installed (or Podman with nvidia support)
- Python 3.10+ on host (for autoscaler & benchmark)
- Internet to pull images and Python packages

## Quick checklist
1. Verify drivers:
   ```bash
   nvidia-smi
