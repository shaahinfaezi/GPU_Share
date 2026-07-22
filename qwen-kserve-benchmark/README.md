# Two-Container Qwen KServe Benchmark

This package compares two identical Qwen services on one approximately 12 GiB NVIDIA GPU:

1. Sequential execution.
2. MPS-only concurrent execution.
3. HAMi concurrent execution without MPS.
4. HAMi plus MPS concurrent execution.

Only files in this directory are used. Existing files under `GPU_Share/kubernetes/` and `GPU_Share/hami-mps-benchmark/` are not modified.

This folder now includes the missing runner and cleanup scripts for the two-container benchmark flow.

## Fixed workload

- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- Model storage: `pvc://qwen-local-pv-claim/`
- Namespace: `llm-demo`
- Services: `qwen-bench-01` and `qwen-bench-02`
- Requests: 10 per service, 20 per mode
- Prompt and generation parameters are fixed in `client.py`
- vLLM: `--gpu-memory-utilization=0.30`, `--max-model-len=512`, `--max-num-seqs=1`
- Prefix caching: disabled

vLLM's normal KV cache remains enabled. This benchmark controls its maximum size and concurrency; it does not claim to disable the KV cache.

## GPU memory calculation

For a 12 GiB GPU:

```text
12 GiB = 12,288 MiB
2 x 0.30 x 12,288 MiB = 7,373 MiB
```

Each HAMi service requests `3,686` MiB and `50` GPU cores in the balanced profile. The two services therefore request approximately 7.2 GiB and 100% aggregate compute. Actual model weights and CUDA overhead are still measured by the benchmark.

## Prerequisites

- Kubernetes and `kubectl`.
- KServe with the `LLMInferenceService` CRD and the gateway from the supplied template.
- `llm-demo` namespace and `qwen-local-pv-claim` already created.
- Local registry image `127.0.0.1:5000/kserve/huggingfaceserver:v0.17.0-gpu` available to cluster nodes.
- HAMi installed for HAMi modes.
- MPS enabled and Kubernetes GPU sharing configured for MPS modes. A normal exclusive `nvidia.com/gpu: 1` resource will not schedule two Pods onto one physical GPU by itself.
- Python 3 with `requests` for the client.

## Run

From a Linux shell on a machine with cluster access:

```bash
cd GPU_Share/qwen-kserve-benchmark
python3 -m pip install -r requirements.txt
chmod +x run-qwen-benchmark.sh cleanup-qwen-benchmark.sh
./run-qwen-benchmark.sh
```

Run the optional HAMi + MPS mode too:

```bash
./run-qwen-benchmark.sh --with-hami-mps
```

Run only a HAMi profile:

```bash
./run-qwen-benchmark.sh --mode hami --hami-profile weighted

Use explicit endpoints if the KServe gateway URL is not reachable from your client machine:

```bash
./run-qwen-benchmark.sh \
  --endpoint-1 https://qwen-bench-01.example.test \
  --endpoint-2 https://qwen-bench-02.example.test
```

Clean up the benchmark resources without deleting the shared namespace or PVC:

```bash
./cleanup-qwen-benchmark.sh
```
```

Results are written to `results/<UTC timestamp>/<mode>/`.

## Metrics

The client records success/failure, request latency, p50/p95 latency, output tokens, output tokens per second, request rate, and raw responses. The runner also records readiness time, Pod placement, Pod restarts, GPU state, and HAMi-related annotations.

Startup time is reported separately from steady-state request metrics.
