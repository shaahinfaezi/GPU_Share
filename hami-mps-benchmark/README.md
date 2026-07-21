# GPU Sharing Benchmark: Sequential vs MPS vs HAMi

This folder contains an isolated Kubernetes benchmark for comparing GPU sharing modes on a non-MIG NVIDIA GPU.

It does not modify or depend on the existing Kubernetes manifests in `GPU_Share/kubernetes/`.

## What This Compares

1. **Sequential baseline**: one CUDA workload runs at a time on a full GPU.
2. **MPS**: multiple workloads run concurrently through NVIDIA Multi-Process Service.
3. **HAMi**: multiple workloads run concurrently with HAMi `gpumem` and `gpucores` quotas.
4. **HAMi + MPS**: HAMi handles scheduling/quota while MPS is enabled for CUDA process sharing.

HAMi is expected to be stronger for Kubernetes-native scheduling, VRAM quota, and isolation. MPS may help throughput for compatible workloads, but it does not provide the same Kubernetes resource model or per-Pod VRAM isolation by itself.

## Prerequisites

- A Kubernetes node with a working NVIDIA GPU.
- NVIDIA container runtime or GPU Operator configured.
- `kubectl` configured for the target cluster.
- HAMi installed before running the HAMi tests.
- A GPU sharing setup that allows the MPS test Pods to schedule together. Depending on your cluster, this may require NVIDIA device-plugin time-slicing or a similar sharing config. Without that, Kubernetes may keep only one `nvidia.com/gpu: 1` Pod scheduled per physical GPU.

KServe and Knative are not required for these tests. Use raw Jobs first so the benchmark measures GPU behavior without model-server or autoscaling noise.

## Files

| File | Purpose |
| --- | --- |
| `00-namespace.yaml` | Creates `gpu-share-bench`. |
| `01-benchmark-configmap.yaml` | Stores the shared PyTorch benchmark script. |
| `10-sequential-job.yaml` | Runs the baseline workload sequentially. |
| `20-mps-daemon.yaml` | Starts the NVIDIA MPS daemon on GPU nodes. |
| `21-mps-parallel-job.yaml` | Runs concurrent workloads through MPS. |
| `30-hami-parallel-job.yaml` | Runs concurrent workloads with HAMi quotas. |
| `31-hami-oom-job.yaml` | Confirms HAMi rejects over-quota VRAM allocation. |
| `40-hami-plus-mps-job.yaml` | Runs concurrent workloads with HAMi quotas plus MPS env/mounts. |
| `run-benchmark.sh` | Runs tests and stores logs. |
| `cleanup-benchmark.sh` | Deletes benchmark resources only. |

## Quick Start

Run all standard tests:

```bash
cd GPU_Share/hami-mps-benchmark
chmod +x run-benchmark.sh cleanup-benchmark.sh
./run-benchmark.sh
```

Run the optional HAMi + MPS test too:

```bash
./run-benchmark.sh --with-hami-mps
```

Collect only one mode manually:

```bash
kubectl apply -f 00-namespace.yaml
kubectl apply -f 01-benchmark-configmap.yaml
kubectl apply -f 10-sequential-job.yaml
kubectl logs -n gpu-share-bench job/gpu-sequential-benchmark
```

Clean up:

```bash
./cleanup-benchmark.sh
```

## Reading Results

Each Pod prints:

- `nvidia-smi` before and after the run.
- CUDA device name and visible memory.
- Requested allocation size.
- Iteration count.
- Wall-clock duration.
- Iterations per second.
- PyTorch allocated and reserved memory.

Compare:

- total runtime across modes,
- per-Pod runtime,
- whether all parallel Pods can run at once,
- GPU memory visibility inside containers,
- whether HAMi OOM is contained to the over-quota Pod.

## Notes

- The benchmark image is `pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime`.
- Default workload allocation is intentionally modest. Increase `BENCH_ALLOC_MB`, `BENCH_MATRIX_SIZE`, or `BENCH_ITERS` in the YAMLs for larger GPUs.
- The OOM test intentionally exits successfully only when the over-quota allocation fails.
