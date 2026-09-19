# KServe custom predictor for DeepSpeed ZeRO-Inference

This package is a focused proof of concept for Box2. It runs a custom KServe
predictor on `new2` and uses DeepSpeed ZeRO stage 3 with parameter offload to
local NVMe. It deliberately does not reuse the retired Docker/shim work in the
rest of this repository.

## What this deploys

The runtime has three physical memory levels:

1. GPU VRAM runs the current parameter group and the forward pass.
2. Host RAM is DeepSpeed's bounded cache for parameters read from NVMe.
3. The local NVMe volume holds the offloaded parameter backing files.

This is **full ZeRO parameter offload**, not llama.cpp's fixed
`--n-gpu-layers` setting. DeepSpeed fetches and releases parameter groups as it
executes the model, and prefetches later groups. The relevant controls are in
`zero-nvme.json`: `stage3_max_live_parameters`,
`stage3_prefetch_bucket_size`, and `offload_param.max_in_cpu`.

The model test is the Hugging Face checkpoint
`Qwen/Qwen2.5-0.5B-Instruct`, placed in a local directory. It is not the GGUF
file from the llama.cpp experiment: DeepSpeed and Transformers need the
original Hugging Face configuration, tokenizer, and safetensors checkpoint.

Qwen2.5-0.5B has 0.49 billion parameters and 24 transformer layers, so it can
fit on an ordinary GPU. It is useful for proving the image, KServe contract,
local storage, and parameter-offload path. It is not evidence of a capacity
gain by itself. The NVMe backing files and DeepSpeed startup log must be
observed before treating the pilot as successful. A later oversized model is
the capacity test.

## Boundaries and current limitations

* This is a single-GPU, one-process runtime. It requests one full
  `nvidia.com/gpu`; do not combine it with HAMi or MPS during this pilot.
  ZeRO's weight streaming needs predictable VRAM, PCIe, CPU RAM, and NVMe
  bandwidth.
* The custom predictor serializes requests. A DeepSpeed ZeRO engine is not a
  safe concurrent `generate()` target in this small single-process design.
* It enables parameter offload only. The upstream ZeRO-Inference example
  provides KV-cache offload adaptations for BLOOM, Llama, and OPT, but not
  Qwen. Keep the Qwen pilot at a modest context and output length.
* The image is intentionally loaded into `new2`'s k3s/containerd image store
  with `imagePullPolicy: Never`. It is not present on `3080-1`; use an
  accessible production registry only after the pilot succeeds.

## Preconditions

Run the following checks before applying any manifest. `new2` must be the
Kubernetes node name, have an NVIDIA device plugin, have enough host RAM, and
have a mounted fast local NVMe filesystem. Do not place the paths below on the
root filesystem merely because the directory exists.

```bash
kubectl get nodes -o wide
kubectl describe node new2
ssh new2 'nvidia-smi; findmnt -T /srv; df -h /srv'
```

If the Kubernetes node name differs, replace every `new2` in `k8s/*.yaml`.
If `new2` is tainted, add the exact matching toleration under
`spec.predictor.tolerations` in `02-inferenceservice.yaml`; do not add a broad
catch-all toleration.

On `new2`, create directories only after confirming that `ZERO_ROOT` resolves
to the intended NVMe mount. This command does not format, mount, or delete any
disk.

```bash
cd box2-zero-inference-kserve
sudo ZERO_ROOT=/srv/gpu-share/zero ./scripts/prepare-new2.sh
```

The supplied values reserve 20 GiB for the model volume and 40 GiB for the
scratch volume. They are advisory Kubernetes capacities; the actual free-space
gate is the filesystem check. Increase them before testing a larger model.

## Build only on `new2`

Copy this package to `new2`, then build and import the image there. The base
image pull, Docker layers, compiled DeepSpeed AIO extension, and resulting
containerd image remain on the worker. Nothing large is stored on `3080-1`.

```bash
ssh new2
cd /path/to/box2-zero-inference-kserve
./scripts/build-on-new2.sh
```

The script expects Docker and k3s. For another containerd installation, set
`CONTAINERD_RUNTIME=ctr` for a standard `ctr -n k8s.io` import. The build compiles
DeepSpeed asynchronous I/O (`DS_BUILD_AIO=1`) against the installed PyTorch;
the script fails if `ds_report` does not report the `async_io` op as installed
or compatible.

The default CUDA wheel is `cu121`. Before building, compare it with `new2`'s
driver reported by `nvidia-smi`. A driver may run an older CUDA runtime only
when NVIDIA's compatibility rules permit it. If necessary, choose a matching
base image and set both `TORCH_CUDA` and `TORCH_VERSION`, for example:

```bash
BASE_IMAGE=nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 \
TORCH_CUDA=cu121 TORCH_VERSION=2.5.1 \
./scripts/build-on-new2.sh
```

## Deploy the storage and model

Apply storage first from the control plane. The static local PVs are pinned to
`new2`; the `WaitForFirstConsumer` storage class prevents a pod from binding
them on another node.

```bash
kubectl apply -f k8s/00-storage.yaml
kubectl apply -f k8s/01-download-qwen.yaml
kubectl -n llm-demo wait --for=condition=complete job/qwen-zero-download --timeout=20m
kubectl -n llm-demo logs job/qwen-zero-download
```

The download job writes a normal local model directory at
`/models/qwen2.5-0.5b-instruct`. Verify it contains `config.json`, tokenizer
files, and safetensors. It does not use an HF cache-tree path, a GGUF, or a
model download sidecar at runtime. The service is configured offline after this
step, so request-time restarts cannot silently consume internet bandwidth or
control-plane disk.

## Deploy and test the InferenceService

```bash
kubectl apply -f k8s/02-inferenceservice.yaml
kubectl -n llm-demo wait --for=condition=Ready inferenceservice/qwen-zero-offload --timeout=25m
kubectl -n llm-demo get inferenceservice,pod,pvc
```

Check the two facts that prove this is a ZeRO NVMe pilot rather than an ordinary
GPU load:

```bash
POD=$(kubectl -n llm-demo get pods -o name | grep qwen-zero-offload | head -n1)
kubectl -n llm-demo logs "$POD" -c kserve-container | grep -Ei 'zero|offload|nvme|async'
kubectl -n llm-demo exec "$POD" -c kserve-container -- \
  sh -lc 'du -sh /opt/zero-nvme; find /opt/zero-nvme -type f | head'
```

For a local protocol test, forward the KServe service or the selected pod and
send a V1 request. The Python model server exposes the standard KServe endpoint
`/v1/models/qwen-zero-offload:predict`.

```bash
kubectl -n llm-demo port-forward "$POD" 8080:8080
```

In another terminal:

```bash
curl -sS http://127.0.0.1:8080/v1/models/qwen-zero-offload:predict \
  -H 'Content-Type: application/json' \
  -d '{"instances":[{"prompt":"What is the capital of France?","max_new_tokens":32,"temperature":0}]}'
```

The response contains a `predictions` array with generated text and a basic
token count. Only after the local test succeeds should the existing KServe
ingress configuration be used. Obtain the URL with:

```bash
kubectl -n llm-demo get inferenceservice qwen-zero-offload -o jsonpath='{.status.url}{"\n"}'
```

## Tuning and promotion rules

The defaults intentionally prioritize evidence and stability over speed:

* `offload_param.device: nvme` forces the disk tier.
* `max_in_cpu: 100000000` bounds the CPU cache to roughly 0.2 GiB of FP16
  parameter elements, excluding buffers and framework overhead. This is small
  enough that the 0.5B model exercises NVMe.
* `pin_memory: false` is the safe initial value. Turning it on can improve DMA
  throughput, but it consumes non-swappable locked RAM and requires a suitable
  `ulimit -l` / container memlock configuration.
* The small Stage-3 live and prefetch thresholds limit temporary VRAM. Raising
  them can improve throughput only after GPU headroom is measured.

For each tuning change, record: model name and revision, DeepSpeed/PyTorch/
CUDA image versions, CPU RAM and NVMe free space, `ds_report`, GPU peak memory,
scratch-volume size, prompt/output lengths, batch size, and p50/p95 latency.
Increase one value at a time. Do not promise low interactive latency: ZeRO
Inference is designed for throughput-oriented batches, and parameter streaming
across PCIe/NVMe adds latency.

Promote this pilot only when all of the following hold:

1. the pod is scheduled on `new2` and consumes one physical GPU;
2. `ds_report` confirms async AIO availability;
3. the local NVMe scratch directory has backing files during model load;
4. requests complete repeatedly without OOM, restart, or model download;
5. the measured latency and throughput meet the workload's stated target.

For production, publish an immutable image digest to a registry reachable from
`new2` (not a registry whose storage is on `3080-1`), set
`imagePullPolicy: IfNotPresent`, use an image-pull secret if needed, restrict
the KServe route with the cluster's authentication policy, and retain the two
local PVs. A model that has no single layer fitting in the assigned GPU still
requires tensor or pipeline parallelism; ZeRO parameter streaming does not make
an indivisible oversized layer executable on one GPU.
