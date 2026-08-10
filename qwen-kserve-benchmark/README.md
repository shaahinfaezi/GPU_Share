# Two-Container Qwen KServe Benchmark

Compares two identical Qwen services sharing one ~12 GiB non-MIG NVIDIA GPU,
across four modes:

1. **sequential** - one service on the GPU at a time. Baseline.
2. **mps** - both services concurrent, NVIDIA MPS.
3. **hami** - both services concurrent, HAMi `gpumem`/`gpucores` quotas.
4. **hami-mps** - HAMi quotas plus MPS.

Architecture: Knative + KServe `LLMInferenceService` (llm-d router/scheduler)
with vLLM behind the `huggingfaceserver` image. Only files in this directory are
used; `GPU_Share/kubernetes/` and `GPU_Share/hami-mps-benchmark/` are untouched.

## What "sequential" means now

The sequential mode used to apply **both** `LLMInferenceServices` at once and
merely tell the client to send its requests one after the other. Two vLLM
processes stayed resident on the GPU for the entire run, so the "baseline" was
actually an unmanaged co-tenancy run — the exact configuration that crashed.

Sequential now deploys `qwen-bench-01` **alone**, drives it, tears it down,
waits for the Pod to actually terminate and release its CUDA context, and only
then deploys `qwen-bench-02`. One CUDA context on the device at a time. The two
phases are merged into one result document, with wall time summed rather than
maxed, because the phases genuinely ran back to back.

## Keeping the four modes comparable

Every mode gives each service the **same absolute VRAM budget: 3686 MiB**
(0.30 x 12288 MiB). Only the sharing mechanism changes.

This is why `--gpu-memory-utilization` is not the same number everywhere:

| Mode | What vLLM sees | Flag | Absolute VRAM |
| --- | --- | --- | --- |
| sequential | 12288 MiB (whole device) | `0.30` | 3686 MiB |
| mps | 12288 MiB (MPS does not virtualize memory) | `0.30` | 3686 MiB |
| hami / hami-mps | 4096 MiB (`nvidia.com/gpumem` quota) | `0.90` | 3686 MiB |

HAMi's vGPU library intercepts the CUDA/NVML memory queries, so under HAMi the
flag is a fraction of the **quota**, not of the device. Reusing `0.30` there
gives vLLM 0.30 x 4096 = 1229 MiB, which cannot hold the 0.5B weights plus a
CUDA context, and the Pod crash-loops. That was a real failure mode in the
previous manifests.

## Fixed workload

- Model: `Qwen/Qwen2.5-0.5B-Instruct`, from `pvc://qwen-local-pv-claim/`
- Namespace: `llm-demo`; services `qwen-bench-01`, `qwen-bench-02`
- 10 measured requests per service, plus 1 warm-up per service that is excluded
  from every metric (it pays for CUDA graph capture and the first KV-cache
  allocations, which would otherwise land entirely in the first latency sample)
- vLLM: `--max-model-len=512`, `--max-num-seqs=1`, prefix caching disabled

vLLM's normal KV cache stays enabled. This benchmark bounds its maximum size and
concurrency; it does not claim to disable it.

In concurrent modes the client runs **one worker thread per service**, each
issuing its requests back to back. The old client queued all 20 requests into a
2-worker pool, which let one service's backlog drain into the other's worker and
made the per-service numbers meaningless.

## HAMi profiles

`31-qwen-hami-compute-profiles.yaml` holds JSON profiles that the runner
actually reads and patches onto the services. Previously `--hami-profile` only
stamped a label, so `weighted` behaved identically to `balanced`.

| Profile | svc01 cores/mem/util | svc02 cores/mem/util | Absolute VRAM |
| --- | --- | --- | --- |
| `balanced` | 50 / 4096 / 0.90 | 50 / 4096 / 0.90 | 3686 + 3686 MiB |
| `conservative` | 25 / 3072 / 0.80 | 25 / 3072 / 0.80 | 2458 + 2458 MiB |
| `weighted` | 75 / 6144 / 0.90 | 25 / 3072 / 0.80 | 5530 + 2458 MiB |

`weighted` is the QoS case from the project goals: one tenant prioritized on the
same physical GPU.

## Prerequisites

- Kubernetes and `kubectl` with cluster access.
- KServe with the `LLMInferenceService` CRD and the `kserve-ingress-gateway`.
- `llm-demo` namespace and `qwen-local-pv-claim` already created.
- Local registry image `127.0.0.1:5000/kserve/huggingfaceserver:v0.17.0-gpu`
  reachable from the cluster nodes.
- HAMi installed. HAMi's scheduler is also what lets two Pods that each request
  `nvidia.com/gpu: 1` land on one physical GPU; with only the stock NVIDIA
  device plugin and no time-slicing config, the second Pod stays Pending and the
  concurrent modes cannot run.
- Python 3 with `requests` on the client machine.

## Run

```bash
cd GPU_Share/qwen-kserve-benchmark
python3 -m pip install -r requirements.txt
chmod +x run-qwen-benchmark.sh cleanup-qwen-benchmark.sh
./run-qwen-benchmark.sh
```

Include the HAMi + MPS mode in a full run:

```bash
./run-qwen-benchmark.sh --with-hami-mps
```

One mode, one profile:

```bash
./run-qwen-benchmark.sh --mode hami --hami-profile weighted
```

### The nine modes and what each one decides

`--mode` takes a comma-separated list or a preset (`all`, `published`, `new`,
`every`).

| Mode | gpucores | MPS | Question it answers |
|---|---|---|---|
| `sequential` | — (whole card) | off | Baseline: one tenant alone. |
| `mem-only` | none | **off** | Is the loss inherent to two CUDA contexts, or does the HAMi launch gate *cause* the switches? |
| `mps` | none | 50% | Compute split from MPS alone. |
| `mps-100` | none | 100% | Does removing the SM ceiling help, or is MPS already work-conserving here? |
| `hami` | 50/50 | off | Compute split from HAMi alone. |
| `hami-mps` | 50/50 | 50% | Both limiters at once. |
| `hami-mps-100` | 50/50 | 100% | Work-conserving MPS with HAMi as a time-averaged backstop. |
| `asym-hami` | **75/25** | off | Control: does `gpucores` bind *at all* here? |
| `asym-mps` | **75/25** | 100% | **The decisive one.** Does `gpucores` bind *under MPS*, or is it blind because NVML reports at device level? |

`asym-*` force the `compute-skew` profile, which varies cores **only** — same
`gpumem`, same vLLM utilization, therefore the same KV cache on both tenants.
Do not use `weighted` for this: it varies cores, memory and utilization at once,
so any split it produces is unattributable.

Read `gpu_per_process.tenant_sm_share_split` in `summary.json`: near
`[0.75, 0.25]` means the limiter binds; near `[0.5, 0.5]` means it does not.

### Repeats

```bash
./run-qwen-benchmark.sh --mode every --repeats 3
```

Repeats run **in place** against the same Pods. Measured on this node, 91% of a
mode's wall time is Pod churn and only 9% is measurement (4 modes = 26 min, of
which 143 s measured), so redeploying per repeat would triple the run to measure
the same thing.

Each pass writes to `pass-N/`. The **median** pass by aggregate throughput is
promoted to the mode directory and published; `repeats.json` records the spread.
The published number is always a real measured pass, never an average — averaging
throughput, utilization and latency across passes produces a composite that never
occurred and whose internal relationships no longer reconcile.

This measures variance of the *measurement* under a fixed deployment. It does
not capture deploy-to-deploy variance.

### Fairness and measurement fixes

Problems that were present and are now handled:

- **Uncontended tail.** Whole-run throughput counted tokens produced after the
  faster tenant had finished and the slower one had the GPU to itself. Metrics
  are now also computed over the **contended window** where both tenants had
  work in flight, with straddling requests prorated by time. Measured impact:
  0.23% on the symmetric published run, but **73% on an asymmetric one** — so
  this mattered only once the `asym-*` modes existed.
- **Staggered start.** Both tenants now release from a `threading.Barrier`, and
  the measurement origin is stamped inside the barrier action. Previously
  whichever driver thread was scheduled first got uncontended GPU.
- **Serial warm-up.** Warm-up ran service 1 to completion first, so it captured
  CUDA graphs on an idle GPU and, under MPS, paid the server start-up alone.
  Warm-up is now concurrent.
- **Identical prompts.** Every request sent the same string. Prompts are now
  unique per request index, deterministic, and identical across modes and
  tenants; repeat passes are offset so pass 2 never replays pass 1 against a
  warm server. `--prompt-mode fixed` restores the old behaviour.
- **Unclipped sampling.** Samplers start before the client, so mean utilization
  averaged in probing, warm-up and idle time — by a different amount per mode.
  All host series are now clipped to the client's measured window.
- **Unmeasured confounds.** Now recorded: node CPU and per-Pod cgroup v2 CFS
  throttling (each Pod is capped at `cpu: "2"`, and a CPU-starved vLLM decodes
  slower for reasons unrelated to GPU sharing); SM clock, temperature and
  throttle reasons (modes run back to back and the GPU heats up); client CPU
  saturation (if the client burns ~1 core per in-flight thread it is the
  bottleneck, not the GPU).

### SM occupancy vs utilization

`utilization.gpu` marks a sample busy when *any* kernel is resident: it measures
**time**, not **width**. Two modes with equal utilization but different token
rates differ in how much of the SM array did useful work.

True occupancy needs DCGM `DCGM_FI_PROF_SM_ACTIVE` / `SM_OCCUPANCY`, which
NVIDIA does not expose on this GeForce part. Two stand-ins are recorded instead:

- `sm_occupancy_proxy.tokens_per_second_per_utilization_point` — useful work per
  unit of busy time. Explicitly flagged `is_proxy: true`.
- `gpu_per_process` from `nvidia-smi pmon` — per-process SM%, which *does* work
  on consumer cards. Without MPS it attributes SM time per tenant. With MPS it
  attributes everything to `nvidia-cuda-mps-server`, which is not a tooling
  limitation but direct evidence that NVML cannot see per-tenant usage once MPS
  owns the context — the reason the HAMi limiter may be blind under MPS.

Explicit endpoints, when the gateway URL in `.status.url` is not reachable from
the client machine:

```bash
./run-qwen-benchmark.sh \
  --endpoint-1 http://94.184.177.237/llm-demo/qwen-bench-01 \
  --endpoint-2 http://94.184.177.237/llm-demo/qwen-bench-02
```

### If kubectl cannot reach the cluster

If the script works under `sudo` but fails as a normal user with
`dial tcp 127.0.0.1:8080: connection refused`, that is kubectl's fallback
address when no kubeconfig is readable — your user has no `~/.kube/config`
while root does. On k3s:

```bash
mkdir -p ~/.kube
sudo cp /etc/rancher/k3s/k3s.yaml ~/.kube/config
sudo chown "$(id -u):$(id -g)" ~/.kube/config
chmod 600 ~/.kube/config
```

Running under `sudo` also works, but then everything in `results/` is owned by
root. The runner checks API access up front and prints this remedy.

### If CRD detection fails

`kubectl api-resources` exits non-zero and can print nothing at all when any
single aggregated APIService in the cluster is unavailable, even though the
KServe CRD is registered and usable. The runner therefore tries four sources in
order: `api-resources` scoped to the group, `api-resources` unscoped,
`kubectl get crd`, and finally a direct probe. If all four fail it prints what
each one actually returned. You can also bypass detection entirely:

```bash
./run-qwen-benchmark.sh --resource-type llminferenceservices.serving.kserve.io
```

Clean up without deleting the shared namespace or PVC:

```bash
./cleanup-qwen-benchmark.sh
```

Results land in `results/<UTC timestamp>/<mode>/`.

## Endpoint path discovery

The gateway URL is path-based (`http://<gw>/llm-demo/qwen-bench-01`) and which
OpenAI prefix sits behind it depends on the route template and the server image.
The client no longer hardcodes one guess: it probes `/v1/chat/completions`,
`/openai/v1/chat/completions` and `/chat/completions` with a 1-token request and
locks onto the first that answers. The probe results are recorded under
`endpoint_probes` in the client JSON, so a 404 storm is diagnosable instead of
looking like a dead service.

## Metrics

Per mode, in `summary.json`:

- **client**: success/failure counts, per-request latency, p50/p95/p99, output
  tokens, tokens/sec, request rate. Per-service rates use that service's own
  active window, so a service that finishes early is not penalised by the other
  one still running.
- **fairness_ratio**: `min(tokens/sec) / max(tokens/sec)` across the two
  services. `1.0` means the shared GPU treated both tenants identically. This is
  the number that answers the QoS question in the project goals. It is `null`
  for sequential, where the services never overlapped.
- **gpu**: peak/mean VRAM and peak/mean utilization, sampled every 2s during the
  measured window from a privileged `qwen-benchmark-gpu-probe` DaemonSet. The
  probe exists because sampling `nvidia-smi` inside a HAMi Pod reports the vGPU
  quota, not the device — that would make the HAMi modes non-comparable. Raw
  samples are in `gpu-timeline.csv`. Each service's own (virtualized) view is
  still captured in `<service>/gpu-smi.txt`, which is how the quota is verified.
- **acceptance**: readiness, OOM kills, restarts, and whether all requests
  succeeded. The expected request count scales with `--requests-per-service`
  instead of being hardcoded to 20.
- **startup_seconds** per service, reported separately from steady-state metrics.

## Notes on MPS

- The MPS daemon leaves the GPU in `DEFAULT` compute mode. `EXCLUSIVE_PROCESS`
  is the classic way to force every client through MPS, but it is a persistent
  node-global driver setting: if a run is interrupted, every later
  single-process run fails with "all CUDA-capable devices are busy or
  unavailable". Workload Pods opt in explicitly via `CUDA_MPS_PIPE_DIRECTORY`.
  If a previous run left the GPU stuck, reset it with `sudo nvidia-smi -c DEFAULT`.
- The daemon's readiness probe checks for the MPS control socket, and the runner
  aborts an MPS mode if the DaemonSet does not become Ready. Running an "MPS
  mode" without MPS produces a plausible number that is indistinguishable from a
  real result.
- The `mps-pipe` volume uses `hostPath` type `Directory` (not
  `DirectoryOrCreate`) for the same reason: a missing pipe directory fails the
  Pod instead of silently mounting an empty one and running without MPS.
- `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` is read from the **client** process
  environment, so it is set on the workload Pods, not on the daemon.

## Files

| File | Purpose |
| --- | --- |
| `00-namespace.yaml` | `llm-demo` namespace |
| `05-gpu-probe.yaml` | Privileged DaemonSet for host-level GPU sampling |
| `10-qwen-sequential-01.yaml` | Sequential phase 1: `qwen-bench-01` alone |
| `10-qwen-sequential-02.yaml` | Sequential phase 2: `qwen-bench-02` alone |
| `20-mps-daemon.yaml` | NVIDIA MPS control daemon |
| `21-qwen-mps.yaml` | Both services, MPS |
| `30-qwen-hami.yaml` | Both services, HAMi quotas |
| `31-qwen-hami-compute-profiles.yaml` | HAMi capacity profiles (read by the runner) |
| `40-qwen-hami-mps.yaml` | Both services, HAMi + MPS |
| `client.py` | Load client, path discovery, metrics, phase merge |
| `run-qwen-benchmark.sh` | Runner |
| `cleanup-qwen-benchmark.sh` | Deletes benchmark resources only |
