# Round 2 — runbook

Order matters. Each step is safe to re-run.

## Once, before any run

```bash
./00-preflight.sh > preflight.txt      # cluster shape, disk, GPU, monitoring
./05-pooling-probe.sh > probe.txt      # vLLM capability + DCGM metric inventory
./verify-models.sh --deep              # all 8 models present and loadable
./07-prefetch-traces.sh                # ShareGPT + Azure corpora onto /data2
kubectl apply -f 01-models-storage.yaml
kubectl apply -f monitoring/60-servicemonitors.yaml
kubectl apply -f monitoring/61-dashboard.yaml
```

Confirm the scrape targets actually came up — a ServiceMonitor whose selector
matches nothing collects nothing, and says nothing about it:

```bash
kubectl -n default port-forward svc/prometheus-operated 9090:9090
# Status -> Targets should list r2-hami-scheduler, r2-hami-device-plugin,
# r2-dcgm, r2-vllm
```

## Each test

```bash
python run.py --list                   # the matrix, and why each test exists
python run.py --test T1 --dry-run      # plan only; touches nothing
python run.py --test T1                # all three arms, ~45-60 min
python render_results.py results/T1-*  # summary.csv, arms.csv, report.html
```

`run.py` refuses to start if the container filesystem is within 6 GiB of
kubelet's eviction threshold, if a GPU is already occupied, or if Prometheus is
unreachable. A three-arm run is about an hour; losing it at minute 40 to a
node-wide eviction is worse than not starting. `--skip-preconditions` overrides.

## The scale-down test

```bash
python 09-scaledown.py --arm hami
python 09-scaledown.py --arm hami-mps
python render_results.py results/T9-*
```

## What each arm means

| arm | co-tenants | gpumem | gpucores | MPS |
|---|---|---|---|---|
| `dedicated` | none — one run **per tenant** | same as shared | no | no |
| `hami` | yes | same | yes | no |
| `hami-mps` | yes | same | yes | yes |

The baseline keeps the **shared** memory quota on purpose. Give it the whole
card and its extra throughput becomes a memory-budget effect wearing a sharing
costume.

## Fairness controls, enforced in code

- one trace per test, replayed by every arm; the SHA-256 `trace_id` is recorded
  per sub-run and compared at the end — divergence fails the run
- every tenant pinned to one physical GPU, and the placement **verified** after
  deployment. HAMi's default GPU policy is spread, so two "co-tenants" can land
  on separate cards and produce a clean result that measures nothing
- the GPU is drained to idle between sub-runs, checked with `nvidia-smi` rather
  than slept through
- a test that asks for `source=azure` and silently gets synthetic aborts
- the client talks straight to the pod IP, so the KServe router is outside the
  measurement in every arm

## Known limits of this hardware

**SM occupancy is not measurable here.** `DCGM_FI_PROF_*` is a datacenter-GPU
feature; neither exporter on this RTX 3080 Ti node publishes it, and
`nvmlDeviceGetProcessUtilization` returns `NotFound`. `DCGM_FI_DEV_GPU_UTIL` is
not a substitute — round one found it pinned at 95–99% in nearly every
configuration, so it cannot separate two tenants. What does the work instead:
per-tenant tokens/s, `vllm:num_requests_running`, queue depth, and power/energy
as a proxy for real SM activity. Say this in the report rather than presenting
GPU_UTIL as occupancy.

**Under MPS, per-process attribution collapses.** Every client's work is
submitted by the single `nvidia-cuda-mps-server` process, so `nvidia-smi pmon`
and NVML attribute all of it there. That is a property of MPS, not a gap in the
instrumentation, and it is worth stating in the report.

**The MPS daemon is running without `-multiuser-server`.** Harmless while every
tenant runs as one uid, which is true today and was true for all of round one.
It stops being harmless the moment an arm runs two different uids — round one
measured the newcomer hanging in CUDA init with no error, backing off to 51 s.
