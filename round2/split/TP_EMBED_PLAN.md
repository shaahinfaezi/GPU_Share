# Box1: a partitioned model alongside a single-GPU embedding model

Status: experiment design selected by the owner on 2026-09-23. Not executed;
not yet wired into the current `52`/`53` runner. No performance results exist
for this combination. Run it as a separate series after the current memory
and replay checks, not as an extra tenant added to TP2+PP2.

## Purpose and selected models

Demonstrate that one model can span two physical GPUs while a different model
actively shares one of those same GPUs. Residency alone is insufficient: both
services must successfully process overlapping traffic.

| Tenant | Existing repository configuration | GPU workers |
|---|---|---:|
| Qwen/Qwen2.5-0.5B-Instruct | `50-render-split.py`, role `tp`, tensor parallel size 2 | 2 |
| BAAI/bge-large-en-v1.5 | `../render.py`, role `embed-small`, single GPU pooling/embedding | 1 |

`embed-small` is a repository role name; the official model name contains
`large`. This is not `bge-small` and not a second Qwen instance. The repository
defines its model directory as `pvc://r2-models-pvc/models/embed-small`, whereas
the split Qwen uses `pvc://qwen-local-pv-claim/`. Check both actual mounted
directories and model configs on the node before deployment; repository
configuration does not establish current on-server availability.

Illustrative placement (the physical GPU numbers may be reversed):

```text
GPU A: Qwen TP rank 0 + BGE embedding worker
GPU B: Qwen TP rank 1
```

This is three GPU workers instead of four in TP2+PP2. It is a plausible way to
reduce host-memory demand, not proof that the combination fits a 16 GiB host.
Serving frontends and engine processes also consume memory.

## Controlled comparison

Measure Qwen TP2 alone, then BGE alone, with the same settings each will have
in the shared arms. A previous TP baseline is reusable only after verifying
the exact runtime/profile, resource budget and workload compatibility. Do not
reuse PP's memory or throughput calibration for BGE.

Then run two separately invoked arms:

1. HAMi, MPS off, with both tenants active.
2. HAMi + MPS, same tenants and quotas, verified daemon/client state.

Use the same low-memory profile across comparable baselines and shared arms;
retain Qwen's TP2 and sequence/context settings. Start from the repository's
4096 MiB quota per Qwen GPU and 2816 MiB BGE quota. The shared GPU's declared
memory budget is therefore 6912 MiB. These quotas still require startup and
live-residency validation. Shared core quotas can start at 50 per allocated
GPU for each tenant: 100 on the shared GPU and 50 on the other GPU. Record
this asymmetry; do not describe both cards as symmetrically shared.

Retain BGE's existing pooling, embedding conversion, FP16, 512-token context,
and batch/sequence settings unless a separately documented memory probe
requires a change. Establish its own solo peak host-memory delta. Admit the
pair only with measured host headroom; do not disable monitoring to claim a
successful report demonstration.

## Required runner changes before this can be executed

- Add a separate scenario/series selector and output namespace; the existing
  split runner assumes roles `tp` and `pp` throughout. Do not rename `pp` to
  `embed-small` in YAML and reuse its calibration.
- Render each tenant with its correct model PVC and endpoint type. Qwen uses
  streaming chat; BGE uses embeddings. Smoke-test both protocol paths.
- Replace the current placement rule for this scenario: Qwen must have exactly
  two distinct GPU UUIDs, BGE exactly one, and BGE's UUID must belong to Qwen's
  set. Both pods must be on `mr-gholami`. Also corroborate UUID allocations
  with process/device residency; annotations alone do not prove execution.
- Verify readiness simultaneously and verify actual MPS state for both
  tenants. Existing single-TP MPS smoke does not establish this mixed pair.
- Freeze a scenario-specific Poisson trace for both shared arms, preserving
  per-service payloads, arrival times, embedding batch sizes, source and
  digest. Calibrate the tenant mix from each service's own achieved request
  capacity, not from PP or token rates across unlike services.
- Keep 300 seconds of arrivals per shared arm, with warm-up and draining
  outside that duration. Retain error counts, client scheduling delay and
  host-memory acceptance checks. Confirm a meaningful interval of overlapping
  successful traffic to both services.
- Save model identities, rendered manifests, pod/GPU placement, replay
  interval, and dashboard evidence. Keep the screenshot hold and select the
  actual replay interval when capturing it.

## Report evidence

Use one readable vertical panel per metric with common time-bin boundaries:
scheduled and actual request arrivals, successful completion rates, p95/p99
latency, and throughput. Preserve failed-request counts. Label Qwen throughput
in output token/s and BGE throughput in embedding requests/s or input texts/s,
with batch size stated. Do not sum token/s and vectors/s into a combined score.
Keep latency quantiles service-specific; a chat tail and an embedding tail
are not the same workload. Document whether bins group requests by completion
or arrival and whether token rate is measured or completion-attributed.

Show raw arrival counts and a clearly labelled smooth trend if requested;
smoothing is a visual aid, not a new arrival distribution or a statistical
proof of Poisson arrivals. The deployment diagram, verified shared GPU UUID,
and overlapping successful traffic establish split-plus-share operation.
Comparative performance and MPS benefit require their own measurements.
