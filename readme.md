# GPU Share

GPU-sharing and model-partitioning research using Kubernetes, KServe, vLLM, HAMi and NVIDIA MPS. The current path is the measured Tier configurations and the TP2+PP2 split experiment. The earlier Docker scheduler and custom shims are historical approaches.

## Start here

- [Persian research report](docs/research/GPU_SHARE_RESEARCH_REPORT_FA.docx) and [PDF preview](docs/research/GPU_SHARE_RESEARCH_REPORT_FA.pdf).
- [Tested service manifests](deployment/configurations/): original configuration copies, organized by workload and arm.
- [Delivery explanation](tools/delivery/README.md) and [Persian operating guide](tools/delivery/OPERATIONS_FA.md).
- [Per-file configuration explanations](tools/delivery/CONFIGURATIONS_FA.md).
- [Three node-pool YAMLs](deployment/pools/) and [Box1 sequential variants](deployment/box1/).
- [Round 2 execution guide](round2/RUNBOOK.md) and [split benchmark](round2/split/README.md).

## Build the local delivery

Use Python with PyYAML and jsonschema installed; Bash is needed for shell syntax checks.

```bash
python tools/delivery/explain_configurations.py
python tools/delivery/build_focused.py
python tools/delivery/validate_focused.py
```

These commands operate on local files only. The generated `deliverable/` directory remains Git-ignored. Commit the source configurations, delivery helpers and report instead. File hashes preserve the exact relationship to the recorded trial; if a source changes, review and update its provenance deliberately.

The package uses the tested Tier and successful split configurations. `configurations/` preserves the original manifests; pool variants identify the destination node, namespace, registry, storage and allocation policy explicitly. Install prerequisites and set those site values before deployment. Do not apply every scenario or arm at once.

## Verified split experiment

On 2026-09-25, Qwen2.5-7B-Instruct-AWQ TP2 and PP2 ran concurrently on the same two RTX 3080 Ti GPUs. Dedicated TP2 completed 151/151 requests and PP2 155/155; HAMi and HAMi+MPS each completed 306/306. MPS state and GPU placement were verified. The workload was 0.5 synthetic Poisson requests/s per tenant, 64 output tokens, 300 seconds plus drain.

This establishes sharing and model partitioning together at that controlled load. Host RAM constraints prevented a suitable saturation experiment for this split setup. Monitoring was temporarily paused to fit the test; historical dashboard screenshots are interface illustrations, not telemetry from that run.

## Repository layout

| Directory | Role |
|---|---|
| `round2/` | Original Tier scripts, workload replay, monitoring and split-series code |
| `deployment/configurations/` | Exact selected tested manifest snapshots |
| `tools/delivery/` | Local assembly, provenance, explanations and offline validation |
| `docs/research/` | Current Persian DOCX and PDF preview |
| `docs/history/` | Earlier Docker/shim introduction |
| `qwen-kserve-benchmark/`, `hami-mps-benchmark/` | Earlier KServe/HAMi/MPS experiments and original support manifests |
| `src/`, `docker/`, `kubernetes/`, `configs/`, `scripts/` | Preserved Docker/shim implementation and supporting files; historical, separate from the current delivery |

Raw trial directories, video scenarios and private working notes are local-only. Docker/shim sources remain tracked as project history; the current production handoff is under `deployment/` and `tools/delivery/`. Benchmark code and configuration provenance explain the measured settings. Do not commit tokens, local credentials or generated run logs.
