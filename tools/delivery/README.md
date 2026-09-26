# GPU Share: tested configurations and pool adaptations

Use the [Persian operating guide](OPERATIONS_FA.md) for installation, placement, model responses and MPS controls. The research report's technical section uses the same guide. The package uses the tested Tier and successful split configurations; original helper files are kept separately under `scripts/`.

| Path | Purpose |
|---|---|
| `pools/dedicated/` | Dedicated GPU allocation via HAMi core admission; model and storage templates |
| `pools/hami/` | Shared GPU allocation; optional TP2+PP2 pair in `split/` |
| `pools/hami-mps/` | Shared GPU allocation with MPS; optional TP2+PP2 pair in `split/` |
| `box1/` | Corresponding sequential demonstrations, pinned to `mr-gholami` and the existing model PVC |
| `infrastructure/hami-values.yaml` | HAMi 2.9.0 fresh-install values with binpack and managed node/namespace selectors |
| `infrastructure/mps-pool.yaml` | Existing supervisor identity restricted to the enabled MPS pool |
| `configurations/` | Thirty unchanged service snapshots from T1, T2, T3, T17 and the successful split run |
| `infrastructure/models-storage.yaml`, `mps-daemon.yaml` | Original infrastructure snapshots; use the pool-specific variants for the new walkthrough |
| `monitoring/` | Scrape definitions and dashboard for an existing monitoring installation |
| `scripts/` | Original optional helpers, retained separately from the deployment walkthrough |
| `validation/` | Pinned KServe schema and local validation receipt |

Each pool contains `00-storage.yaml`, `10-llm.yaml` and `11-embed.yaml`. Replace the storage hostname placeholder, local model path and image registry before destination use. A sample local volume is tied to one host: a multi-node pool needs shared storage or separate per-host deployments/volumes. Box1 variants use its existing `r2-models-pvc`; do not recreate that volume.

`provenance.json` identifies the unchanged copies. `pool-provenance.json` records the 25 derived files and their original source hashes. Runtime arguments remain unchanged; deliberate differences are service names, pool selectors, the existing `nvidia` runtime class, site storage and `gpucores=100` for the dedicated production example. The monitoring adaptation recognizes pool service names without creating a duplicate scrape target. The historical dedicated baseline only removed the neighbor and is not an exclusive allocation policy.

Do not apply this package recursively or install two MPS supervisors. Drain the selected GPU clients before switching modes. The original helper's MPS toggle clears the supervisor node selector and must not be used unchanged across production pools. Use the documented pool labels instead. Pool labels govern placement, not security isolation from privileged or unmanaged workloads.

The local `deliverable/` directory is Git-ignored and must be transferred to Box1 separately. Public sources live in `deployment/` and `tools/delivery/`. From the repository root:

```bash
python tools/delivery/explain_configurations.py
python tools/delivery/build_focused.py
python tools/delivery/validate_focused.py
```

These commands contact no cluster. Checks cover source hashes, preserved runtime settings, YAML/KServe schema, shell syntax and guide paths. Execution evidence comes from the original Tier and successful split runs; `configurations/` preserves those exact manifests, while `pools/` supplies explicit site bindings and the documented dedicated allocation policy.

The verified split workload was 0.5 Poisson requests/s per role for 300 seconds plus drain; both sharing arms completed 306/306 requests. It demonstrates sharing and partitioning together, not MPS saturation or maximum capacity. Monitoring was paused for host RAM/scheduler reservations. With monitoring restored, remeasure concurrent startup/load plus reserve before attempting the pair; do not infer that the previous memory admission still holds.

See [the per-file catalogue](CONFIGURATIONS_FA.md) for model settings and resources. Installation references: [HAMi 2.9.0 Helm deployment](https://project-hami.io/docs/v2.9.0/get-started/deploy-with-helm), [HAMi scheduling](https://project-hami.io/docs/userguide/nvidia-device/scheduling-policy), and [NVIDIA MPS controls](https://docs.nvidia.com/deploy/mps/appendix-tools-and-interface-reference.html).
