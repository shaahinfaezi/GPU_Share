# GPU Share

Tested GPU sharing and model partitioning with Kubernetes, KServe, vLLM, HAMi and NVIDIA MPS. The production handoff uses the recorded Tier configurations and the successful Qwen 7B AWQ TP2+PP2 experiment.

## Production handoff

- [Delivery overview](deliverable/README.md)
- [Operations and short demonstrations](deliverable/OPERATIONS_FA.md): installation, GPU placement, inference, MPS controls and monitoring
- [Configuration explanations](deliverable/CONFIGURATIONS_FA.md)
- [Original tested YAMLs](deployment/configurations/)
- [Three node pools](deployment/pools/): dedicated, HAMi and HAMi+MPS
- [Box1 variants](deployment/box1/): demonstrate the modes sequentially on the existing node

Choose one workload and mode. Set the destination node, image registry and model storage; do not apply every folder recursively. Recorded runtime settings and site-specific differences are documented in the configuration provenance.

## Assemble the package

Python and Bash are required for local assembly and validation:

```bash
python -m pip install -r tools/delivery/requirements.txt
python tools/delivery/explain_configurations.py
python tools/delivery/build_focused.py
python tools/delivery/validate_focused.py
```

These commands do not deploy anything. The three delivery guides are tracked; generated package contents remain ignored and can be built locally or on Box1. Their maintained source guides are under `tools/delivery/`. Persian research documents, private notes and the filming scenario stay local.

## Evidence and history

`round2/` contains the benchmark sources. The split experiment ran TP2 and PP2 together on the same two RTX 3080 Ti GPUs; each sharing mode completed 306/306 requests. Host RAM constraints limited the workload: this demonstrates sharing and partitioning together, not an MPS saturation advantage.

The earlier Docker scheduler and custom shims remain in `src/`, `docker/`, `kubernetes/` and their supporting directories as project history. They are separate from the current production handoff.
