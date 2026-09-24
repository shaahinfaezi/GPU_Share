# Box1 serving-memory candidate — 2026-09-24

The existing TP2+PP2 series is **not admitted**. The owner's September 23 solo
run measured peak host-memory deltas of 5928/5678 MiB. Adding the existing
1024 MiB reserve gives 12630 MiB, versus the latest 7572 MiB available. All ten
PP failures were HTTP read timeouts; this does not establish an OOM. Do not
reuse the old root calibration or reduce admission thresholds to force a run.

## What changes in `SPLIT_PROFILE=lowmem-serve`

The candidate uses the same locally available image and the same
`LLMInferenceService`, model PVC, Qwen identity, CPU limit, 8192-token context,
4096 MiB quota per GPU and 0.70 GPU-memory utilization setting.

- The main container explicitly runs
  `python -m vllm.entrypoints.openai.api_server`, with `/mnt/models` as the
  model directory and port 8000. The controller must still mount the model PVC.
- TP uses tensor parallelism 2 and pipeline parallelism 1; PP explicitly uses
  tensor parallelism 1 and pipeline parallelism 2. The probe records the actual
  pod command/arguments and startup log: do not assume how the old image's
  entrypoint was implemented or that it created extra workers.
- TP's maximum active sequences changes from 32 to 8; PP stays at 4. Both use
  eager execution, a 1024-token scheduling budget with chunked prefill, and the
  `mp` executor on the single node. Long prompts remain supported via chunking.
- The container receives `MALLOC_ARENA_MAX=2`, `OMP_NUM_THREADS=1`,
  `TOKENIZERS_PARALLELISM=false`, and `CUDA_MODULE_LOADING=LAZY`. Some may
  already match runtime defaults; their incremental effect is unmeasured.
- Candidate container host-memory limit is 6 GiB (request remains 4 GiB).
  A limit is a ceiling, not a memory reduction or shared admission guarantee.
  OOM/restarts fail the probe, rather than being counted as a successful fit.

This is an **experimental combined serving profile**, not a proven saving or
a one-variable causal test. Batching/thread changes can change throughput.
It still requires an API process, engine and distributed GPU workers; no claim
is made that it removes those processes. Original and `lowmem-eager` profiles
remain available. The renderer hash/profile in calibration rejects old results.

## Why these alternatives were not used

- `FINDINGS.md` sections 5.14e/f invalidate the apparent multi-GiB saving from
  `--swap-space=0`; that earlier experiment measured the wrong pod lifecycle.
- vLLM 0.15.1's async engine requires a multiprocess engine client. Setting
  `VLLM_ENABLE_V1_MULTIPROCESSING=0` is not an in-process async serving solution.
- No swap file, CPU weight offloading, new image download, infrastructure
  shutdown, lower admission reserve, or changes to the report are involved.

Upstream versioned references:

- [vLLM 0.15.1 engine-client process modes](https://github.com/vllm-project/vllm/blob/v0.15.1/vllm/v1/engine/core_client.py)
- [vLLM 0.15.1 scheduling and parallel CLI options](https://github.com/vllm-project/vllm/blob/v0.15.1/vllm/engine/arg_utils.py)
- [KServe 0.17.0 vLLM wrapper](https://github.com/kserve/kserve/blob/v0.17.0/python/huggingfaceserver/huggingfaceserver/vllm/vllm_model.py)

The local image tag does not guarantee its contents match these sources. The
probe captures installed vLLM/Torch versions and actual pod imageID/arguments.

## Run the bounded probe, not another full calibration

Copy the updated `50-render-split.py` and new `56-serving-probe.py` to this
directory on Box1 before running. Both are needed; changing an environment
variable alone cannot install the profile. Existing `../harness.py`,
`split_checks.py` and `../workload/replay.py` are dependencies. Python's standard
library is sufficient for the probe itself; replay retains its existing deps.

```bash
cd /opt/GPU_Share/round2/split
mkdir -p logs results
python3 -m py_compile 50-render-split.py 56-serving-probe.py
python3 -u 56-serving-probe.py --role tp
```

The default runs `lowmem-eager` then `lowmem-serve`, **one TP tenant at a time**.
Each pass has two warm-up requests and the same seeded Poisson schedule over
60 seconds, 0.25 requests/s, short synthetic prompts and at most 64 output
tokens per live request. Startup and draining are additional. This deliberately
does not reproduce the previous overloaded workload or measure saturation.
`--role pp` selects the equivalent PP probe later; do not run it simultaneously.

The probe:

1. Takes the same `.split-run.lock` as existing stages; refuses existing
   inference objects/pods and busy GPUs, then ensures MPS is off.
2. Requires a stable idle baseline and at least 6656 MiB available before each
   pass. It pins the probe to `mr-gholami` through a node selector.
3. Samples host memory **before creation**, throughout startup and replay.
   A best-effort watchdog aborts below 1536 MiB available. It is not a substitute
   for kernel resource limits or proof against transient unsampled OOMs.
4. Records actual pod spec/imageID, installed versions, per-process PSS/RSS by
   pod cgroup UID, GPU residency, request records, errors, server logs and events.
   PID names alone are not used to assign memory to a pod. PSS snapshots are
   not PSS peaks; the separately sampled host delta includes unrelated activity.
5. Refuses changed/restarted pods or incomplete PSS, missing requests, request
   errors and excess client scheduling delay. A failure stops further passes.
6. Deletes only the labelled object it created, verifies its pods have gone and
   GPUs are idle before the next pass. Ctrl-C/SIGTERM takes the same cleanup
   path; SIGKILL/power loss cannot be cleaned up by a process that has died.

Results go to `results/S-serving-probe-<UTC timestamp>/`. Send both
`*/probe.json`, `*/pss-after-replay.json`, `*/versions.txt` and `*/server.log`.
On failure, send the directory's available logs and the traceback; don't keep
retrying. Every result explicitly says `calibration: false`.

**Passing only establishes a light-load serving probe.** It never writes
`calibration.json`, never launches two tenants, and never clears shared
admission. Review whether a meaningful reduction exists before measuring
full-load peaks. TP2+BGE remains a separate planned experiment; this does not
replace BGE-large with BGE-small or implement its mixed-endpoint runner.

## Local validation

From the workspace root:

```text
python -B -m unittest discover -s GPU_Share/round2/split -p "test_*.py"
```

Tests cover all six candidate role/arm combinations, the original eager
profile, trace identity, cgroup PSS ownership, memory guard, cleanup ownership,
failure evidence, and existing replay/admission checks. No GPU deployment has
been executed by the assistant; only the owner runs server commands.
