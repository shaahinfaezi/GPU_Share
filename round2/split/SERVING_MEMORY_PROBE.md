# Box1 serving-memory candidate — 2026-09-24

## Latest outcome: fork reduced the measured light-load footprint

PP also passed in `S-serving-probe-20260924-093527/lowmem-fork/`: 13/13 live
requests, client-delay p99 0.009795 s, idle/minimum available 7505/3887 MiB,
peak host delta 3618 MiB and post-replay PSS 4038.05 MiB. The log confirms
TP=1, PP=2, two NCCL ranks, vLLM 0.15.1 / Torch 2.9.1, with no captured spawn
override. Host CSV, process PSS sum and replay quality were independently
reconciled against the local raw artifacts for all four serving probes.

The light-load screening sum is **3949 + 3618 + 1024 = 8591 MiB**, versus
7505 MiB available at the latest idle baseline: **1086 MiB short**. Summing
separate host peaks is conservative and includes unrelated node activity;
it is not an observed simultaneous allocation or proof of an OOM. It still
does not clear the existing admission gate. Two simultaneous TP2 instances
would screen at **8922 MiB** using twice the TP observation plus the reserve.
Sequential TP2 runs have only one tenant resident; they can compare serving
configurations but cannot demonstrate sharing between independent models.
Neither short probe is a full-load calibration or an MPS comparison.

Local evidence: workspace `S-serving-probe-20260924-091358/lowmem-fork/`.
Independently verified host CSV minimum and process PSS sum; replay quality
passed for 13/13 live requests. Idle available RAM was 7499 MiB, minimum
3550 MiB, peak host delta 3949 MiB, post-replay pod PSS 4392.82 MiB. The
requested method was fork and no spawn override was found in the captured log.
Versions remain vLLM 0.15.1 / Torch 2.9.1, TP=2, PP=1.

Against the earlier `lowmem-serve` observation, the host delta fell 918 MiB
(18.86%) and post-replay PSS fell 910.08 MiB (17.16%). The earlier observation
now also has local raw artifacts, reconciled against the same checks.
The shared trace remains `1b0e71199aaada4c`. This is a promising single
light-load result, not proof of saturated fit or MPS benefit. PP's separate
light-load result is recorded above.

The owner explicitly requires the RAM motivation, all profile changes and
the ample-RAM restoration guidance in both the final report and YAML handoff.
See [the Persian report/handoff text](RAM_PROFILE_HANDOFF_FA.md). Full calibration
and shared admission remain pending; do not replace the measured heavy-load
memory gate with this short probe's lower consumption.

## First measured outcome: no meaningful host-memory reduction

The owner supplied the completed output of
`S-serving-probe-20260924-084949`. Both TP passes succeeded with identical
trace ID `1b0e71199aaada4c`:

| Profile | Idle available MiB | Minimum available MiB | Peak host delta MiB | Post-replay pod PSS MiB |
|---|---:|---:|---:|---:|
| lowmem-eager | 7475 | 2603 | 4872 | 5230.15 |
| lowmem-serve | 7425 | 2558 | 4867 | 5302.90 |

The candidate reduced the observed host delta by only 5 MiB (about 0.10%) and
had 72.75 MiB higher post-replay PSS. One sequential pair does not establish
statistical equivalence or a regression, but it provides no useful memory
saving for the multi-GiB admission shortfall. Do not promote the candidate as
a fix, repeat full calibration, or run PP simply to repeat this comparison.
The lower deltas than the earlier overloaded solo run reflect a different
workload; do not attribute that difference to the candidate profile. No shared
admission or TP2+BGE fit has been established. Next inspect the already-saved
process PSS, versions, pod command/imageID and startup logs before selecting a
different serving architecture. The owner subsequently supplied the versions,
per-process PSS and startup logs; their interpretation follows.

## Process attribution and second candidate

Both profiles actually ran vLLM 0.15.1 / Torch 2.9.1, with TP=2, PP=1 and
world_size=2. The original was already a vLLM API server, not an additional
Hugging Face inference server wrapped around vLLM. The removed shell accounted
for only 2.65 MiB PSS. Do not claim removing a KServe inference process saved RAM.

| Process group | Eager PSS MiB | Direct candidate PSS MiB |
|---|---:|---:|
| API server | 787.45 | 794.49 |
| EngineCore | 702.11 | 703.43 |
| Two TP workers | 3729.38 | 3796.56 |

Both logs show eager execution, disabled CUDA graphs, and NCCL initialization
with two ranks. Custom all-reduce was disabled because GPU P2P capability was
unavailable or its test failed; this is not a demonstrated cause of the host
RAM footprint. The configured swap-space warning is not proof of an 8 GiB
resident allocation. Do not revive the invalid swap-space attribution test.

**Second candidate: `lowmem-fork`.** It differs from `lowmem-serve` by exactly
one environment setting: `VLLM_WORKER_MULTIPROC_METHOD=fork`. Upstream's
[CLI setup](https://github.com/vllm-project/vllm/blob/v0.15.1/vllm/entrypoints/utils.py)
sets `spawn` when the variable is unset, including the direct Python API-server
entrypoint. Fork may share some imported host-memory pages through copy-on-write;
it does not remove the engine or either GPU worker, nor guarantee savings.
The deployed method of the earlier runs is inferred from this source, not
directly measured in their PSS files. The comparison must verify startup and
memory rather than infer success merely from an environment value.

vLLM's [process context safeguards](https://github.com/vllm-project/vllm/blob/v0.15.1/vllm/utils/system_utils.py)
remain intact: initialized CUDA/Ray/WSL conditions may override the request
back to spawn. The probe records an observed override as failure to exercise
the requested candidate. No monkey patches, forkserver hacks, or CUDA safety
bypasses are used. If startup hangs or fails, keep diagnostics and stop.
The revised probe also records shared/private smaps fields, and captures 500
startup-log lines to retain override messages. Shared memory figures from
different processes must not be summed as unique physical memory; use PSS.

After installing the revised renderer and probe, run just the new TP candidate:

```bash
cd /opt/GPU_Share/round2/split
python3 -u 56-serving-probe.py --role tp --profiles lowmem-fork
```

This reuses the same short trace as the first probe. A comparison against that
earlier pass is sequential across different times, not a randomized causal
estimate. No new full benchmark, PP probe, or shared run is authorized by a
light-load success. If this candidate yields a useful reduction, new full-load
baselines and NCCL/MPS verification are still necessary before comparison arms.

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
