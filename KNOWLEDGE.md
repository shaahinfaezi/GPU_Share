# gpu_share — project knowledge and handoff

Last reviewed: 2026-09-18. Repository HEAD during investigation: `c46cace`.

Latest split continuation: 2026-09-23 (the older report/investigation sections
below are historical snapshots, not a statement that today's report is final).

- The owner confirmed Box1 only, both RTX 3080 Ti GPUs on `mr-gholami`, the same
  15988 MiB host, no RAM upgrade, no SSH by the assistant, and one agent only.
- User-supplied September 23 idle evidence: MemAvailable 7609 MiB, no swap,
  both GPUs idle, only the model-image pinner in `llm-demo`. Their September 19
  solo calibration reports TP/PP RSS 6656/6159 MiB and 940.135/165.888 token/s.
  The pasted `solo.log` was September 15. September 19 CSV request success
  counts are not yet verified. Expired events do not establish why old shared
  pods failed to appear.
- Local split corrections add an opt-in `SPLIT_PROFILE=lowmem-eager` with
  `--enforce-eager` in every arm. It is a candidate for reducing memory, not a
  proven fit in 16 GiB. New solo baselines are mandatory. TP2/PP2, model,
  context/sequence limits and GPU quotas stay the same. Solo now records
  idle-to-peak host memory deltas; sharing uses their sum plus 1024 MiB
  headroom. Host deltas include other host activity and are estimates.
- Shared traffic remains Poisson, 300 seconds of arrivals per arm, separately
  invoked. New tenant mix is capacity-weighted rather than equal: the old
  7.7 req/s equal mix over-offered the slower PP tenant. State this experimental
  change in any new report. A series freezes its trace and calibration; role
  remapping updates the digest. Acceptance checks include request errors,
  missing rows and client scheduling-delay p99 <= 0.5 s. This is not an SLO.
- `round2/split/55-split-diagnose.py` reads solo quality, process PSS and node
  allocation without changing cluster state. README documents rerun commands.
  Local validation: seven offline tests, Bash/embedded Python syntax, twelve
  rendered YAML variants. No deployment or benchmark was executed by the agent.
- The owner additionally selected a separate TP2 + `embed-small` experiment:
  Qwen/Qwen2.5-0.5B-Instruct split over two GPUs, alongside the different model
  BAAI/bge-large-en-v1.5 on one of the same GPUs. See
  `round2/split/TP_EMBED_PLAN.md`. This is a planned three-GPU-worker scenario,
  not an executed result or an extra tenant added to TP2+PP2. Current `52`/`53`
  still only implement TP2+PP2; mixed endpoint handling, placement checks and
  BGE calibration must be added before running the new scenario.
- Report follow-up remains pending successful split execution: vertically
  aligned charts with identical bins for request arrivals, p95/p99 latency and
  throughput; preserve raw data and explicitly label any arrival smoothing.
  Never turn a Poisson arrival time series into an invented bell curve. Main
  result plots should show service metrics; keep the full dashboard and the
  requested screenshots. Add sourced technical explanation of HAMi/MPS and
  YAML controls. Do not claim a universal MPS rule from depth above 100 percent;
  depth, duty/oversubscription percentage and throughput ratio are different.

This document preserves the context from the repository/history investigation and the owner's subsequent clarifications. It is a navigation and evidence guide, not a replacement for source code, raw results, or the research reports. Recheck affected code and data before changing them. Paths below are relative to this repository unless explicitly marked as workspace paths.

## 1. Owner's intent and current scope

The owner is Shaahin. This project studies sharing non-MIG consumer NVIDIA GPUs among inference services, primarily using HAMi, NVIDIA MPS, KServe, and vLLM.

The owner clarified these decisions on 2026-09-16:

1. The immediate deliverable is a **Persian research report for a coworker**.
2. The subsequent deliverable is a **production deployment package of YAML manifests and scripts**, supporting **two videos: one educational and one proof of concept**. Do not describe the existing research harness as already production-qualified.
3. The Docker scheduler and custom CUDA shims are **dead historical code**. Do not revive or spend implementation effort on them unless asked.
4. **No more general benchmarks or reruns are planned.** One split benchmark series will be supplied by the owner after project familiarization. Do not turn evidence limitations into demands for more benchmarks.
5. The owner will provide the split series together. Do not assume it has already been provided or executed.
6. **Do not focus on Box2.**
7. The deployment design uses **different node pools**, described in `../reports-findings/FINDINGS-en.docx`.
8. **Everything available is in this workspace and Git.** There are no additional reruns held elsewhere. Search the local record before asking for missing outputs.
9. The owner did not know the answer to the earlier T9 placement/admission concern; it remains an evidence issue, not a decision to rerun.

The initial investigation was strictly read-only. The owner later explicitly authorized creating this knowledge Markdown and revising a Persian research report **without changing semantics or results**. On 2026-09-18, the maintained report was written to `../reports-findings/GPU_SHARE_RESEARCH_REPORT_FA.docx`, with a companion Markdown file, 65 embedded charts, 26 numbered data tables, and reproducible generators at `../reports-findings/build_gpu_share_research_report_v2.py` and `../reports-findings/build_research_report_assets.py`. The legacy `build_gpu_share_research_report.py` delegates to the v2 builder. It does not overwrite `FINDINGS-fa.docx`. The report defines combined throughput as the sum of each service's recorded token rate; it preserves results while labeling direct evidence, technical interpretation, design targets, and unresolved questions separately. If translation work resumes, preserve numbers, conclusions, table relationships, and technical meaning; report factual contradictions separately rather than silently correcting them as translation.

## 2. Evidence conventions

Keep three categories separate:

- **Verified:** inspected directly in source, Git history, or local raw output.
- **Reported:** stated in a research document, but not independently matched to all necessary local raw evidence.
- **Inference:** a plausible explanation or generalization, not a measured causal result.

Do not treat a report's confident prose as stronger evidence than the underlying run. Do not replace research observations with universal guarantees. In particular, conclusions from two tenants on one RTX 3080 Ti are not automatically fleet-wide, cross-architecture, or security guarantees.

“Artifacts” means saved experimental outputs: `run.json`, `events.jsonl`, `trace.json`, request CSVs, summaries, telemetry, logs, Prometheus exports, and rendered manifests. It does not mean the owner needs to produce a separate deliverable or locate another archive.

## 3. Workspace and source-of-truth map

The workspace root is `C:/Users/shaahin/OneDrive/Desktop/CUDA`; the actual Git repository is its `GPU_Share/` child.

Important workspace paths outside Git:

- `reports-findings/FINDINGS-en.docx` and `.pdf`: consolidated English research report; DOCX text and tables read during this investigation.
- `reports-findings/FINDINGS-fa.docx` and `.pdf`: Persian consolidated report; not fully reviewed in the same way as the English DOCX during this pass.
- `reports-fa/`: eight current Persian Tier DOCX reports, all read: `t1-fa`, `t2-fa`, `t5-fa`, `t7-fa`, `t10-fa`, `t11-fa`, `t13-fa`, `t14-fa`.
- `reports-fa/_old/`: archived Tier reports and `fa-report.docx`. Compared/examined as historical material; do not use them as the current numerical authority. Some archive extraction output was truncated, so do not claim every archived paragraph was reviewed.
- `reports-findings/_old/`: archived consolidated reports.
- Numerous top-level `T*-<timestamp>/` folders: actual Round 2 run outputs. Do not search only `GPU_Share/round2/results/`.
- `BENCH LOGS` variants and other benchmark folders: earlier outputs.
- `model spillting yamls/`: owner-provided reference manifests, including `Llama-3.yaml` and TP/PP references. The spelling is intentional as an existing path.
- `Goal.txt`: Persian statement of the internal enterprise GPU-sharing objective.
- Root English/Persian GPU sharing reports and test plans: earlier reporting context.

Repository documents:

- `CLAUDE.md`: local, Git-ignored working guidance. Fully read, but several status/placement statements are stale.
- `FINDINGS.md`: large, Git-ignored technical record; major sections inspected. It contains evolving observations and superseded explanations.
- `round2/TIERS.md` and `round2/RUNBOOK.md`: definitions and operation notes; some status statements lag code/results.
- READMEs read in root, `kubernetes/`, `hami-mps-benchmark/`, `qwen-kserve-benchmark/`, and `round2/split/`.

For an individual claim, use the relevant raw run plus the code/version that generated it. Reports are essential for interpretation and intent, but neither report recency nor a filename alone resolves contradictions.

## 4. Current architecture

The active project is primarily an experimental orchestration and measurement system around Kubernetes inference services. It is not a single application or a finished autonomous GPU-management controller.

```text
Tier definition / scenario / seed
        |
        v
round2/run.py ----> workload/gen_trace.py ----> trace.json
        |
        +--------> harness.py: teardown, RAM guard, MPS state,
        |                       readiness, placement, telemetry
        |
        +--------> render.py ----> KServe LLMInferenceService / Deployment
        |                              |
        |                         HAMi allocation
        |                              |
        |                      vLLM model tenants
        |                      on one physical GPU
        |                      with MPS off or on
        |
        +--------> workload/replay.py ----> direct tenant HTTP endpoints
                                             |
                      request CSVs / summaries / GPU and host telemetry
                                             |
                                collectors and research reports
```

Main comparison arms:

- `dedicated`: each role runs alone, sequentially, generally retaining its HAMi memory grant. This is not necessarily an unrestricted whole-card baseline.
- `hami`: co-located roles with HAMi and MPS off.
- `hami-mps`: equivalent co-location with MPS on; usual experimental setting is 100% active-thread allowance per client.

The important separation is between **scheduler admission**, **device-memory limits**, **vLLM's internal memory budget**, and **MPS execution provisioning**. They are different mechanisms; matching their numeric values does not make them equivalent.

## 5. Main components and control flow

### Round 2 core

- `round2/run.py`: Tier table, CLI, arm order, trace generation, deployment/replay lifecycle, result statuses. Generates one trace for the main comparison and orchestrates solo/shared subruns.
- `round2/harness.py`: Kubernetes helpers; daemon management; GPU placement and MPS checks; host-memory admission checks; endpoint discovery; GPU and host sampling; telemetry queries.
- `round2/render.py`: models, scenarios, quotas, and generated manifests. Default service path is KServe `LLMInferenceService`; pooling also has a plain Deployment path.
- `round2/workload/gen_trace.py`: ShareGPT-based or Azure-based traces; deterministic generation; warmup and request metadata.
- `round2/workload/replay.py`: open-loop HTTP replay, streaming chat and pooling requests, latency/TTFT/token measurements, request CSV and summaries.

Main runner behavior inspected:

1. Generate a trace and reject an unexpected corpus fallback.
2. Remove previous workloads and attempt to drain the GPU.
3. Set MPS state for the arm.
4. Render manifests and check host-memory capacity before applying them.
5. Sample host memory, apply workloads, wait for readiness, verify placement and MPS clients.
6. Discover direct pod HTTP endpoints and served model names; smoke check.
7. Sample GPU state and replay the trace; export metrics.
8. Write results and tear down workloads.

Operational caveats: some helper return values are ignored; teardown selectors can encompass other matching workloads in the namespace; MPS manipulation assumes control of the node's daemon. This harness belongs on a controlled experimental node, not alongside arbitrary production workloads without adaptation.

### Specialized experiments and helpers

- `round2/09-scaledown.py`: T9 admission/eviction experiment.
- `round2/13-validate.py`: environment and metrics validation.
- `round2/15-calibrate.py`: service-load calibration.
- `round2/18-vram-probe.sh`: memory overhead probing.
- `round2/23-failure-inject.py`: MPS daemon loss, tenant deletion, device assert, illegal-address global fault, managed over-grant allocation, unmanaged VRAM hog.
- `round2/28-series-b.sh`, `33-final-series.sh`, `36-three-body.sh`, `37-final-runs.sh`: series orchestration.
- `round2/30-collect*.py`, `32-collect-faults.py`, `34-collect-c.py`: derived measurements; some encode local paths/assumptions.
- `round2/31-freeram.sh`: temporarily parks optional services.
- `round2/35-swapfile.sh`: swap backstop; swap is not counted as usable admission capacity.

### Earlier live research layers

- `qwen-kserve-benchmark/`: extensive shell/Python harness for two Qwen 0.5B services, true sequential comparison, concurrent replay, repeated passes, HAMi profiles, MPS lifecycle and GPU/CPU telemetry. Its daemon manifest is reused by Round 2.
- `hami-mps-benchmark/`: earlier raw PyTorch allocation/matmul, sharing, and OOM experiments.
- `kubernetes/`: January-era installation/injection/manifests. Some incomplete or misleadingly named resources; not a complete current deployment package.

### Dead history

- `src/gpu_scheduler.py`, `src/server/`, `docker/`, `scripts/`, legacy Compose/configs and CUDA shims describe the original Docker-based approach.
- The scheduler launches a fixed count then records snapshots; it is not a continuous autoscaler.
- The shims do not implement reliable live multi-process memory accounting.
- These observations explain history, not work to prioritize now.

## 6. Hardware, services, state, and deployment boundaries

Recorded lab environment, not independently checked against a live cluster:

- `mr-gholami`, Ubuntu 22.04, single-node k3s 1.33.6.
- Two RTX 3080 Ti cards, 12,288 MiB each; roughly 16 GiB host RAM.
- GPU UUIDs in scripts: GPU 0 `GPU-bc62f652-9fa4-7abf-2b0d-81a84f3a27cf`; GPU 1 `GPU-fa422f59-3462-6e01-aece-96aeafeab30c`.
- HAMi 2.9.0; KServe serving image `127.0.0.1:5000/kserve/huggingfaceserver:v0.17.0-gpu`; recorded vLLM 0.15.1.
- Namespace usually `llm-demo`.
- Model cache `/data2/hf-cache`, local PV/PVC `r2-models-pvc`, flattened model directories using hardlinks.
- Prometheus/Grafana/DCGM and `nvidia-smi` provide telemetry; request replay supplies client metrics.
- Registry restrictions and limited root/containerd disk space shaped image-pinning/local-registry scripts.

Models include Qwen 7B AWQ and 1.5B AWQ decoders, bge-m3 and bge-large embedding roles, a reranker and a vision-language role. Identical-pair roles are aliases over the same model, not distinct model weights.

Persistence is filesystem-based: local model volumes and experimental files. No application database was identified. Benchmark traffic uses direct pod endpoints, bypassing the intended production routing/authentication layer.

No complete current authentication, NetworkPolicy, or tenant-security boundary package was found. Privileged daemonsets, host IPC/PID access, host paths, and shared writable MPS sockets are part of the experimental setup. Node pools express intended separation; their end-to-end enforcement is future deployment work.

The English report proposes:

| Pool | Intended role | Reported configuration direction |
|---|---|---|
| Dedicated | Customer/untrusted code, critical endpoints, uncharacterized workloads | One tenant/card, MPS off, spread, gpucores 100 |
| Shared HAMi | Default shared pool, especially without capacity pressure | MPS off, binpack, default core policy, admission shares |
| Shared HAMi + MPS | Trusted first-party co-located workloads that benefit from concurrency | MPS on, normally 100% per client, binpack, admission shares |

Treat that table as deployment intent. Do not claim dedicated GPU allocation alone establishes complete security isolation, or that the exact workload thresholds were qualified across a fleet.

## 7. Memory, placement, and measurement details

HAMi `gpumem` and vLLM `gpu-memory-utilization` are not interchangeable. The renderer clamps utilization using measured memory outside the vLLM budget and a margin; it also checks model/profile/KV feasibility. Recorded outside-budget values include 448 MiB for the 7B role and a conservative 664 MiB for the 1.5B role. Reusing those values on other architectures is an assumption.

Host memory was a real admission constraint: measured per-role footprint is about 2.3–3.1 GiB in this stack. T4/T18/T22 shared configurations were refused. The portable idea is to budget host memory as well as VRAM; “two tenants maximum” is a lab-machine observation, not a universal limit.

Older notes say KServe GPU pinning cannot work. Commit `162c39e` moved annotations to the correct KServe location, and a later T2 run records successful pinning. Do not repeat the old claim unchanged. The alternative pooling renderer has a separate nested-annotation problem noted below.

MPS is controlled by parking/enabling the `qwen-benchmark-mps-daemon` DaemonSet, not a systemd service. Socket existence, server list, and M+C clients are checked. The main scripts include node-wide assumptions; a daemon can cover multiple cards.

Trace/metric interpretation:

- Default ShareGPT mode uses real prompt text but synthetic target lengths, padding/truncation, and synthetic output lengths.
- T14 uses raw corpus prompts and corpus-derived answer-length estimates.
- T6 uses Azure token-length pairs with generated Poisson arrivals.
- T20 uses actual Azure inter-arrival gaps; its configured `--rate` is not the effective arrival rate.
- An embedding service emits no generated tokens. Zero tok/s does not mean no work; token-only capacity comparisons cannot describe its contribution.
- Report “saturation” is shared throughput divided by solo-sum throughput. Lower values indicate more lost throughput. It is not GPU utilization, SM occupancy, or a universally validated saturation threshold.
- Request e2e timing excludes client dispatch delay. Streaming chunk timing is not always true per-token timing.
- `contended_window` is based on overlapping activity envelopes, not a full reconstruction of simultaneous in-flight requests.
- Trace IDs are stored hashes; replay does not independently recompute them. A copied ID alone cannot prove unchanged requests after transformations.

## 8. Research understanding and evidence cautions

The broad research story is that software co-location works on this non-MIG card, and MPS effects depend on workload and load. Separate throughput, latency, fairness, memory behavior, and failure scope. None substitutes for the others.

The interpretation evolved from “MPS favors the larger model” to a relationship with **in-flight request depth**, estimated from each tenant's solo arrival rate and mean latency. T11 matters because the smaller, heavily loaded model gains more. This is an empirical interpretation across the tested cases, not proof of a universal scheduling law.

Always present per-tenant effects with the aggregate. For example, the inspected T2 run shows 7B mean latency 3.827 → 2.121 seconds, while 1.5B rises 1.375 → 1.707 seconds; throughput is essentially unchanged. “26.7% less pooled time” must not become “26.7% more throughput.”

The T5/T14 repeats provide stronger capacity evidence than a single observation, with reported combined gains around 6% and 7.5–8.7% respectively. Different tenants can still move in different directions.

The four-hour T12 arms each have 86,504 successful requests out of 86,670 measured attempts, with 166 oversized-context HTTP 400 failures. In the inspected CSVs these were not transport failures. Preserve that distinction when describing stability; do not call every attempted request successful.

### Verified issues that affect interpretation

1. **T17 and T21 hit client file-descriptor limits.** Rechecked directly in request CSVs on 2026-09-16:
   - T17 HAMi: 338 `Too many open files` errors; MPS arm: zero such errors.
   - T21 HAMi: 3,254 such errors; MPS arm: 2,212.
   - The reported +19.0%/+20.8% output differences are observations, but not clean isolated measurements of GPU capacity. Some failures occurred before requests reached the server. Do not explain all incompletion as GPU overload or a fixed replay-window cutoff.
2. **Runner success is not request success.** `run.py` can record `ok` because the replay subprocess exits successfully even when many requests failed.
3. **T9 does not establish the reported single-card admission block.** The inspected `T9-hami-20260910-225617` records arrival scheduling/running before eviction, with GPU 0 memory increasing while incumbents occupy GPU 1. Readiness delay and scheduler refusal are different events. No rerun is authorized/planned.
4. **VRAM-hog's 512 MiB field is a collector problem.** The parser takes the first matching allocation message and stops; the loop allocates repeatedly. The same run's OOM log reports about 3.50 GiB allocated by PyTorch and 3.74 GiB process usage. These quantities are different; 512 MiB is not the demonstrated peak. Both victim arms served 885/885 requests in the inspected final hog runs.
5. **Global-fault MPS evidence remains only partially located.** The reports state 307/515 neighbor failures under MPS versus 0/516 under HAMi. Local searches found three `global-fault` HAMi `t10.json` records, including a corrected actual illegal-address injection, but no corresponding MPS `t10.json`. The reports contain the MPS figure and client PID changes. Classify it as reported pending locating its raw backing; do not assume another rerun exists elsewhere or declare the observation fabricated.
6. **Pooling manifest regression:** the KServe pinning fix inserts an entire annotations block into the plain-pooling template's existing annotations block, producing nested annotation structure. Reproduced in memory, not changed.
7. **Main runner `--dry-run` writes files.** It creates output directories/traces/manifests/events. Do not use it during a strict no-file-change task.

### Contradictions inside the reports

The current English DOCX combines material from different stages. Its September 11 cover/footer includes September 12 results. Examples needing editorial reconciliation before final publication:

- T20 is listed as not run, then later fully analyzed.
- A section says no production-arrival replay exists, despite the later T20 section.
- Global illegal-address injection is described as both successful and never successfully performed.
- Daemon loss is described as severe failed requests/model reload in one place and harmless silent fallback in another.
- A table labels T2's 26.7% as tok/s rather than pooled time.
- Memory-cap enforcement is stated strongly while the document also acknowledges its over-grant probe did not establish that claim.
- Run/Tier counts differ between sections; inventory snapshots must be dated and scoped.
- T17/T21 prose does not account for the verified client EMFILE failures.
- “All competing explanations excluded” is stronger than GPU power/utilization/clock telemetry alone establishes; SM occupancy was not measured on this hardware.

The Persian Tier documents likewise have stale passages: T13/T15 says no saturated identical-pair experiment exists; T14 calls itself the only/deepest saturated Tier; T10 includes a newer global-fault figure beside older claims that it was never tested. T1's broad wording about unlike tenants should not be generalized to all heterogeneous models. These are semantic/research issues, not translation fixes.

External citations in the documents have not all been independently reverified in this handoff task. Do not repeat their bibliographic details as verified solely because they appear in a report.

## 9. Split series: pending owner input

`round2/split/` was added at the current HEAD. Read its README and scripts when the owner provides the actual series.

- `50` renderer: TP2 and PP2 Qwen 0.5B configurations spanning both GPUs; per-card quota 4096 MiB and 50 cores in shared mode, utilization 0.70. TP and PP use different max sequence counts (32 versus 4), an important interpretation difference.
- `51` preflight: some readiness/smoke outcomes are not enforced.
- `52` calibration: role filtering/remapping affects actual offered rate; fallback values and calibration provenance need attention.
- `53` shared runner: ignores several MPS/readiness/smoke return values; role remapping does not refresh the trace digest; two busy GPUs do not prove every rank's intended placement; exception handling can still exit successfully.
- `60`–`62` are Box2/custom-model/llama.cpp paths. Set aside per owner.

No completed split results/calibration were located during the investigation. This is a statement about that search, not proof that future owner-provided files do not exist.

## 10. Development history

Reachable history inspected: 267 commits across branches, 264 on main, from 2025-11-09 through 2026-09-15. There is no visible earlier full twelve-month record. No commits were observed in February–May 2026; do not infer no work occurred outside Git.

| Phase | Changes and significance |
|---|---|
| Nov 2025 | Docker/NVML mock autoscaler became a real LLM-serving prototype; project restructured; initial driver shim. |
| Dec 2025 | Runtime CUDA shim, image/launch repairs, MPS plus VRAM experiments. |
| Jan 2026 | Kubernetes manifests, MPS/plugin/injection attempts, model-serving deployment. Some incomplete approaches remain. |
| Jun 2026 | Persian documentation work on a side branch. |
| Jul 2026 | HAMi/PyTorch experiments, then two-model Qwen/KServe benchmark; true sequential baseline and equal memory budgeting became important. |
| Aug 2026 | Repeat/telemetry/core-policy refinements; Round 2 trace-driven architecture introduced. |
| Sep 1–4 | Corrected memory budgeting; discovered MPS had contaminated comparison arms; enforced per-arm state; introduced host-memory guards. |
| Sep 5–10 | Compute-policy, fairness, fault, raw-corpus and repeated experiments; GPU pinning corrected; collectors revised. |
| Sep 11–12 | Soak, real Azure arrivals, high-rate symmetric pair, global-fault/hog work, three-tenant refusal with swap. |
| Sep 15 | Split TP/PP series added; pending review against owner-provided results. |

Ten historical milestones worth inspecting:

1. `f154649` (Nov 9): initial mock Docker/NVML autoscaling concept.
2. `f29d9ba` (Nov 11): real GPT-2-era inference restructure.
3. `4fd14cf` / `c7da3eb` (Nov 19 / Dec 7): driver shim followed by runtime shim.
4. `6dccad7` (Jan 7): Kubernetes infrastructure introduced.
5. `34d1fb7` / `5c0e761` (Jul 21–22): HAMi experiments and Qwen/KServe harness.
6. `95c0c56` / `2e382d5` (Jul 29): true sequential comparison and HAMi memory/placement plus MPS compute design.
7. `d093375` / `b1e833f` (Aug 25–26): Round 2 skeleton and centralized harness.
8. `cb06386` / `19b8899` / `2fc04f7` (Sep 2): repair invalid comparisons by controlling the actual MPS DaemonSet and stale state.
9. `f1dc823` / `162c39e` / `b646708` (Sep 3/9/11): host RAM protection, corrected KServe pinning, actual global-fault probe.
10. `c46cace` (Sep 15): split benchmark series.

The central architectural lesson from history is that experimental validity required controlling actual placement, daemon state, model budgets, load generation, and telemetry—not merely assigning names to comparison arms.

## 11. Local inventory, testing, and practical tool notes

Observed inventory: 138 tracked paths; about 151 repository files including local ignored material; roughly 5,122 workspace files excluding Git internals. These are snapshots, not fixed requirements.

The latest local recheck found:

- 50 parseable `run.json` files, all with `events.jsonl`; 48 containing request CSVs somewhere below their run directory.
- 25 `t10.json` files spanning six fault names.
- Run records for T1–T18 excluding the separate T10 format, and T20–T22. T19 and later proposed definitions should not be mistaken for completed experiments.
- These file counts are not the report's total historical attempt count; some attempted runs/log-only folders may lack the same record format.

Testing is mainly hardware/cluster integration and saved experimental validation. No tracked CI pipeline was found. The legacy smoke test expects a response key that differs from the old server implementation. During investigation, 32 Python source files were parsed in memory successfully; this does not validate runtime behavior or Kubernetes schemas. No GPU benchmark, deployment, or mutating test was run.

Tool environment used for the read-only investigation:

- PowerShell on Windows; Python at `C:/Python314/python.exe`.
- Git at `C:/Program Files/Git/cmd/git.exe`, version 2.36.1; not on PATH.
- Git `-C` encountered unsafe ownership. Reading using explicit `--git-dir=GPU_Share/.git --work-tree=GPU_Share` worked without writing global Git configuration.
- `--no-optional-locks status --porcelain` used for nonmutating status checks.
- DOCX text/tables read directly with Python `zipfile` and ElementTree over `word/document.xml`, without extracting or creating files. This reads textual content, not a complete visual/layout inspection of charts and pagination.
- Python `-B` avoids bytecode files; stdout UTF-8 avoids corrupted Persian output.
- Do not run repository imports casually: legacy modules initialize GPU/Docker clients at import, and orchestration scripts can mutate the cluster.

## 12. Where to start next

Ten key locations: `round2/run.py`, `round2/harness.py`, `round2/render.py`, `round2/workload/`, `round2/23-failure-inject.py`, `round2/09-scaledown.py`, `qwen-kserve-benchmark/`, `round2/split/`, `../reports-findings/FINDINGS-en.docx`, and `../reports-fa/` plus the corresponding raw Tier folders.

For the next report task, preserve existing measurements, map each headline to a run, distinguish reported from verified claims, and reconcile stale passages explicitly. For translation-only work, do not change the research conclusion or numbers. For the later deployment package, translate the intended node-pool design into enforceable, reviewable manifests and scripts rather than directly repackaging lab-wide teardown and daemon controls.

Open items are evidence/editorial work, not permission to run more experiments: locating the reported MPS global-fault backing in the existing record; qualifying T17/T21 client limitations; expressing T9's actual observed sequence; reconciling report counts/statuses; and incorporating the owner's forthcoming split series. Box2 and dead Docker/shim code remain outside the current focus.

## 13. Final Persian research report and corrected Split context

On 2026-09-18, the maintained Persian research report was regenerated at `../reports-findings/GPU_SHARE_RESEARCH_REPORT_FA.docx`; its Markdown companion is beside it. The former document was retained as `GPU_SHARE_RESEARCH_REPORT_FA_BEFORE_FINAL_REVISION.docx` before replacement.

Generation sequence:

1. Run `python ../reports-findings/build_research_report_assets.py` to rebuild the 65 evidence-backed figure assets and `data_manifest.json`.
2. Run `python ../reports-findings/build_gpu_share_research_report_v2.py` to build the compact report. It writes the canonical filename by default; set `GPU_SHARE_REPORT_OUTPUT` only when a second output filename is required because Word has the main file open.
3. `build_gpu_share_research_report.py` now delegates to the v2 layout when directly executed. Its old `build()` function remains only as legacy implementation support for the `Report` class.

Final report invariants:

- Every reported Tier is arranged as formal explanatory prose, its own table and available figures, then a constrained result. T10, T12, and T19 use special evidence tables; T19 has no performance artifact. The Split Tier is the final result section.
- The report embeds all 65 generated charts in 25 compact grouped figures, has 26 numbered data tables, and only two explicit page breaks. It does not create fake zero-performance bars for failed shared starts.
- Visible terminology is `HAMI`, `HAMI + MPS`, and `زیرآزمایش`; do not restore `HAMi`, Persian HAMI spellings, `arm`, or `بازو` in reader-facing Persian prose.
- All Persian document runs have B Mitra (`w:cs`) and report text contains no U+0654 detached kasra. Use plain forms such as `حافظه`, not the combining-kasra spelling.
- References are RTL and begin with the Persian number in the right-aligned reference paragraph; each English URL appears in its own right-aligned LTR paragraph. References 9 and 10 are ShareGPT and Microsoft Azure. Main report text avoids raw run directory and artifact filenames; technical documentation contains clickable GitHub links instead.
- Persian Matplotlib labels must use the logical text directly with B Mitra. Do not call `arabic_reshaper` or `get_display` before rendering: that reverses otherwise correct native Matplotlib shaping in this environment.

Corrected Split facts:

- Dedicated TP2 baseline: 3,620 scheduled, 2,183 successful, 1,437 failed, 913.086 token/s, 102.823 s mean E2E, 164.376 s p99 E2E, 6,575 MiB vLLM RSS, and 2,149 MiB host memory available at snapshot.
- Dedicated PP2 baseline: 3,662 scheduled, 1,025 successful, 2,637 failed, 171.698 token/s, 342.653 s mean E2E, 607.339 s p99 E2E, 6,162 MiB vLLM RSS, and 3,002 MiB host memory available at snapshot.
- The calibration sum is 1,084.784 token/s from independent baselines, not simultaneous combined throughput. Both shared attempts stopped before traffic replay with no pod IP for the TP role; neither supplies a HAMI versus HAMI + MPS performance result.
- The solo trace was deliberately overloaded; most failed requests were client-side EMFILE errors. It is a calibration/stress result, not an interactive latency or SLO measurement.

The final report is structurally validated for 65 drawings, 26 data tables, 2 hard page breaks, B Mitra on Persian runs, no detached kasra, no obsolete visible HAMI terminology, no evidence-level section, and no raw artifact filenames in reader-facing content. The validator also confirms direct in-text citations for references 1 through 14, a reference for every figure and table, sequential RTL references, and no English token with a Persian plural suffix. A separate PDF conversion with Word was not retained because a hidden Office helper stalled; do not reuse that helper without supervision.
