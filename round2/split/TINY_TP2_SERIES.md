# Pythia-70M TP2 + TP2 on Box1

Status: locally implemented and tested; **not yet run on Box1**. This is the
owner's September 24 choice after reviewing the Qwen memory probes. It is a
separate experiment from Qwen TP2+PP2 and the planned TP2+BGE combination.

## Why change the model?

The shortage is host RAM, not GPU VRAM. Box1 has 15988 MiB of host RAM, with
about 7500 MiB available after the OS and cluster services. The Qwen fork TP2
probe used 3416 MiB on each 12288 MiB GPU. Its post-replay pod PSS was 4392.82
MiB: API 667.85, engine 423.43, two workers 1646.72/1644.96, other processes
9.86 MiB. Tensor parallelism divides model computations/weights; it does not
eliminate the host-side API, engine or worker processes.

The light-load Qwen TP2 host delta was 3949 MiB; PP2 was 3618 MiB. The
conservative screen for two Qwen TP2 instances is 3949*2+1024=8922 MiB, above
the latest 7505 MiB idle availability. The TP2+PP2 screen is 8591 MiB.
Neither calculation is an observed concurrent allocation or evidence of an
OOM. Earlier full Qwen TP2+PP2 results required 12630 MiB by the same method.

`lowmem-serve` did not demonstrate a useful RAM saving. The subsequent
`lowmem-fork` TP2 observation reduced host delta by 918 MiB and post-replay PSS
by 910.08 MiB against that candidate. These are different metrics, not two
savings to add. The fork comparison was one sequential light-load observation.

Pythia-70M is a smaller test candidate, **not a guaranteed RAM fix**. Four GPU
worker processes still exist in the shared configuration. Actual solo memory
measurements determine whether the runner may deploy both instances.

## Model and runtime

- Repository: `EleutherAI/pythia-70m`, pinned revision
  `a39f36b100fe8a5377810d56c3f4789b9c53ac42`.
- GPT-NeoX architecture, eight attention heads, six layers, 2048-token context.
  Eight heads divide evenly over TP=2; vLLM 0.15.1 implements this architecture.
- Two independent `LLMInferenceService` objects: `r2-tp-a` and `r2-tp-b`.
  Both use TP=2, PP=1, and run on `mr-gholami`.
- Existing local `huggingfaceserver:v0.17.0-gpu` image; actual image IDs and
  vLLM/Torch versions are captured and must remain identical across arms.
- Model files go into the existing `r2-models-pvc` host directory under
  `/data2`, subdirectory `models/pythia-70m-split-probe`. The script discovers
  the actual PV path, downloads only config/tokenizer/safetensors, and refuses
  an existing directory without matching provenance. No new serving image.
- Base text-generation model: `/v1/completions`, not the chat endpoint.
  This is an infrastructure experiment, not an assessment of answer quality.

Every arm retains `fork`, eager execution, eight active sequences, 1024
scheduled tokens, chunked prefill, the `mp` executor, allocator/thread
settings, 4Gi requested/6Gi limited host memory, and 4096 MiB GPU quota per
device with utilization 0.70. Both GPUs therefore receive two 4096 MiB quotas
in shared arms. Each shared tenant requests 50 GPU cores per card; MPS clients
use 100% active-thread allowance. Actual MPS attachment is checked, not inferred
from YAML. Quotas and host limits do not prove fit or save RAM by themselves.

The 2048 context, model architecture/weights, completion API and synthetic
workload differ from the Qwen experiment. Do not call this merely a model-name
replacement or use old Qwen baselines in Pythia performance ratios.

## One background command

Copy the delivered `split-tiny-series-20260924.zip` into `/opt/GPU_Share`.
This update contains the runner, its model helper, shared probe/replay helpers,
tests, documentation, and reviewable YAML. Run from Box1 as root:

```bash
cd /opt/GPU_Share
python3 -m zipfile -e split-tiny-series-20260924.zip .
cd round2/split
mkdir -p logs results
STAMP=$(date -u +%Y%m%d-%H%M%S)
nohup python3 -u 58-tiny-series.py --rate-per-tenant 2 --seconds 300 --hold-seconds 300 > "logs/tiny-series-${STAMP}.log" 2>&1 < /dev/null &
SERIES_PID=$!
echo "$SERIES_PID" > logs/tiny-series.pid
echo "PID: $SERIES_PID  Log: logs/tiny-series-${STAMP}.log"
tail -f "logs/tiny-series-${STAMP}.log"
```

Ctrl-C exits `tail`; the `nohup` series continues. To request cleanup and stop
the series, first check the stored PID still belongs to this command:

```bash
ps -fp "$(cat logs/tiny-series.pid)"
# Only if that is the current 58-tiny-series.py process:
kill -TERM "$(cat logs/tiny-series.pid)"
```

Do not launch an old `52`/`53` stage alongside it. A shared lock prevents
overlapping updated stages in this checkout. No SSH is performed by the agent.
Preparation requires the existing host Python dependencies (`requests` and
`huggingface_hub`); it does not import torch to load weights on the host.

## Stage order and traffic

| Stage | Resident tenants | MPS | Arrival window |
|---|---|---|---:|
| `dedicated__tp-a` | A alone, TP2 | off | 300 s |
| `dedicated__tp-b` | B alone, TP2 | off | 300 s |
| `hami` | A + B, each TP2 | off | 300 s |
| `hami-mps` | A + B, each TP2 | on | 300 s |

Startup, warm-up, request draining, screenshots and cleanup are additional.
Each stage keeps its pods resident for a 300-second screenshot hold after
replay. In the shared arms initialization is staggered; after both become
Ready they replay **concurrently**. This does not reduce steady-state memory
or waive the admission requirement.

Default traffic: independent seeded Poisson arrivals at 2 requests/s per
tenant, approximately 4/s combined. Actual counts fluctuate. There are two
warm-up requests per tenant, followed by unique short synthetic prompts and
128 generated tokens per measured request (`ignore_eos=true`). The standalone
60-second `57` probe uses its own smaller workload and is optional, not required
before `58`. One frozen full trace is filtered for each solo; both shared arms
replay the identical full file. Seeds, hashes, model revision and settings are
saved. No ShareGPT fallback or old calibration is used.

**The chosen rate is an explicit experimental load, not measured capacity.**
An underloaded result may show little MPS benefit. A capacity knee or saturation
claim would require an additional rate sweep. Changed rates create a new full
series with fresh matching solos; they do not reuse these baselines.

Before either shared arm, the runner requires the sum of both full solo peak
host-memory drops plus 1024 MiB available. It checks again after changing MPS
state. A 0.5-second sampler starts before the first deployment and aborts below
1024 MiB throughout startup, replay and the screenshot hold. This is best-effort
protection, not a kernel-level guarantee against all transient OOMs. Memory
pressure, request errors, restarts, excessive client dispatch delay, incorrect
placement or incomplete MPS attachment stop the series; evidence is retained.

## Dashboard and evidence

The existing round-two PodMonitor extracts `r2_role` from `r2-(.+)`, so it can
label `tp-a` and `tp-b`. The existing dashboard retains its hardware, HAMi and
engine panels. If needed, expose Grafana through your existing access method:

```bash
kubectl -n default port-forward svc/prometheus-grafana 3000:80
```

Each completed replay prints a dashboard URL with an absolute time range and
saves `dashboard-window.json`. The URL's localhost is the machine where the
port-forward is reachable; use your existing Grafana hostname if different.
Select the recorded loaded interval for screenshots, **not** the idle hold.
Screenshot capture is manual. Historical intervals remain available according
to Prometheus retention; the script does not invent screenshots or launch a GUI.

Output: `results/S-tiny-tp2-<timestamp>/`; the latest root is also recorded in
`logs/tiny-series-latest.txt`. Files include manifests, model/runtime identity,
full trace, each replay's request CSV/summary, startup logs, PSS snapshots,
host/GPU time series, admission decisions, MPS evidence, and dashboard windows.
`series.json` records completion/failure. `comparison.json` exists only when
all four stages pass; ratios use each Pythia tenant's own baseline.

Each arm's `intervals.csv` aligns scheduled arrivals, actual sends, successful
and failed completions, p95/p99 end-to-end latency, completions/s and output
tokens/s in the same five-second bins. Latencies are for requests completing
in that bin; tokens are attributed to completion time, not actual token emission
time. Empty latency bins remain empty. Tail drain bins are retained. Plot the
raw arrival series as a line if desired; smoothing must be labelled and must
not be described as evidence proving a Poisson distribution.

## Interpretation and production handoff

Keep Qwen observations and Pythia results separate. This experiment can test
whether two split instances coexist, their response times and MPS's effect
under the declared load. It cannot show that two Qwen instances fit, establish
Qwen speedups, or evaluate language quality. Order is fixed and each arm runs
once, so avoid causal or universal claims beyond the measured comparison.

The same RAM rationale and restoration guidance are maintained in
[RAM_PROFILE_HANDOFF_FA.md](RAM_PROFILE_HANDOFF_FA.md). On a host with more
available RAM, `fork` can be changed to the standard `spawn` method after
compatibility checks. Batch/graph settings must be reconsidered separately;
more RAM does not justify increasing GPU quotas. Pythia's native 2048 context
must remain respected: `SPLIT_PROFILE=original` describes the old Qwen renderer,
not a way to turn Pythia into an 8192-context model. Rebaseline after changes.

HAMi-only YAML omits MPS variables/mounts; the benchmark controller also stops
the MPS daemon. HAMi+MPS adds the pipe mount and variables and starts that daemon
before clients. Environment edits alone do not reliably change actual MPS
state on this node. The runner verifies both workers of each pod among actual
MPS client PIDs, captures both NCCL ranks, and finishes with MPS off after cleanup.
Production rollout should apply these settings to the intended node pool and
verify actual attachment; use measured host requests/limits for the target load.

Sources: [Pythia model/config](https://huggingface.co/EleutherAI/pythia-70m/blob/a39f36b100fe8a5377810d56c3f4789b9c53ac42/config.json),
[vLLM 0.15.1 GPT-NeoX TP implementation](https://github.com/vllm-project/vllm/blob/v0.15.1/vllm/model_executor/models/gpt_neox.py),
[vLLM multiprocessing safeguards](https://github.com/vllm-project/vllm/blob/v0.15.1/vllm/utils/system_utils.py).

Offline check: `python3 -B -m unittest discover -s . -p 'test_*.py'` from this
directory (tests additionally use PyYAML). Cluster behavior and memory fit
remain unverified until the owner runs the series.
