# Round two: the tier matrix, in order

Status of every tier, what each one bought, why the skipped ones were skipped,
and what to run next ranked by how much it can still change a conclusion.

Duty figures are **client-side** — the fraction of the measured window with at
least one request in flight, summed over each tenant's solo run. The nvidia-smi
numbers that earlier drafts used are not occupancy; a card reads 100% with a
single SM busy. FINDINGS 7.2.

---

## Done

### T1 — `llm-embed` @ 6 req/s · 140.4% duty
**Decoder beside a pooling model.** MPS effect **+1.31% on pooled time in
system — a null, and slightly negative.** Two run dirs; the first
(`T1-20260826-175322`) is void, it predates the MPS daemon fix.

Its value is negative evidence, and it turned out to be the most informative
null in the matrix: T1 sits at 140.4% duty and returns nothing while T2@4 sits
at 155.7% and returns −18.8%. Those are close enough that **duty cannot be the
discriminator — composition is.** T2 is two decoders doing sustained
memory-bound work; T1 is a decoder beside a pooling model doing short bursts,
and the pooling tenant pays a per-launch cost it cannot amortise.

### T2 — `llm-llm` @ 6 req/s · 174.0% duty
**The headline latency result: −26.7% pooled time in system.** Swept at 4 and 5
req/s to fill the band — 155.7% → −18.8%, 166.8% → −23.3%, 174.0% → −26.7%.
Monotone in contention.

Also the fairness result: plain HAMi degrades both tenants almost equally
(2.56× and 2.46× solo), and **MPS pulls them apart to 1.42× and 3.06×** — the
small tenant ends up worse *in absolute terms* than it was without MPS. Time
slicing is the fair mechanism; MPS is the fast one.

A free replicate exists from the `vram-escape` fault runs (trace
`1a8278ce7fdd85f2`): −26.65% against T2's −26.67%.

### T3 — `vl-rerank` @ 4 req/s · 93.4% duty
**True null: +0.42%.** The card is not oversubscribed, so there is nothing for
MPS to recover. This is the tier that killed the round-one power hypothesis.

### T4 — `dense4` @ 8 req/s — **not runnable on this node**
Four tenants need ~12.6 GiB of host RAM against ~7.3 available. The two shared
arms were refused by the host-memory guard; the four dedicated arms ran and
**delivered the tier's real value: the per-instance host RSS coefficient.**

2,358 / 2,570 / 2,936 / 3,045 / 3,049 MiB for embed-small / llm 7B / embed /
llm2 1.5B / rerank. **The 7B costs less than the 1.5B**, and a pooling model
with no KV cache costs the same as a generative one — a 1.29× spread across a
4.6× parameter range. Host RAM is the Python/torch/CUDA runtime, not the model.

Report the coefficient, not the tier. This node is a two-tenant box.

### T5 — `llm-llm` @ 10 req/s, 240 s · 193.7% duty
**The capacity result, and the strongest number in the study.**
**+5.94% combined throughput, −77.95% pooled time in system, TTFT p99
43.5 s → 3.5 s.** The tier that vindicates round one: throughput gains require
saturation.

### T5 `--core-policy disable` — the control that removed the confound
Re-ran T5 on the **same trace** (`198fa8fe46ce4904`) with the one policy that
provably short-circuits `rate_limiter()`.

Disarming the limiter moved throughput by −0.20% and −0.12%. **None of T5's
33.8 s mean latency was HAMi throttling; all of it was contention.** MPS effect
reproduces at +6.03% / −77.76%.

Two by-products worth more than the run cost:
- **Reproducibility: 0.04–0.36% on throughput**, same trace, two days apart.
  This retires the earlier worry about 34% run-to-run variance.
- **The limiter never bound in any co-tenancy tier.** `render.py` sets
  `share = 100 // len(roles)`, so two tenants each get `gpucores: 50` — which
  is what they would take anyway. Every co-tenancy number in this study is
  contention, not throttling.

### T7 — `llm-only` @ 6 req/s — **the only tier where the limiter binds**
Four runs, complete. One tenant, no neighbour, so any delta is the limiter
alone. All four replayed the identical trace `f9bb269f531e383e`, 1,757
requests, seed 20260826, 660,530 prompt tokens submitted on every arm.

| gpucores | policy | dedicated | **hami** | hami-mps |
|---|---|---|---|---|
| unset (=100) | unset | 836.19 | **835.92** | 835.00 |
| llm=50 | unset | 835.39 | **835.72** | 833.59 |
| llm=50 | **disable** | 835.54 | **835.90** | 834.41 |
| llm=50 | **force** | 835.55 | **548.42** | 834.51 |

**The three-way is settled from behaviour: `disable` ≈ `default` ≈ no limit at
all, and `force` is the lone outlier.** That is the documented semantic —
`force` limits unconditionally; **`default` and `disable` never limit.** T16
closed the last cell: 25/25 under contention with the policy left at default
returns 429.27 tok/s, indistinguishable from unclamped. Behaviour settles it,
and a latch defect in `rate_limiter()` explains it — the early return on
`cached_util_switch == 0` sits in front of the only writer of the flag the
monitor reads, so once the switch is observed off it stays off for the life of
the pod.27 tok/s**,
indistinguishable from the 429.49 unclamped. 5.14b's prediction from HAMi's docs
was half right — the two policies differ, but the documented *reason* does not
survive measurement.

**`force` throttles to 65.6% of solo throughput, 21.6× mean latency, 45.8×
TTFT p99. Adding MPS to that same forced pod returns it to 834.51 — +52.2%,
99.9% of dedicated.** MPS does not weaken the limiter, it erases it, and every
request still succeeds so nothing surfaces the fact.

This tier also corrected two published claims: "HAMi's limiter is free" (that
run was `gpucores: 100` — nothing to limit) and "`default` and `force` are
identical in code" (they are plainly not identical in behaviour). FINDINGS 8.2.

### T10 — failure injection · **5 faults, 3 of them with data**
Not a tier in `run.py`; it lives in `23-failure-inject.py`.

- **`mps-kill` — the serious one.** RECOMPUTED 2026-09-06 after a defect in
  the window code (it put refused requests into the latency percentiles, so an
  outage reported as a 54.5% p99 *improvement*). Corrected: the 60 s outage
  window failed **96 of 173 (55.5%)**, and the following 120 s failed **253 of
  343 (73.8%)** — a HIGHER rate, so **no recovery was observed inside the
  instrumented period at all.** 338 of 349 failures are connection-refused.
  The earlier "~140 s to recovery" described the end of the run, not a measured
  return to service. Both tenants died together.
- `tenant-kill` — benign, survivor served through re-init.
- `vram-escape` — **contained on both arms** by HAMi-core's own OOM path, the
  log line matching `allocator.c:52` exactly. HAMi's memory isolation survives
  MPS because it hooks `cuMemAlloc_v2` and never consults NVML.
- `vram-hog` — **run twice, VOID both times.** The hog pod was created and
  never allocated a byte: device 0 peak 1.0 MiB, device 1 peak 11,549 MiB,
  identical to the normal two-tenant level. Zero victim failures proves nothing
  when nothing was held. Pinning to the tenants' card was fixed and did not
  help; the injector now captures the hog's own log. FINDINGS 8.14.
- `tenant-fault` — **run 4 times; the alarming result did NOT replicate, and
  the claim built on it is withdrawn.** A genuine device-side assert
  (`cudaErrorAssert`, dead context) was **contained on every arm that has a
  clean measurement**: the `hami` replicate and both `hami-mps` runs show 0
  victim failures and flat VRAM. One earlier `hami` run lost 384 of 885 victim
  requests with an 8 GiB VRAM collapse; it did not reproduce, and its VRAM
  signature is a pod restart rather than fault propagation. The probe spawns a
  third torch process on a node with no swap, so a host OOM kill explains every
  observation — but the injector never sampled host memory, so it cannot be
  confirmed. **A sticky CUDA error appears to kill only the process that caused
  it.** FINDINGS 8.16.

### T11 — noisy neighbour @ 12 req/s, 90/10 · **done 2026-09-06**
The aggressor takes 10.82 req/s against its own knee of ~6; the victim gets
1.22 req/s, well inside capacity.

| | solo | HAMi | HAMi+MPS |
|---|---|---|---|
| `llm` 7B, well-behaved | 1.235 s | 3.038 s (2.46×) | **2.181 s (1.77×)** |
| `llm2` 1.5B, aggressor | 0.783 s | 3.179 s (4.06×) | **2.045 s (2.61×)** |

Combined **+0.86% tokens/s, −34.96% pooled time**. Three findings:
**noisy-neighbour damage is real** (the victim degrades 2.46× while nowhere
near its own capacity); **MPS protects the victim** (−28.2%), the first
positive isolation result MPS earns in this study; and **it protects the
aggressor more** (−35.7%), which **retires the round-two claim that MPS
reallocates toward the larger tenant.**

It reallocates toward **in-flight depth**. T5's 7B ran 11.11 concurrent against
2.84 and won; T11's hammered 1.5B ran 8.47 against 1.51 and won. Same law,
opposite sign on size. FINDINGS 8.5.

### T13 — `llm2-llm2` @ 10 req/s · symmetric pair · **done 2026-09-06**
Two identical `Qwen2.5-1.5B-AWQ` instances. Solo in-flight depth 3.04 against
3.14 — the closest matched pair available. Degradation spread **1.019 under
HAMi and 1.031 under MPS**, against 2.16× in T2 and 2.84× in T5 between unlike
tenants. **MPS creates no winner when depth is equal**, which is what retires
"MPS favours the larger tenant" and establishes that the asymmetry was never
about model size.

Caveat that limits it: the HAMi arm reached 99.9% of the solo throughput sum,
so **T13 is unsaturated**. It settles fairness and says nothing about capacity.
FINDINGS 8.6.

### T15 — `llm2-llm2` @ 20 req/s · **done, and it failed at its job**
T13 at double the rate, intended to make the symmetric pair saturate. It did
not: two 1.5B tenants at 20 req/s still serve essentially all of the offered
load. Reported anyway, because a tier that fails to reach its own operating
point is a result about the harness that the next attempt needs.
See "5. A genuinely saturated symmetric pair" below.

### T14 — `llm-llm` @ 10 req/s, raw prompts · **done ×2 · the study's best tier**
The tier that retires the corpus objection. It replays ShareGPT prompts
**verbatim** — no filler padding, no truncation, real answer lengths — where
every other tier draws prompt and output lengths from a lognormal and pads.

Two replicates agreeing to **0.15%**: MPS gives **+8.72% / +8.52% throughput**
against T5's +5.9%, pooled time −70%, fairness spread collapsing 5.65× → 1.67×.
At **86.3% saturation it is the most throughput-saturated tier in the study**,
so **capacity claims should rest on T14, not T5.** FINDINGS 8.13.

T14 also carries the overbook sweep — the same trace at Σ MPS 200 / 150 / 100%
— which is where the "matched configuration is the worst one" result comes
from. See the corrections section.

### T16 — `llm-llm` @ 6 req/s, `gpucores 25/25`, policy `default` · **done**
The last empty cell in the policy table, and the operationally important one,
because most clusters run the default. Quotas set well below the natural share
**with a neighbour present**, so the limiter has something to bind on:
`llm` returns **429.27 tok/s** against 429.49 unclamped and 298.37 under
`force`. **`default` does not limit even under contention, even below the
natural share.** FINDINGS 8.11.

### T8 — QoS ceiling on T11's mix · **redefined 2026-09-06, run, reported**
The original T8 was removed (see "Skipped, and why"). It was redefined as the
question its slot was worth: does an MPS thread-percentage ceiling protect a
victim from a noisy neighbour? T11's 90/10 mix, MPS at 50 instead of 100.

**A clean null with a known cause.** Combined throughput 1,581.81 against
1,580.51 (+0.08%), victim mean latency −3.7%, aggressor +1.3%. The cap was
genuinely applied — the manifest carries 50, peak VRAM is 146 MiB below the
100% run exactly as NVIDIA documents, and the card sits 11.8 points lower
against its power limit. **It did not bind because neither tenant was using
more than half the SMs to begin with**: the tier runs at 99.2% of the solo
throughput sum, so nothing was being withheld from anyone. A ceiling only
closes when set below what the tenant would otherwise take.

### T18 — `dense3` @ 9 req/s · **defined, not yet run**
Three tenants. Every co-tenancy result in this study is two-body, and this is
the largest structural gap in the matrix. It may be refused by the host-RAM
guard before it deploys — **if so, that refusal is the result.** See item 11
under "What to run next".

---

## Corrections that are settled (tracked here so they travel with the repo)

**The overbook curve is measured, and the "matched" config is the worst one.**
T14, the only saturated tier, varying only the MPS percentage on one trace:
Σ 200% → **+8.6%**, Σ 150% → **+3.7%**, Σ 100% → **−6.9%**. Setting `gpucores`
equal to `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` — the intuitive way to make the
scheduler's model honest — delivers **less throughput than not running MPS at
all**, and the fairness spread explodes 1.67 → 4.26 → **15.20** as the cap
tightens. A cap does not protect the small tenant; it strands the large one,
which is the only tenant that could have used the SMs the small one leaves
idle. **On a saturated card the choice is binary: MPS at 100, or no MPS.** The
`hami` control replicates to 0.24% across all four runs. FINDINGS 8.17.

**`GPU_CORE_UTILIZATION_POLICY`: `default` and `force` are NOT the same.** An
earlier source review claimed they were identical in code and that our
835.72-vs-548.42 measurement was unexplained. Withdrawn. T7 ran all three
policies on one identical trace, one tenant, `gpucores: 50`:
`default` 835.72 / `disable` 835.90 / **`force` 548.42**. `force` limits
unconditionally; **`default` and `disable` never limit.** T16 closed the last
cell: `gpucores 25/25` under contention with the policy left at default returns
**429.27 tok/s**, indistinguishable from the 429.49 unclamped. Behaviour settles
it, and a latch defect explains it — `rate_limiter()`'s early return on
`cached_util_switch == 0` sits in front of the only writer of the flag the
vGPUmonitor reads, so once the switch is observed off it stays off for the life
of the pod. `force` bypasses the shared region and is immune.

**The compute limiter never bound in any co-tenancy tier.** `render.py` uses
`share = 100 // len(roles)`, so two tenants get 50 each -- which is what they
would take anyway. T5 re-run with `--core-policy disable` moved throughput by
-0.20%. Every co-tenancy number in this study is contention, not throttling.

**The compute limiter DOES work under co-tenancy, and MPS erases it exactly.**
`T2-20260906-102419` ran T2 at `gpucores 25/25` with `force` -- quotas below
the natural share, so the limiter genuinely binds with a neighbour present.
The HAMi arm lost 30.5% of the 7B's throughput and took a 93.7x TTFT p99; the
1.5B was untouched (+0.1%), because equal quotas cost unequal amounts. The MPS
arm on the same clamped config came in at **-0.04% throughput and +0.38%
pooled time against the UNCLAMPED baseline** -- the limiter is not weakened, it
is absent. Within the clamped run MPS is worth +18.9% throughput and -95.2%
pooled time. **There is no configuration on this stack giving both a per-tenant
compute guarantee and efficient use of the card.**

**CORRECTED 2026-09-06. `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` IS enforced on
this GeForce.** An earlier version of this section claimed neither compute
control constrains anything under MPS, and left open whether the MPS knob is
ignored on consumer silicon. **The study's own round-one archive refutes it.**
`BenchLOGS_B1b/asym-mpsonly`, same node, `CUDA_DEVICE_SM_LIMIT=0` on both pods
so no HAMi core limit is involved, MPS set to 75 and 25:

| MPS % | svc-01 | svc-02 | ratio |
|---|---|---|---|
| 75 / 25 | 1117.4 / 1112.1 / 1083.6 | **624.3 / 622.2 / 614.1** | **1.79:1** |
| 100 / 100 | 916.9 / 914.3 / 913.0 | 915.2 / 912.1 / 909.0 | 1.00:1 |

Three passes each. The knob binds, and it binds hard when set asymmetrically.

**What T8 actually showed** is narrower and still useful: a SYMMETRIC 50/50 cap
did not bind, on a card that was only 99.2% saturated, running a
bandwidth-bound decode. Restricting a tenant to half the SMs does not halve its
tokens/s when the bottleneck is HBM and nobody is competing for SMs anyway.

**So the honest statement is:** `nvidia.com/gpucores` is void under MPS (T7,
T2@25/25 -- mechanism confirmed in HAMi-core source). The MPS thread percentage
is NOT void; it is a real ceiling that does nothing until it is set below what
the tenant would actually use, and then it strands capacity the same way HAMi's
`force` does. `T7@mps=50` cannot test this -- 50 symmetric is precisely the
non-binding case. Test asymmetrically, or at 25.

**The MPS thread percentage is a REAL, monotone compute control.** Single
tenant, whole card, identical trace, only `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE`
varying: **100% -> 835.00, 50% -> 560.98, 25% -> 297.91 tok/s.** Enforced and monotone.
It is **not** evidence of sub-linear SM scaling: the 100% arm was never at
capacity (304.7 s wall against a 300 s trace, peak in-flight 61, 4.06 s mean
latency), so 835.00 is the offered rate and 67.1%/35.7% are upper bounds on
retention rather than measurements of it. The one interval with both ends
saturated is 50 -> 25: **0.531 for a halving, elasticity 0.913 — essentially
proportional.** The 163 lost requests at 25% are client-side connection errors
at ~1,000 concurrent threads, a load-generator ceiling, not a tenant breakdown. And on the same trace HAMi's `gpucores: 50`
+ `force` costs 548.42 (65.6%) — **the two controls both labelled "50%" cost
within 1.5 points of each other.** What differs is that HAMi's is void the
moment MPS is on and MPS's is not. FINDINGS 8.12.

**The corpus objection is dead, and the effect is BIGGER with real text.** T14
replays ShareGPT prompts verbatim — no filler padding, no truncation, real
answer lengths. Two replicates agreeing to **0.15%**: MPS gives **+8.72% /
+8.52% throughput** against T5's +5.9%, pooled time −70%, and the fairness
spread collapses **5.65x -> 1.67x**. At 86.3% saturation it is also the most
throughput-saturated tier in the study, so **capacity claims should rest on
T14, not T5.** FINDINGS 8.13.

**`default` does not limit, even below the natural share under contention.**
T16 at `gpucores 25/25`, policy default: llm returns **429.27 tok/s** against
429.49 unclamped and 298.37 under `force`. The last empty cell in the policy
table, and it is the operationally important one — most clusters run the
default. FINDINGS 8.11.

**MPS creates no winner when in-flight depth is equal (T13).** Two identical
1.5B instances, solo depth 3.04 against 3.14: degradation spread is 1.019 under
HAMi and 1.031 under MPS, against 2.16x (T2) and 2.84x (T5) between unlike
tenants. The asymmetry was never about the models. Caveat: T13 is unsaturated
-- the HAMi arm hit 99.9% of the solo throughput sum -- so it settles fairness
and says nothing about capacity.

**MPS reallocates toward IN-FLIGHT DEPTH, not model size.** The round-two claim
that MPS favours the larger tenant is retired. T5's 7B ran 11.11 concurrent
against the 1.5B's 2.84 and won; T11's hammered 1.5B ran 8.47 against the 7B's
1.51 and won. Measure `arrival rate x mean latency` per tenant, not parameters.

**The ~3 GiB per-tenant host RAM is NOT recoverable by configuration.** The
`--swap-space` A/B test was invalid (KServe never sets `r2-role`, so the
teardown selector matched nothing) and was never redone. It does not need to
be: `--swap-space` defaults to 4 GiB, and a stock tenant's total resident
footprint is 3,276 MiB. A 4 GiB buffer cannot fit inside a 3.3 GiB RSS, so it
was never allocated. Upstream agrees -- `num_cpu_blocks` is hard-coded to zero
in vLLM V1 (vllm#27984), which preempts by RECOMPUTE, never SWAP.

**What `source: sharegpt` actually means.** The corpus does load -- all 13 run
dirs record it, and `run.py` raises if a tier asks for it and gets synthetic.
But `load_sharegpt` fills `pool` and leaves `lengths` None, so **ShareGPT
supplies prompt TEXT only**: both the prompt-length target and the output
length come from a lognormal. The filler fraction rises with length -- 33.2%
below 500 chars, 98.0% above 4,000, 100% above 12,000 (the corpus cap), 58.3%
overall in T5's trace. This cannot bias an arm-versus-arm comparison, since
every arm replays one trace byte for byte. It does mean any conclusion that
depends on prompt or output LENGTH is untested against real text, which is
what T14 is for.

## Answered, no tier needed

**RAM offloading / vLLM `swap_space`** — **done, negative result.**
`27-hostmem-attribution.sh` measured −40 MiB from `--swap-space=0`, after an
earlier run claiming 2,121 MiB was invalidated by a selector bug (KServe never
sets `r2-role`). The reason is upstream: `swap_space` is **dead code in vLLM
V1** — `num_cpu_blocks` is hard-coded to zero in `vllm/v1/engine/core.py` and
"always has been according to the git history" (vllm#27984). V1 preempts by
RECOMPUTE, not SWAP. Nothing to tune, nothing further to test.

**Does T9 use HAMi?** — **yes, already.** `09-scaledown.py` takes
`--arm {dedicated,hami,hami-mps}` and **defaults to `hami`**, renders through
`render.py --arm`, and toggles the MPS daemon to match the arm. It runs one arm
per invocation, so a comparison is two runs.

---

## Skipped, and why

### T8 as originally defined — **removed 2026-09-03, then the slot reused**
It was the round-one control, meant to let the two rounds be compared directly.
It could not do that job: T8 was `llm-embed` and round one was **two decoders**
(2× Qwen2.5-0.5B). Varying the corpus on a pairing that measured null twice
tests nothing. The reconciliation between rounds is established algebraically
instead (FINDINGS 6.1).

**The slot was then redefined** as an MPS QoS-ceiling test on T11's mix, and in
that form it ran and is reported — see the Done section.

### Ada and Blackwell — **out of scope by decision (2026-09-10)**
Every number here is one RTX 3080 Ti (GA102), and the target fleet is GeForce
30/40/50-series, so a second architecture would raise external validity.
**The user has ruled it out**, and it is recorded here so nobody re-opens it:
no Ada or Blackwell part is available to this project, and inventing a
prediction for one would be worse than the honest limitation.

What that costs, stated plainly so the report can carry it: SM count, L2 size,
memory bandwidth and the MPS server implementation all move between Ampere,
Ada and Blackwell. The three numbers most likely to shift are `outside_mib`
(CUDA context plus graph pool, measured 448–664 MiB here), the saturation point
at which MPS starts paying, and the ~3 GiB host-RAM coefficient — the last of
which is a Python/torch/driver property and the most likely of the three to
travel unchanged. **Everything in this study is therefore stated as "measured
on 3080 Ti", in the executive summary rather than a footnote.**

---

## Specified in full: T6, T9, T12

These three were carried as "skipped" for weeks on priority grounds. That was
right while tiers with larger effects were unrun; it is no longer, because
nothing decisive is left in the two-body throughput question. Each is written
out here to the point where it can be launched without further thought.

### T12 — soak, 4 hours · `llm-llm` @ 6 req/s

**What it answers.** Every other run in this matrix is 300 seconds. That is
long enough for a latency distribution and far too short for anything
operational. Four hours is the shortest window in which KV-cache
fragmentation, a slow leak in either tenant, thermal steady state, clock drift
and MPS daemon degradation can appear at all. **A tenant that degrades at hour
three is invisible to every other tier here.**

**It is required for a production sign-off and unlikely to change a
conclusion** — which is exactly why it is worth the wall-clock rather than the
attention.

    nohup python3 -u run.py --test T12 --arms hami-mps \
      > ~/T12-mps.log 2>&1 &

**Arms.** Run `hami-mps` first and alone. The soak question is about MPS
survival, and `hami` adds four hours for a configuration nothing suspects.
Run the `hami` arm only if the MPS arm moves.

**Pass criteria, all four:**

| quantity | pass |
|---|---|
| throughput, last hour vs first | within ±0.42% (the measured noise floor) |
| tenant VRAM, hour 4 vs hour 1 | flat to within sampling resolution |
| MPS daemon | same PID throughout, both client PIDs attached |
| host RAM low-water | never below 1,200 MiB |

**Cost.** 4 h of replay plus ~4 min init. Budget 4.5 h for the one arm.

**Known risk.** This tier was queued once and killed: it was in the overnight
batch that died because `23-failure-inject.py` had no teardown at the time, so
T5 rep2, T5 rep3 and T12 all exited `rc=2` against pods that were never
removed. **The teardown is fixed.** Re-check with `25-node-check.sh` before
launching, because a four-hour run that dies at minute two costs a night.

### T6 — `llm-llm` @ 6 req/s on the Azure trace

**What it answers.** Whether the headline −26.7% pooled time-in-system is a
property of GPU sharing or an artifact of ShareGPT's particular output-length
spread. Same two decoders, same offered rate, same 300 s as T2 — the only
thing that changes is the (input, output) token-length pairs, drawn from
Microsoft's public Azure inference traces. `gen_trace.load_azure` keeps the
length pairs and regenerates a Poisson arrival process, so what differs is the
prefill/decode ratio and its variance, **not** burstiness.

    nohup ./28-series-b.sh T6 > ~/T6.log 2>&1 &

**Registered prediction: it reproduces.** T2's gain was already flat across
output-length buckets — −38.0% at ≤32 output tokens to −45.0% above 512. A
result that is flat in the variable you are about to change is a result that
survives changing it.

**Pass criterion.** Pooled time-in-system within ±8 percentage points of T2's
−26.7%. Outside that band, the ShareGPT corpus objection returns and T14 —
which already retires it with real complete prompts — becomes the only
external-validity evidence in the study.

**Cost.** Three arms × 300 s plus init ≈ 35 min.

**Pre-flight.** `python3 workload/gen_trace.py --source azure --help` must
resolve the trace file before you spend the run; `load_azure` is the only
code path in the harness that reads an external corpus.

### T9 — scale-down and admission · `09-scaledown.py`

**What it answers.** What happens to a surviving tenant while a neighbour is
evicted and a replacement is admitted on the same card. This is the only tier
that measures a **transition** rather than a steady state, and scale-to-zero is
the feature most often assumed to make GPU sharing economic.

**It was unrun for a real reason and for a bad one.** The real one: model init
of 113–250 s dominates the measurement, so the headline number it produces is
a KServe and Knative property, not a GPU-sharing one. The bad one: it crashed
100% of the time, ~15 minutes in, after both incumbents had deployed and
warmed.

**That crash is now fixed.** `H.smoke_test` returns `(ok, detail)`;
`09-scaledown.py:299` called `.items()` on the tuple. The arrival-side call at
line 388 did not crash but tested the wrong thing — `"ok" in str((False, {…}))`
is satisfied by any detail string containing those letters, so a failed
arrival could be marked as serving. Both sites now unpack.

    nohup python3 -u 09-scaledown.py --arm hami-mps > ~/T9-mps.log 2>&1 &
    # then, for the comparison:
    nohup python3 -u 09-scaledown.py --arm hami     > ~/T9-hami.log 2>&1 &

**It already uses HAMi.** `--arm {dedicated,hami,hami-mps}` defaults to `hami`,
renders through `render.py --arm`, and toggles the MPS daemon to match. One arm
per invocation, so a comparison is two runs.

**Report these, and not the wall-clock to serving:**

| quantity | why it is the interesting one |
|---|---|
| survivor latency during the transition | the only GPU-sharing quantity in the tier |
| survivor failed requests during eviction | does losing a co-tenant disturb the other |
| `t_serving − t_admitted`, split into scheduling vs model load | separates the KServe cost from the GPU cost |
| VRAM low-water during the overlap | admission before eviction completes is the failure mode |

**Cost.** ~25 min per arm, most of it init.

---

## What to run next, ranked by decisiveness

Rewritten 2026-09-07. Everything that was on the previous list has now RUN —
T8, T13, T14 (x2), T15, T16, T7 at MPS 50 and 25, `tenant-fault`, `vram-hog`.
What follows is what those runs opened, not what they closed.

Criterion is unchanged: how much can this run change a conclusion that is
already written down?

**The connection-pool blocker is FIXED** (it gated items 2, 5 and 8).
`workload/replay.py` called module-level `requests.post` with no `Session` and
no `HTTPAdapter`, so every in-flight thread opened and discarded its own TCP
connection. Raising `--max-inflight` to 4096 removed the concurrency ceiling
and exposed a connection-establishment ceiling instead: `T7-20260906-225515`
lost 163 requests to `ConnectionError` at ~1,000 concurrent, and shipped
593,178 prompt tokens against 660,530 on every comparable arm. `replay.py` now
holds a module-level `Session` whose pool is sized from `--max-inflight` at
startup, with `max_retries=0` — deliberately, because a retry would silently
repair exactly the failures this harness exists to measure. Any tier driven
past ~500 in flight is now safe to run.

### 1. `tenant-fault` on the `hami-mps` arm — the highest-value run outstanding
~9 minutes, no new code, and it decides a recommendation.

The `hami` arm (`T10-tenant-fault-hami-20260906-185601`, MPS daemon DOWN)
showed a device-side assert inside one tenant taking the untouched neighbour
offline **9 s later for 131.7 s**, 384 of 885 requests lost, while the faulting
tenant served on for a further 78 s. So the fault domain is the physical GPU,
not the MPS server — which falsified the premise that HAMi-without-MPS gives
you an unshared blast radius.

**What is still unknown is whether MPS makes it worse.** NVIDIA documents an
MPS server FAULT state that refuses new clients with
`CUDA_ERROR_MPS_SERVER_NOT_READY` until every existing client exits, which
predicts a *longer* outage and a slower recovery on the MPS arm. If that
reproduces, "MPS widens an already-shared fault domain" is measured rather than
quoted. If the two arms are indistinguishable, the honest headline becomes
"sharing a card at all is the risk; MPS is not the marginal one".

    python3 -u 23-failure-inject.py --fault tenant-fault --arm hami-mps

The run exists only on the node — it was executed once and never transferred.
Check before re-running.

### 2. The true SM-percentage curve — the 100% anchor is not a capacity number
`T7-20260904-111030`'s MPS arm finished in 304.7 s against a 300 s trace with a
peak of 61 requests in flight and a 4.06 s mean latency. **It was serving the
offered load, not saturating.** So 835.00 tok/s is the arrival rate, and the
67.1% / 35.7% retentions at 50% and 25% are upper bounds rather than
measurements. The only interval with both ends saturated is 50 -> 25, which
gives 0.531 for a halving — elasticity 0.913, essentially proportional.

To measure the curve, drive the 100% arm past its own knee so every point is a
capacity measurement:

    python3 -u run.py --test T7 --rate 24 --mps-percent llm=100
    python3 -u run.py --test T7 --rate 24 --mps-percent llm=50

Needs the connection fix above; rate 24 will exceed 500 in flight.

### 3. Verify the `default` latch mechanism — one run, decisive
T16 established the behaviour: `default` does not limit, at any quota, with or
without a neighbour. The source explains it as a latch defect —
`rate_limiter()` returns early on `cached_util_switch == 0`, *before* reaching
`set_recent_kernel(2)`, the only writer of the flag the vGPUmonitor reads — so
once the switch is observed off it stays off for the life of the pod.

That is a source reading, not a measurement. Re-run T16 with the monitor and
the container both talking:

    kubectl -n kube-system set env ds/hami-device-plugin -v=5
    # and on the tenant: LIBCUDA_LOG_LEVEL to make hami-core print
    #   "init_utilization_watcher: util_switch=…"
    python3 -u run.py --test T16

The log either shows the switch going off and never coming back, or it does
not. **This is the difference between reporting a defect and reporting a
behaviour**, and HAMi's maintainers cannot act on the second.

### 4. Replicate the two `force` results — the study's central negative rests on n=1
`T7 gpucores 50 + force` = **548.42** and `T2 gpucores 25/25 + force` =
**298.37** are each measured exactly once, and round one shows throttled arms
spreading up to 15.3% across passes of a single run. The directions are safe;
the magnitudes — "-34.4%", "+52.17%", "21.6x", "-30.5%" — are not yet
quotable to two decimals.

    python3 -u run.py --test T7 --gpucores llm=50 --core-policy force
    python3 -u run.py --test T2 --gpucores llm=25,llm2=25 --core-policy force

Two runs, ~55 minutes, and they harden the most-quoted table in the report.

### 5. A genuinely saturated symmetric pair — T15 did not manage it
`T15-20260906-182642` doubled T13's rate to 20 and still returned **99.5%** of
the solo sum; its dedicated arms finished in 241 s against a 240 s trace. Two
1.5B models do not fill this card at rate 20 either. The fairness prediction
held (spread 1.008 HAMi, 1.029 MPS) but the saturation prediction failed, so
the symmetric capacity point is **still missing**.

The test is on the *dedicated* arm, not the shared one: raise the rate until a
single tenant's wall clock overruns the trace window by >15%.

    python3 -u run.py --test T13 --rate 32

### 6. `deviceMemoryScaling = 1.5` — the untested cell with the worst failure mode
Confirmed not enabled on this node (`devicememoryscaling: 1` in the
`hami-device-plugin` ConfigMap), so this is genuinely untested rather than
accidentally on.

VRAM is the one axis where HAMi's enforcement provably works — it hooks
`cuMemAlloc_v2` and hard-fails at the per-tenant cap regardless of MPS. And the
source says the overcommit is scheduler-only: `register.go:145` inflates the
advertised memory, `server.go:656` sets `CUDA_OVERSUBSCRIBE=true` in the
container, and `grep -r OVERSUBSCRIBE` over the whole HAMi-core tree returns
**nothing**. No paging, no eviction, no fallback. `oom_check` compares each
container against its own limit and has no knowledge of the physical device.

Predicted: both pods admitted, the second dies with a raw
`CUDA_ERROR_OUT_OF_MEMORY` during vLLM profiling, no HAMi event or metric
explaining it, while a virtualised `nvidia-smi` shows headroom. For an 8-12 GiB
GeForce fleet that is the most dangerous switch a customer could flip.

    helm upgrade hami hami-charts/hami -n kube-system --reuse-values \
      --set devicePlugin.deviceMemoryScaling=1.5
    python3 -u run.py --test T2 --gpumem llm=8192,llm2=8192

Cluster-config change — run it awake, and revert afterwards.

### 7. Fix `vram-hog`, then run it
Two attempts, both VOID: the pod was created and allocated **zero bytes** on
either card (device 0 peak 1.0 MiB, device 1 unchanged at 11,549 MiB). Pinning
to the tenants' device was added and did not help. The injector now captures
the pod's own log and prints a NEVER ALLOCATED banner, so the third attempt
will at least say why — the likeliest cause is the image, since this node
returns 403 from `registry-1.docker.io`.

**Nothing about VRAM containment is testable until the pod is confirmed holding
memory.** Do not report a quiet victim as containment.

### 8. Where the depth law's re-ranking turns over
Nine runs out of nine show the deeper tenant taking the larger share of the MPS
gain. Where the depth gap is large the shallower tenant is made *absolutely
worse* (T14: 2.96x -> 6.38x degradation; T5: 2.89x -> 5.42x). Where it is small
nobody loses (T13, T15: spread ~1.01).

Unknown is where the turnover sits. One asymmetric tier at a *moderate* depth
ratio, around 2x, would find it:

    python3 -u run.py --test T5 --rate 10 --mix llm=0.35,llm2=0.65

This is the run that turns the depth law from a nine-point correlation into a
control with a threshold.

### 9. The power-limit sweep — demoted, but it settles a withdrawn claim
The claim that this card is power-limited and that compute quotas are therefore
theatre has been **withdrawn** — a busy GPU sits near its board limit whatever
the bottleneck is, and T7's SM curve shows SM allocation demonstrably matters.
Power is now used only as a *diagnostic* that a throttle engaged.

If anyone wants the causal question closed:

    sudo nvidia-smi -pl 250 && python3 -u run.py --test T7 --arms dedicated
    sudo nvidia-smi -pl 350

Throughput tracking the limit would resurrect the claim; flat would bury it.

### 10. T12 soak, T6 Azure trace, T9 scale-down
All three are now **fully specified above** with commands, pass criteria and
costs, and T9's deterministic crash is fixed. Priority among them is unchanged:
T12 is required for a production sign-off and unlikely to change a conclusion,
T6 is external validity and is predicted to reproduce, T9 measures a transition
whose headline number belongs to KServe rather than to the GPU.

### 11. Three tenants — T18, the largest structural gap left
Every co-tenancy result in this study is **two-body**. The depth law, the
fairness reversal and the overbook curve were all measured on pairs, and
nothing in the mechanism says they extend to three. T18 (`dense3` @ 9 req/s) is
defined and ready.

It may not run: three tenants need roughly 8.3 GiB of host RAM against ~8.2
available, and T4 was already refused at four. **If the guard refuses it, that
is the result** — report it as the density ceiling rather than working around
it. If it does run, the prediction is that the deepest tenant takes the largest
share of the MPS gain and the shallowest is made worst-off, exactly as in the
two-body tiers. A three-body run is the first chance for that to fail.

---

## Tiers worth defining if the matrix is extended

Ranked by what each could still change. The first four have concrete
mechanisms behind them; the last two are coverage.

### T19 — the true MPS SM-percentage curve, at real saturation
The 100% anchor in the existing T7 sweep **was never at capacity** — 304.7 s of
wall against a 300 s trace and 61 peak in flight. So the published elasticity
covers 50→25 only, where it is 0.913: essentially proportional, not the
sub-linear "you keep more than your SM fraction" that was withdrawn. Re-run the
sweep at a rate that saturates every point, and the curve becomes quotable.
`llm-only`, MPS at 100 / 75 / 50 / 25, rate raised until the 100% arm is itself
saturated. Four arms, ~40 min.

### T20 — does `default` latch off, and when
`GPU_CORE_UTILIZATION_POLICY=default` measurably never limits, three ways.
The source says it should: `get_utilization_switch()` returns a region value
initialised to 1 and never written to 0. The candidate mechanism is that
`rate_limiter()` returns early on `cached_util_switch == 0` **before** reaching
`set_recent_kernel(2)` — the only writer of the flag vGPUmonitor reads — so the
limiter latches off for the pod's lifetime. One run with the pod started
**already loaded** would separate "never armed" from "armed then latched off",
and it is decisive either way.

### T21 — `deviceMemoryScaling = 1.5`, the untested cell with the worst failure mode
Memory overcommit is the one HAMi knob nobody here has touched, and its failure
mode is an OOM inside a tenant rather than a slowdown. Note that
`deviceCoreScaling` is a **one-way door** — above 1 the exclusivity gate
`Totalcore == 100 && Coresreq == 100` becomes permanently unreachable — and
`deviceMemoryScaling` should be assumed to behave the same until shown
otherwise. Test on a node you can rebuild.

### T22 — a global-scope fault
`tenant-fault` raises `CUDA_EXCEPTION_12`, scoped **per warp** — the most
containable class there is. The `global-fault` probe in `23-failure-inject.py`
raises `CUDA_EXCEPTION_10` instead, an illegal global address, via
`torch.as_strided` (no bounds check, needs no nvcc in the image). **Until this
runs on both arms, this study does not support a fault-isolation claim.**

### T23 — where the depth law's re-ranking turns over
MPS reallocates toward in-flight depth, and nine runs agree. The open question
is the crossover: at what depth ratio does the advantage change hands? A sweep
of one pair at several rate splits would turn a qualitative law into a
threshold an operator could schedule against.

### T24 — `gpucores` above the admission ceiling, by the only route there is
`Fit()` rejects on `Totalcore − Usedcores < Coresreq` and `Coresreq` is
independently validated 0–100, so **the per-card sum is hard-capped at 100**:
75/75 and 100/100 are unschedulable, full stop. The only way to a sum above 100
is `deviceCoreScaling`, and that is the one-way door above. A tier that
demonstrates the rejection and then demonstrates the scaled admission would
close the scheduling question properly.

### Deliberately still not planned
- **A second architecture (Ada, Blackwell).** Ruled out by decision — see
  "Skipped, and why".
- **MPS static SM partitioning** (`CUDA_MPS_SM_PARTITION`). Research, not
  shippable: it needs the control daemon driven by hand per node, is exposed by
  neither HAMi nor KServe nor the device plugin, and nothing would reconcile it
  with `nvidia.com/gpucores`.
- **CUDA green contexts.** Proposed in HAMi-core#179, not implemented in 2.9.0
  — not testable here at all.

---

## References

Every external claim in this matrix and in the tier reports resolves to one of
these. Measured numbers are ours; the entries below are what we relied on for
mechanism, scope and vocabulary.

**NVIDIA**

- *Multi-Process Service* (MPS documentation). The `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE`
  contract, the client/server architecture, and the fault statement this study
  quotes: a fatal fault from one client may bring down a different user's
  client sharing the GPU. Volta's improvement is server *survival*, not client
  isolation. <https://docs.nvidia.com/deploy/mps/>
- *CUDA-GDB* — the exception table that scopes each fault.
  `CUDA_EXCEPTION_12` (device-side assert) is **per warp**;
  `CUDA_EXCEPTION_10` (illegal global address) and hardware stack overflow are
  **global**. This is the source of the fault-scope taxonomy in T10 and T22.
  <https://docs.nvidia.com/cuda/cuda-gdb/>
- *NVML / `nvidia-smi`* — and the reason `DCGM_FI_DEV_GPU_UTIL` is **not** SM
  occupancy: a card reads 100% with a single SM busy. `DCGM_FI_PROF_*` is
  absent on this hardware and `nvmlDeviceGetProcessUtilization` returns
  `NotFound`, so SM occupancy is not measurable here and is not claimed.
- *MIG user guide* — cited only to record that it does not apply. The target
  fleet is consumer GeForce, where MIG does not exist; that is the premise of
  the whole study rather than a limitation of it.

**HAMi**

- Project documentation and `nvidia.com/gpucores` semantics.
  <https://github.com/Project-HAMi/HAMi>
- `pkg/device/nvidia/device.go` — `Fit()` rejects on
  `dev.Totalcore - dev.Usedcores < k.Coresreq`, and `Coresreq` is independently
  validated 0–100. **This is the source of the hard 100-per-card cap** and of
  the exclusivity gate `Totalcore == 100 && Coresreq == 100` that
  `deviceCoreScaling > 1` makes unreachable.
- HAMi-core, `src/multiprocess/multiprocess_memory_limit.c` and the
  `rate_limiter()` path — the mechanism behind "`default` never limits, only
  `force` does", including `get_utilization_switch()` and `set_recent_kernel`.
- **Issue #1864**, filed against v2.9.0 — an independent report that
  `gpucores` does not limit under the default policy. Our measurement and this
  issue were arrived at separately and agree.
- HAMi-core **#179** — the green-contexts proposal. Not implemented in 2.9.0,
  which is why it is untestable here.
- RiseUnion's published isolation test — a third independent observation of the
  same `default`-policy behaviour.

**vLLM and KServe**

- vLLM `--gpu-memory-utilization` semantics: a fraction that **omits** the CUDA
  context and the graph pool. This is why `resident = util × quota + outside`
  and why the safe test is `util ≤ 1 − outside/quota`. Measured `outside`:
  664 MiB (1.5B), 448 MiB (7B AWQ).
- **vllm#27984** and `vllm/v1/engine/core.py` — `swap_space` is dead code in
  vLLM V1; `num_cpu_blocks` is hard-coded to zero and V1 preempts by
  RECOMPUTE, not SWAP. This is why the RAM-offloading question is closed with
  a negative result rather than a tier.
- `vllm:num_requests_running` — the production metric that corresponds to the
  in-flight depth the MPS depth law keys on.
- KServe `LLMInferenceService`: `spec.template` is a bare `PodSpec` with no
  `metadata`, so labels and annotations are dropped. Pods carry
  `app.kubernetes.io/name=r2-<role>` and `kserve.io/component=workload`.
  `WorkloadSpec.Labels`/`Annotations` are siblings of `Template`, and KServe
  0.17 ships propagation for them — which is why the GPU pin belongs in
  `spec.annotations`, not `metadata.annotations`.

**Papers and formulas**

- Little, J. D. C. (1961). "A Proof for the Queuing Formula: L = λW."
  *Operations Research* 9(3), 383–387. The in-flight depth used throughout is
  `arrival_rate × mean_latency` from each tenant's SOLO arm — Little's law, and
  the reason depth rather than model size predicts who wins under MPS.
- arXiv **2605.26461** — fault injection under MPS on L40 and H100: 19
  scenarios, with **all five SM-fault scenarios propagating** to every client,
  because the SM global TRAP path carries no per-channel attribution.
  Device-side assert is not among their five, so it does not contradict our
  containment result — it bounds what that result covers.
- Published co-location work on resource-profile complementarity: three
  independent systems, keying on three different signals, converge on the same
  rule — co-location gain depends on the tenants' resource profiles being
  complementary, not on how busy the device is. Memory-bound pairs can make
  **both** tenants worse. This is what T1 and T3's nulls look like from the
  literature's side.
- KV-cache and weight VRAM estimation follows the standard transformer
  formulas: weights = parameters × bytes-per-parameter (0.5 for 4-bit AWQ),
  KV = 2 × layers × kv_heads × head_dim × bytes × sequence × batch. Model
  config values come from each repository's `config.json` on Hugging Face.

**Corpora**

- **ShareGPT** — prompt text for every tier except T6 and T14. It supplied
  prompt TEXT only: lengths and output sizes were drawn lognormal, and filler
  rose from 33% of prompts at short lengths to 100% above 12,000 characters.
  This is the corpus objection, and **T14 retires it** by replaying complete
  real prompts with zero filler.
- **Azure public inference traces** (Microsoft AzurePublicDataset) — the
  (input, output) token-length pairs behind T6.

---

### Also outstanding, not tiers

- ~~Harness: connection pooling in `replay.py`~~ — **done.** Module-level
  `Session`, pool sized from `--max-inflight`, `max_retries=0`.
- ~~Harness: `09-scaledown.py:299` crashes deterministically~~ — **done.** Both
  smoke-test call sites now unpack `(ok, detail)`.
- ~~Recompute every pre-`9afafe6` fault artifact~~ — **done offline.**
  `scratchpad/windows.py` recomputed all of them from the saved
  `replay.requests.csv` files, percentiles over successful requests only. No
  re-runs were needed. The `mps-kill` outage that had been reported as
  "victim p99 −54.5%" is now 96/173 during and 253/343 after.
- ~~Tier reports~~ — **done, eight of them**, one per tier where a tier earned
  one and merged where it did not: T7, T14, T13+T15, T2+T16, T11+T8, T10, T5,
  T1+T3+T4.
- ~~Persian translations~~ — **done.** All eight, as HTML, DOCX and PDF, in
  `reports-fa/`. Technical vocabulary is left in Latin script by design and
  every Latin run is bidi-isolated; a Persian comma does not separate two
  Latin terms, it joins them and they swap silently.
- **T2 and T5 repeats to n≥3.** T5 has n=2 at 0.04–0.36% agreement and T14 has
  n=2 at 0.03–0.33%, so the case for more of these is weak — the `force` arms
  in item 4 are where replication is actually needed.
- **Two flags that belong in any executive summary**, neither of which a
  further tier removes cheaply: (a) every co-tenancy result here is two-body,
  which T18 addresses and may not be able to on this node; (b) the only fault
  class tested is the most containable one in NVIDIA's taxonomy, which the
  `global-fault` probe addresses.
