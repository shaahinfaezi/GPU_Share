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
`force` limits unconditionally, `default` limits only under contention — and it
is what 5.14b predicted from HAMi's docs before any of these runs existed.

**`force` throttles to 65.6% of solo throughput, 21.6× mean latency, 45.8×
TTFT p99. Adding MPS to that same forced pod returns it to 834.51 — +52.2%,
99.9% of dedicated.** MPS does not weaken the limiter, it erases it, and every
request still succeeds so nothing surfaces the fact.

This tier also corrected two published claims: "HAMi's limiter is free" (that
run was `gpucores: 100` — nothing to limit) and "`default` and `force` are
identical in code" (they are plainly not identical in behaviour). FINDINGS 8.2.

### T10 — failure injection · **3 of 5 faults**
Not a tier in `run.py`; it lives in `23-failure-inject.py`.

- **`mps-kill` — the serious one.** 681 of 1,753 requests failed (39%),
  ~100 s of total outage, ~140 s to recovery, gated by model reload. Both
  tenants died together.
- `tenant-kill` — benign, survivor served through re-init.
- `vram-escape` — **contained on both arms** by HAMi-core's own OOM path, the
  log line matching `allocator.c:52` exactly. HAMi's memory isolation survives
  MPS because it hooks `cuMemAlloc_v2` and never consults NVML.
- `vram-hog` — **built, never run.**
- `tenant-fault` — **added 2026-09-06, never run.** Triggers a real fatal GPU
  fault (device-side assert), which is the fault class NVIDIA documents as
  poisoning the whole MPS server.

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

---

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

### T6 — `llm-llm` @ 6 on the Azure trace — **deprioritised, not abandoned**
Asks whether −26.7% survives production token-length distributions rather than
ShareGPT's. **Low expected information:** T2's gain was already flat across
output-length buckets — −38.0% at ≤32 tokens to −45.0% above 512 — which
predicts the result survives a different length mix. Worth running for external
validity in the final report, not worth running ahead of anything below.

### T8 — **removed 2026-09-03**
It was the round-one control, meant to let the two rounds be compared directly.
It could not do that job: T8 was `llm-embed` and round one was **two decoders**
(2× Qwen2.5-0.5B). Varying the corpus on a pairing that measured null twice
tests nothing. The reconciliation between rounds is established algebraically
instead (FINDINGS 6.1).

### T9 — scale-down / admission — **script exists (`09-scaledown.py`), unrun**
Init time of 113–250 s dominates the measurement, so what it reports is a
KServe and Knative property rather than a GPU-sharing one. Lowest information
per minute in the matrix.

### T12 — **not skipped, killed**
Queued in an overnight batch that died because T10 had no teardown at the time;
T5 rep2, T5 rep3 and T12 all exited `rc=2` against pods that were never
removed. The teardown is fixed. **T12 has not been re-queued and is still
wanted.** T11 was in the same batch and has since been run -- see above.

---

## What to run next, ranked by decisiveness

Criterion: how much can this run change a conclusion that is already written
down?

Run them one at a time, in this order. Static SM partitioning is deliberately
last: it is a research question about what MPS *could* do, not a configuration
anyone would ship, and it should not displace tiers that describe the system
the customer will actually run.

Run them one at a time, in this order.

### 1. T8 — matched partition — **run this one now**
`gpucores: 50` **and** `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE: 50`, policy `force`,
two tenants at 10 req/s. Zero new code — the tier pins its own knobs.

Every shared arm so far ran gpucores 50 with MPS at **100**: HAMi asking for
half a card while MPS was told to use all of it. That is correct for isolating
"MPS on/off" and it is **not what anyone would deploy** — an operator would set
both to 50 and expect a 50% guarantee. T7 proved HAMi's limiter is void under
MPS, so the MPS thread percentage is the only thing left that can bound a
tenant. Prediction: a ceiling near 50% but **no floor**, because NVIDIA states
the limit *"does not reserve dedicated resources... kernels launched from
different MPS client contexts may execute on the same SM."*

```
python3 -u run.py --test T8
```

### 2. T13 — symmetric pair, `llm2a` + `llm2b` — **built, ready to run**
Every pairing we ran was asymmetric, so "does MPS create systematic winners and
losers" is currently answered by mechanism rather than measurement. Two copies
of Qwen2.5-1.5B-AWQ fit: 2 × 5,632 = 11,264 MiB against `CARD_MIB = 12288`.
Two 7Bs do not (2 × 8,192).

Predicts near-symmetric degradation on both arms and an aggregate MPS gain at
or above T5's, since two identical decoders is the most composition-similar
pair possible. Also yields the **Gini coefficient** the external review asked
for, comparable to the published 0.35-MPS / 0.12-MIG figures.

**Built and verified 2026-09-06** (commit `1ab4531`): roles `llm2a`/`llm2b`
share one `model_dir`, so `render.py`'s PVC path now follows the model rather
than the role name. Renders at 2 x 5,632 = 11,264 MiB, 91.7% of the card, util
clamped to 0.855 on both.

```
python3 -u run.py --test T13
```

### 2b. T14 — raw prompts, the corpus control — **built, ready to run**
`make_prompt()` pads chat prompts with a fixed FILLER sentence to hit a
synthetic target length and truncates the overshoots. In T7's trace **1,033 of
1,749 prompts (59.1%)** carry it. That cannot bias an arm-versus-arm
comparison — every arm replays the identical trace — but the prompts are not
pure ShareGPT, and a reviewer is entitled to ask whether the effect survives
real text.

T14 is T5 with `--raw-prompts`: corpus prompts verbatim, no padding, no
truncation, prefill lengths as ShareGPT actually contains them. The flag is
recorded in the trace's own `params`, so a run is self-describing. Absolute
numbers will differ from T5 and are not meant to match — **the claim under test
is only that the MPS effect reproduces in sign and rough magnitude.** If it
does, the filler is retired as an objection; if it does not, every
length-sensitive conclusion needs revisiting.

```
python3 -u run.py --test T14
```

### 3. T10 `vram-hog` and `tenant-fault`
Completes the fault matrix, which the T10 report needs before it can be
written. `tenant-fault` is the one that matters: it triggers the fault class
NVIDIA documents as moving the MPS server to `FAULT` and refusing new clients
with `CUDA_ERROR_MPS_SERVER_NOT_READY`. If that reproduces, the blast-radius
finding gets much stronger than `mps-kill` alone made it.

### 4. T12 — 4-hour soak
Required for a production sign-off, unlikely to change a conclusion. Watching
for KV fragmentation, memory growth, thermal steady state versus the 60–90 s
ramp every short run shows, and whether the MPS daemon survives four hours of
continuous multiplexing. Run it on a night when nothing else is queued.

### 5. T6 — Azure trace
External validity. Predicted to reproduce.

### 6. T9 — scale-down
Init time dominates it.

### 7. MPS static SM partitioning — **last, and it is research, not production**
Deferred deliberately. NVIDIA documents a mechanism that might give a compute
ceiling back under MPS: **static SM partitioning** (MPS guide 2.3.10,
`sm_partition add` with `CUDA_MPS_SM_PARTITION`), enforced by the MPS server
itself -- unlike `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE`, which NVIDIA states
explicitly *"does not reserve dedicated resources for any MPS client context."*

If it worked it would soften FINDINGS 8.3 from "MPS erases your compute
ceiling" to "...unless you partition statically". But it is **not something the
customer would deploy**: it needs the MPS control daemon driven by hand per
node, it is not exposed by HAMi, KServe or the device plugin, and nothing in
the toolchain would reconcile it with `nvidia.com/gpucores`. The fault-domain
objection survives it regardless.

Support on consumer GeForce silicon is also unverified. When it is finally
reached, the first step is a five-minute capability probe, not a tier.

CUDA green contexts would close the same gap at the driver level and are the
right long-term fix, but they are proposed in HAMi-core#179 and **not
implemented in 2.9.0** -- not testable here at all.

### Also outstanding, not tiers
- **T2 and T5 repeats to n≥3.** T5 now has n=2 with 0.04–0.36% agreement, so
  the case for more repeats is much weaker than it was.
- **T7 report rewrite** — its central claim changed.
- **Persian translations** of every report.
