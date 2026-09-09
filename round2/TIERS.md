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
the pod. 5.14b's prediction from HAMi's docs was half right — the two policies
differ, but the documented *reason* does not survive measurement.

**`force` throttles to 65.6% of solo throughput, 21.6× mean latency, 45.8×
TTFT p99. Adding MPS to that same forced pod returns it to 834.51 — +52.2%,
99.9% of dedicated.** MPS does not weaken the limiter, it erases it, and every
request still succeeds so nothing surfaces the fact.

This tier also corrected two published claims: "HAMi's limiter is free" (that
run was `gpucores: 100` — nothing to limit) and "`default` and `force` are
identical in code" (they are plainly not identical in behaviour). FINDINGS 8.2.

### T10 — failure injection · **3 of 5 faults**
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
- `tenant-fault` — **run 4 times; the alarming result did NOT replicate.**
  A genuine device-side assert (`cudaErrorAssert`, dead context) was
  **contained on every arm with a clean measurement** — the `hami` replicate
  and both `hami-mps` runs show 0 victim failures and flat VRAM. One earlier
  `hami` run lost 384 of 885 victim requests with an 8 GiB VRAM collapse; it
  did not reproduce and is most plausibly a host OOM kill, which the injector
  cannot confirm because it never sampled host memory. **A sticky CUDA error
  appears to kill only the process that caused it.** FINDINGS 8.16.
- **MPS static SM partitioning** (`sm_partition`, `CUDA_MPS_SM_PARTITION`).
  Research, not shippable: it needs the MPS control daemon driven by hand per
  node, is not exposed by HAMi, KServe or the device plugin, and nothing would
  reconcile it with `nvidia.com/gpucores`.
- **CUDA green contexts.** Proposed in HAMi-core#179, not implemented in
  2.9.0 — not testable here at all.

### Also outstanding, not tiers

- **Harness: connection pooling in `replay.py`** (see the note at the top of
  this section). Gates items 2, 5 and 8.
- **Harness: `09-scaledown.py:299` crashes deterministically** — `smoke.items()`
  on the tuple `smoke_test()` returns. T9 fails 100% of the time, ~15 min in,
  after both incumbents have deployed and warmed.
- **Recompute every pre-`9afafe6` fault artifact.** The saved `t10.json` files
  put refused requests into the latency percentiles, so outages read as
  improvements. `scratchpad/windows.py` recomputes them correctly from the
  saved CSVs — no re-runs needed.
- **Tier reports.** T7 is rebuilt. Outstanding: T2+T16 and T5+T14 need updating,
  T8+T11, T13+T15 and T10 need writing, T1+T3 should be merged into one nulls
  report. Six reports for twelve tiers.
- **T2 and T5 repeats to n≥3.** T5 has n=2 at 0.04–0.36% agreement and T14 has
  n=2 at 0.03–0.33%, so the case for more of these is weak — the `force` arms
  in item 4 are where replication is actually needed.
- **Persian translations** of every report.
