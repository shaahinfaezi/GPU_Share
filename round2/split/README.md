# Split × Share series

Does model **splitting** (the coworker's TP/PP, llama.cpp offload, custom runtime)
still work when **GPU sharing** (HAMi, HAMi+MPS) is in the path — and does sharing
still pay when the tenants are split?

Two arms only, `hami` and `hami-mps`, because those are the only two answers a
HAMi node can give. Everything else would not change what anyone does on Monday.

## Box 1 — this box (`mr-gholami`, 2× 3080 Ti)

Two 2-way-split vLLM tenants on two cards = **four ranks, two per card**. Every
card is simultaneously a sharing target and a splitting target.

```bash
cd /opt/GPU_Share/round2/split && chmod +x *.sh *.py

# 1. gates (~15 min, foreground — you want to read it)
./51-split-preflight.sh 2>&1 | tee logs/preflight-$(date -u +%Y%m%d-%H%M%S).log

# 2. solo baselines + rate calibration (~25 min)
nohup ./52-split-solo.sh  > logs/solo.log  2>&1 &
tail -f logs/solo.log

# 3. hami then hami-mps (~35 min)
nohup ./53-split-share.sh > logs/share.log 2>&1 &
tail -f logs/share.log

# 4. read it
python3 54-split-collect.py results/S-share-<stamp>
```

| arm | what |
|---|---|
| S1 `tp` dedicated | `--tensor-parallel-size 2`, alone on both cards |
| S2 `pp` dedicated | `--pipeline-parallel-size 2`, alone on both cards |
| S3 `hami` | both co-resident, HAMi quotas, no MPS |
| S4 `hami-mps` | identical, MPS daemon up, every client at 100% |

## Box 2 — the coworker's box

```bash
nohup ./60-box2-split.sh > logs/box2.log 2>&1 &
python3 62-box2-collect.py results/B-box2-<stamp>
```

Same four-arm shape: llama.cpp solo, custom solo, both under `hami`, both under
`hami-mps`. This pair is the generalisation test — two runtimes with nothing in
common. If HAMi and MPS behave the same across them, the mechanism belongs to the
layer and not to vLLM, which is the only reason to believe round two's numbers
apply to the fleet's real mix.

## Why stage 2 cannot be skipped

Every number in stage 3 is a ratio against stage 2. The **saturation ratio** —
combined co-tenancy tok/s ÷ sum of solo tok/s — is the only variable that
predicted the MPS effect across round two's five points:

| saturation | MPS gain |
|---|---|
| 99.6% | +0.06% |
| 86.3% | +8.6% |
| 80.9% | +19.0% |
| 55.9% | +20.8% |

Run the sharing arms at a rate that leaves the card unsaturated and MPS correctly
shows nothing — and the report says *"MPS doesn't help split models"* when what it
measured was the rate. `52-split-solo.sh` measures each tenant's real capacity and
writes `calibration.json`; `53-split-share.sh` refuses to start without it.

## Changes made to the supplied manifests

Deployment-wrapper changes only. `--tensor-parallel-size 2`,
`--pipeline-parallel-size 2` and `--n-gpu-layers 10` are untouched — they are the
thing under test.

| # | change | left alone → |
|---|---|---|
| 1 | both vLLM files were `metadata.name: qwen2-0b5-instruct-local` → renamed `r2-tp` / `r2-pp` | the second overwrites the first; the co-tenancy arm is silently a solo arm |
| 2 | dropped the `router:` block | `llm-d-inference-scheduler` 403s on this node (measured in round two); pod never goes Ready |
| 3 | added `gpumem` + `gpucores` to **all four** | a pod naming neither is uncapped **and** consumes no scheduler quota — `Fit()` compares against `Coresreq`, absent = 0. Two such pods don't share a card, they race for it |
| 4 | `--gpu-memory-utilization` 0.75/0.5 → 0.70 | the fraction is of the **quota**, and vLLM doesn't count the CUDA context or graph pool inside it. `util ≤ 1 − outside/quota` → 0.83 ceiling at 4096 MiB |
| 5 | **removed `CUDA_VISIBLE_DEVICES: "0"`** from the custom runtime | HAMi *sets* this to communicate the device it allocated. Hardcoding it overrides the assignment — the container addresses the driver's device 0, not its own. `model.py`'s `cuda:0` is already correct: index 0 of what HAMi exposed |

Per-card budget: `2 ranks × 4096 MiB = 8192` of 12,045 usable; `2 × 50 gpucores = 100`,
exactly the `Fit()` ceiling.

## Two things to fix before box 2 runs

1. **`model.py` will not start as pasted.** `logging.getLogger(name)` →
   `__name__`, and `if name == "main":` → `if __name__ == "__main__":`. Without
   the second, the module defines everything and exits without calling
   `load_model()` or `uvicorn.run()` — the container goes Ready-never and the
   probe fails 20 times before anyone reads the log.
2. **Check the custom model's real size** against the 6144 MiB quota. Too small
   and torch raises on load — crashloop, not graceful degradation.

## Known limits of this series

- **No TTFT on box 2.** The custom runtime serves `/v1/models/{name}:predict`,
  one JSON blob, no streaming — there is no first token to time. Recorded as
  absent, not as zero. `61-box2-load.py` exists because `replay.py` speaks only
  streaming OpenAI chat and would 404 on every request.
- **`pp` is throttled by `--max-num-seqs 4`** (the coworker's value, kept). It
  will be slower than `tp` in every arm. The collector compares each tenant to
  *its own* solo baseline so that throttle is never reported as a sharing effect.
- **The fault question is not asked.** Round two's most decisive finding: a
  global-scope CUDA fault is contained under HAMi (0/516) and **propagates under
  MPS** (307/515, 59.6%). Splitting should make that strictly worse — a 2-rank
  tenant has twice the surface and sits on both cards, so one global fault under
  MPS could plausibly reach every tenant on the box rather than one card's worth.
  **That is a prediction, not a measurement**, and it is written down here so it
  is not later mistaken for one. Measuring it needs `23-failure-inject.py` taught
  about multi-device roles.
- **NCCL under MPS is the live risk.** TP all-reduces across ranks every layer;
  under MPS both ranks are clients of one daemon. Preflight gate 5 serves one
  real request through that path. If it fails, do not run S4 — it will hang, not
  crash. That failure is itself a publishable result.
