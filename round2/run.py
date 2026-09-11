#!/usr/bin/env python3
"""Run one round-2 test across all three arms and save everything needed to
re-render the result later.

    python run.py --list
    python run.py --test T1                        # all three arms
    python run.py --test T1 --arms hami,hami-mps   # skip the baseline
    python run.py --test T1 --dry-run              # print the plan, touch nothing

WHAT AN ARM MEANS HERE

  dedicated   requirement 2: the same environment WITHOUT sharing. For a
              two-tenant scenario this is not one run, it is one run PER
              TENANT -- llm alone, then embed alone. Each replays only its own
              rows from the shared trace, so the arrival times a tenant sees
              are byte-identical to the ones it sees under contention. That is
              the only way "what did sharing cost me" has an answer.

              The gpumem quota stays at the shared value on purpose. Handing
              the baseline the whole card would make its extra throughput a
              memory-budget effect wearing a sharing costume.

  hami        co-tenants, HAMi gpumem + gpucores, no MPS.
  hami-mps    identical to hami plus the MPS env block. Diffing the two
              rendered manifests shows only those lines.

FAIRNESS, ENFORCED RATHER THAN INTENDED (requirement 7)

  * one trace per test, replayed by every arm; its SHA-256 trace_id is stored
    in each sub-run and compared at the end. Divergence fails the run.
  * co-tenants forced onto ONE physical GPU, and the placement VERIFIED after
    deployment. HAMi ships gpuSchedulerPolicy=spread, so two "co-tenants" would
    otherwise land on separate cards and produce a clean result that measures
    nothing. Co-location comes from the scheduler (10-hami-binpack.sh), because
    the per-pod pin does not survive KServe. Verification is against nvidia-smi
    as well as HAMi's annotation, and a split aborts the arm.
  * the GPU is drained to idle between sub-runs, verified by nvidia-smi rather
    than by sleeping.
  * the client talks straight to the pod IP, so the KServe router is outside
    the measurement in every arm.

WHAT LANDS ON DISK (requirement 3)

  results/<test>-<stamp>/
      run.json          the whole plan, plus a verdict per sub-run
      trace.json        the exact bytes every arm replayed
      events.jsonl      phase boundaries, for annotating the screen recording
      <arm>__<roles>/
          rendered/         the manifests as applied
          replay.summary.json, replay.requests.csv, replay.itl.json
          prom.json         every dashboard series over the run window
          gpu.csv           1 Hz nvidia-smi, finer than the 5 s DCGM scrape
          pods.json         placement, images, resource blocks as admitted
"""
import argparse
import datetime
import json
import os
import subprocess
import sys

import harness as H

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

# ShareGPT is fetched by hf_hub_download at trace-generation time and is ~650
# MB. Left to default it lands in ~/.cache/huggingface on `/`, which has ~13
# GiB free against a node-wide eviction threshold of 4.84 GiB. Point it at the
# same /data2 cache the model weights already use. 07-prefetch-traces.sh stages
# both corpora ahead of time so a run never downloads anything.
os.environ.setdefault("HF_HOME", "/data2/hf-cache")
_AZ = "/data2/hf-cache/traces/AzureLLMInferenceTrace_conv.csv"
if os.path.exists(_AZ):
    os.environ.setdefault("AZURE_TRACE_CSV", _AZ)

# Scenario -> roles, mirroring render.py and workload/gen_trace.py.
ROLES = {
    "llm-only": ["llm"], "embed-only": ["embed"], "rerank-only": ["rerank"],
    "llm2-only": ["llm2"], "vl-only": ["vl"], "embed-small-only": ["embed-small"],
    "llm-embed": ["llm", "embed"], "llm-llm": ["llm", "llm2"],
    "llm2-llm2": ["llm2a", "llm2b"],
    "dense3": ["llm2", "embed-small", "embed"],
    "vl-rerank": ["vl", "rerank"],
    "dense4": ["llm2", "embed", "embed-small", "rerank"],
}

# The matrix. Rates are offered load in requests/second, Poisson-distributed --
# an OPEN loop, so an overloaded arm builds a queue instead of quietly reducing
# its own offered load the way a fixed-concurrency client does.
TESTS = {
    "T1": dict(scenario="llm-embed", rate=6.0, duration=300, source="sharegpt",
               why="Headline. 7B-AWQ + bge-m3 fills 91.7% of the card against "
                   "round one's measured 62.7-69.3% (peak VRAM on GPU 0, from "
                   "its own gpu-timeline.csv). The bigger gap is the workload: "
                   "round one used 51-token prompts and fixed 256-token "
                   "outputs at concurrency 4, which is decode-bound and never "
                   "queues. Registered prediction: filling the card and "
                   "letting it queue widens the MPS advantage."),
    "T2": dict(scenario="llm-llm", rate=6.0, duration=300, source="sharegpt",
               why="Two generative tenants, and the HIGHEST-POWER scenario in "
                   "the matrix. T1 established that MPS is bounded by power, "
                   "not contention: every arm holding the 7B sat at 343 W of a "
                   "350 W board limit and MPS returned +0.0%, where round one "
                   "at 249-256 W returned +24.9%. Three predictions registered "
                   "before the run. (a) MPS again returns ~0%: adding a 1.5B "
                   "to a card already at its ceiling cannot create headroom. "
                   "(b) Sharing costs MORE than T1's +3.1% e2e p99, because "
                   "the neighbour is a steady decode consumer rather than a "
                   "bursty encoder. (c) The clock-residency gain that made "
                   "bge-m3 7.4% FASTER will NOT appear for llm2 -- it only "
                   "existed because the embedder alone left the card at 3.5% "
                   "utilisation and 1,665 MHz. A 1.5B alone will already hold "
                   "clocks near peak, so there is nothing for a neighbour to "
                   "recover. T4 is the opposite extreme and the real test of "
                   "the power hypothesis. "
                   "OUTCOME (2026-09-02): the first attempt was VOID -- the MPS daemon was left up for every arm, so hami and hami-mps were the same configuration. Re-run with per-arm daemon control returned pooled time-in-system -26.7%, 7B -44.6%, 1.5B +24.2%. Power was never the mechanism; contention is. See FINDINGS 6."),
    "T3": dict(scenario="vl-rerank", rate=4.0, duration=300, source="sharegpt",
               why="Vision encoder plus a cross-encoder reranker: two "
                   "compute-bound, non-autoregressive neighbours. The case "
                   "where gpucores SHOULD matter most if it ever does. "
                   "Registered after T2: (a) power lands BELOW the 343-346 W "
                   "both previous tiers hit, because neither tenant is a 7B -- "
                   "if it does not, the power hypothesis is in trouble before "
                   "T4 even runs. (b) The ITL-convergence result does NOT "
                   "transfer: it is a property of two autoregressive decoders "
                   "interleaving token by token, and rerank does not decode at "
                   "all. Expect the T1 shape (small throughput cost, tail "
                   "latency cost) rather than the T2 shape."),
    "T4": dict(scenario="dense4", rate=8.0, duration=300, source="sharegpt",
               why="DENSITY TEST -- no longer a power test. The power "
                   "hypothesis is dead (FINDINGS 6.2): T2 gained 26.7% with "
                   "1.4 W of headroom while T3 gained nothing with 36-57 W, "
                   "and round one's power/gain correlation runs the wrong way "
                   "(r=+0.717). Contention gates MPS, not watts. "
                   "PREDICTION, registered before the run: T4 shows NO MPS "
                   "effect. Scaling measured solo duty cycles to this "
                   "scenario's per-tenant rates (llm2 2.4/s, embed 2.4/s, "
                   "embed-small 1.6/s, rerank 1.6/s) gives a combined demand "
                   "near 59% -- BELOW T3's 67%, which was a true null at "
                   "p=0.90. dense4 is structurally low-contention because "
                   "three of its four tenants are pooling models that sit "
                   "near-idle; raising the rate cannot fix that without "
                   "pushing llm2 past its own knee and conflating saturation "
                   "with contention. "
                   "WHAT IT DOES TEST, and nothing else in the matrix does: "
                   "(a) HAMi admitting FOUR co-tenants on one card at 97.9% "
                   "memory fill; (b) four vLLM engines profiling and "
                   "capturing cudagraphs SIMULTANEOUSLY, which is exactly the "
                   "init-peak collision 4.6a warns about and the densest "
                   "memory moment in the study; (c) whether the per-tenant "
                   "cost of sharing stays flat going from two tenants to "
                   "four. Requirement 1 asks for number of models as a "
                   "variable; this is its top end."),
    # T5 is the ONLY tier that can produce a throughput number, and it took
    # until 2026-09-02 to understand why. Every other tier replays open-loop at
    # a rate all three arms serve completely (1753/1753, zero queueing), so
    # tokens/s is set by the load generator and cannot respond to anything.
    # Round one reported "+24.9% throughput" only because it was CLOSED-loop
    # with fixed work, where tokens/s is algebraically 1/latency.
    #
    # Here the offered rate deliberately EXCEEDS what the pair can serve. Once
    # the queue is permanently non-empty, achieved tokens/s IS the service
    # rate, so the hami-vs-hami-mps delta answers the question the client
    # actually has: how much more work fits on the cards they already own.
    #
    # Read ONLY throughput from this tier. End-to-end latency is meaningless
    # here -- the queue grows without bound by design, so p99 measures the
    # backlog, not the system.
    #
    # llm-llm, not llm-embed: the embedder pairing measured null twice (T1, and
    # T3's equivalent), and a null pairing cannot show a capacity gain.
    # Dedicated llm alone already reaches 96% duty at 2.93 req/s, so 5 req/s
    # per tenant is deep saturation for the pair. Duration is cut to 240 s to
    # keep the client-side backlog bounded.
    "T5": dict(scenario="llm-llm", rate=10.0, duration=240, source="sharegpt",
               why="CAPACITY. The only tier whose offered rate exceeds what "
                   "the pair can serve, and therefore the only one where "
                   "achieved tokens/s equals the service rate rather than the "
                   "load generator's setting (FINDINGS 6.1). Answers 'how much "
                   "more work fits on the existing cards'. READ ONLY "
                   "THROUGHPUT: the queue grows unbounded by design, so e2e "
                   "latency here measures backlog, not the system. Prediction: "
                   "MPS delivers a real aggregate throughput gain, because T2 "
                   "showed a 26.7% service-rate increase that the open-loop "
                   "harness could not express as tokens. If throughput is FLAT "
                   "here while T2's latency gain was real, then the T2 result "
                   "is a scheduling reallocation with no capacity behind it, "
                   "and the practical value of MPS is much smaller than T2 "
                   "suggests."),
    # T6 repurposed from llm-embed. Pointing a robustness check at a pairing
    # that measured null twice would have tested nothing.
    "T6": dict(scenario="llm-llm", rate=6.0, duration=300, source="azure",
               why="ROBUSTNESS OF THE HEADLINE. Same two decoders and same "
                   "offered rate as T2, with real production TOKEN-LENGTH "
                   "distributions from Azure's public inference traces. "
                   "Arrivals stay Poisson at the requested rate on purpose -- "
                   "gen_trace.load_azure keeps the (input, output) pairs and "
                   "regenerates the arrival process -- so what differs from T2 "
                   "is the prefill/decode ratio and its variance, not "
                   "burstiness. Question: does the -26.7% survive production "
                   "length distributions, or was it an artifact of ShareGPT's "
                   "particular output-length spread? T2's gain was flat across "
                   "output-length buckets (-38.0% at <=32 tokens to -45.0% "
                   "above 512), which predicts it survives."),
    "T21": dict(scenario="llm2-llm2", rate=40.0, duration=240, source="sharegpt",
                why="A SYMMETRIC PAIR THAT ACTUALLY SATURATES. T13 at 10 req/s "
                    "and T15 at 20 req/s both answered the FAIRNESS question -- "
                    "degradation spread 1.006 to 1.031, so MPS creates no winner "
                    "when in-flight depth is equal -- and neither answered the "
                    "CAPACITY question, because both stayed above 99.5% of their "
                    "solo-sum throughput. T15 was commissioned to saturate and "
                    "failed: doubling the rate roughly doubled delivered "
                    "throughput (1,307 -> 2,589 tok/s), which is what "
                    "not-yet-saturated looks like. "
                    "So double it again. Two Qwen2.5-1.5B-AWQ instances have "
                    "delivered 1,327 tok/s each; 40 req/s demands far more than "
                    "the card can serve, which is the only way to learn whether "
                    "the MPS throughput gain measured on UNLIKE pairs (T5 +6.4%, "
                    "T14 +7.6%) also appears when the tenants are identical. "
                    "Registered prediction: it does, and at a similar magnitude, "
                    "because the gain comes from removing context-switch cost "
                    "rather than from any asymmetry between the tenants. If it "
                    "does NOT appear, then every throughput number in this study "
                    "depends on the pair being unlike, which would be a "
                    "significant and previously unsuspected caveat. "
                    "Watch the host-RAM guard: two 1.5B tenants fit, but this "
                    "tier is the most in-flight requests the node has held."),
    "T20": dict(scenario="llm-llm", rate=6.0, duration=300, source="azure",
                replay_arrivals=True,
                why="THE ARRIVAL PROCESS, FOR ONCE. Every other tier in this "
                    "study -- T6 included -- regenerates arrivals as Poisson "
                    "at a requested rate, so offered load stays a controlled "
                    "variable. That is defensible and it is what the Splitwise "
                    "authors did for cluster sizing, but it means NO tier "
                    "anywhere replays a production arrival process, and the "
                    "findings report has to say so. T20 replays the real "
                    "inter-arrival gaps from a contiguous window of "
                    "AzureLLMInferenceTrace_conv.csv. "
                    "WHAT TO EXPECT, registered before the run: not much, and "
                    "that is the point. The conversation trace measures at an "
                    "inter-arrival CV of 1.057-1.094, which is close to "
                    "Poisson, and no gap in it exceeds ~4 s. So if the T6 "
                    "result holds here, the Poisson substitution used "
                    "everywhere else is VINDICATED for this corpus, and that "
                    "is a stronger statement than any single tier's effect "
                    "size. If it does NOT hold, every tier in the study "
                    "inherits a caveat. Either way the answer is worth one "
                    "run. "
                    "NOTE the rate argument no longer controls offered load: "
                    "real gaps set it, and the achieved rate was 7.41/s "
                    "against 6.00 requested when the trace was generated."),
    "T7": dict(scenario="llm-only", rate=6.0, duration=300, source="sharegpt",
               why="Prefill-heavy, single tenant. Round one's second "
                   "registered prediction: the sub-linear SM-to-throughput "
                   "mapping straightens under prefill-dominated load. Also the "
                   "clean single-tenant control -- the only tier where no "
                   "co-tenancy exists at all."),
    # T8 was REMOVED on 2026-09-03 and REINSTATED on 2026-09-06 with a new job.
    # The old T8 was the round-one control (llm-embed against round one's two
    # decoders) and could not do that job; the reconciliation is algebraic
    # instead (FINDINGS 6.1). The slot now holds the question that matters more.
    "T8": dict(scenario="llm-llm", rate=12.0, duration=300, source="sharegpt",
               mix="llm=0.1,llm2=0.9", mps_percent="llm=50,llm2=50",
               why="IS THE MPS CEILING A USABLE QoS KNOB? T8's first draft was "
                   "a 'matched partition' tier -- gpucores 50 with MPS 50, on "
                   "T5's balanced load, to make the scheduler's model of the "
                   "card true. That framing is WRONG and was withdrawn before "
                   "it ran. Scheduler honesty and oversubscription are the "
                   "same dial, not two things: honest means reservations, "
                   "which means the percentages sum to 100, which means no "
                   "statistical multiplexing and therefore no reason to run "
                   "MPS at all. There is no configuration that is both "
                   "matched AND oversubscribed, so 'should the numbers agree' "
                   "is not a question anyone should be asking. What you can "
                   "choose is an OVERBOOKING RATIO -- and if HAMi's "
                   "deviceCoreScaling is set to N, the coherent pairing is "
                   "MPS at 100 with the scheduler overbooked by the same N, "
                   "never 1:1. "
                   "The question that survives is a QoS one: does an MPS "
                   "ceiling BOUND THE DAMAGE one tenant can do to another? "
                   "That needs a real aggressor, which T5's balanced load "
                   "does not have and T11's 90/10 does. So T8 is T11 with "
                   "both tenants capped at 50%. T11 measured the uncapped "
                   "case: the aggressor ran 8.47 concurrent against the "
                   "victim's 1.51 and took the card, and MPS helped the "
                   "aggressor MORE than the victim (-35.7% against -28.2%). "
                   "Prediction: the cap binds on the AGGRESSOR, whose demand "
                   "exceeds 50%, and not on the victim, whose ~25% share is "
                   "already inside it -- so aggressor throughput falls and "
                   "victim latency improves beyond T11's. If instead nothing "
                   "moves, then CUDA_MPS_ACTIVE_THREAD_PERCENTAGE is as inert "
                   "as nvidia.com/gpucores was, and MPS has NO usable QoS "
                   "knob at all -- which is the single most consequential "
                   "negative result still available in this study."),
    "T11": dict(scenario="llm-llm", rate=12.0, duration=300, source="sharegpt",
                mix="llm=0.1,llm2=0.9",
                why="NOISY NEIGHBOUR. T2 measured fairness under MATCHED load "
                    "and found MPS more than doubles the imbalance (1.05 -> "
                    "2.19) in favour of the larger tenant. Nothing has tested "
                    "ADVERSARIAL load. Here llm2 takes 90% of a 12 req/s trace "
                    "-- about 10.8 req/s against its own knee of roughly 6 -- "
                    "while llm receives 1.2 req/s, well inside its capacity. "
                    "Question: can a hammering small tenant damage a "
                    "well-behaved large one, and does MPS make that better or "
                    "worse? Given T2, the prediction is that MPS PROTECTS the "
                    "large tenant, because it reallocates toward size rather "
                    "than toward demand -- which would make MPS a crude "
                    "isolation mechanism as well as a density one."),
    "T12": dict(scenario="llm-llm", rate=6.0, duration=14400, source="sharegpt",
                why="SOAK, four hours. Every other run is 300 seconds, which "
                    "is long enough for a latency distribution and far too "
                    "short for anything operational. Watching for: KV cache "
                    "fragmentation, memory growth in either tenant, thermal "
                    "steady state versus the 60-90 s ramp seen in every short "
                    "run, clock drift, and whether the MPS daemon survives "
                    "four hours of continuous multiplexing. A tenant that "
                    "degrades at hour three is invisible to the rest of this "
                    "matrix."),
    "T18": dict(scenario="dense3", rate=9.0, duration=300, source="sharegpt",
                why="THREE TENANTS. Every co-tenancy result in this study is a "
                    "PAIR -- the depth law, the fairness reversal and the "
                    "overbook curve are all two-body results, and nothing says "
                    "they extend to three. This is the largest structural gap "
                    "in the matrix. WARNING: it may not run on this node. "
                    "Three tenants need roughly 8.3 GiB of host RAM against "
                    "~8.2 available, and T4 was already refused at four. If "
                    "the host guard refuses this too, THAT IS THE RESULT and "
                    "it should be reported as the density ceiling rather than "
                    "worked around -- host memory, not VRAM, is what caps "
                    "tenants per box. Prediction if it does run: the deepest "
                    "tenant takes the largest share of the MPS gain and the "
                    "shallowest is made worst-off, exactly as in the two-body "
                    "tiers. A three-body run is the first chance for that to "
                    "fail."),
    "T13": dict(scenario="llm2-llm2", rate=10.0, duration=240, source="sharegpt",
                why="SYMMETRIC PAIR. Every co-tenancy tier in this study pairs "
                    "UNLIKE models, so 'does MPS create systematic winners and "
                    "losers' has only ever been answered by argument. Two "
                    "identical Qwen2.5-1.5B-AWQ instances remove the confound: "
                    "same weights, same quota, same kernels, same trace "
                    "weight. FINDINGS 8 established that MPS reallocates "
                    "toward IN-FLIGHT DEPTH rather than model size -- T5's 7B "
                    "ran 11.11 concurrent against the 1.5B's 2.84 and won; "
                    "T11's hammered 1.5B ran 8.47 against 1.51 and won. "
                    "Identical tenants under identical load have identical "
                    "depth, so the prediction is NEAR-SYMMETRIC degradation on "
                    "both arms and an aggregate MPS gain at or above T5's, two "
                    "identical decoders being the most composition-similar "
                    "pair possible. Also yields the Gini coefficient the "
                    "external review asked for, comparable to the published "
                    "0.35-MPS / 0.12-MIG figures."),
    "T14": dict(scenario="llm-llm", rate=10.0, duration=240, source="sharegpt",
                raw_prompts=True,
                why="RAW PROMPTS -- the methodological control for the corpus. "
                    "make_prompt() pads a chat prompt with a fixed FILLER "
                    "sentence to hit a synthetic target length, and truncates "
                    "the ones that overshoot; in T7's trace 1033 of 1749 "
                    "prompts (59.1%) carry it. That CANNOT bias an "
                    "arm-versus-arm comparison, because every arm replays the "
                    "identical trace -- but the prompts are not pure ShareGPT, "
                    "and a reviewer is entitled to ask whether the effect "
                    "survives real text. This tier is T5 with --raw-prompts: "
                    "corpus prompts verbatim, no padding, no truncation, so "
                    "the prefill length distribution is whatever ShareGPT "
                    "actually contains. Absolute numbers will differ from T5 "
                    "and are not meant to match. The claim under test is only "
                    "that the MPS EFFECT reproduces in sign and rough "
                    "magnitude. If it does, the filler is retired as an "
                    "objection; if it does not, every length-sensitive "
                    "conclusion in the study needs revisiting."),
    "T15": dict(scenario="llm2-llm2", rate=20.0, duration=240, source="sharegpt",
                why="SATURATED SYMMETRIC PAIR. T13 answered fairness and could "
                    "not answer capacity: its hami arm reached 99.9% of the sum "
                    "of the two solo throughputs, so the card was never "
                    "compute-starved and MPS had no deficit to recover "
                    "(-0.19% throughput, -22.98% pooled). Only T5 in this "
                    "entire study is genuinely saturated, at 92.3%, which "
                    "means every capacity claim we make rests on ONE tier. "
                    "T15 doubles T13's offered rate to force saturation while "
                    "keeping the tenants identical. Prediction: a throughput "
                    "gain appears where T13 had none, and the degradation "
                    "spread STAYS near 1.0 -- because 8.9 says asymmetry comes "
                    "from unequal in-flight depth, and identical tenants under "
                    "identical load have identical depth no matter how hard "
                    "they are driven. If the spread opens up under saturation, "
                    "the depth law is incomplete."),
    "T16": dict(scenario="llm-llm", rate=6.0, duration=300, source="sharegpt",
                gpucores="llm=25,llm2=25",
                why="THE LAST EMPTY CELL IN THE POLICY TABLE. 5.14b's semantic "
                    "-- force limits unconditionally, default limits only "
                    "under contention -- has never been tested where it "
                    "actually differs. Every co-tenancy tier ran quotas at or "
                    "ABOVE the natural share (100//N), where default cannot "
                    "engage; T2-20260906-102419 went below at 25/25 but used "
                    "FORCE. This is that run with the policy left at default, "
                    "same trace, so the only variable is the policy. If the "
                    "7B's throughput falls toward the 298.37 tok/s that force "
                    "produced, then DEFAULT ALSO CLAMPS whenever quotas are "
                    "conservative -- and 'the default is safe' is wrong for "
                    "most real clusters, which set quotas below natural share "
                    "precisely to pack more tenants. If it stays near 429, "
                    "default is confirmed safe and the distinction is real. "
                    "Read the THROUGHPUT column: the noise floor there is "
                    "0.42% against 9.51% for latency (8.10)."),
}

ARMS = ["dedicated", "hami", "hami-mps"]


def run(cmd, cwd=HERE, timeout=3600, echo=True):
    if echo:
        print("    $ " + " ".join(str(c) for c in cmd))
    p = subprocess.run([str(c) for c in cmd], cwd=cwd, capture_output=True,
                       text=True, timeout=timeout)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def gen_trace(test, cfg, out_path, seed, warmup):
    """One trace, generated once, replayed by everything."""
    cmd = [PY, os.path.join("workload", "gen_trace.py"),
           "--scenario", cfg["scenario"], "--rate", cfg["rate"],
           "--duration", cfg["duration"], "--source", cfg["source"],
           "--seed", seed, "--warmup", warmup, "--out", out_path]
    if cfg.get("fixed_output"):
        cmd += ["--fixed-output", cfg["fixed_output"]]
    if cfg.get("mix"):
        cmd += ["--mix", cfg["mix"]]
    if cfg.get("raw_prompts"):
        cmd += ["--raw-prompts"]
    # T20 replays the Azure trace's real inter-arrival gaps instead of
    # regenerating them. gen_trace.py refuses rather than falling back to
    # Poisson if the CSV is missing, which is the whole point of the flag.
    if cfg.get("replay_arrivals"):
        cmd += ["--replay-arrivals"]
    rc, out = run(cmd)
    if rc != 0:
        raise RuntimeError("gen_trace failed:\n" + out[-3000:])
    doc = H.read_json(out_path, {})
    tid = doc.get("trace_id") or doc.get("meta", {}).get("trace_id")
    n = len(doc.get("requests", []))

    # gen_trace falls back to synthetic when a corpus is missing, and says so
    # in one warning line that is easy to miss inside a long run. T6 exists
    # ONLY to replay production arrival shapes -- a silent fallback would leave
    # it measuring exactly what T1 measures while the report claims otherwise.
    got = (doc.get("params") or {}).get("source")
    if got and got != cfg["source"]:
        raise RuntimeError(
            "test %s asked for source=%s but the trace resolved to %s.\n"
            "The corpus is not staged. Fix that rather than running a test "
            "that would quietly measure something else:\n"
            "    ./07-prefetch-traces.sh" % (test, cfg["source"], got))
    print("    trace_id=%s  requests=%d  source=%s" % (tid, n, got))
    return tid, n


# Failure text that means "the node cannot run this, and will not be able to
# run the next one either". Everything here was observed on this cluster.
_NODE_LEVEL = (
    "node unregistered",          # HAMi's scheduler extender lost the node
    "nodes are available",        # generic FailedScheduling preamble
    "no available node",          # HAMi FilteringFailed
    "Insufficient nvidia.com",    # device plugin not advertising
    "node(s) had untolerated taint",
)


def _abort_reason(rec):
    """Is this sub-run's failure one that will simply repeat?"""
    err = rec.get("error") or ""
    if "host RAM:" in err:
        # The guard refused this arm. A wider arm will be refused too, but a
        # NARROWER one (the dedicated arm, one tenant at a time) still fits,
        # so this is not on its own a reason to stop the run.
        return None
    blob = err + "\n" + "\n".join(
        (rec.get("ready_diag") or {}).values())
    for needle in _NODE_LEVEL:
        if needle in blob:
            return "node cannot schedule GPU pods (%r). Run ./25-node-check.sh --fix" % needle
    return None


def subrun(run_dir, ev_global, test, cfg, arm, roles, trace_path, trace_id,
           pin_uuid, pooling_via, mps_percent, warmup, dry, gpucores="",
           core_policy=""):
    """Deploy, measure, tear down -- one arm, one set of co-resident roles."""
    tag = "%s__%s" % (arm, "+".join(roles))
    d = os.path.join(run_dir, tag)
    os.makedirs(d, exist_ok=True)
    ev = H.Events(os.path.join(d, "events.jsonl"))
    rec = dict(tag=tag, arm=arm, roles=roles, trace_id=trace_id,
               scenario=cfg["scenario"], test=test, status="started")

    print("\n--- %s" % tag)
    hsampler = None
    try:
        # 1. clean slate, verified rather than assumed --------------------
        # Skipped under --dry-run: both of these shell out to the cluster, and
        # a dry run that needs a cluster is not a dry run.
        if not dry:
            ev.mark("teardown_begin")
            H.teardown()
            H.wait_gpu_idle(ev=ev)

            # 1b. MPS state, set BEFORE any pod starts ---------------------
            # Attachment happens at CUDA init, so this cannot be done later.
            # And it has to be the daemon, not the pod spec: a container with
            # no MPS mount and no MPS env still attached in T1 and T2, because
            # the container runtime and HAMi's device plugin can inject the
            # pipe after the spec is written. Only the daemon's absence is a
            # real control.
            ok, how = H.set_mps_daemon(arm == "hami-mps", ev=ev)
            if not ok:
                raise RuntimeError(how)
            print("      mps daemon: %s (%s)"
                  % ("up" if arm == "hami-mps" else "down", how))

        # 2. render -------------------------------------------------------
        rdir = os.path.join(d, "rendered")
        cmd = [PY, "render.py", "--scenario", cfg["scenario"], "--arm", arm,
               "--out", rdir, "--pooling-via", pooling_via]
        for r in roles:
            cmd += ["--only", r]
        if pin_uuid:
            cmd += ["--pin-gpu", pin_uuid]
        if gpucores:
            cmd += ["--gpucores", gpucores]
        if core_policy:
            cmd += ["--core-policy", core_policy]
        if mps_percent:
            cmd += ["--mps-percent", mps_percent]
        rc, out = run(cmd)
        if rc != 0:
            raise RuntimeError("render failed:\n" + out[-2000:])
        rec["render"] = out.strip().splitlines()[-4:]

        if dry:
            rec["status"] = "dry-run"
            return rec

        # 3. apply and wait ----------------------------------------------
        # Host RAM is checked here, not in preconditions, because it depends on
        # how many tenants THIS arm deploys: the dedicated arm runs one at a
        # time and always fits, the shared arm runs all of them at once and is
        # where T4 was killed.
        hprob, hdetail = H.host_memory_problems(roles, ev=ev)
        rec["host_memory"] = hdetail
        if hprob:
            raise RuntimeError("; ".join(hprob))
        print("      host ram: need ~%d MiB, %d MiB available%s"
              % (hdetail.get("need_mib", 0), hdetail.get("available_mib", 0),
                 "" if hdetail.get("measured") else " (per-role RSS unmeasured)"))

        # Sampling starts BEFORE the pods do. The host memory peak is during
        # model load, not during replay.
        hsampler = H.HostSampler(os.path.join(d, "host.csv"))
        hsampler.start()

        ev.mark("apply")
        rc, out = H.kubectl(["apply", "-f", rdir], timeout=180)
        if rc != 0:
            raise RuntimeError("kubectl apply failed:\n" + out[-2000:])

        ev.mark("wait_ready", roles=",".join(roles))
        ok, diag = H.wait_ready(roles, ev=ev)
        if not ok:
            for role, text in diag.items():
                print("      %s did not become ready:\n%s" % (role, text[:2500]))
            rec["ready_diag"] = {k: v[:4000] for k, v in diag.items()}
            raise RuntimeError("roles never became ready: %s" % ", ".join(diag))

        # 4. placement -- the check that protects the whole comparison ----
        colocated, detail = H.assert_colocated(roles, ev=ev)
        rec["placement"] = detail
        if not colocated:
            raise RuntimeError(
                "tenants landed on DIFFERENT GPUs: %s\nThis arm would not "
                "actually be sharing anything. Run ./10-hami-binpack.sh, or set "
                "the HAMi scheduler's gpuSchedulerPolicy to binpack."
                % json.dumps(detail))
        if detail.get("warning"):
            print("      NOTE: " + detail["warning"])

        # 4b. MPS state -- the check that protects the OTHER comparison ---
        # assert_colocated protects dedicated-vs-shared. This protects
        # hami-vs-hami-mps, and it is the one whose absence cost T1 and T2:
        # every arm ran as an MPS client, so the two sharing arms were the
        # same configuration and returned +0.0% and -0.008%.
        mps_ok, mps_detail = H.assert_mps_matches_arm(arm, roles, ev=ev)
        rec["mps"] = mps_detail
        print("      mps check: %s  daemon=%s clients=%s"
              % ("OK" if mps_ok else "FAILED",
                 mps_detail["daemon_running"], mps_detail["mps_client_pids"]))
        if not mps_ok:
            raise RuntimeError(mps_detail.get("reason", "MPS state mismatch"))

        # 5. endpoints and model names -----------------------------------
        eps = H.endpoints(roles, ev=ev)
        models = {}
        for role, base in eps.items():
            m = H.served_model(base)
            if m:
                models[role] = m
        rec["endpoints"], rec["models"] = eps, models

        H.write_json(os.path.join(d, "pods.json"),
                     {r: H.pods_for(r) for r in roles})

        # 6. the measured run --------------------------------------------
        # Warm-up is not a separate pass: the trace already carries warmup
        # rows (t=-1, warmup=True), replay fires them before the clock rows,
        # and summarize() excludes them. That matters because the first
        # request into a cold vLLM pays CUDA graph capture and torch.compile;
        # inside the measured window it lands in p99 and gets reported as a
        # sharing cost, which it is not.
        # A tenant that cannot serve should cost 30 seconds, not 45 minutes.
        # T1's first attempt ran four arms to completion with 0/885 successful
        # LLM requests: the pods were Ready, placement was verified, the trace
        # matched across arms, and every generative request failed. Readiness
        # only proves /health answers -- it says nothing about whether the
        # endpoint accepts the request shape the client is about to send.
        import render as _R
        _models = {r: _R.MODELS[r]["repo"] for r in roles}
        _kinds = {r: _R.MODELS[r]["kind"] for r in roles}
        smoke_ok, smoke_detail = H.smoke_test(eps, _models, _kinds, ev=ev)
        rec["smoke"] = smoke_detail
        if not smoke_ok:
            lines = "\n  ".join("%s: %s" % kv for kv in sorted(smoke_detail.items()))
            raise RuntimeError("smoke test failed, refusing to spend a "
                               "measurement window on it:\n  " + lines)

        sampler = H.GpuSampler(os.path.join(d, "gpu.csv"))
        sampler.start()
        t0 = ev.mark("replay_begin")
        rc, out = run([PY, os.path.join("workload", "replay.py"),
                       "--trace", trace_path, "--arm", arm, "--out-dir", d,
                       "--note", "%s %s" % (test, tag),
                       # replay.py defaults to 512 worker threads. Past that the
                       # client stops issuing at the trace's rate and the wait
                       # lands in client_delay, which latency_e2e EXCLUDES -- so
                       # a client-throttled arm reports better latency than it
                       # delivered. Worse, the pool is SHARED across tenants, so
                       # the slower one holds more workers and throttles its own
                       # offered load: a feedback loop that artificially
                       # equalises two tenants, which is exactly what a symmetry
                       # tier is trying to measure. Highest peak ever recorded on
                       # disk is 291 (T5 hami), so 4096 cannot bind at any rate
                       # in this matrix.
                       "--max-inflight", "4096"]
                      + sum([["--only", r] for r in roles], [])
                      + sum([["--endpoint", "%s=%s" % (r, u)]
                             for r, u in eps.items()], [])
                      + sum([["--model", "%s=%s" % (r, m)]
                             for r, m in models.items()], []),
                      timeout=int(cfg["duration"]) + 1800)
        t1 = ev.mark("replay_end", rc=rc)
        sampler.stop()
        rec["replay_rc"] = rc
        rec["replay_tail"] = out.strip().splitlines()[-25:]
        if rc != 0:
            print("      replay exited %d:\n%s" % (rc, out[-2500:]))

        # 7. server-side metrics, saved as data not as a dashboard link ---
        H.prom_collect(os.path.join(d, "prom.json"), t0, t1, ev=ev)

        if hsampler is not None:
            rec["host_min_available_mib"] = hsampler.min_available

        summ = H.read_json(os.path.join(d, "replay.summary.json"), {})
        rec["summary"] = summ
        got = summ.get("trace_id") or summ.get("meta", {}).get("trace_id")
        rec["trace_id_seen"] = got
        rec["status"] = "ok" if rc == 0 else "replay-failed"
        rec["t0"], rec["t1"] = t0, t1

    except Exception as exc:
        rec["status"] = "error"
        rec["error"] = "%s: %s" % (type(exc).__name__, exc)
        print("    ERROR %s" % rec["error"])
    finally:
        if hsampler is not None:
            hsampler.stop()
            rec.setdefault("host_min_available_mib", hsampler.min_available)
        if not dry:
            # teardown_end used to be marked BEFORE the teardown ran, so a
            # teardown that timed out looked instantaneous in the event log.
            # T4's hami arm "ended" at 13:39:28 and the process did not exit
            # until 13:43:58, with the pods still up.
            waited = H.teardown(quiet=True)
            rec["teardown_s"] = round(waited, 1)
            rec["teardown_left"] = list(getattr(H.teardown, "pods_left", []))
            ev.mark("teardown_end", waited_s=round(waited, 1),
                    left=len(rec["teardown_left"]))
        else:
            ev.mark("teardown_end")
        ev.close()
        H.write_json(os.path.join(d, "subrun.json"), rec)
    return rec


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test", help="T1..T8; see --list")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    ap.add_argument("--seed", type=int, default=20260826)
    # Default OFF. The annotation is inert on the KServe path -- KServe drops
    # pod annotations -- so pinning does nothing there, and co-location now
    # comes from the scheduler (10-hami-binpack.sh, verified co-locating both
    # T1 tenants on one card). But it DOES land on the plain-Deployment path,
    # so a mixed render would pin the Deployment tenants to this UUID while
    # binpack sent the KServe tenants wherever it liked -- a guaranteed split,
    # caused by the very flag meant to prevent one.
    #
    # Set it only when EVERY tenant renders as a Deployment (--pooling-via vllm
    # on an all-pooling scenario), where it is then the only mechanism.
    ap.add_argument("--pin-gpu", default="",
                    help="pin every tenant to this device UUID. Leave empty: "
                         "KServe ignores it and binpack handles co-location. "
                         "Only useful when every tenant is a plain Deployment.")
    ap.add_argument("--pooling-via", choices=("kserve", "vllm"), default="kserve")
    ap.add_argument("--core-policy", default="",
                    choices=("", "default", "force", "disable"),
                    help="GPU_CORE_UTILIZATION_POLICY on the HAMi arms. Round "
                         "one ran 'force'; every round-two tier so far ran "
                         "nothing. Pair with --gpucores to test whether the "
                         "limiter binds: T7 is one tenant, so any throughput "
                         "delta is the limiter and not a neighbour.")
    ap.add_argument("--gpucores", default="",
                    help="override gpucores per role, e.g. llm=25. Exists to "
                         "test ENFORCEMENT: run T7 (one tenant, no neighbour) "
                         "at 25/50/100 and any throughput difference is HAMi's "
                         "rate limiter, with contention held out of it.")
    ap.add_argument("--mps-percent", default="",
                    help="per-role CUDA_MPS_ACTIVE_THREAD_PERCENTAGE, e.g. "
                         "llm=75,embed=25. Default 100 for every role, which "
                         "isolates 'MPS on/off' from 'MPS partitioned'.")
    ap.add_argument("--warmup", type=int, default=8,
                    help="warm-up requests per role, carried IN the trace and "
                         "excluded from the summary. Do not set this to 0: the "
                         "first request into a cold engine pays graph capture "
                         "and would be scored as a sharing cost.")
    ap.add_argument("--duration", type=float, help="override the test's duration")
    ap.add_argument("--rate", type=float, help="override the test's offered rate")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--skip-preconditions", action="store_true")
    a = ap.parse_args()

    if a.list or not a.test:
        print("%-5s %-12s %6s %6s %-10s  %s"
              % ("test", "scenario", "rate", "dur", "source", "roles"))
        for k in sorted(TESTS):
            c = TESTS[k]
            print("%-5s %-12s %6.1f %6d %-10s  %s"
                  % (k, c["scenario"], c["rate"], c["duration"], c["source"],
                     ", ".join(ROLES[c["scenario"]])))
            for line in _wrap(c["why"], 72):
                print("      %s" % line)
        print("\nT9 (admission / scale-down) lives in 09-scaledown.py -- its "
              "shape is different enough that folding it in here would make "
              "both harder to read.")
        return 0

    if a.test not in TESTS:
        sys.exit("unknown test %s; try --list" % a.test)
    cfg = dict(TESTS[a.test])
    if a.duration:
        cfg["duration"] = a.duration
    if a.rate:
        cfg["rate"] = a.rate

    # A tier may pin gpucores / mps-percent / core-policy as part of its
    # DEFINITION rather than leaving them to the caller -- T8 is a partition
    # test, so "both knobs at 50" is the tier, not an option on it. An explicit
    # flag still wins, so the tier stays sweepable from the command line.
    for key, flag in (("gpucores", "gpucores"),
                      ("mps_percent", "mps_percent"),
                      ("core_policy", "core_policy")):
        if cfg.get(key) and not getattr(a, flag):
            setattr(a, flag, cfg[key])
            print("    %s from the tier definition: %s" % (flag, cfg[key]))
    arms = [x.strip() for x in a.arms.split(",") if x.strip()]
    bad = [x for x in arms if x not in ARMS]
    if bad:
        sys.exit("unknown arm(s): %s" % bad)
    roles = ROLES[cfg["scenario"]]

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(a.out, "%s-%s" % (a.test, stamp))
    os.makedirs(run_dir, exist_ok=True)
    ev = H.Events(os.path.join(run_dir, "events.jsonl"))

    print("=== %s  %s  %s" % (a.test, cfg["scenario"], run_dir))
    print("    %s" % "\n    ".join(_wrap(cfg["why"], 74)))
    print()

    if not a.dry_run and not a.skip_preconditions:
        problems = H.preconditions(ev)
        if problems:
            print("REFUSING TO START:")
            for p in problems:
                print("  - %s" % p)
            print("\nA three-arm run is roughly an hour. Losing it at minute 40 "
                  "to a node-wide eviction is worse than not starting.\n"
                  "Override with --skip-preconditions if you disagree.")
            return 2

    trace_path = os.path.join(run_dir, "trace.json")
    trace_id, n_req = gen_trace(a.test, cfg, trace_path, a.seed, a.warmup)
    ev.mark("trace", trace_id=trace_id, requests=n_req)

    plan = []
    for arm in arms:
        if arm == "dedicated":
            # requirement 2: one baseline PER TENANT, each alone on the card
            plan += [(arm, [r]) for r in roles]
        else:
            plan.append((arm, roles))

    print("\nplan: %d sub-run(s)" % len(plan))
    for arm, rs in plan:
        print("  %-10s %s" % (arm, "+".join(rs)))

    results = []
    aborted = None
    try:
        consecutive = 0
        for arm, rs in plan:
            if aborted:
                print("\n--- %s__%s  SKIPPED (%s)"
                      % (arm, "+".join(rs), aborted))
                continue
            r = subrun(run_dir, ev, a.test, cfg, arm, rs, trace_path,
                       trace_id, a.pin_gpu, a.pooling_via,
                       a.mps_percent, a.warmup, a.dry_run, a.gpucores,
                       a.core_policy)
            results.append(r)

            if r.get("status") == "ok":
                consecutive = 0
                continue
            consecutive += 1

            # A node that cannot place a GPU pod will not place the next one
            # either. After T4's OOM cascade the node came back Ready with a
            # clean nvidia-smi but HAMi had lost its registration, so every
            # pod sat Pending -- and this loop was about to spend four times
            # 900 seconds discovering that four times. Node-level faults abort
            # at once; anything else gets one more chance, because a single
            # arm can fail for reasons the next arm does not share.
            why = _abort_reason(r)
            if why:
                aborted = why
                print("\n!!! ABORTING the remaining sub-runs: %s" % why)
                ev.mark("aborted", reason=why, after_tag=r.get("tag"))
            elif consecutive >= 2:
                aborted = "two consecutive sub-run failures"
                print("\n!!! ABORTING the remaining sub-runs: %s" % aborted)
                ev.mark("aborted", reason=aborted, after_tag=r.get("tag"))
    finally:
        # Leave the daemon up however the run ended. 13-validate.py requires
        # it, and the next run sets it per arm anyway -- but a series that
        # aborts mid-dedicated would otherwise leave the node in a state its
        # own gate rejects.
        if not a.dry_run:
            H.set_mps_daemon(True, ev=ev)

    # ---- the fairness assertion, made loudly ---------------------------
    seen = {r["tag"]: r.get("trace_id_seen") for r in results
            if r.get("status") == "ok"}
    mismatched = {k: v for k, v in seen.items() if v and v != trace_id}
    doc = dict(test=a.test, config=cfg, roles=roles, arms=arms,
               run_dir=run_dir, stamp=stamp, seed=a.seed,
               trace_id=trace_id, requests=n_req,
               pin_gpu=a.pin_gpu, pooling_via=a.pooling_via,
               mps_percent=a.mps_percent, warmup=a.warmup, gpucores=a.gpucores,
               core_policy=a.core_policy,
               subruns=results, trace_id_mismatch=mismatched,
               aborted=aborted, planned_subruns=len(plan))
    H.write_json(os.path.join(run_dir, "run.json"), doc)

    print("\n=== summary")
    for r in results:
        s = r.get("summary") or {}
        con = s.get("contended") or {}
        bits = []
        if s.get("requests_ok") is not None:
            bits.append("%d/%d ok" % (s["requests_ok"], s.get("requests_total", 0)))
        if con.get("aggregate_tokens_per_second"):
            bits.append("%.1f tok/s agg" % con["aggregate_tokens_per_second"])
        if con.get("ratio"):
            bits.append("A:B %.3f" % con["ratio"])
        print("  %-28s %-14s %s" % (r["tag"], r["status"], "  ".join(bits)))
    if mismatched:
        print("\n  FAIRNESS VIOLATION: these arms replayed a different trace:")
        for k, v in mismatched.items():
            print("    %s saw %s, expected %s" % (k, v, trace_id))
        print("  The comparison between arms is NOT valid. Do not report it.")
    bad = [r for r in results if r["status"] not in ("ok", "dry-run")]
    print("\n  %d/%d sub-runs ok" % (len(results) - len(bad), len(results)))
    print("  results: %s" % run_dir)
    print("  render artifacts with: python render_results.py %s" % run_dir)
    ev.close()
    return 1 if (bad or mismatched) else 0


def _wrap(text, width):
    words, line, out = text.split(), "", []
    for w in words:
        if len(line) + len(w) + 1 > width:
            out.append(line)
            line = w
        else:
            line = (line + " " + w).strip()
    if line:
        out.append(line)
    return out


if __name__ == "__main__":
    sys.exit(main())
