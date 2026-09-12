#!/usr/bin/env python3
"""Generate ONE workload trace, replayed identically by every arm.

Requirement 7 -- dedicated, HAMi and HAMi+MPS must see the same inputs -- is met
by making the workload a FILE rather than a live generator. The same trace is
replayed in each arm, so prompt text, arrival times, target output lengths and
request ordering are bit-identical across arms. Nothing is drawn at run time.

The arrival process is Poisson: inter-arrival gaps are exponential with mean
1/rate. That is what a real serving front end sees, and it is the thing a fixed
concurrency loop cannot reproduce -- a closed loop cannot build a queue, so it
can never show the latency blow-up that sharing actually causes.

SOURCES
  sharegpt   real multi-turn chat prompts (vLLM's own benchmark corpus)
  azure      real production trace: input/output token-count pairs
  synthetic  offline fallback, length distribution shaped like sharegpt

Usage
  python gen_trace.py --scenario llm-embed --rate 6 --duration 300 --out t.json
  python gen_trace.py --list-scenarios
"""
import argparse
import hashlib
import json
import os
import random
import sys

SHAREGPT = "anon8231489123/ShareGPT_Vicuna_unfiltered"
SHAREGPT_FILE = "ShareGPT_V3_unfiltered_cleaned_split.json"

# Scenario = which services receive traffic and in what proportion. Each entry
# is (service_role, share_of_requests, kind). The role names are what the
# deployment manifests and the client use to address a service.
SCENARIOS = {
    "llm-only":    [("llm", 1.0, "chat")],
    "embed-only":  [("embed", 1.0, "embed")],
    "rerank-only": [("rerank", 1.0, "rerank")],
    # Single-tenant scenarios for every role, so 15-calibrate.py can drive one
    # model at a known rate with no weighting in between.
    "llm2-only":   [("llm2", 1.0, "chat")],
    "vl-only":     [("vl", 1.0, "chat")],
    "embed-small-only": [("embed-small", 1.0, "embed")],
    "llm-embed":   [("llm", 0.5, "chat"), ("embed", 0.5, "embed")],
    "llm-llm":     [("llm", 0.5, "chat"), ("llm2", 0.5, "chat")],
    # SYMMETRIC PAIR. Identical model on both sides, so any asymmetry in
    # the result is the mechanism and not the tenants. Every other
    # co-tenancy scenario pairs unlike models, which is why "does MPS
    # create winners and losers" was answerable only by argument.
    "llm2-llm2":   [("llm2a", 0.5, "chat"), ("llm2b", 0.5, "chat")],
    "dense3b":     [("llm2a", 0.4, "chat"), ("llm2b", 0.4, "chat"),
                    ("embed-small", 0.2, "embed")],
    "dense3":      [("llm2", 0.4, "chat"), ("embed-small", 0.3, "embed"),
                    ("embed", 0.3, "embed")],
    "vl-rerank":   [("vl", 0.5, "chat"), ("rerank", 0.5, "rerank")],
    # Must match render.py's SCENARIOS exactly: a trace naming a service that
    # was never deployed fails at replay, after the deploy-and-warm cost.
    "dense4":      [("llm2", 0.3, "chat"), ("embed", 0.3, "embed"),
                    ("embed-small", 0.2, "embed"), ("rerank", 0.2, "rerank")],
}

FILLER = ("Explain the trade-off, list the failure modes, and give a concrete "
          "example drawn from a production system. Be specific about the "
          "mechanism rather than the symptom. ")


def load_sharegpt(limit, rng):
    """Real chat prompts. Cached on disk, so every arm reads the same corpus."""
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(SHAREGPT, SHAREGPT_FILE, repo_type="dataset")
    except Exception as exc:
        print("[warn] sharegpt unavailable (%s), falling back to synthetic"
              % exc.__class__.__name__, file=sys.stderr)
        return None
    with open(path, encoding="utf-8") as fh:
        convs = json.load(fh)
    out = []
    for c in convs:
        turns = c.get("conversations") or []
        if len(turns) < 2:
            continue
        prompt = (turns[0].get("value") or "").strip()
        answer = (turns[1].get("value") or "").strip()
        # Drop degenerate rows, but keep the long tail -- it is the point.
        if 16 <= len(prompt) <= 12000 and answer:
            out.append((prompt, max(16, min(1024, len(answer) // 4))))
        if len(out) >= limit * 4:
            break
    rng.shuffle(out)
    return out[:limit] if out else None


def load_azure(limit, rng):
    """Production trace: real (input, output) token-count pairs.

    AzurePublicDataset ships CSVs from live LLM inference services. We take the
    token PAIRS only -- the arrival process is regenerated as Poisson at the
    requested rate, so offered load stays a controlled variable rather than
    whatever that particular day happened to look like.
    """
    import csv
    for env in ("AZURE_TRACE_CSV", "R2_AZURE_TRACE"):
        p = os.environ.get(env)
        if not p or not os.path.exists(p):
            continue
        rows = []
        with open(p, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                try:
                    ctx = int(float(r.get("ContextTokens") or r.get("input_tokens")))
                    gen = int(float(r.get("GeneratedTokens") or r.get("output_tokens")))
                except (TypeError, ValueError):
                    continue
                if 8 <= ctx <= 16000 and 1 <= gen <= 4096:
                    rows.append((ctx, gen))
        if rows:
            rng.shuffle(rows)
            return rows[:limit]
    print("[warn] no Azure trace CSV (set AZURE_TRACE_CSV); using synthetic lengths",
          file=sys.stderr)
    return None


def load_azure_arrivals(duration, rng):
    """Real (offset, ctx, gen) triples from a contiguous window of the CSV.

    Unlike load_azure this keeps TIMESTAMP. A contiguous window is taken rather
    than a shuffled sample, because the whole point is to preserve the shape of
    the arrival process -- shuffling would destroy exactly what is being
    measured. The window start is chosen at random among the windows that
    actually contain the requested span.
    """
    import csv, datetime
    for env in ("AZURE_TRACE_CSV", "R2_AZURE_TRACE"):
        path = os.environ.get(env)
        if not path or not os.path.exists(path):
            continue
        rows = []
        with open(path, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                ts = (r.get("TIMESTAMP") or "").strip()
                try:
                    ctx = int(float(r.get("ContextTokens")))
                    gen = int(float(r.get("GeneratedTokens")))
                except (TypeError, ValueError):
                    continue
                if not (8 <= ctx <= 16000 and 1 <= gen <= 4096) or not ts:
                    continue
                # .NET ticks: 7 fractional digits, more precision than
                # datetime accepts. Truncate to microseconds.
                if "." in ts:
                    head, frac = ts.split(".", 1)
                    ts = head + "." + frac[:6]
                try:
                    when = datetime.datetime.strptime(ts, "%Y-%m-%d %H:%M:%S.%f")
                except ValueError:
                    try:
                        when = datetime.datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
                    except ValueError:
                        continue
                rows.append((when, ctx, gen))
        if len(rows) < 16:
            continue
        rows.sort(key=lambda x: x[0])
        span = (rows[-1][0] - rows[0][0]).total_seconds()
        if span < duration:
            print("[warn] azure trace spans %.0f s, shorter than the requested "
                  "%.0f s; using the whole file" % (span, duration),
                  file=sys.stderr)
            start = rows[0][0]
        else:
            latest = rows[0][0] + datetime.timedelta(
                seconds=(span - duration))
            offset = rng.uniform(0.0, (latest - rows[0][0]).total_seconds())
            start = rows[0][0] + datetime.timedelta(seconds=offset)
        out = []
        for when, ctx, gen in rows:
            dt = (when - start).total_seconds()
            if 0.0 <= dt < duration:
                out.append((dt, ctx, gen))
        if out:
            gaps = [out[i][0] - out[i - 1][0] for i in range(1, len(out))]
            if gaps:
                mean = sum(gaps) / len(gaps)
                var = sum((g - mean) ** 2 for g in gaps) / len(gaps)
                cv = (var ** 0.5 / mean) if mean else 0.0
                print("azure arrivals  n=%d  span=%.0fs  mean gap=%.3fs  CV=%.3f"
                      % (len(out), duration, mean, cv))
            return out
    print("[warn] no Azure trace CSV for --replay-arrivals "
          "(set AZURE_TRACE_CSV)", file=sys.stderr)
    return None


def synth_lengths(rng):
    """Lognormal in, lognormal out -- shaped like the sharegpt corpus."""
    ctx = int(min(8192, max(16, rng.lognormvariate(5.5, 1.05))))
    gen = int(min(1024, max(8, rng.lognormvariate(4.7, 0.85))))
    return ctx, gen


def make_prompt(ctx_tokens, pool, idx, raw=False):
    """A prompt of roughly ctx_tokens tokens, from the real pool where possible.

    With raw=True the corpus prompt is used VERBATIM: no FILLER padding and
    no truncation, so ctx_tokens is ignored and the length distribution is
    whatever ShareGPT actually contains.

    This exists because the default path pads 59.1% of chat prompts (T7,
    measured) and truncates the ones that overshoot. Neither can bias an
    arm-versus-arm comparison, since every arm replays the identical
    trace -- but it does mean the prompts are not pure ShareGPT, and a
    reviewer is entitled to ask whether the effect survives real text.
    """
    base = pool[idx % len(pool)][0] if pool else FILLER
    if raw:
        return base
    want = ctx_tokens * 4          # ~4 chars/token for English
    if len(base) >= want:
        return base[:want]
    pad = FILLER * (1 + (want - len(base)) // len(FILLER))
    return (base + " " + pad)[:want]


def build(a, rng, pool, lengths):
    mix = SCENARIOS[a.scenario]
    roles = [r for r, _, _ in mix]
    weights = [s for _, s, _ in mix]

    # Per-role share override. A scenario's weights describe a balanced mix;
    # the noisy-neighbour tier needs a deliberately unbalanced one, and
    # inventing a whole scenario for that would duplicate the role list and
    # invite it to drift out of step with render.py.
    if getattr(a, "mix", ""):
        over = {}
        for kv in filter(None, a.mix.split(",")):
            k, v = kv.split("=")
            over[k.strip()] = float(v)
        unknown = [k for k in over if k not in roles]
        if unknown:
            sys.exit("--mix names role(s) %s not in scenario %s (%s)"
                     % (unknown, a.scenario, roles))
        weights = [over.get(r, w) for r, w in zip(roles, weights)]
        print("mix override  %s"
              % ", ".join("%s=%.3g" % (r, w) for r, w in zip(roles, weights)))
    kind_of = dict((r, k) for r, _, k in mix)

    reqs = []
    # Warm-up first, stamped t < 0 so it is unambiguous which rows are excluded.
    for role in roles:
        for i in range(a.warmup):
            ctx, _ = synth_lengths(rng)
            reqs.append(dict(t=-1.0, service=role, kind=kind_of[role], warmup=True,
                             prompt=make_prompt(min(ctx, 256), pool, i),
                             inputs=[make_prompt(64, pool, i + j) for j in range(4)],
                             query=make_prompt(24, pool, i),
                             documents=[make_prompt(64, pool, i + j) for j in range(4)],
                             max_tokens=32))

    # Real production inter-arrival gaps, when asked for. The trace carries
    # no service identity, so roles are still drawn from the scenario weights;
    # what is replayed is WHEN requests arrive and how big they are, which is
    # exactly what is claimed for it and no more.
    schedule = getattr(a, "_arrivals", None)

    t, idx = 0.0, 0
    while True:
        if schedule is not None:
            if idx >= len(schedule):
                break
            t, _ctx, _gen = schedule[idx]
        else:
            t += rng.expovariate(a.rate)     # Poisson arrivals
            if t >= a.duration:
                break
        role = rng.choices(roles, weights=weights, k=1)[0]
        kind = kind_of[role]
        if schedule is not None:
            ctx, gen = _ctx, _gen
        elif lengths:
            ctx, gen = lengths[idx % len(lengths)]
        else:
            ctx, gen = synth_lengths(rng)
        # SHAREGPT SUPPLIES TEXT, NOT LENGTHS. load_sharegpt fills `pool` and
        # leaves `lengths` None, so a "sharegpt" run takes its prompt-length
        # target and its output length from the synthetic lognormals above and
        # uses the corpus only for prompt TEXT, padded or truncated to fit.
        # load_sharegpt does compute a real answer length -- pool[i][1] -- and
        # nothing has ever read it. In raw mode we do: the whole point of the
        # tier is a request shaped like the corpus rather than like our
        # lognormal, and that means the output length too, not just the prompt.
        if getattr(a, "raw_prompts", False) and pool:
            gen = pool[idx % len(pool)][1]
        if a.fixed_output:
            gen = a.fixed_output
        row = dict(t=round(t, 6), service=role, kind=kind, warmup=False)
        if kind == "chat":
            # raw mode applies to the MEASURED chat prompt only. Warm-up
            # prompts stay bounded: their content is irrelevant and an
            # unbounded corpus prompt there just slows the warm-up.
            row["prompt"] = make_prompt(ctx, pool, idx,
                                        raw=getattr(a, "raw_prompts", False))
            row["max_tokens"] = int(gen)
        elif kind == "embed":
            per = max(16, ctx // a.embed_batch)
            row["inputs"] = [make_prompt(per, pool, idx + j) for j in range(a.embed_batch)]
        elif kind == "rerank":
            per = max(16, ctx // a.rerank_docs)
            row["query"] = make_prompt(24, pool, idx)
            row["documents"] = [make_prompt(per, pool, idx + j) for j in range(a.rerank_docs)]
        reqs.append(row)
        idx += 1
    return reqs


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__)
    ap.add_argument("--scenario", default="llm-embed", choices=sorted(SCENARIOS))
    ap.add_argument("--rate", type=float, default=6.0,
                    help="aggregate requests/second (Poisson)")
    ap.add_argument("--duration", type=float, default=300.0, help="seconds of traffic")
    ap.add_argument("--mix", default="",
                    help="override the scenario's per-role request share, e.g. "
                         "llm=0.1,llm2=0.9. Used by the noisy-neighbour tier "
                         "to drive one tenant far past its knee while the "
                         "other stays inside its capacity. Weights are "
                         "relative, not required to sum to 1.")
    ap.add_argument("--source", choices=("sharegpt", "azure", "synthetic"),
                    default="sharegpt")
    ap.add_argument("--seed", type=int, default=20260825)
    ap.add_argument("--fixed-output", type=int, default=0,
                    help="force every reply to exactly N tokens (comparability arm); "
                         "0 keeps the trace's own variable lengths")
    ap.add_argument("--embed-batch", type=int, default=32,
                    help="documents per embedding request")
    ap.add_argument("--rerank-docs", type=int, default=16,
                    help="candidates per rerank request")
    ap.add_argument("--warmup", type=int, default=8,
                    help="per-service warm-up requests, excluded from metrics")
    ap.add_argument("--out")
    ap.add_argument("--raw-prompts", action="store_true",
                    help="use corpus prompts verbatim -- no FILLER "
                         "padding and no truncation. Changes the offered "
                         "load, so compare a raw run only against another "
                         "raw run.")
    ap.add_argument("--replay-arrivals", action="store_true",
                    help="replay the Azure trace's REAL inter-arrival gaps "
                         "instead of regenerating them as Poisson. Requires "
                         "--source azure. Burstiness survives into the run, "
                         "so --rate no longer controls offered load and two "
                         "runs are only comparable if they share a seed.")
    ap.add_argument("--list-scenarios", action="store_true")
    a = ap.parse_args()

    if a.list_scenarios:
        for k, v in sorted(SCENARIOS.items()):
            parts = ", ".join("%s %.0f%% %s" % (r, s * 100, kd) for r, s, kd in v)
            print("%-12s %s" % (k, parts))
        return
    if not a.out:
        ap.error("--out is required")

    rng = random.Random(a.seed)
    n_est = int(a.rate * a.duration * 1.3) + 64
    pool, lengths = None, None
    if a.source == "sharegpt":
        pool = load_sharegpt(n_est, rng)
    elif a.source == "azure":
        if getattr(a, "replay_arrivals", False):
            a._arrivals = load_azure_arrivals(a.duration, rng)
            if not a._arrivals:
                sys.exit("--replay-arrivals asked for the Azure trace and did "
                         "not get it. Refusing rather than silently falling "
                         "back to Poisson, which is the whole point of this "
                         "flag. Stage the CSV with 07-prefetch-traces.sh and "
                         "set AZURE_TRACE_CSV.")
            lengths = [(c, g) for _, c, g in a._arrivals]
        else:
            lengths = load_azure(n_est, rng)

    reqs = build(a, rng, pool, lengths)

    body = json.dumps(reqs, ensure_ascii=False, sort_keys=True)
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]
    resolved = a.source if (pool or lengths) else "synthetic"
    doc = dict(
        version=1,
        # The digest is the proof of requirement 7: every arm's result file
        # records it, and a mismatch means the comparison is not like-for-like.
        trace_id=digest,
        params=dict(scenario=a.scenario, rate=a.rate, duration=a.duration,
                    source=resolved, seed=a.seed, fixed_output=a.fixed_output,
                    embed_batch=a.embed_batch, rerank_docs=a.rerank_docs,
                    warmup=a.warmup,
                    raw_prompts=bool(getattr(a, "raw_prompts", False)),
                    replay_arrivals=bool(getattr(a, "replay_arrivals", False))),
        services=sorted(set(r["service"] for r in reqs)),
        requests=reqs,
    )
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, ensure_ascii=False)

    live = [r for r in reqs if not r["warmup"]]
    per = {}
    for r in live:
        per[r["service"]] = per.get(r["service"], 0) + 1
    toks = sorted(r["max_tokens"] for r in live if r["kind"] == "chat")
    print("trace_id    %s" % digest)
    print("scenario    %s   source %s" % (a.scenario, resolved))
    print("requests    %d live + %d warm-up over %.0fs  (%.2f/s achieved, %.2f/s requested)"
          % (len(live), len(reqs) - len(live), a.duration,
             len(live) / a.duration, a.rate))
    print("per service %s" % per)
    if toks:
        q = lambda p: toks[min(len(toks) - 1, int(len(toks) * p / 100))]
        print("output tok  p50 %d  p95 %d  max %d" % (q(50), q(95), toks[-1]))
    print("wrote       %s" % a.out)


if __name__ == "__main__":
    main()
