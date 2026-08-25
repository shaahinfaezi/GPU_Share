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
    "llm-embed":   [("llm", 0.5, "chat"), ("embed", 0.5, "embed")],
    "llm-llm":     [("llm", 0.5, "chat"), ("llm2", 0.5, "chat")],
    "vl-rerank":   [("vl", 0.5, "chat"), ("rerank", 0.5, "rerank")],
    "dense4":      [("llm", 0.3, "chat"), ("llm2", 0.3, "chat"),
                    ("embed", 0.2, "embed"), ("rerank", 0.2, "rerank")],
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


def synth_lengths(rng):
    """Lognormal in, lognormal out -- shaped like the sharegpt corpus."""
    ctx = int(min(8192, max(16, rng.lognormvariate(5.5, 1.05))))
    gen = int(min(1024, max(8, rng.lognormvariate(4.7, 0.85))))
    return ctx, gen


def make_prompt(ctx_tokens, pool, idx):
    """A prompt of roughly ctx_tokens tokens, from the real pool where possible."""
    base = pool[idx % len(pool)][0] if pool else FILLER
    want = ctx_tokens * 4          # ~4 chars/token for English
    if len(base) >= want:
        return base[:want]
    pad = FILLER * (1 + (want - len(base)) // len(FILLER))
    return (base + " " + pad)[:want]


def build(a, rng, pool, lengths):
    mix = SCENARIOS[a.scenario]
    roles = [r for r, _, _ in mix]
    weights = [s for _, s, _ in mix]
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

    t, idx = 0.0, 0
    while True:
        t += rng.expovariate(a.rate)     # Poisson arrivals
        if t >= a.duration:
            break
        role = rng.choices(roles, weights=weights, k=1)[0]
        kind = kind_of[role]
        if lengths:
            ctx, gen = lengths[idx % len(lengths)]
        else:
            ctx, gen = synth_lengths(rng)
        if a.fixed_output:
            gen = a.fixed_output
        row = dict(t=round(t, 6), service=role, kind=kind, warmup=False)
        if kind == "chat":
            row["prompt"] = make_prompt(ctx, pool, idx)
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
                    warmup=a.warmup),
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
