#!/usr/bin/env python3
"""Show what a trace actually contains: prompts, arrivals, and the Poisson shape.

A trace is the one thing every arm shares, so it is worth being able to look at
one directly rather than trusting that it is what the parameters said.

    python3 20-show-trace.py results/T2-.../trace.json
    python3 20-show-trace.py results/T2-.../trace.json --rows 12 --chars 300
"""
import argparse
import json
import math
import statistics as st


def load(path):
    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)
    return doc.get("params", {}), doc.get("requests", doc if isinstance(doc, list) else [])


def preview(row, chars):
    if "prompt" in row:
        return "prompt", row["prompt"]
    if "inputs" in row:
        return "inputs[%d]" % len(row["inputs"]), row["inputs"][0]
    if "query" in row:
        return "query + %d docs" % len(row.get("documents", [])), row["query"]
    return "?", ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--rows", type=int, default=8)
    ap.add_argument("--chars", type=int, default=220)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    params, reqs = load(a.trace)
    live = [r for r in reqs if not r.get("warmup")]
    print("=== %s" % a.trace)
    for k in ("scenario", "source", "rate", "duration", "seed"):
        if k in params:
            print("    %-9s %s" % (k, params[k]))
    print("    %d live requests, %d warm-up"
          % (len(live), len(reqs) - len(live)))

    # ---- arrivals -------------------------------------------------------
    ts = [r["t"] for r in live]
    gaps = [b - a_ for a_, b in zip(ts, ts[1:])]
    mean = st.mean(gaps)
    cv = st.pstdev(gaps) / mean
    print("\n=== arrivals")
    print("    span            %.1fs, %d requests -> %.2f/s achieved"
          % (ts[-1] - ts[0], len(ts), len(ts) / (ts[-1] - ts[0])))
    print("    mean gap        %.4fs   (1/rate = %.4fs)"
          % (mean, 1.0 / params.get("rate", 1) if params.get("rate") else float("nan")))
    print("    gap CV          %.3f    (exponential = 1.000; fixed rate = 0.000)" % cv)
    print("    shortest gap    %.4fs" % min(gaps))
    print("    longest gap     %.4fs   <- %.1fx the mean; this is the burst/idle"
          % (max(gaps), max(gaps) / mean))
    print("    gaps < 0.1s     %d of %d (%.1f%%)  <- arrivals that pile up"
          % (sum(1 for g in gaps if g < 0.1), len(gaps),
             100.0 * sum(1 for g in gaps if g < 0.1) / len(gaps)))

    # A Poisson process has exponentially distributed gaps: P(gap > x) = e^(-x/mean).
    print("\n    gap distribution vs the exponential it should be")
    print("    %-16s %8s %8s" % ("", "observed", "expected"))
    edges = [0.05, 0.1, 0.25, 0.5, 1.0]
    for x in edges:
        obs = 100.0 * sum(1 for g in gaps if g > x) / len(gaps)
        exp = 100.0 * math.exp(-x / mean)
        print("    %-16s %7.1f%% %7.1f%%" % ("gap > %.2fs" % x, obs, exp))

    # ---- how many are in flight ----------------------------------------
    # Bursts are the point of an open loop, so show the worst second.
    per_s = {}
    for t in ts:
        per_s[int(t)] = per_s.get(int(t), 0) + 1
    busiest = max(per_s.values())
    print("\n    busiest 1s      %d arrivals   (mean %.2f)  <- %.1fx"
          % (busiest, len(ts) / (ts[-1] - ts[0]),
             busiest / (len(ts) / (ts[-1] - ts[0]))))
    print("    quietest 1s     %d arrivals   over %d seconds with none"
          % (min(per_s.values()),
             int(ts[-1]) - int(ts[0]) + 1 - len(per_s)))

    # ---- the prompts ----------------------------------------------------
    import random
    rng = random.Random(a.seed)
    print("\n=== %d requests, sampled" % a.rows)
    for r in sorted(rng.sample(live, min(a.rows, len(live))), key=lambda r: r["t"]):
        kind, text = preview(r, a.chars)
        text = " ".join(text.split())
        extra = ""
        if "max_tokens" in r:
            extra = "  max_tokens=%d" % r["max_tokens"]
        print("\n  t=%8.3fs  %-6s %-16s ~%d chars%s"
              % (r["t"], r["service"], kind, len(text), extra))
        print("    %s%s" % (text[:a.chars], "..." if len(text) > a.chars else ""))

    # ---- lengths --------------------------------------------------------
    outs = [r["max_tokens"] for r in live if "max_tokens" in r]
    if outs:
        outs.sort()
        q = lambda p: outs[min(len(outs) - 1, int(p * len(outs)))]
        print("\n=== output length (max_tokens), the decode work")
        print("    p50 %d   p90 %d   p95 %d   p99 %d   max %d"
              % (q(.5), q(.9), q(.95), q(.99), outs[-1]))
        print("    %.1fx spread p50 -> p99. A fixed-length benchmark hides this."
              % (q(.99) / max(1, q(.5))))


if __name__ == "__main__":
    main()
