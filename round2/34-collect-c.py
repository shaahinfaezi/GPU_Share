#!/usr/bin/env python3
"""Collect the T18 / T20 / T21 / T22 series -- and compute the two numbers
those tiers exist to produce, rather than leaving them to be done by hand.

WHY THIS EXISTS RATHER THAN A PATCH TO 30-collect-b.py
------------------------------------------------------
30-collect-b.py hardcodes the run dates it will read:

    NEW = ("20260910-22", "20260910-23", "20260911")

so it silently reports nothing for anything run later, which is how a
collector rots. This one takes patterns on the command line and defaults to
the current series. It also reads the trace metadata from doc["params"],
where gen_trace.py actually writes it -- 30-collect-b.py looks for
doc["meta"], which has never existed, so its `source=` and `seed=` lines have
been printing None for every run.

    ./34-collect-c.py                       # T18/T20/T21/T22 from 2026-09-12
    ./34-collect-c.py T20-20260912 T21      # any substring of the run dir name

WHAT IT DERIVES THAT THE RAW FILES DO NOT CONTAIN
-------------------------------------------------
SATURATION RATIO (section 3), the point of T21. Combined co-tenancy
throughput divided by the sum of the solo arms. Lower means more saturated.
Only T14 (86.3-86.8%) and T5 (92.3%) have ever been saturated in this study;
T13 and T15 both stayed above 99.5%, which is why they answered the fairness
question and not the capacity one.

ARRIVAL-PROCESS STATISTICS (section 2), the point of T20. Coefficient of
variation of the inter-arrival gaps, the max gap, and the burst ratio. A
Poisson process has CV = 1 exactly. The registered prediction is that the
Azure conversation trace measures close to 1 and therefore changes little --
and that IS the result, because it vindicates the Poisson substitution used
by every other tier in the study.

HOST MEMORY AND SWAP (section 5), the point of T18 and T22. A refusal is a
result here, so the refusal is reported as data: what was needed, what was
available, and whether any swap was touched. A tier that only fit because it
paged has measured paging, not co-tenancy.
"""
import json, os, glob, csv, sys, statistics

R = os.environ.get("R2_RESULTS", "/opt/GPU_Share/round2/results")
DEFAULT = ("T18-20260912", "T20-20260912", "T21-20260912", "T22-20260912")


def jload(p):
    try:
        with open(p, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def runs(pats):
    for d in sorted(glob.glob(os.path.join(R, "*"))):
        b = os.path.basename(d)
        if any(k in b for k in pats):
            yield d, b


def arms_of(d):
    return [a for a in sorted(os.listdir(d))
            if os.path.isdir(os.path.join(d, a))]


def summary(d, a):
    return jload(os.path.join(d, a, "replay.summary.json")) or {}


def total_tps(s):
    return sum((v.get("tokens_per_second") or 0.0)
               for v in (s.get("per_service") or {}).values())


PATS = sys.argv[1:] or list(DEFAULT)
FOUND = list(runs(PATS))

print("#" * 74)
print("# 0. INVENTORY  -- what actually exists on disk")
print("#" * 74)
print("results root : %s" % R)
print("patterns     : %s" % " ".join(PATS))
if not FOUND:
    print("\n!!! nothing matched. ls the results dir and pass a substring.")
    sys.exit(1)
for d, b in FOUND:
    arms = arms_of(d)
    done = [a for a in arms
            if os.path.exists(os.path.join(d, a, "replay.summary.json"))]
    run = jload(os.path.join(d, "run.json")) or {}
    print("\n%s" % b)
    print("   scenario=%-9s arms=%d  with results=%d"
          % (run.get("scenario") or (run.get("params") or {}).get("scenario"),
             len(arms), len(done)))
    for a in arms:
        ok = "ok " if a in done else "NO RESULT"
        print("     %-9s %s" % (ok, a))

print("\n" + "#" * 74)
print("# 1. TRACE PROVENANCE  -- did each tier get the corpus it asked for?")
print("#" * 74)
for d, b in FOUND:
    t = jload(os.path.join(d, "trace.json"))
    if not t:
        print("\n%s   (no trace.json)" % b)
        continue
    p = t.get("params") or {}
    reqs = t.get("requests") or []
    live = [r for r in reqs if not r.get("warmup")]
    gen = sorted(int(r["max_tokens"]) for r in live if "max_tokens" in r)
    # No ctx_tokens field is stored; the prompt TEXT is. ~4 chars/token is the
    # same ratio make_prompt() uses to build it, so this inverts that and is
    # an estimate, not a tokenizer count. Labelled as such.
    ctx = sorted(len(str(r.get("prompt", ""))) // 4 for r in live if "prompt" in r)
    print("\n%s" % b)
    print("   trace_id=%s  source=%s  seed=%s"
          % (t.get("trace_id"), p.get("source"), p.get("seed")))
    print("   scenario=%-9s rate=%-6s duration=%-6s warmup=%s"
          % (p.get("scenario"), p.get("rate"), p.get("duration"), p.get("warmup")))
    print("   raw_prompts=%-6s replay_arrivals=%s   <-- T20 needs this True"
          % (p.get("raw_prompts"), p.get("replay_arrivals")))
    print("   requests=%d total, %d measured (%d warm-up)"
          % (len(reqs), len(live), len(reqs) - len(live)))
    if gen:
        print("   gen tokens   min/med/max = %d / %d / %d"
              % (gen[0], gen[len(gen) // 2], gen[-1]))
    if ctx:
        print("   ctx est.     min/med/max = %d / %d / %d  (chars/4, NOT tokenised)"
              % (ctx[0], ctx[len(ctx) // 2], ctx[-1]))
    per_role = {}
    for r in live:
        per_role[r.get("service")] = per_role.get(r.get("service"), 0) + 1
    print("   per role     %s" % ", ".join("%s=%d" % kv for kv in sorted(per_role.items())))

print("\n" + "#" * 74)
print("# 2. ARRIVAL PROCESS  -- T20's whole question. Poisson has CV = 1.000")
print("#" * 74)
for d, b in FOUND:
    t = jload(os.path.join(d, "trace.json"))
    if not t:
        continue
    p = t.get("params") or {}
    ts = sorted(float(r["t"]) for r in (t.get("requests") or [])
                if not r.get("warmup") and "t" in r)
    if len(ts) < 3:
        continue
    gaps = [b_ - a_ for a_, b_ in zip(ts, ts[1:])]
    mean = statistics.fmean(gaps)
    sd = statistics.pstdev(gaps)
    cv = sd / mean if mean else 0.0
    gs = sorted(gaps)
    print("\n%s   replay_arrivals=%s" % (b, p.get("replay_arrivals")))
    print("   n=%d  span=%.1fs  rate=%.3f/s" % (len(ts), ts[-1] - ts[0],
                                                len(ts) / max(1e-9, ts[-1] - ts[0])))
    print("   gap mean=%.4fs  sd=%.4fs  CV=%.3f   (1.000 = Poisson)" % (mean, sd, cv))
    print("   gap p50=%.4f  p99=%.4f  max=%.4f  burst ratio (max/mean)=%.1f"
          % (gs[len(gs) // 2], gs[int(len(gs) * 0.99)], gs[-1], gs[-1] / mean if mean else 0))

print("\n" + "#" * 74)
print("# 3. THROUGHPUT / LATENCY / SATURATION  -- T21's whole question")
print("#" * 74)
for d, b in FOUND:
    arms = [a for a in arms_of(d)
            if os.path.exists(os.path.join(d, a, "replay.summary.json"))]
    if not arms:
        print("\n%s   (no arm produced a replay summary)" % b)
        continue
    print("\n%s" % b)
    solo = {}
    combined = {}
    for a in arms:
        s = summary(d, a)
        per = s.get("per_service") or {}
        tot = total_tps(s)
        print("  %-36s TOTAL %9.2f tok/s" % (a, tot))
        for r in sorted(per):
            v = per[r]
            lat = v.get("latency_e2e") or {}
            ttft = v.get("ttft") or {}
            tps = v.get("tokens_per_second") or 0.0
            print("      %-14s tps=%8.2f  mean=%.3f p95=%.3f p99=%.3f  "
                  "ttft p99=%.3f  n=%s ok=%s"
                  % (r, tps, lat.get("mean", 0), lat.get("p95", 0),
                     lat.get("p99", 0), ttft.get("p99", 0),
                     v.get("requests"), v.get("succeeded")))
            if a.startswith("dedicated"):
                solo[r] = tps
        if not a.startswith("dedicated"):
            combined[a] = (tot, per)

    # Saturation ratio: combined divided by the sum of the solo arms. This is
    # the number that separates "the card had spare capacity" from "the card
    # was full", and it is what T13 and T15 failed to reach.
    if solo and combined:
        ssum = sum(solo.values())
        print("\n  -- saturation (combined / sum of solo). LOWER = more saturated")
        print("     solo sum = %.2f tok/s  (%s)"
              % (ssum, ", ".join("%s %.2f" % kv for kv in sorted(solo.items()))))
        for a in sorted(combined):
            tot, per = combined[a]
            print("     %-30s %8.2f  ratio %6.1f%%" % (a, tot, 100.0 * tot / ssum))
            # Per-tenant retention says WHO paid for the shortfall.
            for r in sorted(per):
                if solo.get(r):
                    keep = 100.0 * (per[r].get("tokens_per_second") or 0.0) / solo[r]
                    print("         %-14s keeps %6.1f%% of solo" % (r, keep))
        # Fairness spread: max/min of per-tenant retention, the T13 metric.
        for a in sorted(combined):
            _, per = combined[a]
            keeps = [(per[r].get("tokens_per_second") or 0.0) / solo[r]
                     for r in per if solo.get(r)]
            if len(keeps) > 1 and min(keeps) > 0:
                print("     %-30s fairness spread (max/min) = %.2f"
                      % (a, max(keeps) / min(keeps)))

    # MPS delta: the hami-mps arm against the plain hami arm.
    base = next((a for a in arms if a.startswith("hami__")), None)
    mps = next((a for a in arms if a.startswith("hami-mps")), None)
    if base and mps:
        tb, tm = total_tps(summary(d, base)), total_tps(summary(d, mps))
        if tb:
            print("\n  -- MPS effect: %.2f -> %.2f tok/s  (%+.1f%%)"
                  % (tb, tm, 100.0 * (tm - tb) / tb))

print("\n" + "#" * 74)
print("# 4. GPU / POWER / VRAM  -- device-side, card 1, per arm")
print("#" * 74)
for d, b in FOUND:
    for a in arms_of(d):
        p = os.path.join(d, a, "gpu.csv")
        if not os.path.exists(p):
            continue
        pw, vr, ut, cl = [], [], [], []
        with open(p, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                try:
                    if str(row.get("index")) != "1":
                        continue
                    pw.append(float(row.get("power_w") or 0))
                    vr.append(float(row.get("mem_used_mib") or 0))
                    ut.append(float(row.get("util_gpu") or 0))
                    if row.get("clock_sm_mhz"):
                        cl.append(float(row["clock_sm_mhz"]))
                except Exception:
                    pass
        if pw:
            print("  %-24s %-28s power mean=%5.1f max=%5.1f | vram peak=%5.0f "
                  "| util mean=%5.1f%s"
                  % (b, a, statistics.fmean(pw), max(pw), max(vr) if vr else 0,
                     statistics.fmean(ut) if ut else 0,
                     " | sm clk mean=%.0f" % statistics.fmean(cl) if cl else ""))

print("\n" + "#" * 74)
print("# 5. HOST MEMORY, REFUSALS AND SWAP  -- T18/T22's result, not their error")
print("#" * 74)
for d, b in FOUND:
    print("\n%s" % b)
    said = False
    # The refusal itself, from the run-level and arm-level event logs.
    for evp in [os.path.join(d, "events.jsonl")] + \
               [os.path.join(d, a, "events.jsonl") for a in arms_of(d)]:
        if not os.path.exists(evp):
            continue
        with open(evp, encoding="utf-8") as fh:
            for line in fh:
                try:
                    e = json.loads(line)
                except Exception:
                    continue
                if e.get("event") != "host_memory":
                    continue
                said = True
                print("   host_memory ok=%-5s roles=%-28s need=%-6s avail=%-6s "
                      "swap_total=%s"
                      % (e.get("ok"), ",".join(e.get("roles") or []),
                         e.get("need_mib"), e.get("available_mib"),
                         e.get("swap_total_mib")))
    if not said:
        print("   (no host_memory marks)")
    # Whether anything was actually paged, from the 1 Hz host sampler.
    for a in arms_of(d):
        p = os.path.join(d, a, "host.csv")
        if not os.path.exists(p):
            continue
        avail, used = [], []
        with open(p, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                try:
                    avail.append(float(row["mem_available"]))
                    st, sf = float(row["swap_total"]), float(row["swap_free"])
                    used.append(st - sf)
                except Exception:
                    pass
        if avail:
            u0, u1 = (used[0], max(used)) if used else (0, 0)
            flag = "  *** PAGED: latency figures VOID" if (u1 - u0) > 64 else ""
            print("   %-28s min avail=%6.0f MiB | swap %.0f -> peak %.0f MiB%s"
                  % (a, min(avail), u0, u1, flag))
