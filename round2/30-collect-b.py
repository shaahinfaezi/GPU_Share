#!/usr/bin/env python3
"""Dump every number series B produced, plus the provenance each claim needs."""
import json, os, glob, csv, sys

R = "/opt/GPU_Share/round2/results"
NEW = ("20260910-22", "20260910-23", "20260911")


def jload(p):
    try:
        with open(p) as fh:
            return json.load(fh)
    except Exception:
        return None


def runs():
    for d in sorted(glob.glob(os.path.join(R, "*"))):
        b = os.path.basename(d)
        if any(k in b for k in NEW):
            yield d, b


print("#" * 72)
print("# 1. TRACE PROVENANCE  -- did each tier get the corpus it asked for?")
print("#" * 72)
for d, b in runs():
    t = jload(os.path.join(d, "trace.json"))
    if not t:
        continue
    m = t.get("meta", t)
    reqs = t.get("requests") or []
    ctx = [r.get("ctx_tokens") or r.get("prompt_tokens") for r in reqs]
    gen = [r.get("max_tokens") or r.get("gen_tokens") for r in reqs]
    ctx = [x for x in ctx if isinstance(x, int)]
    gen = [x for x in gen if isinstance(x, int)]
    filler = sum(1 for r in reqs if "FILLER" in str(r.get("prompt", ""))
                 or "Explain the trade-off" in str(r.get("prompt", "")))
    print("\n%s" % b)
    print("   source=%-10s seed=%-9s trace_id=%s"
          % (m.get("source"), m.get("seed"), m.get("trace_id")))
    print("   requests=%-6d filler=%d (%.1f%%)"
          % (len(reqs), filler, 100.0 * filler / max(1, len(reqs))))
    if ctx:
        ctx.sort(); gen.sort()
        print("   ctx  min/med/max = %d / %d / %d" %
              (ctx[0], ctx[len(ctx) // 2], ctx[-1]))
        print("   gen  min/med/max = %d / %d / %d" %
              (gen[0], gen[len(gen) // 2], gen[-1]))

print("\n" + "#" * 72)
print("# 2. THROUGHPUT / LATENCY  -- every arm of every tier")
print("#" * 72)
for d, b in runs():
    arms = [a for a in sorted(os.listdir(d))
            if os.path.exists(os.path.join(d, a, "replay.summary.json"))]
    if not arms:
        continue
    print("\n%s" % b)
    for a in arms:
        s = jload(os.path.join(d, a, "replay.summary.json")) or {}
        per = s.get("per_service", {})
        tot, parts = 0.0, []
        for r in sorted(per):
            v = per[r]
            tps = v.get("tokens_per_second") or 0.0
            lat = v.get("latency_e2e") or {}
            ttft = v.get("ttft") or {}
            tot += tps
            parts.append("%s tps=%.2f mean=%.3f p99=%.3f ttft99=%.3f n=%s ok=%s"
                         % (r, tps, lat.get("mean", 0), lat.get("p99", 0),
                            ttft.get("p99", 0), v.get("requests"),
                            v.get("succeeded")))
        print("  %-34s TOTAL %9.2f" % (a, tot))
        for p in parts:
            print("      " + p)

print("\n" + "#" * 72)
print("# 3. FAULTS  -- windows, recovery, and what the probe ACTUALLY raised")
print("#" * 72)
for d, b in runs():
    t = jload(os.path.join(d, "t10.json"))
    if not t:
        continue
    print("\n%s" % b)
    print("   fault=%-14s arm=%-9s host_min_avail=%s recovered=%s"
          % (t.get("fault"), t.get("arm"), t.get("host_min_available_mib"),
             t.get("recovered")))
    w = t.get("windows") or {}
    for k in ("before", "during", "after"):
        x = w.get(k) or {}
        print("   %-7s failed=%-5s n=%-5s mean=%.3f p99=%.3f"
              % (k, x.get("failed"), x.get("n"), x.get("mean", 0),
                 x.get("p99", 0)))
    mr = t.get("mps_recovery")
    if mr:
        print("   mps_recovery: %s" % json.dumps(mr))
    fd = t.get("fault_detail") or {}
    for key in ("probe", "hog_phase", "hog_held_mib", "hog_events",
                "hog_container_state", "how", "daemon_after"):
        if key in fd:
            print("   %-20s %s" % (key, json.dumps(fd[key])[:400]))
    if "out_tail" in fd:
        print("   out_tail: %s" % fd["out_tail"][-400:].replace("\n", " | "))

print("\n" + "#" * 72)
print("# 4. T9 SCALE-DOWN  -- the transition tier")
print("#" * 72)
for d, b in runs():
    t = jload(os.path.join(d, "t9.json"))
    if not t:
        continue
    print("\n%s" % b)
    for k, v in sorted(t.items()):
        if isinstance(v, (int, float, str, bool)):
            print("   %-24s %s" % (k, v))
        elif k in ("marks", "smoke_incumbent", "smoke_arrival", "windows"):
            print("   %-24s %s" % (k, json.dumps(v)[:500]))

print("\n" + "#" * 72)
print("# 5. GPU / POWER / VRAM  -- device-side, per arm")
print("#" * 72)
for d, b in runs():
    for a in sorted(os.listdir(d)):
        p = os.path.join(d, a, "gpu.csv")
        if not os.path.exists(p):
            continue
        pw, vr, ut = [], [], []
        with open(p, newline="") as fh:
            for row in csv.DictReader(fh):
                try:
                    if str(row.get("index")) != "1":
                        continue
                    pw.append(float(row.get("power_w") or row.get("power.draw") or 0))
                    vr.append(float(row.get("mem_used_mib") or row.get("memory.used") or 0))
                    ut.append(float(row.get("util_gpu") or row.get("utilization.gpu") or 0))
                except Exception:
                    pass
        if pw:
            print("  %-28s %-30s power mean=%.1f max=%.1f | vram peak=%.0f | util mean=%.1f"
                  % (b, a, sum(pw) / len(pw), max(pw), max(vr), sum(ut) / len(ut)))
