#!/usr/bin/env python3
"""Collect the split series into the two tables the report needs.

WHAT IT REPORTS, AND WHY EACH ONE
---------------------------------
SATURATION RATIO. Combined co-tenancy tok/s divided by the sum of the two solo
tok/s. Lower means the card is more saturated. This is the ONLY variable that
predicted the MPS effect across round two's five measured points, so it is
printed before any MPS number and every MPS claim is qualified by it. Quoting
an MPS delta without it is how a rate choice gets reported as a mechanism.

IN-FLIGHT DEPTH, from Little's law on the SOLO arm: depth = lambda x W, arrival
rate times mean residency. Round two's depth law is that MPS reallocates
capacity toward the tenant with more in-flight work, not toward the bigger
model. With tp at max-num-seqs 32 against pp at 4, this series has a large
built-in depth asymmetry, so the law predicts MPS should favour tp. Printing
both depths next to the per-tenant deltas is what makes that checkable rather
than assertable.

PER-TENANT AGAINST ITS OWN SOLO BASELINE. pp is throttled by the coworker's
--max-num-seqs 4 and will be slower than tp in every arm. Comparing the two
tenants to each other would report that throttle as a sharing effect. Each is
compared only to itself.

    python3 54-split-collect.py results/S-share-20260914-101500
"""
import glob
import json
import os
import sys

ROLES = ("tp", "pp")


def load(p, default=None):
    try:
        with open(p, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return default


def summary(d, role):
    s = load(os.path.join(d, "replay.summary.json"))
    if not s:
        return None
    ps = s.get("per_service", {}).get(role)
    if not ps:
        return None
    return {
        "tok_s":   ps["tokens_per_second"],
        "req_s":   ps["requests_per_second"],
        "ok":      ps["requests"],
        "failed":  ps.get("failed", 0),
        "lat":     ps["latency_e2e"]["mean"],
        "lat_p99": ps["latency_e2e"]["p99"],
        "ttft":    ps["ttft"]["mean"],
        "wall":    s.get("wall_seconds"),
        "sent":    s.get("requests_total"),
    }


def gpu_stats(d):
    """Peak VRAM per card, mean power, mean SM clock -- from gpu.csv.

    Reads the RAW nvidia-smi column names. They are not normalised anywhere in
    the pipeline, and a collector that looked for tidy names once reported
    every power and VRAM figure in a series as zero.
    """
    p = os.path.join(d, "gpu.csv")
    if not os.path.exists(p):
        return {}
    import csv
    per_card, pw, sm = {}, [], []
    with open(p, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            try:
                i = int(row["index"])
                per_card[i] = max(per_card.get(i, 0), float(row["memory.used"]))
                pw.append(float(row["power.draw"]))
                sm.append(float(row["clocks.sm"]))
            except (KeyError, ValueError, TypeError):
                continue
    return {
        "peak_vram": {k: per_card[k] for k in sorted(per_card)},
        "power_w":   sum(pw) / len(pw) if pw else 0.0,
        "sm_mhz":    sum(sm) / len(sm) if sm else 0.0,
    }


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    share = sys.argv[1].rstrip("/\\")
    root = os.path.dirname(share) or "."

    cal = load("calibration.json") or load(os.path.join(root, "calibration.json"))
    if not cal:
        solos = sorted(glob.glob(os.path.join(root, "S-solo-*", "calibration.json")))
        cal = load(solos[-1]) if solos else None
    if not cal or "solo" not in cal:
        sys.exit("no calibration.json -- run 52-split-solo.sh first")

    solo = cal["solo"]
    arms = {}
    for arm in ("hami", "hami-mps"):
        d = os.path.join(share, "%s__tp+pp" % arm)
        if not os.path.isdir(d):
            continue
        arms[arm] = {"dir": d, "gpu": gpu_stats(d),
                     "roles": {r: summary(d, r) for r in ROLES}}

    missing = [a for a in ("hami", "hami-mps") if a not in arms]
    if missing:
        print("!! missing arm(s): %s -- read the run log before reading below"
              % ", ".join(missing))

    # ------------------------------------------------------------ solo table
    print("\nSOLO BASELINES (dedicated, one tenant on both cards)")
    print("  %-4s %10s %9s %9s %9s %8s" %
          ("role", "tok/s", "req/s", "lat mean", "TTFT", "depth"))
    depth = {}
    for r in ROLES:
        s = solo.get(r)
        if not s:
            print("  %-4s  (missing)" % r)
            continue
        # Little's law: in-flight work = arrival rate x mean residency.
        depth[r] = s["requests_per_second"] * s["latency_mean"]
        print("  %-4s %10.1f %9.2f %8.2fs %8.3fs %8.1f"
              % (r, s["tokens_per_second"], s["requests_per_second"],
                 s["latency_mean"], s["ttft_mean"], depth[r]))
    solo_total = sum(v["tokens_per_second"] for v in solo.values())
    print("  %-4s %10.1f" % ("sum", solo_total))

    # ------------------------------------------------------- co-tenancy table
    print("\nCO-TENANCY")
    print("  %-9s %-4s %10s %9s %9s %9s %8s %7s" %
          ("arm", "role", "tok/s", "req/s", "lat mean", "lat p99", "TTFT", "fail"))
    for arm in ("hami", "hami-mps"):
        a = arms.get(arm)
        if not a:
            continue
        for r in ROLES:
            s = a["roles"].get(r)
            if not s:
                print("  %-9s %-4s  (missing)" % (arm, r))
                continue
            print("  %-9s %-4s %10.1f %9.2f %8.2fs %8.2fs %7.3fs %7d"
                  % (arm, r, s["tok_s"], s["req_s"], s["lat"], s["lat_p99"],
                     s["ttft"], s["failed"]))

    # ------------------------------------------------------------- the ratios
    print("\nSATURATION AND MPS EFFECT")
    tot = {}
    for arm in ("hami", "hami-mps"):
        a = arms.get(arm)
        if not a or not all(a["roles"].get(r) for r in ROLES):
            continue
        tot[arm] = sum(a["roles"][r]["tok_s"] for r in ROLES)
        sat = 100.0 * tot[arm] / solo_total if solo_total else 0.0
        print("  %-9s combined %8.1f tok/s   saturation %5.1f%% of solo sum"
              % (arm, tot[arm], sat))

    if "hami" in tot and "hami-mps" in tot:
        sat_h = 100.0 * tot["hami"] / solo_total
        delta = 100.0 * (tot["hami-mps"] - tot["hami"]) / tot["hami"]
        print("\n  MPS effect on combined throughput: %+.2f%%" % delta)
        print("  measured at %.1f%% saturation on the HAMi arm." % sat_h)
        # The standing rule: never quote an MPS number without the regime it
        # was measured in, and never without the depth asymmetry beside it.
        if sat_h > 95:
            print("  REGIME: the card was NOT saturated. Round two measured")
            print("  +0.06% at 99.6% and +8.6% at 86.3%; a small number here")
            print("  says the rate was low, NOT that MPS fails on split models.")
            print("  Re-run 53-split-share.sh with SECS=240 and a higher rate")
            print("  before drawing any conclusion about MPS.")
        elif sat_h > 88:
            print("  REGIME: lightly saturated. Comparable to round two's 86.3%")
            print("  point (+8.6%). A modest gain here is consistent, not weak.")
        else:
            print("  REGIME: genuinely saturated, the band where round two saw")
            print("  +19.0% (80.9%) and +20.8% (55.9%). This is the point the")
            print("  capacity claim for split models should rest on.")

        print("\n  per tenant, each against its OWN solo baseline:")
        for r in ROLES:
            h = arms["hami"]["roles"].get(r)
            m = arms["hami-mps"]["roles"].get(r)
            s = solo.get(r)
            if not (h and m and s):
                continue
            print("    %-4s  hami %6.1f%% of solo -> mps %6.1f%% of solo  (MPS %+.2f%%), depth %.1f"
                  % (r,
                     100.0 * h["tok_s"] / s["tokens_per_second"],
                     100.0 * m["tok_s"] / s["tokens_per_second"],
                     100.0 * (m["tok_s"] - h["tok_s"]) / h["tok_s"],
                     depth.get(r, float("nan"))))
        if len(depth) == 2:
            hi = max(depth, key=depth.get)
            print("\n  DEPTH LAW: %s carries the deeper queue (%.1f vs %.1f), so"
                  % (hi, depth[hi], depth[min(depth, key=depth.get)]))
            print("  the round-two law predicts MPS should favour %s. Check that" % hi)
            print("  against the two per-tenant deltas above before quoting either.")

        # Fairness, the number T14 was judged on.
        for arm in ("hami", "hami-mps"):
            sh = [100.0 * arms[arm]["roles"][r]["tok_s"] / solo[r]["tokens_per_second"]
                  for r in ROLES]
            print("  fairness spread (%-8s): %5.2f points" % (arm, max(sh) - min(sh)))

    # -------------------------------------------------------------- hardware
    print("\nHARDWARE")
    for arm in ("hami", "hami-mps"):
        a = arms.get(arm)
        if not a or not a["gpu"]:
            continue
        g = a["gpu"]
        vram = "  ".join("card %d %.0f MiB" % (k, v)
                         for k, v in g["peak_vram"].items())
        print("  %-9s %s | %.0f W | %.0f MHz SM" % (arm, vram, g["power_w"], g["sm_mhz"]))
        if len(g["peak_vram"]) < 2:
            print("            only one card recorded -- the split did not hold")

    print("\nHOST MEMORY (the density ceiling, not VRAM)")
    for arm in ("hami", "hami-mps"):
        a = arms.get(arm)
        if not a:
            continue
        hm = load(os.path.join(a["dir"], "hostmem.json"), {})
        if hm:
            print("  %-9s available %s MiB, vLLM RSS %s MiB, problems: %s"
                  % (arm, hm.get("host", {}).get("available"),
                     hm.get("vllm_rss_mib"), hm.get("problems") or "none"))
    print()


if __name__ == "__main__":
    main()
