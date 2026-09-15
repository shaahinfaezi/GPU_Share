#!/usr/bin/env python3
"""Collect the box-2 run into the same two ratios box 1 reports.

Deliberately the same shape as 54-split-collect.py so the two boxes can be put
side by side without re-deriving anything. TTFT is absent here and printed as
"--" rather than as zero: the custom runtime does not stream, so there is no
first token to time. That is a property of the runtime, not a gap in the run.

    python3 62-box2-collect.py results/B-box2-20260914-120000
"""
import json
import os
import sys

ROLES = ("llamacpp", "custom")


def load(p):
    try:
        with open(p, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def per(run, sub, role):
    d = load(os.path.join(run, sub, "load.summary.json"))
    return (d or {}).get("per_service", {}).get(role)


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    run = sys.argv[1].rstrip("/\\")

    solo = {r: per(run, "dedicated__%s" % r, r) for r in ROLES}
    arms = {a: {r: per(run, "%s__llamacpp+custom" % a, r) for r in ROLES}
            for a in ("hami", "hami-mps")}

    print("\nSOLO BASELINES (one tenant on the card)")
    print("  %-9s %10s %9s %10s" % ("role", "tok/s", "req/s", "lat mean"))
    for r in ROLES:
        s = solo[r]
        if not s:
            print("  %-9s  (missing -- ratios below are not computable)" % r)
            continue
        print("  %-9s %10.1f %9.2f %9.2fs"
              % (r, s["tokens_per_second"], s["requests_per_second"],
                 s["latency_e2e"]["mean"]))

    print("\nCO-TENANCY")
    print("  %-9s %-9s %10s %9s %10s %10s %6s"
          % ("arm", "role", "tok/s", "req/s", "lat mean", "lat p99", "fail"))
    for a in ("hami", "hami-mps"):
        for r in ROLES:
            s = arms[a][r]
            if not s:
                print("  %-9s %-9s  (missing)" % (a, r))
                continue
            print("  %-9s %-9s %10.1f %9.2f %9.2fs %9.2fs %6d"
                  % (a, r, s["tokens_per_second"], s["requests_per_second"],
                     s["latency_e2e"]["mean"], s["latency_e2e"]["p99"],
                     s["failed"]))

    if not all(solo.values()):
        print("\nno complete solo baseline -- stop here, the ratios would be fiction\n")
        return

    solo_total = sum(s["tokens_per_second"] for s in solo.values())
    print("\nSATURATION AND MPS EFFECT")
    tot = {}
    for a in ("hami", "hami-mps"):
        if not all(arms[a].values()):
            continue
        tot[a] = sum(s["tokens_per_second"] for s in arms[a].values())
        print("  %-9s combined %8.1f tok/s   saturation %5.1f%% of solo sum"
              % (a, tot[a], 100.0 * tot[a] / solo_total))

    if len(tot) == 2:
        sat = 100.0 * tot["hami"] / solo_total
        d = 100.0 * (tot["hami-mps"] - tot["hami"]) / tot["hami"]
        print("\n  MPS effect: %+.2f%%, measured at %.1f%% saturation." % (d, sat))
        if sat > 95:
            print("  The card was not saturated. A near-zero MPS number here")
            print("  reports the offered rate, not the mechanism -- raise RATE")
            print("  and re-run before concluding anything.")
        print("\n  per tenant, each against its OWN solo baseline:")
        for r in ROLES:
            h, m, s = arms["hami"][r], arms["hami-mps"][r], solo[r]
            print("    %-9s hami %6.1f%% -> mps %6.1f%%  (MPS %+.2f%%)"
                  % (r, 100.0 * h["tokens_per_second"] / s["tokens_per_second"],
                     100.0 * m["tokens_per_second"] / s["tokens_per_second"],
                     100.0 * (m["tokens_per_second"] - h["tokens_per_second"])
                     / h["tokens_per_second"]))
        for a in ("hami", "hami-mps"):
            sh = [100.0 * arms[a][r]["tokens_per_second"] / solo[r]["tokens_per_second"]
                  for r in ROLES]
            print("  fairness spread (%-8s): %5.2f points" % (a, max(sh) - min(sh)))
    print()


if __name__ == "__main__":
    main()
