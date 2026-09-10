#!/usr/bin/env python3
"""Dump the headline numbers from one or more result directories.

Exists because every analysis so far has been a bespoke one-liner, and two
published claims came from reading the wrong field. This prints exactly the
keys the reports quote, and says plainly when one is missing rather than
substituting a zero.

    python3 30-collect.py                     # every run dir, newest first
    python3 30-collect.py T14 T18             # only dirs whose name matches
    python3 30-collect.py --since 20260909    # only runs from this date on
"""
import json, os, sys, glob

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def load(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def g(d, *path, **kw):
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return kw.get("default")
        cur = cur[k]
    return cur


def fmt(v, nd=2):
    if v is None:
        return "--"
    if isinstance(v, float):
        return ("%%.%df" % nd) % v
    return str(v)


def show_replay(run):
    """A normal tier run: one sub-directory per arm."""
    arms = sorted(d for d in os.listdir(run)
                  if os.path.isdir(os.path.join(run, d)))
    rows = []
    for arm in arms:
        s = load(os.path.join(run, arm, "replay.summary.json"))
        if not s:
            continue
        per = g(s, "per_service", default={}) or {}
        tot = 0.0
        parts = []
        for role in sorted(per):
            tps = g(per, role, "tokens_per_second")
            mean = g(per, role, "latency_e2e", "mean")
            p99 = g(per, role, "latency_e2e", "p99")
            ttft = g(per, role, "ttft", "p99")
            n = g(per, role, "requests")
            ok = g(per, role, "ok")
            if tps:
                tot += tps
            parts.append("      %-12s tok/s %9s  mean %8s  p99 %8s  ttft99 %8s  %s/%s"
                         % (role, fmt(tps), fmt(mean, 3), fmt(p99, 3),
                            fmt(ttft, 3), fmt(ok, 0), fmt(n, 0)))
        rows.append((arm, tot, parts))

    if not rows:
        return False
    print("  arms: %d" % len(rows))
    for arm, tot, parts in rows:
        print("    %-26s combined %9s tok/s" % (arm, fmt(tot)))
        for p in parts:
            print(p)

    # the comparison the reports actually quote
    by = {a: t for a, t, _ in rows}
    hami = by.get("hami__" + _scen(by), None)
    h = [v for k, v in by.items() if k.startswith("hami__")]
    m = [v for k, v in by.items() if k.startswith("hami-mps__")]
    if h and m and h[0]:
        print("    >> MPS effect on combined throughput: %+.2f%%"
              % ((m[0] - h[0]) / h[0] * 100.0))
    return True


def _scen(by):
    for k in by:
        if "__" in k:
            return k.split("__", 1)[1]
    return ""


def show_fault(run):
    """A 23-failure-inject.py run: one t10.json at the top."""
    t = load(os.path.join(run, "t10.json"))
    if not t:
        return False
    print("  fault   : %s" % g(t, "fault", default="?"))
    print("  arm     : %s" % g(t, "arm", default="?"))
    for key in ("victim", "survivor"):
        w = g(t, key, default=None)
        if isinstance(w, dict):
            for win in ("before", "during", "after"):
                d = g(w, win, default={}) or {}
                print("    %-8s %-7s failed %s / %s   p99 %s"
                      % (key, win, fmt(d.get("failed"), 0),
                         fmt(d.get("total"), 0), fmt(d.get("p99"), 3)))
    for k in ("mps_daemon_state", "mps_recovery", "hog_phase",
              "hog_container_state", "grabbed_mib", "sees_free_mib",
              "sees_total_mib", "host_min_available_mib",
              "context_alive_after", "error_class"):
        v = g(t, k, default=None)
        if v is not None:
            print("    %-24s %s" % (k, v))
    ev = g(t, "hog_events", default=None)
    if ev:
        print("    hog_events:")
        for e in (ev if isinstance(ev, list) else [ev])[:8]:
            print("      %s" % e)
    return True


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    since = None
    for a in sys.argv[1:]:
        if a.startswith("--since"):
            since = a.split("=", 1)[-1] if "=" in a else None
    if since is None and "--since" in sys.argv:
        i = sys.argv.index("--since")
        if i + 1 < len(sys.argv):
            since = sys.argv[i + 1]
            args = [x for x in args if x != since]

    runs = sorted(glob.glob(os.path.join(R, "*")), reverse=True)
    runs = [r for r in runs if os.path.isdir(r)]
    if args:
        runs = [r for r in runs
                if any(a.lower() in os.path.basename(r).lower() for a in args)]
    if since:
        runs = [r for r in runs if since in os.path.basename(r)
                or os.path.basename(r).split("-")[-2:-1] and
                _date_of(r) >= since]

    if not runs:
        print("no matching run directories under %s" % R)
        return 1

    for run in runs:
        print("=" * 78)
        print(os.path.basename(run))
        if not (show_fault(run) or show_replay(run)):
            print("  (no replay.summary.json and no t10.json)")
    return 0


def _date_of(run):
    parts = os.path.basename(run).split("-")
    for p in parts:
        if len(p) == 8 and p.isdigit():
            return p
    return "00000000"


if __name__ == "__main__":
    sys.exit(main())
