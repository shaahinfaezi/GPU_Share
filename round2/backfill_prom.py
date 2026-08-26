#!/usr/bin/env python3
"""Recover the server-side metrics a run failed to collect.

WHY THIS EXISTS
The first T1 run lost prom.json for every sub-run. GpuSampler subclassed
threading.Thread and assigned self._stop = threading.Event(), shadowing
Thread's private _stop() METHOD. join() calls that method once the thread
finishes, so sampler.stop() raised

    TypeError: 'Event' object is not callable

in the calling thread -- one line before H.prom_collect(). Every sub-run
printed `replay_end rc=0` and then `ERROR`, and the run looked healthy apart
from one odd line.

Nothing irreplaceable was lost. The client-side data (replay.summary.json,
replay.requests.csv, replay.itl.json) comes from a separate process that exited
0, and gpu.csv was flushed as it went. Only the Prometheus snapshot is missing,
and Prometheus still HAS that data -- kube-prometheus-stack retains ~10 days by
default. This re-queries it for each sub-run's measurement window.

The window comes from each sub-run's OWN events.jsonl. replay.requests.csv
stores times as offsets from an in-process perf_counter origin with no absolute
anchor, so it cannot supply one. run.py keeps two phase logs -- one per sub-run
and one at the run root -- and replay_begin/replay_end are marked inside the
sub-run, so they land in the sub-run's file. An earlier version of this script
read only the root log, found nothing, and skipped every sub-run.

replay.py now also records epoch_start and epoch_end bracketing the post-warmup
measurement window, so future runs anchor themselves and this script prefers
those when present.

    python backfill_prom.py results/T1-20260826-175322
    python backfill_prom.py results/T1-*            # several runs
    python backfill_prom.py results/T1-... --dry-run
"""
import argparse
import glob
import json
import os
import sys
import time

import harness as H


def _pairs(path):
    """(t0, t1) replay windows from one phase log, in order."""
    if not os.path.exists(path):
        return []
    begins, pairs = [], []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("phase") == "replay_begin":
                begins.append(rec["t"])
            elif rec.get("phase") == "replay_end" and begins:
                pairs.append((begins.pop(0), rec["t"]))
    return pairs


def windows_from_events(run_dir):
    """Ordered (t0, t1) pairs from the RUN-ROOT log, if it has any.

    run.py keeps two phase logs: one per sub-run in the sub-run directory, and
    one at the run root. replay_begin/replay_end are marked inside the sub-run,
    so they land in the sub-run's own file and the root log has none. This is
    only the fallback; subrun_window() prefers the per-sub-run file, which needs
    no ordering inference at all.
    """
    return _pairs(os.path.join(run_dir, "events.jsonl"))


def subrun_dirs(run_dir):
    """Sub-run directories in the order they were created, which is plan order."""
    out = []
    for name in os.listdir(run_dir):
        d = os.path.join(run_dir, name)
        if os.path.isdir(d) and "__" in name:
            out.append(d)
    return sorted(out, key=lambda d: os.path.getmtime(d))


def subrun_window(d):
    """The measurement window for one sub-run, best source first.

    Returns ((t0, t1), source) or (None, None).
    """
    summ = H.read_json(os.path.join(d, "replay.summary.json"), {}) or {}
    t0, t1 = summ.get("epoch_start"), summ.get("epoch_end")
    if t0 and t1:
        return (float(t0), float(t1)), "summary epoch"

    own = _pairs(os.path.join(d, "events.jsonl"))
    if own:
        # The last pair: a sub-run replays once, but a re-run into the same
        # directory would append, and the most recent one is the live data.
        return own[-1], "sub-run events.jsonl"
    return None, None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="re-collect even where prom.json already exists")
    a = ap.parse_args()

    roots = []
    for pat in a.run_dirs:
        roots.extend(sorted(glob.glob(pat)) or [pat])

    base = H.prometheus_url()
    if not base:
        sys.exit("Prometheus ClusterIP not resolvable; nothing to query")
    print("prometheus %s" % base)

    total, done, skipped, failed = 0, 0, 0, 0
    for root in roots:
        if not os.path.isdir(root):
            print("  %s: not a directory" % root)
            continue
        print("\n=== %s" % root)
        subs = subrun_dirs(root)
        ev_windows = windows_from_events(root)

        for i, d in enumerate(subs):
            total += 1
            name = os.path.basename(d)
            out = os.path.join(d, "prom.json")
            if os.path.exists(out) and not a.force:
                print("   %-26s prom.json present, skipping" % name)
                skipped += 1
                continue

            win, src = subrun_window(d)
            if win is None:
                if i < len(ev_windows):
                    win, src = ev_windows[i], "run-root events.jsonl (by order)"
                else:
                    print("   %-26s no window available, skipping" % name)
                    skipped += 1
                    continue
            t0, t1 = win

            age_d = (time.time() - t0) / 86400.0
            span = t1 - t0
            print("   %-26s %s  %.0fs window, %.1f day(s) ago"
                  % (name, src, span, age_d))
            if age_d > 9:
                print("      WARNING: near the default 10-day retention; "
                      "Prometheus may already have dropped this")
            if a.dry_run:
                continue

            doc = H.prom_collect(out, t0, t1, base=base)
            if not doc:
                print("      collection failed")
                failed += 1
                continue
            n = sum(len(v.get("result", []))
                    for v in doc["series"].values() if isinstance(v, dict))
            empty = [k for k, v in doc["series"].items()
                     if isinstance(v, dict) and not v.get("result")]
            print("      wrote %s: %d series" % (os.path.basename(out), n))
            if empty:
                print("      %d query/queries returned nothing: %s"
                      % (len(empty), ", ".join(sorted(empty)[:6])))
            done += 1

        # repair run.json's per-sub-run bookkeeping, which the same exception
        # skipped: status, the client summary, and the trace_id comparison.
        rj = os.path.join(root, "run.json")
        doc = H.read_json(rj)
        if doc and not a.dry_run:
            changed = False
            for rec in doc.get("subruns", []):
                d = os.path.join(root, rec.get("dir", ""))
                summ = H.read_json(os.path.join(d, "replay.summary.json"), {}) or {}
                if summ and rec.get("status") in ("error", None):
                    rec["summary"] = summ
                    rec["trace_id_seen"] = (summ.get("trace_id")
                                            or summ.get("meta", {}).get("trace_id"))
                    rec["status"] = "ok" if rec.get("replay_rc") == 0 else rec.get("status")
                    rec["backfilled"] = True
                    changed = True
            if changed:
                H.write_json(rj, doc)
                print("   run.json: repaired sub-run bookkeeping")

    print("\n%d sub-run(s): %d collected, %d skipped, %d failed"
          % (total, done, skipped, failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
