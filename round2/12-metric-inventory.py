#!/usr/bin/env python3
"""Ask Prometheus which metrics actually exist, and check our queries against it.

WHY THIS EXISTS
The T1 backfill reported seven queries returning nothing:

    hami_cores, hami_limit, hami_mem, itl_p95, itl_p99, kv_usage, ...

Two different causes hide behind that, and they need different fixes:

  * NOT SCRAPED -- the HAMi series. If the ServiceMonitors were never applied,
    or their selectors match nothing, Prometheus has no data and never did.
    Unrecoverable for a run that has already happened.

  * WRONG NAME -- vllm:time_per_output_token_seconds_bucket and
    vllm:gpu_cache_usage_perc. TTFT queries worked in the same run, so vLLM
    metrics are being scraped; those two names are simply wrong for 0.15.1.
    Recoverable by backfilling once the name is corrected, since the data is
    there under a different label.

Guessing which is which is how this went wrong in the first place. This asks.

    python 12-metric-inventory.py            # what exists, what our queries need
    python 12-metric-inventory.py --all      # dump every vllm:/hami_/DCGM_ name
"""
import argparse
import json
import re
import sys
import urllib.parse
import urllib.request

import harness as H


def label_values(base, label="__name__", timeout=30):
    url = "%s/api/v1/label/%s/values" % (base, label)
    with urllib.request.urlopen(url, timeout=timeout) as r:
        doc = json.loads(r.read().decode("utf-8"))
    if doc.get("status") != "success":
        raise RuntimeError(doc.get("error", "non-success"))
    return doc.get("data", [])


def targets(base, timeout=30):
    url = "%s/api/v1/targets?state=active" % base
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            doc = json.loads(r.read().decode("utf-8"))
        return doc.get("data", {}).get("activeTargets", [])
    except Exception:
        return []


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true", help="list every matching name")
    a = ap.parse_args()

    base = H.prometheus_url()
    if not base:
        sys.exit("Prometheus ClusterIP not resolvable")
    print("prometheus %s\n" % base)

    names = set(label_values(base))
    fams = {
        "vllm:":  sorted(n for n in names if n.startswith("vllm:")),
        "hami_":  sorted(n for n in names if n.startswith("hami_")),
        "DCGM_":  sorted(n for n in names if n.startswith("DCGM_")),
    }
    print("=== what Prometheus actually holds")
    for pref, got in fams.items():
        print("  %-8s %d metric name(s)" % (pref, len(got)))
        if a.all:
            for n in got:
                print("      %s" % n)
    if not a.all:
        print("  (--all to list them)")

    # --- are the exporters even being scraped? ---------------------------
    print("\n=== scrape targets relevant to round two")
    act = targets(base)
    if not act:
        print("  could not read /api/v1/targets")
    else:
        seen = {}
        for t in act:
            job = t.get("labels", {}).get("job", "?")
            if re.search(r"hami|dcgm|vllm|r2-", job, re.I):
                seen.setdefault(job, [0, 0])
                seen[job][0] += 1
                seen[job][1] += (t.get("health") == "up")
        if not seen:
            print("  NONE. No HAMi, DCGM or vLLM target is being scraped at all.")
            print("  Apply the monitors and re-check:")
            print("      kubectl apply -f monitoring/60-servicemonitors.yaml")
        for job, (n, up) in sorted(seen.items()):
            flag = "" if up == n else "   <== some DOWN"
            print("  %-46s %d/%d up%s" % (job, up, n, flag))

    # --- cross-check every query the harness relies on --------------------
    print("\n=== harness.QUERIES vs reality")
    missing, ok = [], 0
    for key, tmpl in sorted(H.QUERIES.items()):
        expr = tmpl % {"ns": H.NS}
        used = set(re.findall(r"\b((?:vllm:|hami_|DCGM_)[A-Za-z_:0-9]+)", expr))
        absent = sorted(m for m in used if m not in names)
        if absent:
            missing.append((key, absent))
        else:
            ok += 1
    print("  %d/%d queries reference only metrics that exist" % (ok, len(H.QUERIES)))
    for key, absent in missing:
        print("  %-14s MISSING: %s" % (key, ", ".join(absent)))
        for m in absent:
            stem = m.split(":")[-1]
            stem = re.sub(r"_(bucket|sum|count|total|perc)$", "", stem)
            near = sorted(n for n in names
                          if stem.split("_")[0] in n and n.startswith(m.split(":")[0]))
            if near:
                print("      candidates: %s" % ", ".join(near[:6]))

    if missing:
        print("\nFix the names in harness.QUERIES, then re-collect for a past run:")
        print("    python backfill_prom.py results/<run> --force")
        print("A metric that was never SCRAPED cannot be recovered that way --")
        print("check the target list above before assuming a rename.")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
