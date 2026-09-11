#!/usr/bin/env python3
"""Collect the 2026-09-11 fault re-runs, and count the run inventory.

Two jobs.

1. The fault runs from the 15:52 series used the FIXED probes -- the PTX
   kernel for global-fault and the corrected poll order for vram-hog -- so
   this is the first chance either has had to record something. Both have
   produced false negatives before (three ways and four ways respectively),
   so the probe's own output matters more than the windows: a run where
   nothing touched the GPU also shows 0/0/0.

2. Count run directories and measured arms. The consolidated report carried
   an ESTIMATE of 63 runs that was never counted. This replaces it with the
   real number.
"""
import json, os, glob, collections

R = "/opt/GPU_Share/round2/results"
NEW = ("20260911-15", "20260911-16")


def jload(p):
    try:
        with open(p) as fh:
            return json.load(fh)
    except Exception:
        return None


print("#" * 72)
print("# 1. THE FAULT RE-RUNS  -- did the probe actually do anything?")
print("#" * 72)

for d in sorted(glob.glob(os.path.join(R, "*"))):
    b = os.path.basename(d)
    if not any(k in b for k in NEW):
        continue
    t = jload(os.path.join(d, "t10.json"))
    if not t:
        continue
    print()
    print("=" * 70)
    print(b)
    print("   fault=%-14s arm=%-9s host_min_avail=%s recovered=%s"
          % (t.get("fault"), t.get("arm"), t.get("host_min_available_mib"),
             t.get("recovered")))
    w = t.get("windows") or {}
    for k in ("before", "during", "after"):
        x = w.get(k) or {}
        print("   %-7s failed=%-5s n=%-5s mean=%-8.3f p99=%.3f"
              % (k, x.get("failed"), x.get("n"), x.get("mean", 0),
                 x.get("p99", 0)))
    mr = t.get("mps_recovery")
    if mr:
        print("   mps_recovery  %s" % json.dumps(mr))
    fd = t.get("fault_detail") or {}

    # global-fault: the probe's own words are the whole story. A host-side
    # rejection (RuntimeError from torch) means no kernel launched; a real
    # CUDA_EXCEPTION_10 should leave context_alive_after false.
    if "probe" in fd:
        print("   PROBE         %s" % json.dumps(fd["probe"]))
    for key in ("launch_rc", "hog_phase", "hog_held_mib",
                "hog_container_state", "hog_gpu_uuid", "how",
                "daemon_after", "mps_clients_after"):
        if key in fd:
            print("   %-13s %s" % (key, json.dumps(fd[key])[:300]))
    if "hog_events" in fd:
        print("   hog_events    %s"
              % (fd["hog_events"] or "").replace("\n", " | ")[:400])
    if "hog_log_tail" in fd:
        print("   hog_log_tail  %s"
              % (fd["hog_log_tail"] or "").replace("\n", " | ")[-400:])
    if "out_tail" in fd:
        print("   out_tail      %s"
              % (fd["out_tail"] or "").replace("\n", " | ")[-500:])

print()
print("#" * 72)
print("# 2. RUN INVENTORY  -- the real counts, for the report")
print("#" * 72)

tiers = collections.Counter()
arms = 0
for d in sorted(glob.glob(os.path.join(R, "*"))):
    if not os.path.isdir(d):
        continue
    tiers[os.path.basename(d).split("-")[0]] += 1
    for a in os.listdir(d):
        if os.path.exists(os.path.join(d, a, "replay.summary.json")):
            arms += 1

print()
print("run directories by tier:")
for k in sorted(tiers, key=lambda x: (len(x), x)):
    print("   %-24s %d" % (k, tiers[k]))
print()
print("   TOTAL run directories : %d" % sum(tiers.values()))
print("   TOTAL measured arms   : %d" % arms)
