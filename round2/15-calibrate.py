#!/usr/bin/env python3
"""Find the rate each model can actually sustain, before the matrix uses one.

WHY
T1 offered 6.0 req/s to llm-embed, so ~3/s to the 7B, and produced 0/885
successful LLM requests in every arm. The smoke test then showed a single chat
request succeeding, so nothing was broken -- the rate was simply above what the
card sustains. An OPEN loop does not self-limit: past capacity the queue grows
without bound and every request eventually exceeds the client timeout.

That rate was picked from the plan and never measured. It also means T1 (6/s)
and T5 (12/s) were probably both past the knee, i.e. the same test run twice.

WHAT THIS MEASURES
A single tenant, dedicated, at a ladder of offered rates. The knee is where
ACHIEVED throughput stops tracking OFFERED load -- that is the definition that
does not depend on run length, unlike a latency threshold, because past
capacity latency grows with duration rather than settling.

    achieved / offered  >= 0.95   and   success >= 0.98    -> sustainable

    python 15-calibrate.py --role llm
    python 15-calibrate.py --role llm --rates 0.5,1,1.5,2,3,4
    python 15-calibrate.py --role embed --step 60

Writes calibration/<role>.json and prints the rates the matrix should use.
"""
import argparse
import datetime
import json
import os
import subprocess
import sys
import time

import harness as H
import render as R

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

# What fraction of the scenario's total rate each role receives, from
# gen_trace.SCENARIOS. Used to turn a per-role knee into a scenario rate.
WEIGHTS = {
    "llm-embed": {"llm": 0.5, "embed": 0.5},
    "llm-llm": {"llm": 0.5, "llm2": 0.5},
    "vl-rerank": {"vl": 0.5, "rerank": 0.5},
    "dense4": {"llm2": 0.3, "embed": 0.3, "embed-small": 0.2, "rerank": 0.2},
}


def run(cmd, timeout=3600):
    p = subprocess.run([str(c) for c in cmd], cwd=HERE, capture_output=True,
                       text=True, timeout=timeout)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def one_rate(role, rate, step, seed, workdir, endpoint, model, ev):
    """Offer `rate` req/s for `step` seconds; report what came back."""
    tag = "r%05.2f" % rate
    trace = os.path.join(workdir, "trace-%s.json" % tag)
    rc, out = run([PY, os.path.join("workload", "gen_trace.py"),
                   "--scenario", "%s-only" % role, "--rate", rate,
                   "--duration", step, "--source", "sharegpt",
                   "--seed", seed, "--warmup", 4, "--out", trace])
    if rc != 0:
        return {"rate": rate, "error": "gen_trace: " + out[-500:]}

    outdir = os.path.join(workdir, tag)
    os.makedirs(outdir, exist_ok=True)
    t0 = time.time()
    rc, out = run([PY, os.path.join("workload", "replay.py"),
                   "--trace", trace, "--arm", "calibrate", "--out-dir", outdir,
                   "--only", role, "--endpoint", "%s=%s" % (role, endpoint),
                   "--model", "%s=%s" % (role, model),
                   "--note", "calibrate %s @ %.2f/s" % (role, rate)],
                  timeout=step * 4 + 600)
    t1 = time.time()
    s = H.read_json(os.path.join(outdir, "replay.summary.json"), {}) or {}
    per = (s.get("per_service") or {}).get(role, {})
    wall = s.get("wall_seconds") or (t1 - t0)
    n_ok = s.get("requests_ok", 0)
    n_tot = s.get("requests_total", 0)
    row = {
        "rate": rate,
        "offered": rate,
        "requests_total": n_tot,
        "requests_ok": n_ok,
        "success": (n_ok / n_tot) if n_tot else 0.0,
        "achieved": (n_ok / wall) if wall else 0.0,
        "wall": wall,
        "tokens_per_second": per.get("tokens_per_second", 0.0),
        "ttft_p95": (per.get("ttft") or {}).get("p95", 0.0),
        "ttft_p99": (per.get("ttft") or {}).get("p99", 0.0),
        "e2e_p99": (per.get("latency_e2e") or {}).get("p99", 0.0),
        "itl_p95": (per.get("itl") or {}).get("p95", 0.0),
        "replay_rc": rc,
    }
    row["tracking"] = (row["achieved"] / rate) if rate else 0.0
    if rc != 0:
        row["replay_tail"] = out.strip().splitlines()[-8:]
    ev.mark("rate_done", rate=rate, ok=n_ok, total=n_tot,
            achieved=round(row["achieved"], 2))
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--role", default="llm", choices=sorted(R.MODELS))
    ap.add_argument("--rates", default="",
                    help="comma-separated offered rates; default is a ladder "
                         "chosen per model kind")
    ap.add_argument("--step", type=int, default=90,
                    help="seconds of traffic per rate (default 90)")
    ap.add_argument("--seed", type=int, default=20260826)
    ap.add_argument("--settle", type=int, default=20,
                    help="idle seconds between rates, so a queue built at one "
                         "rate does not contaminate the next")
    a = ap.parse_args()

    role = a.role
    m = R.MODELS[role]
    scen = "%s-only" % role
    if scen not in R.SCENARIOS:
        sys.exit("no single-tenant scenario '%s'; add it to render.py, "
                 "gen_trace.py and run.py" % scen)

    if a.rates:
        rates = [float(x) for x in a.rates.split(",") if x.strip()]
    elif m["kind"] == "chat":
        # Generative models are the slow ones; start well below T1's 3/s.
        rates = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    else:
        # Encoders are fast: bge-m3 served 868/868 at 3/s with p99 0.425s.
        rates = [2.0, 4.0, 8.0, 16.0, 24.0, 32.0]

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    workdir = os.path.join(HERE, "calibration", "%s-%s" % (role, stamp))
    os.makedirs(workdir, exist_ok=True)
    ev = H.Events(os.path.join(workdir, "events.jsonl"))

    print("=== calibrating %s (%s)" % (role, m["repo"]))
    print("    rates  %s req/s" % ", ".join("%g" % r for r in rates))
    print("    step   %ds each, %ds settle  -> about %d min of traffic"
          % (a.step, a.settle,
             round((len(rates) * (a.step + a.settle)) / 60.0)))
    print("    out    %s\n" % workdir)

    problems = H.preconditions(ev)
    for p in problems:
        print("  precondition: %s" % p)

    # --- deploy the single tenant, dedicated -----------------------------
    H.teardown(quiet=True)
    H.wait_gpu_idle(timeout=240, ev=ev)
    rdir = os.path.join(workdir, "rendered")
    rc, out = run([PY, os.path.join(HERE, "render.py"), "--scenario", scen,
                   "--arm", "dedicated", "--out", rdir])
    if rc != 0:
        sys.exit("render failed:\n" + out[-1500:])
    rc, out = H.kubectl(["apply", "-f", rdir], timeout=180)
    if rc != 0:
        sys.exit("apply failed:\n" + out[-1500:])
    ev.mark("apply")

    rows = []
    try:
        good, diag = H.wait_ready([role], timeout=1200, ev=ev)
        if not good:
            sys.exit("%s never became Ready:\n%s"
                     % (role, list(diag.values())[0][:2000]))
        eps = H.endpoints([role], ev=ev)
        good, detail = H.smoke_test(eps, {role: m["repo"]},
                                    {role: m["kind"]}, ev=ev)
        if not good:
            sys.exit("smoke test failed, nothing to calibrate:\n  %s"
                     % detail.get(role))
        print("  smoke: %s\n" % detail[role])

        hdr = ("  %-7s %-9s %-9s %-8s %-9s %-9s %-9s %s"
               % ("offered", "achieved", "tracking", "success", "tok/s",
                  "TTFTp95", "TTFTp99", "e2e p99"))
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for rate in rates:
            row = one_rate(role, rate, a.step, a.seed, workdir,
                           eps[role], m["repo"], ev)
            rows.append(row)
            if "error" in row:
                print("  %-7g  ERROR %s" % (rate, row["error"][:80]))
                continue
            flag = ""
            if row["tracking"] < 0.95 or row["success"] < 0.98:
                flag = "   <== past the knee"
            print("  %-7g %-9.2f %-9.2f %-8.1f%% %-9.1f %-9.3f %-9.3f %-8.3f%s"
                  % (rate, row["achieved"], row["tracking"],
                     100 * row["success"], row["tokens_per_second"],
                     row["ttft_p95"], row["ttft_p99"], row["e2e_p99"], flag))
            time.sleep(a.settle)
    finally:
        H.teardown(quiet=True)
        H.wait_gpu_idle(timeout=240)
        ev.close()

    # --- verdict ---------------------------------------------------------
    # Throughput tracking alone is too noisy to locate the knee. `achieved`
    # is measured against the NOMINAL rate, but a Poisson trace of 90 s at
    # 0.5/s holds ~40 requests, not 45, so tracking reads 0.89 at a rate the
    # model handles with 100% success. The first run of this flagged 0.5/s as
    # "past the knee" while 6/s passed -- exactly backwards.
    #
    # Latency is the sharp signal. Below capacity, added load costs almost
    # nothing per token; past it, queueing shows up in inter-token spacing
    # immediately. On the 7B the ladder ran 0.011, 0.011, 0.012, 0.013, 0.018,
    # 0.022, then 0.071 -- a 3.2x jump between 4/s and 6/s.
    clean = [r for r in rows if "error" not in r]
    base_itl = min((r["itl_p95"] for r in clean if r["itl_p95"] > 0), default=0.0)
    base_ttft = min((r["ttft_p95"] for r in clean if r["ttft_p95"] > 0), default=0.0)

    def sustainable(r):
        if r["success"] < 0.98:
            return False, "success %.1f%%" % (100 * r["success"])
        # Offered load as the TRACE actually contains it, not the nominal rate.
        offered = (r["requests_total"] / r["wall"]) if r["wall"] else 0.0
        if offered and r["achieved"] / offered < 0.95:
            return False, "throughput fell behind the trace"
        if base_itl and r["itl_p95"] > 2.0 * base_itl:
            return False, ("ITL p95 %.4f is %.1fx the unloaded %.4f"
                           % (r["itl_p95"], r["itl_p95"] / base_itl, base_itl))
        if not base_itl and base_ttft and r["ttft_p95"] > 2.0 * base_ttft:
            return False, ("TTFT p95 %.3f is %.1fx the unloaded %.3f"
                           % (r["ttft_p95"], r["ttft_p95"] / base_ttft, base_ttft))
        return True, ""

    verdicts = {}
    good_rows = []
    for r in clean:
        okr, why = sustainable(r)
        verdicts[r["rate"]] = (okr, why)
        r["sustainable"] = okr
        r["why_not"] = why
        if okr:
            good_rows.append(r)
    knee = max((r["rate"] for r in good_rows), default=0.0)
    print("\n  per-rate verdict (latency baseline: ITL p95 %.4f)" % base_itl)
    for r in clean:
        okr, why = verdicts[r["rate"]]
        print("    %-6g %s%s" % (r["rate"], "sustainable" if okr else "OVER",
                                 ("  -- " + why) if why else ""))
    peak_tok = max((r["tokens_per_second"] for r in rows
                    if "error" not in r), default=0.0)

    doc = {"role": role, "repo": m["repo"], "kind": m["kind"],
           "step_seconds": a.step, "seed": a.seed, "rows": rows,
           "knee_req_s": knee, "peak_tokens_per_second": peak_tok,
           "measured_at": time.time()}
    H.write_json(os.path.join(workdir, "calibration.json"), doc)
    H.write_json(os.path.join(HERE, "calibration", "%s.json" % role), doc)

    print("\n" + "=" * 70)
    if knee <= 0:
        print("  NO sustainable rate found. Even the lowest offered rate did not")
        print("  track. Either the ladder starts too high, or something other")
        print("  than load is wrong -- check %s/r*/replay.summary.json" % workdir)
        print("=" * 70)
        return 1
    print("  %s sustains %.2f req/s  (peak %.0f tok/s)" % (role, knee, peak_tok))
    print("\n  Suggested matrix rates, PER SCENARIO (the scenario rate is split")
    print("  across roles, so a shared scenario needs the total scaled up):")
    for scen_name, w in sorted(WEIGHTS.items()):
        if role not in w:
            continue
        below = knee * 0.7 / w[role]
        above = knee * 1.5 / w[role]
        print("    %-12s  below knee %5.1f/s   past knee %5.1f/s"
              % (scen_name, below, above))
    print("\n  Note this is the DEDICATED knee. Under contention it drops, so a")
    print("  rate at 70%% of it should stay sustainable in the sharing arms")
    print("  while leaving room for the difference to show as latency rather")
    print("  than as timeouts.")
    print("=" * 70)
    print("\nwrote %s" % os.path.join(HERE, "calibration", "%s.json" % role))
    return 0


if __name__ == "__main__":
    sys.exit(main())
