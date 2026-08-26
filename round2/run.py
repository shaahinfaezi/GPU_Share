#!/usr/bin/env python3
"""Run one round-2 test across all three arms and save everything needed to
re-render the result later.

    python run.py --list
    python run.py --test T1                        # all three arms
    python run.py --test T1 --arms hami,hami-mps   # skip the baseline
    python run.py --test T1 --dry-run              # print the plan, touch nothing

WHAT AN ARM MEANS HERE

  dedicated   requirement 2: the same environment WITHOUT sharing. For a
              two-tenant scenario this is not one run, it is one run PER
              TENANT -- llm alone, then embed alone. Each replays only its own
              rows from the shared trace, so the arrival times a tenant sees
              are byte-identical to the ones it sees under contention. That is
              the only way "what did sharing cost me" has an answer.

              The gpumem quota stays at the shared value on purpose. Handing
              the baseline the whole card would make its extra throughput a
              memory-budget effect wearing a sharing costume.

  hami        co-tenants, HAMi gpumem + gpucores, no MPS.
  hami-mps    identical to hami plus the MPS env block. Diffing the two
              rendered manifests shows only those lines.

FAIRNESS, ENFORCED RATHER THAN INTENDED (requirement 7)

  * one trace per test, replayed by every arm; its SHA-256 trace_id is stored
    in each sub-run and compared at the end. Divergence fails the run.
  * co-tenants forced onto ONE physical GPU, and the placement VERIFIED after
    deployment. HAMi ships gpuSchedulerPolicy=spread, so two "co-tenants" would
    otherwise land on separate cards and produce a clean result that measures
    nothing. Co-location comes from the scheduler (10-hami-binpack.sh), because
    the per-pod pin does not survive KServe. Verification is against nvidia-smi
    as well as HAMi's annotation, and a split aborts the arm.
  * the GPU is drained to idle between sub-runs, verified by nvidia-smi rather
    than by sleeping.
  * the client talks straight to the pod IP, so the KServe router is outside
    the measurement in every arm.

WHAT LANDS ON DISK (requirement 3)

  results/<test>-<stamp>/
      run.json          the whole plan, plus a verdict per sub-run
      trace.json        the exact bytes every arm replayed
      events.jsonl      phase boundaries, for annotating the screen recording
      <arm>__<roles>/
          rendered/         the manifests as applied
          replay.summary.json, replay.requests.csv, replay.itl.json
          prom.json         every dashboard series over the run window
          gpu.csv           1 Hz nvidia-smi, finer than the 5 s DCGM scrape
          pods.json         placement, images, resource blocks as admitted
"""
import argparse
import datetime
import json
import os
import subprocess
import sys

import harness as H

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

# ShareGPT is fetched by hf_hub_download at trace-generation time and is ~650
# MB. Left to default it lands in ~/.cache/huggingface on `/`, which has ~13
# GiB free against a node-wide eviction threshold of 4.84 GiB. Point it at the
# same /data2 cache the model weights already use. 07-prefetch-traces.sh stages
# both corpora ahead of time so a run never downloads anything.
os.environ.setdefault("HF_HOME", "/data2/hf-cache")
_AZ = "/data2/hf-cache/traces/AzureLLMInferenceTrace_conv.csv"
if os.path.exists(_AZ):
    os.environ.setdefault("AZURE_TRACE_CSV", _AZ)

# Scenario -> roles, mirroring render.py and workload/gen_trace.py.
ROLES = {
    "llm-only": ["llm"], "embed-only": ["embed"], "rerank-only": ["rerank"],
    "llm-embed": ["llm", "embed"], "llm-llm": ["llm", "llm2"],
    "vl-rerank": ["vl", "rerank"],
    "dense4": ["llm2", "embed", "embed-small", "rerank"],
}

# The matrix. Rates are offered load in requests/second, Poisson-distributed --
# an OPEN loop, so an overloaded arm builds a queue instead of quietly reducing
# its own offered load the way a fixed-concurrency client does.
TESTS = {
    "T1": dict(scenario="llm-embed", rate=6.0, duration=300, source="sharegpt",
               why="Headline. 7B-AWQ + bge-m3 fills 91.7% of the card against "
                   "round one's measured 62.7-69.3% (peak VRAM on GPU 0, from "
                   "its own gpu-timeline.csv). The bigger gap is the workload: "
                   "round one used 51-token prompts and fixed 256-token "
                   "outputs at concurrency 4, which is decode-bound and never "
                   "queues. Registered prediction: filling the card and "
                   "letting it queue widens the MPS advantage."),
    "T2": dict(scenario="llm-llm", rate=6.0, duration=300, source="sharegpt",
               why="Two generative tenants. Both contend for decode, so ITL "
                   "should degrade symmetrically -- unlike T1, where one "
                   "tenant is prefill-dominated."),
    "T3": dict(scenario="vl-rerank", rate=4.0, duration=300, source="sharegpt",
               why="Vision encoder plus a cross-encoder reranker: two "
                   "compute-bound, non-autoregressive neighbours. The case "
                   "where gpucores SHOULD matter most if it ever does."),
    "T4": dict(scenario="dense4", rate=8.0, duration=300, source="sharegpt",
               why="Four tenants on one card at 97.9% fill: one generative and "
                   "three encoders. Requirement 1 asks for number of models as "
                   "a variable; this is its top end. Quotas are tightened "
                   "per-scenario -- see SCENARIO_QUOTAS in render.py."),
    "T5": dict(scenario="llm-embed", rate=12.0, duration=300, source="sharegpt",
               why="Same as T1 past the knee. Queue depth and preemptions "
                   "become the story; p99 separates from p50."),
    "T6": dict(scenario="llm-embed", rate=6.0, duration=300, source="azure",
               why="Real production arrival trace rather than a synthetic "
                   "Poisson process -- burstiness the generator cannot "
                   "invent (requirement 4)."),
    "T7": dict(scenario="llm-only", rate=6.0, duration=300, source="sharegpt",
               fixed_output=32,
               why="Prefill-heavy, single tenant. Round one's second "
                   "registered prediction: the sub-linear SM-to-throughput "
                   "mapping straightens under prefill-dominated load."),
    "T8": dict(scenario="llm-embed", rate=6.0, duration=300, source="synthetic",
               fixed_output=128,
               why="Round-one control. Fixed output length and synthetic "
                   "prompts reproduce the old conditions, so the two rounds "
                   "can be compared directly instead of by assertion."),
}

ARMS = ["dedicated", "hami", "hami-mps"]


def run(cmd, cwd=HERE, timeout=3600, echo=True):
    if echo:
        print("    $ " + " ".join(str(c) for c in cmd))
    p = subprocess.run([str(c) for c in cmd], cwd=cwd, capture_output=True,
                       text=True, timeout=timeout)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def gen_trace(test, cfg, out_path, seed, warmup):
    """One trace, generated once, replayed by everything."""
    cmd = [PY, os.path.join("workload", "gen_trace.py"),
           "--scenario", cfg["scenario"], "--rate", cfg["rate"],
           "--duration", cfg["duration"], "--source", cfg["source"],
           "--seed", seed, "--warmup", warmup, "--out", out_path]
    if cfg.get("fixed_output"):
        cmd += ["--fixed-output", cfg["fixed_output"]]
    rc, out = run(cmd)
    if rc != 0:
        raise RuntimeError("gen_trace failed:\n" + out[-3000:])
    doc = H.read_json(out_path, {})
    tid = doc.get("trace_id") or doc.get("meta", {}).get("trace_id")
    n = len(doc.get("requests", []))

    # gen_trace falls back to synthetic when a corpus is missing, and says so
    # in one warning line that is easy to miss inside a long run. T6 exists
    # ONLY to replay production arrival shapes -- a silent fallback would leave
    # it measuring exactly what T1 measures while the report claims otherwise.
    got = (doc.get("params") or {}).get("source")
    if got and got != cfg["source"]:
        raise RuntimeError(
            "test %s asked for source=%s but the trace resolved to %s.\n"
            "The corpus is not staged. Fix that rather than running a test "
            "that would quietly measure something else:\n"
            "    ./07-prefetch-traces.sh" % (test, cfg["source"], got))
    print("    trace_id=%s  requests=%d  source=%s" % (tid, n, got))
    return tid, n


def subrun(run_dir, ev_global, test, cfg, arm, roles, trace_path, trace_id,
           pin_uuid, pooling_via, mps_percent, warmup, dry):
    """Deploy, measure, tear down -- one arm, one set of co-resident roles."""
    tag = "%s__%s" % (arm, "+".join(roles))
    d = os.path.join(run_dir, tag)
    os.makedirs(d, exist_ok=True)
    ev = H.Events(os.path.join(d, "events.jsonl"))
    rec = dict(tag=tag, arm=arm, roles=roles, trace_id=trace_id,
               scenario=cfg["scenario"], test=test, status="started")

    print("\n--- %s" % tag)
    try:
        # 1. clean slate, verified rather than assumed --------------------
        # Skipped under --dry-run: both of these shell out to the cluster, and
        # a dry run that needs a cluster is not a dry run.
        if not dry:
            ev.mark("teardown_begin")
            H.teardown()
            H.wait_gpu_idle(ev=ev)

        # 2. render -------------------------------------------------------
        rdir = os.path.join(d, "rendered")
        cmd = [PY, "render.py", "--scenario", cfg["scenario"], "--arm", arm,
               "--out", rdir, "--pooling-via", pooling_via]
        for r in roles:
            cmd += ["--only", r]
        if pin_uuid:
            cmd += ["--pin-gpu", pin_uuid]
        if mps_percent:
            cmd += ["--mps-percent", mps_percent]
        rc, out = run(cmd)
        if rc != 0:
            raise RuntimeError("render failed:\n" + out[-2000:])
        rec["render"] = out.strip().splitlines()[-4:]

        if dry:
            rec["status"] = "dry-run"
            return rec

        # 3. apply and wait ----------------------------------------------
        ev.mark("apply")
        rc, out = H.kubectl(["apply", "-f", rdir], timeout=180)
        if rc != 0:
            raise RuntimeError("kubectl apply failed:\n" + out[-2000:])

        ev.mark("wait_ready", roles=",".join(roles))
        ok, diag = H.wait_ready(roles, ev=ev)
        if not ok:
            for role, text in diag.items():
                print("      %s did not become ready:\n%s" % (role, text[:2500]))
            raise RuntimeError("roles never became ready: %s" % ", ".join(diag))

        # 4. placement -- the check that protects the whole comparison ----
        colocated, detail = H.assert_colocated(roles, ev=ev)
        rec["placement"] = detail
        if not colocated:
            raise RuntimeError(
                "tenants landed on DIFFERENT GPUs: %s\nThis arm would not "
                "actually be sharing anything. Run ./10-hami-binpack.sh, or set "
                "the HAMi scheduler's gpuSchedulerPolicy to binpack."
                % json.dumps(detail))
        if detail.get("warning"):
            print("      NOTE: " + detail["warning"])

        # 5. endpoints and model names -----------------------------------
        eps = H.endpoints(roles, ev=ev)
        models = {}
        for role, base in eps.items():
            m = H.served_model(base)
            if m:
                models[role] = m
        rec["endpoints"], rec["models"] = eps, models

        H.write_json(os.path.join(d, "pods.json"),
                     {r: H.pods_for(r) for r in roles})

        # 6. the measured run --------------------------------------------
        # Warm-up is not a separate pass: the trace already carries warmup
        # rows (t=-1, warmup=True), replay fires them before the clock rows,
        # and summarize() excludes them. That matters because the first
        # request into a cold vLLM pays CUDA graph capture and torch.compile;
        # inside the measured window it lands in p99 and gets reported as a
        # sharing cost, which it is not.
        # A tenant that cannot serve should cost 30 seconds, not 45 minutes.
        # T1's first attempt ran four arms to completion with 0/885 successful
        # LLM requests: the pods were Ready, placement was verified, the trace
        # matched across arms, and every generative request failed. Readiness
        # only proves /health answers -- it says nothing about whether the
        # endpoint accepts the request shape the client is about to send.
        import render as _R
        _models = {r: _R.MODELS[r]["repo"] for r in roles}
        _kinds = {r: _R.MODELS[r]["kind"] for r in roles}
        smoke_ok, smoke_detail = H.smoke_test(eps, _models, _kinds, ev=ev)
        rec["smoke"] = smoke_detail
        if not smoke_ok:
            lines = "\n  ".join("%s: %s" % kv for kv in sorted(smoke_detail.items()))
            raise RuntimeError("smoke test failed, refusing to spend a "
                               "measurement window on it:\n  " + lines)

        sampler = H.GpuSampler(os.path.join(d, "gpu.csv"))
        sampler.start()
        t0 = ev.mark("replay_begin")
        rc, out = run([PY, os.path.join("workload", "replay.py"),
                       "--trace", trace_path, "--arm", arm, "--out-dir", d,
                       "--note", "%s %s" % (test, tag)]
                      + sum([["--only", r] for r in roles], [])
                      + sum([["--endpoint", "%s=%s" % (r, u)]
                             for r, u in eps.items()], [])
                      + sum([["--model", "%s=%s" % (r, m)]
                             for r, m in models.items()], []),
                      timeout=int(cfg["duration"]) + 1800)
        t1 = ev.mark("replay_end", rc=rc)
        sampler.stop()
        rec["replay_rc"] = rc
        rec["replay_tail"] = out.strip().splitlines()[-25:]
        if rc != 0:
            print("      replay exited %d:\n%s" % (rc, out[-2500:]))

        # 7. server-side metrics, saved as data not as a dashboard link ---
        H.prom_collect(os.path.join(d, "prom.json"), t0, t1, ev=ev)

        summ = H.read_json(os.path.join(d, "replay.summary.json"), {})
        rec["summary"] = summ
        got = summ.get("trace_id") or summ.get("meta", {}).get("trace_id")
        rec["trace_id_seen"] = got
        rec["status"] = "ok" if rc == 0 else "replay-failed"
        rec["t0"], rec["t1"] = t0, t1

    except Exception as exc:
        rec["status"] = "error"
        rec["error"] = "%s: %s" % (type(exc).__name__, exc)
        print("    ERROR %s" % rec["error"])
    finally:
        ev.mark("teardown_end")
        if not dry:
            H.teardown(quiet=True)
        ev.close()
        H.write_json(os.path.join(d, "subrun.json"), rec)
    return rec


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test", help="T1..T8; see --list")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    ap.add_argument("--seed", type=int, default=20260826)
    # Default OFF. The annotation is inert on the KServe path -- KServe drops
    # pod annotations -- so pinning does nothing there, and co-location now
    # comes from the scheduler (10-hami-binpack.sh, verified co-locating both
    # T1 tenants on one card). But it DOES land on the plain-Deployment path,
    # so a mixed render would pin the Deployment tenants to this UUID while
    # binpack sent the KServe tenants wherever it liked -- a guaranteed split,
    # caused by the very flag meant to prevent one.
    #
    # Set it only when EVERY tenant renders as a Deployment (--pooling-via vllm
    # on an all-pooling scenario), where it is then the only mechanism.
    ap.add_argument("--pin-gpu", default="",
                    help="pin every tenant to this device UUID. Leave empty: "
                         "KServe ignores it and binpack handles co-location. "
                         "Only useful when every tenant is a plain Deployment.")
    ap.add_argument("--pooling-via", choices=("kserve", "vllm"), default="kserve")
    ap.add_argument("--mps-percent", default="",
                    help="per-role CUDA_MPS_ACTIVE_THREAD_PERCENTAGE, e.g. "
                         "llm=75,embed=25. Default 100 for every role, which "
                         "isolates 'MPS on/off' from 'MPS partitioned'.")
    ap.add_argument("--warmup", type=int, default=8,
                    help="warm-up requests per role, carried IN the trace and "
                         "excluded from the summary. Do not set this to 0: the "
                         "first request into a cold engine pays graph capture "
                         "and would be scored as a sharing cost.")
    ap.add_argument("--duration", type=float, help="override the test's duration")
    ap.add_argument("--rate", type=float, help="override the test's offered rate")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--skip-preconditions", action="store_true")
    a = ap.parse_args()

    if a.list or not a.test:
        print("%-5s %-12s %6s %6s %-10s  %s"
              % ("test", "scenario", "rate", "dur", "source", "roles"))
        for k in sorted(TESTS):
            c = TESTS[k]
            print("%-5s %-12s %6.1f %6d %-10s  %s"
                  % (k, c["scenario"], c["rate"], c["duration"], c["source"],
                     ", ".join(ROLES[c["scenario"]])))
            for line in _wrap(c["why"], 72):
                print("      %s" % line)
        print("\nT9 (admission / scale-down) lives in 09-scaledown.py -- its "
              "shape is different enough that folding it in here would make "
              "both harder to read.")
        return 0

    if a.test not in TESTS:
        sys.exit("unknown test %s; try --list" % a.test)
    cfg = dict(TESTS[a.test])
    if a.duration:
        cfg["duration"] = a.duration
    if a.rate:
        cfg["rate"] = a.rate
    arms = [x.strip() for x in a.arms.split(",") if x.strip()]
    bad = [x for x in arms if x not in ARMS]
    if bad:
        sys.exit("unknown arm(s): %s" % bad)
    roles = ROLES[cfg["scenario"]]

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(a.out, "%s-%s" % (a.test, stamp))
    os.makedirs(run_dir, exist_ok=True)
    ev = H.Events(os.path.join(run_dir, "events.jsonl"))

    print("=== %s  %s  %s" % (a.test, cfg["scenario"], run_dir))
    print("    %s" % "\n    ".join(_wrap(cfg["why"], 74)))
    print()

    if not a.dry_run and not a.skip_preconditions:
        problems = H.preconditions(ev)
        if problems:
            print("REFUSING TO START:")
            for p in problems:
                print("  - %s" % p)
            print("\nA three-arm run is roughly an hour. Losing it at minute 40 "
                  "to a node-wide eviction is worse than not starting.\n"
                  "Override with --skip-preconditions if you disagree.")
            return 2

    trace_path = os.path.join(run_dir, "trace.json")
    trace_id, n_req = gen_trace(a.test, cfg, trace_path, a.seed, a.warmup)
    ev.mark("trace", trace_id=trace_id, requests=n_req)

    plan = []
    for arm in arms:
        if arm == "dedicated":
            # requirement 2: one baseline PER TENANT, each alone on the card
            plan += [(arm, [r]) for r in roles]
        else:
            plan.append((arm, roles))

    print("\nplan: %d sub-run(s)" % len(plan))
    for arm, rs in plan:
        print("  %-10s %s" % (arm, "+".join(rs)))

    results = []
    for arm, rs in plan:
        results.append(subrun(run_dir, ev, a.test, cfg, arm, rs, trace_path,
                              trace_id, a.pin_gpu, a.pooling_via,
                              a.mps_percent, a.warmup, a.dry_run))

    # ---- the fairness assertion, made loudly ---------------------------
    seen = {r["tag"]: r.get("trace_id_seen") for r in results
            if r.get("status") == "ok"}
    mismatched = {k: v for k, v in seen.items() if v and v != trace_id}
    doc = dict(test=a.test, config=cfg, roles=roles, arms=arms,
               run_dir=run_dir, stamp=stamp, seed=a.seed,
               trace_id=trace_id, requests=n_req,
               pin_gpu=a.pin_gpu, pooling_via=a.pooling_via,
               mps_percent=a.mps_percent, warmup=a.warmup,
               subruns=results, trace_id_mismatch=mismatched)
    H.write_json(os.path.join(run_dir, "run.json"), doc)

    print("\n=== summary")
    for r in results:
        s = r.get("summary") or {}
        con = s.get("contended") or {}
        bits = []
        if s.get("requests_ok") is not None:
            bits.append("%d/%d ok" % (s["requests_ok"], s.get("requests_total", 0)))
        if con.get("aggregate_tokens_per_second"):
            bits.append("%.1f tok/s agg" % con["aggregate_tokens_per_second"])
        if con.get("ratio"):
            bits.append("A:B %.3f" % con["ratio"])
        print("  %-28s %-14s %s" % (r["tag"], r["status"], "  ".join(bits)))
    if mismatched:
        print("\n  FAIRNESS VIOLATION: these arms replayed a different trace:")
        for k, v in mismatched.items():
            print("    %s saw %s, expected %s" % (k, v, trace_id))
        print("  The comparison between arms is NOT valid. Do not report it.")
    bad = [r for r in results if r["status"] not in ("ok", "dry-run")]
    print("\n  %d/%d sub-runs ok" % (len(results) - len(bad), len(results)))
    print("  results: %s" % run_dir)
    print("  render artifacts with: python render_results.py %s" % run_dir)
    ev.close()
    return 1 if (bad or mismatched) else 0


def _wrap(text, width):
    words, line, out = text.split(), "", []
    for w in words:
        if len(line) + len(w) + 1 > width:
            out.append(line)
            line = w
        else:
            line = (line + " " + w).strip()
    if line:
        out.append(line)
    return out


if __name__ == "__main__":
    sys.exit(main())
