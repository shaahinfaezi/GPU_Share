#!/usr/bin/env python3
"""T9 -- admission under pressure: a new model arrives on a full card.

Requirement 6, with the evict-to-zero semantics chosen for this project:

    two models are serving; five minutes later a third arrives that cannot
    fit; the incumbent is scaled to zero and only then is the newcomer
    admitted.

    python 09-scaledown.py --arm hami
    python 09-scaledown.py --arm hami-mps --incumbent llm --incumbent embed \\
                           --arrival llm2 --wait 300

WHY THIS IS NOT JUST "kubectl scale"

The interesting quantity is not that the newcomer eventually starts. It is the
DEAD TIME -- the gap between deciding to evict and the newcomer serving its
first token -- and who pays for it. On this card that gap is dominated by three
things nothing in the Kubernetes API surfaces:

  * a vLLM process holds its VRAM until the process exits, so `replicas: 0`
    returning is not the memory being free. This script waits on nvidia-smi,
    not on the API.
  * the newcomer then reloads weights from the PVC and re-runs CUDA graph
    capture, which on a 7B AWQ model is most of the wait.
  * HAMi's refusal is a SCHEDULING event, not an error condition. The pod sits
    Pending with a reason buried in events, and if you are not reading them the
    system looks hung rather than correctly full.

All three are recorded with timestamps, and traffic keeps running against the
incumbent throughout so the cost of eviction lands in a latency series rather
than in prose. The failed requests during the drain are the point, not noise.

Everything lands in the same results layout run.py uses, so render_results.py
draws T9 from the same code path as the rest.
"""
import argparse
import datetime
import json
import os
import subprocess
import sys
import threading
import time

import harness as H
from render import MODELS, SCENARIO_QUOTAS, effective_util

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

# Same reasoning as run.py: keep the 650 MB ShareGPT corpus off the root
# filesystem, where kubelet's eviction threshold lives.
os.environ.setdefault("HF_HOME", "/data2/hf-cache")

# gpumem budgets from render.py, in MiB, against a 12288 MiB card:
#   llm 8192 | embed 3072 | llm2 3840 (SCENARIO_QUOTAS) | rerank 4608
# The newcomer is blocked for TWO reasons, and HAMi reports whichever
# it hits first: memory (1024 MiB free vs 3840 needed) and gpucores
# (the two incumbents already hold 100 of 100). Read arrival_reasons
# before describing this as a memory-admission result.
# The default pairing leaves 11264 of 12288 allocated, so the 5632 MiB newcomer
# cannot be admitted without the incumbent leaving. That is the whole setup.
DEFAULT_INCUMBENTS = ["llm", "embed"]
DEFAULT_ARRIVAL = "llm2"
SCENARIO = "llm-embed"          # supplies llm + embed
ARRIVAL_SCENARIO = "llm-llm"    # supplies llm2


def run(cmd, timeout=3600, echo=True):
    if echo:
        print("    $ " + " ".join(str(c) for c in cmd))
    p = subprocess.run([str(c) for c in cmd], cwd=HERE, capture_output=True,
                       text=True, timeout=timeout)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def render(scenario, arm, roles, out, pin, pooling_via, mps_percent=""):
    cmd = [PY, "render.py", "--scenario", scenario, "--arm", arm,
           "--out", out, "--pooling-via", pooling_via]
    for r in roles:
        cmd += ["--only", r]
    if pin:
        cmd += ["--pin-gpu", pin]
    if mps_percent:
        cmd += ["--mps-percent", mps_percent]
    rc, txt = run(cmd, echo=False)
    if rc != 0:
        raise RuntimeError("render failed:\n" + txt[-2000:])
    return out


def pod_reason(role, ns=H.NS):
    """Why is this role's pod not running? HAMi's refusal lives in events."""
    out = {"phase": None, "waiting": None, "events": []}
    for p in H.pods_for(role, ns):
        out["phase"] = p.get("status", {}).get("phase")
        for c in (p.get("status", {}).get("conditions") or []):
            if c.get("type") == "PodScheduled" and c.get("status") != "True":
                out["waiting"] = "%s: %s" % (c.get("reason"), c.get("message"))
        for cs in (p.get("status", {}).get("containerStatuses") or []):
            w = cs.get("state", {}).get("waiting") or {}
            if w:
                out["waiting"] = "%s: %s" % (w.get("reason"), w.get("message"))
        rc, txt = H.kubectl(["get", "events", "-n", ns, "--field-selector",
                             "involvedObject.name=%s" % p["metadata"]["name"],
                             "--sort-by", ".lastTimestamp",
                             "-o", "custom-columns=REASON:.reason,MSG:.message",
                             "--no-headers"], timeout=30)
        if rc == 0:
            out["events"] = [l.strip() for l in txt.strip().splitlines()][-6:]
    return out


def scale_to_zero(role, ns=H.NS):
    """Evict-to-zero across both deployment paths.

    LLMInferenceService and Deployment take the same `replicas` field but live
    at different API groups, so try both rather than branching on how the role
    happened to be rendered.
    """
    done = []
    for kind in ("llminferenceservice", "deployment"):
        rc, out = H.kubectl(["patch", kind, "r2-%s" % role, "-n", ns,
                             "--type", "merge",
                             "-p", '{"spec":{"replicas":0}}'], timeout=60)
        if rc == 0:
            done.append(kind)
    if not done:
        # Nothing patchable: delete instead. Still evict-to-zero, just blunter.
        H.kubectl(["delete", "llminferenceservice,deployment", "-n", ns,
                   "-l", "r2-role=%s" % role, "--ignore-not-found",
                   "--wait=false"], timeout=60)
        done.append("deleted")
    return done


class Traffic(threading.Thread):
    """Keep the incumbent under load for the whole experiment.

    Without this the eviction is measured on an idle system, where it costs
    nothing and proves nothing. With it, the drain shows up as failed requests
    and a latency cliff in exactly the series the dashboard is already drawing.
    """

    def __init__(self, trace, arm, out_dir, roles, eps, models, note):
        super().__init__(daemon=True)
        self.args = (trace, arm, out_dir, roles, eps, models, note)
        self.rc, self.out = None, ""

    def run(self):
        trace, arm, out_dir, roles, eps, models, note = self.args
        # --timeout 60, not the 600 s default: after eviction the endpoint is
        # a pod IP that no longer exists, and a hung request holds a worker for
        # ten minutes. The refusal IS the measurement here, so it needs to
        # arrive promptly.
        cmd = [PY, os.path.join("workload", "replay.py"), "--trace", trace,
               "--arm", arm, "--out-dir", out_dir, "--note", note,
               "--timeout", "60"]
        for r in roles:
            cmd += ["--only", r]
        for r, u in eps.items():
            cmd += ["--endpoint", "%s=%s" % (r, u)]
        for r, m in models.items():
            cmd += ["--model", "%s=%s" % (r, m)]
        self.rc, self.out = run(cmd, timeout=7200, echo=False)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", default="hami", choices=("dedicated", "hami", "hami-mps"))
    ap.add_argument("--incumbent", action="append", default=[],
                    help="repeatable; default llm + embed")
    ap.add_argument("--arrival", default=DEFAULT_ARRIVAL)
    ap.add_argument("--wait", type=float, default=300.0,
                    help="seconds the incumbents serve before the newcomer arrives")
    ap.add_argument("--admit-timeout", type=float, default=120.0,
                    help="how long to let the newcomer sit Pending before "
                         "evicting. Its whole purpose is to prove it CANNOT be "
                         "admitted while the card is full.")
    ap.add_argument("--evict", default="",
                    help="which incumbent to scale to zero; default the first")
    ap.add_argument("--ready-timeout", type=float, default=900.0)
    ap.add_argument("--drain", type=float, default=120.0,
                    help="extra seconds of traffic after the newcomer is up")
    ap.add_argument("--rate", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=20260826)
    # Default empty, matching run.py. A UUID default pins the Deployment
    # tenants while binpack sends the KServe tenants wherever it likes -- a
    # guaranteed split, caused by the flag meant to prevent one.
    ap.add_argument("--pin-gpu", default="")
    ap.add_argument("--pooling-via", choices=("kserve", "vllm"), default="kserve")
    ap.add_argument("--mps-percent", default="")
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    ap.add_argument("--skip-preconditions", action="store_true")
    a = ap.parse_args()

    incumbents = a.incumbent or DEFAULT_INCUMBENTS
    evict = a.evict or incumbents[0]
    if evict not in incumbents:
        sys.exit("--evict %s is not among the incumbents %s" % (evict, incumbents))

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(a.out, "T9-%s-%s" % (a.arm, stamp))
    d = os.path.join(run_dir, "%s__scaledown" % a.arm)
    os.makedirs(d, exist_ok=True)
    ev = H.Events(os.path.join(run_dir, "events.jsonl"))
    marks = {}

    print("=== T9 admission / scale-down   arm=%s" % a.arm)
    print("    incumbents %s -> newcomer %s arrives after %.0fs"
          % ("+".join(incumbents), a.arrival, a.wait))
    print("    evict-to-zero target: %s" % evict)
    print("    %s\n" % run_dir)

    if not a.skip_preconditions:
        problems = H.preconditions(ev)
        if problems:
            print("REFUSING TO START:")
            for p in problems:
                print("  - %s" % p)
            ev.close()
            return 2

    rec = dict(test="T9", arm=a.arm, tag="%s__scaledown" % a.arm,
               scenario=SCENARIO, roles=list(incumbents),
               incumbents=incumbents, arrival=a.arrival,
               evict=evict, wait=a.wait, rate=a.rate, seed=a.seed,
               run_dir=run_dir, stamp=stamp, status="started")
    sampler = None
    traffic = None
    try:
        # ---- 0. clean slate ---------------------------------------------
        ev.mark("teardown_begin")
        H.teardown()
        H.wait_gpu_idle(ev=ev)

        # MPS state is set BEFORE any pod starts, because attachment happens at
        # CUDA init and cannot be changed afterwards. Skipping this is what
        # voided T1 and T2: the host daemon rests in the UP state, a container
        # attaches to it even with no pipe mount and no MPS environment, and
        # both the hami and dedicated arms silently ran THROUGH MPS while
        # reporting clean numbers. The pod spec proves nothing here.
        ok, how = H.set_mps_daemon(a.arm == "hami-mps", ev=ev)
        if not ok:
            raise RuntimeError(how)
        rec["mps_daemon"] = how
        print("    mps daemon: %s (%s)"
              % ("up" if a.arm == "hami-mps" else "down", how))

        # ---- 1. a trace long enough to cover the whole experiment --------
        total = a.wait + a.admit_timeout + a.ready_timeout * 0.5 + a.drain
        trace = os.path.join(run_dir, "trace.json")
        rc, out = run([PY, os.path.join("workload", "gen_trace.py"),
                       "--scenario", SCENARIO, "--rate", a.rate,
                       "--duration", total, "--source", "sharegpt",
                       "--seed", a.seed, "--out", trace], echo=False)
        if rc != 0:
            raise RuntimeError("gen_trace failed:\n" + out[-2000:])
        tdoc = H.read_json(trace, {})
        rec["trace_id"] = tdoc.get("trace_id")
        ev.mark("trace", trace_id=rec["trace_id"],
                requests=len(tdoc.get("requests", [])), seconds=round(total))

        # ---- 2. incumbents ----------------------------------------------
        rdir = os.path.join(d, "rendered-incumbent")
        render(SCENARIO, a.arm, incumbents, rdir, a.pin_gpu, a.pooling_via,
               a.mps_percent)
        ev.mark("apply_incumbents", roles=",".join(incumbents))
        rc, out = H.kubectl(["apply", "-f", rdir], timeout=180)
        if rc != 0:
            raise RuntimeError("apply incumbents failed:\n" + out[-2000:])
        ok, diag = H.wait_ready(incumbents, timeout=a.ready_timeout, ev=ev)
        if not ok:
            raise RuntimeError("incumbents never ready: %s"
                               % json.dumps({k: v[:600] for k, v in diag.items()}))
        colo, detail = H.assert_colocated(incumbents, ev=ev)
        rec["placement_incumbent"] = detail
        if not colo:
            raise RuntimeError("incumbents on different GPUs: %s" % detail)

        mps_ok, mps_detail = H.assert_mps_matches_arm(a.arm, incumbents, ev=ev)
        rec["mps_incumbent"] = mps_detail
        print("    mps check (incumbents): %s  daemon=%s clients=%s"
              % ("OK" if mps_ok else "FAILED", mps_detail["daemon_running"],
                 mps_detail["mps_client_pids"]))
        if not mps_ok:
            raise RuntimeError(mps_detail.get("reason", "MPS state mismatch"))

        eps = H.endpoints(incumbents, ev=ev)
        models = {r: H.served_model(u) for r, u in eps.items()}
        models = {r: m for r, m in models.items() if m}

        # Serve one real request before spending the measurement window. T1's
        # first attempt ran four arms to completion with 0/885 successful
        # requests because readiness was taken as proof of service.
        kinds = {r: MODELS[r]["kind"] for r in incumbents if r in MODELS}
        smoke_ok, smoke = H.smoke_test(eps, models, kinds, ev=ev)
        rec["smoke_incumbent"] = smoke
        bad = [r for r, v in smoke.items() if "ok" not in str(v).lower()]
        if not smoke_ok and not bad:
            bad = sorted(smoke)
        if bad:
            raise RuntimeError("incumbent smoke test failed for %s: %s"
                               % (bad, json.dumps(smoke)))

        # ---- 3. traffic + sampling start --------------------------------
        sampler = H.GpuSampler(os.path.join(d, "gpu.csv"))
        sampler.start()
        traffic = Traffic(trace, a.arm, d, incumbents, eps, models, "T9 incumbent")
        traffic.start()
        marks["t_traffic"] = ev.mark("traffic_begin")

        # ---- 4. serve for --wait seconds --------------------------------
        print("\n    incumbents serving; newcomer arrives in %.0fs" % a.wait)
        time.sleep(a.wait)

        # ---- 5. the newcomer arrives on a full card ---------------------
        adir = os.path.join(d, "rendered-arrival")
        render(ARRIVAL_SCENARIO, a.arm, [a.arrival], adir, a.pin_gpu,
               a.pooling_via, a.mps_percent)
        marks["t_arrive"] = ev.mark("arrival_applied", role=a.arrival)
        rc, out = H.kubectl(["apply", "-f", adir], timeout=180)
        if rc != 0:
            raise RuntimeError("apply arrival failed:\n" + out[-2000:])

        # It SHOULD fail to be admitted. Watch it fail, and record why.
        print("    watching %s for %.0fs -- it should NOT be admitted"
              % (a.arrival, a.admit_timeout))
        t0 = time.time()
        admitted_early = False
        reasons = []
        while time.time() - t0 < a.admit_timeout:
            r = pod_reason(a.arrival)
            if r not in reasons:
                reasons.append(r)
            ready = any(
                c.get("type") == "Ready" and c.get("status") == "True"
                for p in H.pods_for(a.arrival)
                for c in (p.get("status", {}).get("conditions") or []))
            if ready:
                admitted_early = True
                break
            time.sleep(5)
        rec["arrival_reasons"] = reasons
        rec["admitted_without_eviction"] = admitted_early
        marks["t_blocked"] = ev.mark("arrival_blocked",
                                     admitted_early=admitted_early)
        if admitted_early:
            # Not a script failure -- a finding. It means the card had room,
            # so the premise of the test (a full card) did not hold and the
            # gpumem budget needs re-checking before this run means anything.
            print("\n    NOTE: %s was admitted WITHOUT eviction. The card was "
                  "not full.\n    Check the gpumem budget in render.py before "
                  "reporting this run." % a.arrival)
        else:
            print("    blocked as expected. Last reason: %s"
                  % json.dumps(reasons[-1] if reasons else {})[:300])

            # ---- 6. evict to zero --------------------------------------
            marks["t_evict"] = ev.mark("evict_begin", role=evict)
            rec["evict_via"] = scale_to_zero(evict)
            # A patch returning 0 is not an eviction -- the KServe controller
            # may reconcile replicas back to 1 -- and replicas:0 is not the
            # memory being free, because the vLLM process holds its allocation
            # until it exits. Wait for BOTH, and measure the drop rather than
            # comparing against a hardcoded floor.
            gone, freed, released = _wait_release(evict, SCENARIO, ev=ev)
            rec["evict_pods_gone"] = gone
            rec["vram_released_mib"] = released
            marks["t_freed"] = ev.mark("vram_released", ok=freed,
                                       released_mib=released)
            if not gone:
                raise RuntimeError(
                    "%s still has pods after scale-to-zero; the controller "
                    "reconciled it back. Nothing measured after this point "
                    "would mean anything." % evict)

            # ---- 7. admission ------------------------------------------
            ok, diag = H.wait_ready([a.arrival], timeout=a.ready_timeout, ev=ev)
            marks["t_admitted"] = ev.mark("arrival_ready", ok=ok)
            rec["arrival_ready"] = ok
            if ok:
                # Ready is /health, not a token. The timeline claims to report
                # the gap to the newcomer SERVING, so serve something.
                aeps = H.endpoints([a.arrival], ev=ev)
                amodels = {r: H.served_model(u) or r for r, u in aeps.items()}
                akinds = {a.arrival: MODELS.get(a.arrival, {}).get("kind", "chat")}
                asmoke_ok, asmoke = H.smoke_test(aeps, amodels, akinds, ev=ev)
                rec["smoke_arrival"] = asmoke
                marks["t_serving"] = ev.mark("arrival_serving", ok=asmoke_ok)
                # And the MPS state must still hold with the new tenant set.
                survivors = [r for r in incumbents if r != evict] + [a.arrival]
                m2_ok, m2 = H.assert_mps_matches_arm(a.arm, survivors, ev=ev)
                rec["mps_after_admission"] = m2
                if not m2_ok:
                    raise RuntimeError(m2.get("reason", "MPS state mismatch "
                                              "after admission"))
            if not ok:
                rec["arrival_diag"] = {k: v[:1200] for k, v in diag.items()}
                print("    newcomer STILL not ready after eviction:")
                for k, v in diag.items():
                    print(v[:1500])

        # ---- 8. let the tail run, then collect --------------------------
        print("    draining for %.0fs" % a.drain)
        time.sleep(a.drain)
        marks["t_end"] = ev.mark("drain_end")

    except Exception as exc:
        rec["status"] = "error"
        rec["error"] = "%s: %s" % (type(exc).__name__, exc)
        print("\nERROR %s" % rec["error"])
    finally:
        if traffic is not None:
            # The trace is `total` seconds long and the main thread reaches
            # here well before that. replay.py writes replay.summary.json only
            # after its pool drains, so a short join loses the per-request data
            # this whole test exists to capture -- and the timeline still
            # renders, so the run LOOKS fine.
            spent = time.time() - marks.get("t_traffic", time.time())
            wait_for = max(300.0, total - spent + 300.0)
            print("    waiting up to %.0fs for the traffic thread to finish"
                  % wait_for)
            traffic.join(timeout=wait_for)
            rec["traffic_rc"] = traffic.rc
            rec["traffic_finished"] = not traffic.is_alive()
        if sampler is not None:
            sampler.stop()
        t0 = marks.get("t_traffic", time.time() - 60)
        t1 = marks.get("t_end", time.time())
        H.prom_collect(os.path.join(d, "prom.json"), t0, t1, ev=ev)
        rec["summary"] = H.read_json(os.path.join(d, "replay.summary.json"), {})
        rec["marks"] = marks
        rec["t0"] = t0
        rec["t1"] = t1
        rec["timeline"] = _timeline(marks)
        if rec["status"] == "started":
            rec["status"] = "ok"
        H.write_json(os.path.join(d, "subrun.json"), rec)
        H.write_json(os.path.join(run_dir, "run.json"),
                     dict(test="T9", config=rec, subruns=[rec],
                          run_dir=run_dir, trace_id=rec.get("trace_id"),
                          pin_gpu=a.pin_gpu, pooling_via=a.pooling_via,
                          trace_id_mismatch={}, arms=[a.arm],
                          roles=list(incumbents) + [a.arrival]))
        H.set_mps_daemon(True, ev=ev)
        ev.mark("teardown_end")
        H.teardown(quiet=True)
        ev.close()

    print("\n=== timeline")
    for k, v in rec.get("timeline", {}).items():
        print("  %-34s %s" % (k, v))
    print("\n  results: %s" % run_dir)
    print("  render with: python render_results.py %s" % run_dir)
    if rec.get("admitted_without_eviction"):
        print("\n  EXIT 3: the newcomer was admitted without eviction, so the "
              "premise of this test -- a full card -- did not hold. Nothing "
              "here measures scale-down.")
        return 3
    return 0 if rec["status"] == "ok" else 1


def _rendered_footprint(scenario, role):
    """What the role actually occupies, from render.py's own numbers.

    The previous version retyped a budget table into this file, and it had
    already drifted: it said llm2 was 5632 MiB when SCENARIO_QUOTAS overrides
    it to 3840 under llm-llm. A second copy of a number is a copy that goes
    stale.
    """
    m = dict(MODELS[role])
    over = SCENARIO_QUOTAS.get(scenario, {}).get(role)
    if over:
        m["gpumem"] = over
    util, _cap, _clamped = effective_util(m)
    return int(m["gpumem"] * util)


def _wait_release(role, scenario, timeout=300, ev=None, poll=3.0):
    """Wait for the evicted tenant's pods to go AND its VRAM to come back.

    Returns (pods_gone, released_enough, mib_released).

    Measuring the DROP rather than waiting for an absolute floor is the point.
    The old code compared against a hardcoded remaining-budget figure, which
    for some --evict choices sat above what the survivors actually hold, so
    wait_gpu_idle returned True on its first poll and the headline
    "VRAM release after replicas=0" was reported as roughly zero seconds --
    a measurement produced by not waiting.
    """
    want = _rendered_footprint(scenario, role)
    before = sum(H.gpu_used_mib().values())
    t0 = time.time()
    gone = False
    released = 0
    while time.time() - t0 < timeout:
        if not gone:
            gone = not H.pods_for(role)
        released = before - sum(H.gpu_used_mib().values())
        if gone and released >= 0.7 * want:
            if ev:
                ev.mark("vram_release_detail", role=role, expected_mib=want,
                        released_mib=released,
                        after_s=round(time.time() - t0, 1))
            return True, True, released
        time.sleep(poll)
    if ev:
        ev.mark("vram_release_timeout", role=role, expected_mib=want,
                released_mib=released, pods_gone=gone)
    return gone, False, released


def _timeline(m):
    """Human-readable durations, which are the actual deliverable of T9."""
    out = {}

    def gap(a, b, label):
        if a in m and b in m:
            out[label] = "%.1fs" % (m[b] - m[a])

    gap("t_traffic", "t_arrive", "incumbent served for")
    gap("t_arrive", "t_blocked", "newcomer sat Pending for")
    gap("t_evict", "t_freed", "VRAM release after replicas=0")
    gap("t_freed", "t_admitted", "newcomer load + graph capture")
    gap("t_evict", "t_admitted", "TOTAL dead time (evict -> serving)")
    gap("t_arrive", "t_admitted", "TOTAL from arrival to serving")
    return out


if __name__ == "__main__":
    sys.exit(main())
