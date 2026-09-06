#!/usr/bin/env python3
"""T10 -- failure and blast radius on a shared GPU.

Every tier so far measures a healthy system. This one breaks it on purpose,
mid-serve, and measures what the SURVIVING tenant experiences.

For shared infrastructure that is the first question an operator asks, and the
matrix has no answer to it. Worse, this study has already been bitten once by a
silent failure: the MPS daemon was left running for every arm, every client
attached to it regardless of configuration, and two whole tiers measured a
thing against itself and reported clean numbers. Nothing alerted. That is
exactly the failure class this script exists to characterise.

    python3 23-failure-inject.py --fault mps-kill     --arm hami-mps
    python3 23-failure-inject.py --fault tenant-kill  --arm hami
    python3 23-failure-inject.py --fault vram-hog     --arm hami
    python3 23-failure-inject.py --list

FAULTS

  mps-kill      Park the MPS DaemonSet mid-run, so the control pipe disappears
                under two attached clients. Answers: do in-flight requests
                fail, or does CUDA silently fall back to time-slicing? How
                large is the latency step, and how long does it take? This is
                the failure that invalidated T1 and T2, measured deliberately.

  tenant-kill   Delete one tenant's pod mid-run. The survivor keeps serving
                through its neighbour's death AND through a full re-init --
                measured elsewhere at 113 s for the 1.5B and 250 s for the VL
                model. Compile and cudagraph capture compete for the same SMs,
                so the survivor is expected to degrade DURING the recovery,
                not during the outage.

  vram-escape   Ask a tenant HAMi IS managing to allocate past its own
                nvidia.com/gpumem grant, from a second process inside that
                tenant's own container. This is the isolation test proper.
                vram-hog below asks whether an outsider can hurt the tenants;
                this asks whether the partitioning is real at all. If a tenant
                can exceed its grant, then every quota in the study is advisory
                and one leaking tenant takes its neighbour down with it -- which
                would make co-location unsafe regardless of any latency result.
                Records what the tenant's own CUDA runtime reports as total
                device memory (HAMi is supposed to virtualise this to the
                grant, not the physical 12288 MiB), how much the probe got
                before being refused, and how far the card's real usage moved.

  vram-hog      Start a pod that allocates raw VRAM outside HAMi's accounting.
                Answers the question HAMi's memory partitioning is supposed to
                make unaskable: can a workload that is not playing by the rules
                take the tenants down? A pass here is HAMi holding its caps; a
                fail is one or both tenants OOMing.

WHAT IT RECORDS

Per-request latency in three windows -- before, during and after the fault --
for the surviving tenant, from replay.requests.csv. Plus gpu.csv across the
whole run, the exact injection timestamp, and whether the victim recovered.
The windows are what matters: an aggregate over the whole run would average a
30-second outage into 300 seconds of health and report nothing.
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import harness as H                                          # noqa: E402

PY = sys.executable or "python3"
SCENARIO = "llm-llm"
ROLES = ["llm", "llm2"]

# Imported, not duplicated. A second copy of the image tag is a copy that goes
# stale the day the registry changes, and this script would then test a
# different runtime than the tiers it is supposed to sit beside.
sys.path.insert(0, HERE)
from render import IMAGE                                     # noqa: E402

FAULTS = {
    "mps-kill": "park the MPS DaemonSet under two attached clients",
    "tenant-kill": "delete one tenant's pod, survivor serves through re-init",
    "vram-hog": "allocate raw VRAM outside HAMi's accounting",
    "vram-escape": "make a MANAGED tenant try to exceed its own gpumem grant",
}

# Runs inside the tenant's container, as a second process sharing that
# container's HAMi quota. Deliberately not a pod: the whole question is whether
# the cap binds a process HAMi has already admitted and is already policing.
ESCAPE_PROBE = r"""
import json, torch
out = {}
try:
    free, total = torch.cuda.mem_get_info()
    out["sees_free_mib"] = free // 1048576
    out["sees_total_mib"] = total // 1048576
except Exception as exc:
    out["mem_get_info_error"] = "%s: %s" % (type(exc).__name__, exc)
held, step = [], 256
try:
    while True:
        held.append(torch.empty(step * 1024 * 1024, dtype=torch.uint8,
                                device="cuda"))
        # Runaway guard at twice the physical card. If we ever reach this the
        # answer is already "no cap at all" and there is no point continuing.
        if len(held) * step >= 24576:
            out["stopped"] = "GUARD: reached 24576 MiB without refusal"
            break
except Exception as exc:
    out["stopped"] = "%s: %s" % (type(exc).__name__, str(exc)[:180])
out["grabbed_mib"] = len(held) * step
print("R2ESCAPE " + json.dumps(out), flush=True)
"""


# HAMi's own documented verification of memory hard isolation, from
# https://project-hami.io/blog/kai-scheduler-hami-gpu-memory-hard-isolation
#
# Our probe alone proves only the BEHAVIOUR -- an allocation was refused. These
# prove the MECHANISM: that the quota reached the container, that HAMi-core is
# present, that it is actually preloaded, and that it has rewritten the device
# size the tenant sees. Without them "contained" could equally mean the card was
# full, or torch's allocator gave up, or the probe stopped politely at the
# virtualised free figure.
#
# This matters more than it looks. HAMi's COMPUTE limiter is silently void under
# MPS -- configured, inert, reporting nothing (FINDINGS 5.14d). So whether the
# chain is intact is exactly the question to ask of the memory limiter too, and
# it is the one thing our own probe cannot answer.
CHAIN_CHECK = r"""
echo "CUDA_DEVICE_MEMORY_LIMIT=${CUDA_DEVICE_MEMORY_LIMIT:-<unset>}"
echo "CUDA_DEVICE_SM_LIMIT=${CUDA_DEVICE_SM_LIMIT:-<unset>}"
echo "GPU_CORE_UTILIZATION_POLICY=${GPU_CORE_UTILIZATION_POLICY:-<unset>}"
if [ -f /usr/local/vgpu/libvgpu.so ]; then
  echo "libvgpu.so=present $(stat -c%s /usr/local/vgpu/libvgpu.so) bytes"
else
  echo "libvgpu.so=MISSING"
fi
echo "ld.so.preload=$(cat /etc/ld.so.preload 2>/dev/null || echo '<absent>')"
echo "nvidia-smi-total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1)"
"""


def _exec_sh(pod, ns, src, timeout=120):
    """Run a shell snippet inside a pod, via stdin."""
    try:
        pr = subprocess.run(["kubectl", "exec", "-i", "-n", ns, pod, "--",
                             "sh", "-s"],
                            input=src, capture_output=True, text=True,
                            timeout=timeout)
    except subprocess.TimeoutExpired:
        return 124, "timeout"
    return pr.returncode, (pr.stdout or "") + (pr.stderr or "")


def _chain(pod, ns):
    """Parse HAMi's official chain checks into a dict."""
    rc, out = _exec_sh(pod, ns, CHAIN_CHECK)
    d = {"rc": rc, "raw": out[-800:]}
    for line in out.splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            d[k.strip()] = v.strip()
    return d


def _exec_python(pod, ns, src, timeout=240):
    """Run a python snippet inside a running pod, via stdin.

    stdin rather than -c so the snippet never passes through a shell and no
    quoting can corrupt it.
    """
    try:
        pr = subprocess.run(["kubectl", "exec", "-i", "-n", ns, pod, "--",
                             "python3", "-"],
                            input=src, capture_output=True, text=True,
                            timeout=timeout)
    except subprocess.TimeoutExpired:
        return 124, "timeout after %ds" % timeout
    return pr.returncode, (pr.stdout or "") + (pr.stderr or "")


def _granted_gpumem(pod, ns):
    """The gpumem limit actually on the pod, read from the API rather than
    recomputed from render.py -- what was granted is what matters here."""
    rc, out = H.kubectl(["-n", ns, "get", "pod", pod, "-o",
                         r"jsonpath={.spec.containers[0].resources.limits"
                         r".nvidia\.com/gpumem}"], timeout=30)
    try:
        return int(out.strip())
    except Exception:
        return None

# A pod that grabs VRAM without asking HAMi for any. It requests no
# nvidia.com/* resource at all, so HAMi's admission never sees it -- which is
# the point. NVIDIA_VISIBLE_DEVICES is set directly, bypassing the device
# plugin the same way a mis-configured or hostile workload would.
VRAM_HOG = """---
apiVersion: v1
kind: Pod
metadata:
  name: r2-vram-hog
  namespace: %(ns)s
  labels: {r2-role: vram-hog}
spec:
  restartPolicy: Never
  containers:
    - name: hog
      image: %(image)s
      command: ["python3", "-c"]
      args:
        - |
          import torch, time, sys
          held = []
          # Claim in 512 MiB steps until the allocator refuses, then hold.
          while True:
              try:
                  held.append(torch.empty(512 * 1024 * 1024 // 2,
                                          dtype=torch.float16, device="cuda"))
                  print("held %%d MiB" %% (len(held) * 512), flush=True)
              except Exception as exc:
                  print("stopped at %%d MiB: %%s" %% (len(held) * 512, exc),
                        flush=True)
                  break
          time.sleep(%(hold)d)
      env:
        - name: NVIDIA_VISIBLE_DEVICES
          value: "all"
        - name: NVIDIA_DRIVER_CAPABILITIES
          value: compute,utility
      resources:
        limits: {cpu: "2", memory: 8Gi}
"""


def sh(cmd, timeout=120):
    return H.sh(cmd, timeout=timeout)


def inject(fault, ns, image, hold, ev, outdir):
    """Apply the fault. Returns (label, detail) and stamps the event log."""
    t = time.time()
    if fault == "mps-kill":
        ok, how = H.set_mps_daemon(False, ev=ev)
        detail = {"ok": ok, "how": how,
                  "daemon_after": H.mps_daemon_running(),
                  "mps_clients_after": H.mps_client_pids()}
    elif fault == "tenant-kill":
        pods = H.pods_for("llm2", ns)
        names = [p["metadata"]["name"] for p in pods]
        rc, out = H.kubectl(["-n", ns, "delete", "pod"] + names +
                            ["--wait=false"], timeout=60)
        detail = {"rc": rc, "deleted": names, "out": out[-300:]}
    elif fault == "vram-escape":
        role = ESCAPE_ROLE[0]
        pods = H.pods_for(role, ns)
        if not pods:
            raise SystemExit("no pod for role %s -- nothing to test" % role)
        pod = pods[0]["metadata"]["name"]
        quota = _granted_gpumem(pod, ns)
        chain = _chain(pod, ns)
        before = H.gpu_used_mib()
        rc, out = _exec_python(pod, ns, ESCAPE_PROBE)
        after = H.gpu_used_mib()
        probe = {}
        for line in out.splitlines():
            if line.startswith("R2ESCAPE "):
                try:
                    probe = json.loads(line[len("R2ESCAPE "):])
                except Exception:
                    pass
        rise = None
        if before and after:
            rise = max(after.values()) - max(before.values())
        detail = {"escaper_role": role, "escaper_pod": pod,
                  "granted_gpumem_mib": quota, "probe": probe, "rc": rc,
                  "chain": chain,
                  "card_before_mib": before, "card_after_mib": after,
                  "card_rise_mib": rise, "out_tail": out[-400:]}
        detail["verdict"] = _escape_verdict(detail)
    elif fault == "vram-hog":
        path = os.path.join(outdir, "r2-vram-hog.yaml")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(VRAM_HOG % dict(ns=ns, image=image, hold=hold))
        rc, out = H.kubectl(["apply", "-f", path], timeout=60)
        detail = {"rc": rc, "out": out[-300:], "manifest": path}
    else:
        raise SystemExit("unknown fault %s" % fault)
    ev.mark("fault_injected", fault=fault, at=t, **{
        k: v for k, v in detail.items() if isinstance(v, (int, float, str, bool))})
    return t, detail


# Set from --escaper in main(); a list so inject() can read it without
# threading another argument through a signature the other faults do not need.
ESCAPE_ROLE = ["llm2"]


def _escape_verdict(d):
    """Did HAMi's cap hold? Three independent checks, reported separately.

    They can disagree, and the disagreement is informative: HAMi can virtualise
    what the tenant SEES without enforcing what it GETS, or enforce without
    virtualising. Only the third check is about safety.
    """
    q = d.get("granted_gpumem_mib")
    p = d.get("probe") or {}
    seen_total = p.get("sees_total_mib")
    seen_free = p.get("sees_free_mib")
    grabbed = p.get("grabbed_mib")
    rise = d.get("card_rise_mib")
    v = {}

    if q and seen_total:
        # HAMi is supposed to rewrite the device's reported size to the grant.
        v["view_virtualised"] = abs(seen_total - q) <= max(256, 0.1 * q)
        v["sees_total_mib"] = seen_total
        v["physical_card_mib"] = H.CARD_MIB

    if seen_free is not None and grabbed is not None:
        # Did the allocator refuse near the free figure it advertised?
        v["stopped_near_advertised_free"] = grabbed <= seen_free + 512

    if q is not None and rise is not None and seen_free is not None:
        # THE SAFETY CHECK. The card's real usage must not rise by more than
        # the tenant had left inside its own grant. If it does, the tenant has
        # taken memory that belongs to its neighbour.
        v["card_rise_mib"] = rise
        v["headroom_in_grant_mib"] = seen_free
        v["ESCAPED"] = rise > seen_free + 512

    if p.get("stopped", "").startswith("GUARD"):
        v["ESCAPED"] = True
        v["note"] = "probe reached 24 GiB on a 12 GiB card without refusal"
    return v


def windows(csv_path, t_fault, role, pre=60, during=60, post=120, epoch0=None):
    """Split the survivor's requests into before / during / after the fault.

    An aggregate over the whole run would dissolve a 30-second outage into 300
    seconds of health. The windows are the measurement.

    UNITS. replay.requests.csv stores t_sent RELATIVE to the start of the
    replay -- the first row reads 0.001 -- while the fault timestamp is an
    absolute epoch, 1788672842.87. Comparing them put every request outside
    every window, and the first six fault runs all reported

        window   reqs  failed   p50  p95  p99
        before      0       0     -    -    -

    which reads like a clean run and is actually no measurement at all. Pass
    epoch0 (replay.summary.json's epoch_start) and the fault time is converted
    to the same relative scale.
    """
    if epoch0 and t_fault > 1e6:
        t_fault = t_fault - epoch0
    import csv as _csv
    rows = []
    with open(csv_path, encoding="utf-8") as fh:
        for r in _csv.DictReader(fh):
            if r.get("service") != role:
                continue
            try:
                sent = float(r["t_sent"])
                last = float(r["t_last"]) if r.get("t_last") else None
            except (TypeError, ValueError):
                continue
            rows.append((sent, (last - sent) if last else None,
                         (r.get("ok") or "").lower() in ("true", "1")))
    out = {}
    for name, lo, hi in (("before", t_fault - pre, t_fault),
                         ("during", t_fault, t_fault + during),
                         ("after", t_fault + during, t_fault + during + post)):
        sel = [r for r in rows if lo <= r[0] < hi]
        lat = sorted(x[1] for x in sel if x[1] is not None)
        n_fail = sum(1 for x in sel if not x[2])
        q = (lambda p: lat[min(len(lat) - 1, int(p * len(lat)))]) if lat else (lambda p: None)
        out[name] = dict(n=len(sel), failed=n_fail,
                         p50=q(.5), p95=q(.95), p99=q(.99),
                         mean=(sum(lat) / len(lat)) if lat else None)
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fault", choices=sorted(FAULTS))
    ap.add_argument("--arm", default="hami-mps",
                    choices=("hami", "hami-mps"),
                    help="dedicated is meaningless here -- a blast radius "
                         "needs something to blast")
    ap.add_argument("--victim", default="llm",
                    help="the tenant whose experience is measured")
    ap.add_argument("--escaper", default="llm2",
                    help="vram-escape only: the tenant that tries to exceed "
                         "its own grant. Must not be --victim, or the blast "
                         "radius and the perpetrator are the same pod.")
    ap.add_argument("--at", type=float, default=90.0,
                    help="seconds into the replay to inject")
    ap.add_argument("--rate", type=float, default=6.0)
    ap.add_argument("--duration", type=int, default=300)
    ap.add_argument("--seed", type=int, default=20260826)
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    ESCAPE_ROLE[0] = a.escaper
    if a.fault == "vram-escape" and a.escaper == a.victim:
        raise SystemExit("--escaper and --victim are both %r; the point is to "
                         "measure what the NEIGHBOUR suffers" % a.victim)

    if a.list or not a.fault:
        print("faults:")
        for k, v in sorted(FAULTS.items()):
            print("  %-14s %s" % (k, v))
        return 0

    stamp = time.strftime("%Y%m%d-%H%M%S")
    d = os.path.join(a.out, "T10-%s-%s-%s" % (a.fault, a.arm, stamp))
    os.makedirs(d, exist_ok=True)
    ev = H.Events(os.path.join(d, "events.jsonl"))
    print("=== T10 %s  arm=%s  victim=%s  inject at +%.0fs" %
          (a.fault, a.arm, a.victim, a.at))
    print("    out %s" % d)

    problems = H.preconditions(ev)          # returns a LIST of strings
    if problems:
        print("  preconditions reported %d problem(s); continuing anyway -- a "
              "failure test is allowed to start from a scuffed state"
              % len(problems))
        for pr in problems:
            print("    - %s" % pr)

    ev.mark("teardown_begin")
    H.teardown()
    H.wait_gpu_idle(ev=ev)

    # The arm's MPS state must be set and asserted BEFORE anything starts, the
    # same as every tier. A failure test that cannot say what it was running
    # under measures nothing.
    ok, how = H.set_mps_daemon(a.arm == "hami-mps", ev=ev)
    if not ok:
        raise SystemExit(how)
    print("    mps daemon: %s (%s)" % ("up" if a.arm == "hami-mps" else "down", how))

    rdir = os.path.join(d, "rendered")
    rc, out = H.sh([PY, os.path.join(HERE, "render.py"), "--scenario", SCENARIO,
                    "--arm", a.arm, "--out", rdir], timeout=120)
    if rc != 0:
        raise SystemExit("render failed:\n" + out[-2000:])

    ev.mark("apply")
    rc, out = H.kubectl(["apply", "-f", rdir], timeout=180)
    if rc != 0:
        raise SystemExit("apply failed:\n" + out[-2000:])

    ready, diag = H.wait_ready(ROLES, ev=ev)
    if not ready:
        for role, text in diag.items():
            print("  %s never became ready:\n%s" % (role, text[:2000]))
        raise SystemExit("roles never became ready")

    colocated, place = H.assert_colocated(ROLES, ev=ev)
    if not colocated:
        raise SystemExit("tenants are not co-located: %s" % json.dumps(place))
    mps_ok, mps_detail = H.assert_mps_matches_arm(a.arm, ROLES, ev=ev)
    print("    mps check: %s  daemon=%s clients=%s"
          % ("OK" if mps_ok else "FAILED", mps_detail["daemon_running"],
             mps_detail["mps_client_pids"]))
    if not mps_ok:
        raise SystemExit(mps_detail.get("reason", "MPS state mismatch"))

    eps = H.endpoints(ROLES, ev=ev)
    models = {}
    for role, base in eps.items():
        models[role] = H.served_model(base) or role

    trace = os.path.join(d, "trace.json")
    rc, out = H.sh([PY, os.path.join(HERE, "workload", "gen_trace.py"),
                    "--scenario", SCENARIO, "--rate", str(a.rate),
                    "--duration", str(a.duration), "--source", "sharegpt",
                    "--seed", str(a.seed), "--warmup", "8", "--out", trace],
                   timeout=600)
    if rc != 0:
        raise SystemExit("gen_trace failed:\n" + out[-2000:])

    sampler = H.GpuSampler(os.path.join(d, "gpu.csv"))
    sampler.start()

    cmd = ([PY, os.path.join(HERE, "workload", "replay.py"),
            "--trace", trace, "--arm", a.arm, "--out-dir", d,
            "--note", "T10 %s %s" % (a.fault, a.arm)]
           + sum([["--only", r] for r in ROLES], [])
           + sum([["--endpoint", "%s=%s" % (r, u)] for r, u in eps.items()], [])
           + sum([["--model", "%s=%s" % (r, m)] for r, m in models.items()], []))

    t0 = ev.mark("replay_begin")
    proc = subprocess.Popen(cmd, cwd=HERE, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)

    # Inject on a timer rather than after the replay, so the fault lands in the
    # middle of live traffic. This is the whole point -- a fault applied to an
    # idle system measures deployment, not blast radius.
    fault_at = [None]
    detail = [None]

    def fire():
        time.sleep(a.at)
        if proc.poll() is None:
            print("\n  >>> injecting %s at +%.0fs\n" % (a.fault, a.at), flush=True)
            fault_at[0], detail[0] = inject(a.fault, H.NS, IMAGE,
                                            a.duration, ev, d)

    timer = threading.Thread(target=fire, daemon=True)
    timer.start()

    tail = []
    for line in proc.stdout:
        tail.append(line.rstrip())
        if len(tail) > 400:
            tail.pop(0)
    rc = proc.wait()
    t1 = ev.mark("replay_end", rc=rc)
    sampler.stop()
    timer.join(timeout=5)

    print("\n".join(tail[-12:]))

    # Recovery: did the victim's neighbour come back at all?
    recovered = None
    if a.fault in ("tenant-kill",):
        ready2, _ = H.wait_ready(["llm2"], timeout=600, ev=ev)
        recovered = bool(ready2)

    csv_path = os.path.join(d, "replay.requests.csv")
    win = None
    if fault_at[0] and os.path.exists(csv_path):
        summ = H.read_json(os.path.join(d, "replay.summary.json"), {})
        win = windows(csv_path, fault_at[0], a.victim,
                      epoch0=summ.get("epoch_start"))
        if win and not any(win[k]["n"] for k in win):
            print("  WARNING: all three windows are empty. The victim role (%s) "
                  "may not appear in the csv, or the fault fired outside the "
                  "replay window." % a.victim)

    doc = dict(test="T10", fault=a.fault, arm=a.arm, victim=a.victim,
               inject_at_s=a.at, rate=a.rate, duration=a.duration,
               seed=a.seed, out=d, replay_rc=rc,
               fault_time=fault_at[0], fault_detail=detail[0],
               mps=mps_detail, placement=place, recovered=recovered,
               windows=win, t0=t0, t1=t1)
    H.write_json(os.path.join(d, "t10.json"), doc)

    print("\n=== %s / %s" % (a.fault, a.arm))
    if win:
        print("  %-8s %6s %7s %9s %9s %9s"
              % ("window", "reqs", "failed", "p50", "p95", "p99"))
        for k in ("before", "during", "after"):
            w = win[k]
            f = lambda v: ("%9.3f" % v) if isinstance(v, float) else "        -"
            print("  %-8s %6d %7d %s %s %s"
                  % (k, w["n"], w["failed"], f(w["p50"]), f(w["p95"]), f(w["p99"])))
        b, dur = win["before"], win["during"]
        if b["p99"] and dur["p99"]:
            print("\n  blast radius: victim p99 %+.1f%% during the fault"
                  % ((dur["p99"] / b["p99"] - 1) * 100))
        if dur["failed"]:
            print("  %d victim request(s) FAILED during the fault" % dur["failed"])
        else:
            print("  no victim requests failed")
    else:
        print("  no windows computed -- fault may not have fired")
    if a.fault == "vram-escape" and detail[0]:
        dd = detail[0]
        v = dd.get("verdict") or {}
        pr = dd.get("probe") or {}
        print("\n  --- VRAM isolation")
        print("  escaper            : %s (%s)"
              % (dd.get("escaper_role"), dd.get("escaper_pod")))
        ch = dd.get("chain") or {}
        print("  --- HAMi's own chain checks (project-hami.io verification)")
        for k in ("CUDA_DEVICE_MEMORY_LIMIT", "CUDA_DEVICE_SM_LIMIT",
                  "GPU_CORE_UTILIZATION_POLICY", "libvgpu.so", "ld.so.preload",
                  "nvidia-smi-total"):
            print("    %-28s %s" % (k, ch.get(k, "<not reported>")))
        intact = (ch.get("libvgpu.so", "").startswith("present")
                  and "libvgpu" in ch.get("ld.so.preload", ""))
        print("    %-28s %s" % ("ENFORCEMENT CHAIN",
              "INTACT" if intact else "*** BROKEN -- a refusal here proves nothing ***"))
        print("  granted gpumem     : %s MiB" % dd.get("granted_gpumem_mib"))
        print("  tenant sees        : %s MiB total, %s MiB free"
              % (pr.get("sees_total_mib"), pr.get("sees_free_mib")))
        print("  probe grabbed      : %s MiB" % pr.get("grabbed_mib"))
        print("  refused with       : %s" % pr.get("stopped"))
        print("  card usage rose by : %s MiB" % dd.get("card_rise_mib"))
        if v.get("view_virtualised") is not None:
            print("  view virtualised   : %s  (HAMi rewrites the device size "
                  "to the grant)" % v["view_virtualised"])
        if "ESCAPED" in v:
            if v["ESCAPED"]:
                print("  VERDICT            : *** ESCAPED *** the tenant took "
                      "memory outside its grant.")
                print("                       Every quota in this study is "
                      "advisory and co-location is unsafe without a second "
                      "mechanism.")
            else:
                print("  VERDICT            : contained -- the cap held.")
        else:
            print("  VERDICT            : inconclusive (see t10.json)")

    if recovered is not None:
        print("  neighbour recovered: %s" % recovered)
    print("\n  results: %s" % d)

    # Always clean up the hog; it holds VRAM until the pod is gone.
    if a.fault == "vram-hog":
        H.kubectl(["-n", H.NS, "delete", "pod", "r2-vram-hog",
                   "--ignore-not-found", "--wait=false"], timeout=60)

    # AND THE TENANTS. This was missing, and it cost an overnight batch: the
    # run left both tenants resident, so the next test in the chain hit
    # preconditions one second later with "GPU 1 already holds 7984 MiB, only
    # 3377 MiB of host RAM available" and refused. Three runs -- two T5
    # repeats and a four-hour soak -- died on a mess this script made. The
    # gate was right; the script was not.
    ev.mark("teardown_begin")
    waited = H.teardown()
    ev.mark("teardown_end", waited_s=round(waited, 1),
            left=len(getattr(H.teardown, "pods_left", [])))
    H.wait_gpu_idle(ev=ev)
    H.set_mps_daemon(True, ev=ev)
    return 0 if rc == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
