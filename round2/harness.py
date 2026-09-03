#!/usr/bin/env python3
"""Shared machinery for the round-2 runs: deploy, wait, measure, tear down.

Everything here exists to serve requirement 7 -- the three arms must differ
ONLY in the sharing mechanism. That is easy to say and easy to violate by
accident, so the fairness controls live in code rather than in a checklist:

  * ONE trace file is generated per test and replayed byte-identically by every
    arm. Its SHA-256 trace_id is recorded in each result and asserted equal at
    the end of the run. If an arm somehow got different work, the run fails
    loudly instead of producing a comparison that looks fine.

  * every arm is PINNED TO ONE PHYSICAL GPU. This node has two idle 3080 Tis,
    and nothing stops HAMi from placing two "co-tenants" on separate cards --
    which would silently make the sharing arm an uncontended arm and produce a
    beautiful, meaningless result. See pin_uuid().

  * the GPU is drained to idle between sub-runs and the wait is verified, not
    slept through. A previous tenant still unloading weights is indistinguish-
    able from a slow start otherwise.

  * the client talks STRAIGHT TO THE POD IP, not through the KServe router or a
    port-forward. Both would add a hop that exists in some arms and not others,
    and at TTFT timescales that hop is not noise. We are measuring engines, not
    ingress.

Nothing here is round-2 specific enough to belong in run.py alone -- the
scale-down test in 09-scaledown.py uses the same primitives.
"""
import csv
import errno
import json
import os
import re
import shlex
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

NS = os.environ.get("R2_NS", "llm-demo")
CARD_MIB = 12288

# Both devices on mr-gholami. Index 0 is the default target; the second is here
# so an arm can be pinned away from a busy card if one is ever in use.
GPU_UUIDS = [
    "GPU-bc62f652-9fa4-7abf-2b0d-81a84f3a27cf",   # index 0
    "GPU-fa422f59-3462-6e01-aece-96aeafeab30c",   # index 1
]


# --------------------------------------------------------------------- shell
def sh(cmd, timeout=120, check=False):
    """Run a command, return (rc, stdout+stderr). Never raises on non-zero
    unless asked -- most callers want to inspect the failure, not unwind."""
    argv = cmd if isinstance(cmd, list) else shlex.split(cmd)
    p = None
    for attempt in (0, 1, 2):
        try:
            p = subprocess.run(argv, capture_output=True, text=True,
                               timeout=timeout)
            break
        except OSError as exc:
            # When the node is out of memory, fork() itself fails with ENOMEM
            # and EVERY subprocess call raises -- kubectl, nvidia-smi, all of
            # it. T4 died exactly here: the OSError unwound out of the polling
            # loop inside wait_ready and aborted the run with a traceback that
            # said nothing about memory, 325 seconds into a 900 second wait.
            # Back off and retry: whatever is eating the RAM is usually about
            # to be OOM-killed, and a transient fork failure should not cost a
            # multi-hour run.
            if exc.errno != errno.ENOMEM or attempt == 2:
                raise
            time.sleep(5)
    out = (p.stdout or "") + (p.stderr or "")
    if check and p.returncode != 0:
        raise RuntimeError("command failed (%d): %s\n%s"
                           % (p.returncode, cmd, out[-2000:]))
    return p.returncode, out


def kubectl(args, timeout=120, check=False):
    return sh(["kubectl"] + args, timeout=timeout, check=check)


def kjson(args, timeout=60):
    rc, out = kubectl(args + ["-o", "json"], timeout=timeout)
    if rc != 0:
        return None
    try:
        return json.loads(out)
    except Exception:
        return None


# --------------------------------------------------------------------- events
class Events:
    """A timestamped phase log, written as it happens.

    This is what lets the Grafana screen recording be annotated after the fact:
    every panel is a time series, and without a record of when warmup ended and
    measurement began, any spike near a boundary is unattributable. It is also
    the only durable record of ordering when a run is later re-rendered from
    saved data (requirement 3).
    """

    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._fh = open(path, "a", encoding="utf-8")

    def mark(self, phase, **kw):
        rec = dict(t=time.time(), iso=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                   phase=phase, **kw)
        self._fh.write(json.dumps(rec) + "\n")
        self._fh.flush()
        extra = " ".join("%s=%s" % (k, v) for k, v in kw.items())
        print("  [%s] %s %s" % (time.strftime("%H:%M:%S"), phase, extra))
        return rec["t"]

    def close(self):
        try:
            self._fh.close()
        except Exception:
            pass


# ------------------------------------------------------------------ lifecycle
def teardown(ns=NS, timeout=300, quiet=False):
    """Remove every round-2 workload and wait until the pods are really gone.

    `kubectl delete` returning is not the same as the GPU being free: a vLLM
    pod holds its allocation until the process exits, which takes seconds to
    tens of seconds. Deleting and immediately deploying the next arm is how you
    get an OOM that looks like a sharing failure.
    """
    for sel in ("r2-role", "app.kubernetes.io/part-of=llminferenceservice"):
        for kind in ("llminferenceservice", "deployment", "service", "pod"):
            kubectl(["delete", kind, "-n", ns, "-l", sel,
                     "--ignore-not-found", "--wait=false"], timeout=60)
    t0 = time.time()
    while time.time() - t0 < timeout:
        if not all_r2_pods(ns):
            break
        time.sleep(3)
    else:
        # NOT suppressed by quiet. Leftover pods poison every later arm and
        # every later tier: T4's crash left four vLLM engines running after
        # the harness exited, and they kept OOM-thrashing the node for four
        # hours. A teardown that did not tear down is not a quiet event.
        left = [p["metadata"]["name"] for p in all_r2_pods(ns)]
        print("  WARNING: teardown left %d pod(s) after %ds: %s"
              % (len(left), timeout, ", ".join(left[:6])))
        teardown.pods_left = left
        return time.time() - t0
    teardown.pods_left = []
    return time.time() - t0


HOST_HEADROOM_MIB = 1024      # kernel, the harness itself, and its kubectl forks

# Per-tenant host RSS. PROVISIONAL -- these are not yet measured per role, they
# are read off the OOM killer's own accounting during T4:
#     VLLM::EngineCor  anon-rss 3,033,212 kB .. 3,368,392 kB
#     vllm             anon-rss   941,244 kB
# The default is set so that the two-tenant tiers that are known to work (T1,
# T2, T3) pass the guard and the four-tenant tier that died does not. Replace
# it with real numbers from host.csv -- see 24-host-calibrate below.
HOST_RSS_MIB = {}
HOST_RSS_DEFAULT = 2400   # provisional: passes 2 tenants, refuses 3+


def host_mem_mib():
    """MemTotal / MemAvailable / SwapFree, straight from /proc/meminfo.

    MemAvailable, not MemFree: page cache is reclaimable and MemFree wildly
    understates what a new process can actually get.
    """
    out = {}
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                k, _, v = line.partition(":")
                m = re.match(r"\s*(\d+)", v)
                if m:
                    out[k] = int(m.group(1)) // 1024
    except Exception:
        return {}
    return {"total": out.get("MemTotal", 0),
            "available": out.get("MemAvailable", 0),
            "free": out.get("MemFree", 0),
            "swap_total": out.get("SwapTotal", 0),
            "swap_free": out.get("SwapFree", 0)}


def vllm_rss_mib():
    """Total resident memory of the vLLM process trees, and how many there are.

    Matches the comm names the OOM killer reported: 'VLLM::EngineCor' for the
    engine and 'vllm' for the API server.
    """
    rc, out = sh(["ps", "-eo", "rss=,comm="], timeout=30)
    if rc != 0:
        return 0, 0
    tot = n = 0
    for line in out.splitlines():
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        rss, comm = parts
        if comm.startswith("VLLM") or comm.strip() == "vllm":
            try:
                tot += int(rss)
            except ValueError:
                continue
            n += 1
    return tot // 1024, n


def host_memory_problems(roles, ev=None):
    """Would these tenants fit in RAM? Ask before deploying, not after.

    The VRAM side of this has been modelled since section 2.10 -- gpumem quota
    minus CUDA context and graph pool. The host side had nothing at all, which
    is why a scenario whose GPU arithmetic was correct to 97.9% of the card
    still could not run.
    """
    mem = host_mem_mib()
    if not mem:
        return [], {}
    need = sum(HOST_RSS_MIB.get(r, HOST_RSS_DEFAULT) for r in roles)
    need += HOST_HEADROOM_MIB
    detail = {"roles": list(roles), "need_mib": need,
              "available_mib": mem["available"], "total_mib": mem["total"],
              "swap_total_mib": mem["swap_total"],
              "measured": sorted(r for r in roles if r in HOST_RSS_MIB)}
    problems = []
    if need > mem["available"]:
        problems.append(
            "host RAM: %d tenants need about %d MiB but only %d MiB is "
            "available (total %d MiB, swap %d MiB). This is what killed T4 -- "
            "the OOM killer took two vLLM engines and Prometheus. Use fewer "
            "tenants, or measure real per-role RSS into HOST_RSS_MIB."
            % (len(roles), need, mem["available"], mem["total"],
               mem["swap_total"]))
    if ev:
        ev.mark("host_memory", ok=not problems, **detail)
    return problems, detail


class HostSampler(threading.Thread):
    """1 Hz host memory sampling into host.csv, running from apply to teardown.

    Deliberately wider than GpuSampler's window. GpuSampler starts at replay,
    but the host memory peak is during INITIALISATION -- four engines loading
    weights at once -- which is where T4 actually died, minutes before any
    replay would have started.
    """

    def __init__(self, path, interval=1.0):
        super().__init__(daemon=True)
        self.path = path
        self.interval = interval
        self._halt = threading.Event()
        self.rows = 0
        self.min_available = None

    def run(self):
        cols = ["t", "mem_total", "mem_available", "mem_free",
                "swap_total", "swap_free", "vllm_rss", "vllm_procs"]
        with open(self.path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(cols)
            while not self._halt.is_set():
                m = host_mem_mib()
                rss, n = vllm_rss_mib()
                if m:
                    w.writerow([round(time.time(), 3), m["total"],
                                m["available"], m["free"], m["swap_total"],
                                m["swap_free"], rss, n])
                    self.rows += 1
                    if self.min_available is None or m["available"] < self.min_available:
                        self.min_available = m["available"]
                    fh.flush()
                self._halt.wait(self.interval)

    def stop(self):
        self._halt.set()
        self.join(timeout=10)


def gpu_used_mib():
    """Per-device memory.used, straight from nvidia-smi."""
    rc, out = sh(["nvidia-smi", "--query-gpu=index,memory.used",
                  "--format=csv,noheader,nounits"], timeout=30)
    if rc != 0:
        return {}
    vals = {}
    for line in out.strip().splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) == 2 and parts[0].isdigit():
            vals[int(parts[0])] = int(parts[1])
    return vals


def wait_gpu_idle(baseline_mib=300, timeout=240, ev=None):
    """Block until every device is back near baseline.

    baseline defaults to 300 MiB because the MPS server itself holds ~30 MiB
    per device and that never goes away while the daemon runs.
    """
    t0 = time.time()
    while time.time() - t0 < timeout:
        used = gpu_used_mib()
        if used and max(used.values()) <= baseline_mib:
            if ev:
                ev.mark("gpu_idle", waited_s=round(time.time() - t0, 1),
                        used=json.dumps(used))
            return True
        time.sleep(5)
    if ev:
        ev.mark("gpu_idle_TIMEOUT", used=json.dumps(gpu_used_mib()))
    return False


# KServe does NOT copy an LLMInferenceService's metadata.labels onto the pods
# it creates -- spec.template is a bare PodSpec, which has no metadata field, so
# there is nowhere to put them. Confirmed on the real object; the workload pod
# came up carrying only:
#
#   app.kubernetes.io/name=r2-embed
#   app.kubernetes.io/component=llminferenceservice-workload
#   kserve.io/component=workload
#   llm-d.ai/role=both
#   hami.io/vgpu-node=mr-gholami
#
# A plain vllm Deployment (--pooling-via vllm) DOES carry r2-role, because we
# write its pod template ourselves. So selection has to cover both, and the
# KServe half must also exclude the router-scheduler pod, which shares
# app.kubernetes.io/name with the workload.
KSERVE_SEL = "app.kubernetes.io/name=r2-%s,kserve.io/component=workload"


def pods_for(role, ns=NS):
    seen, out = set(), []
    for sel in ("r2-role=%s" % role, KSERVE_SEL % role):
        d = kjson(["get", "pods", "-n", ns, "-l", sel])
        for p in (d or {}).get("items", []):
            name = p.get("metadata", {}).get("name")
            if name and name not in seen:
                seen.add(name)
                out.append(p)
    return out


def all_r2_pods(ns=NS):
    """Every round-2 pod, whichever path created it."""
    seen, out = set(), []
    for sel in ("r2-role", "kserve.io/component=workload"):
        d = kjson(["get", "pods", "-n", ns, "-l", sel])
        for p in (d or {}).get("items", []):
            name = p.get("metadata", {}).get("name")
            if name and name not in seen:
                seen.add(name)
                out.append(p)
    return out


def wait_ready(roles, ns=NS, timeout=900, ev=None):
    """Wait for one Ready pod per role, and surface WHY if it never happens.

    A vLLM pod that fails to start is usually one of three things -- OOM on the
    gpumem quota, HAMi refusing to place it, or the model path being wrong --
    and all three look like "not ready yet" until you read the events.
    """
    t0 = time.time()
    pending = set(roles)
    while time.time() - t0 < timeout:
        for role in sorted(pending):
            for p in pods_for(role, ns):
                conds = {c["type"]: c["status"]
                         for c in (p.get("status", {}).get("conditions") or [])}
                if conds.get("Ready") == "True":
                    pending.discard(role)
                    if ev:
                        ev.mark("ready", role=role,
                                after_s=round(time.time() - t0, 1))
                    break
        if not pending:
            return True, {}
        time.sleep(5)

    diag = {}
    for role in sorted(pending):
        chunks = []
        for p in pods_for(role, ns):
            name = p["metadata"]["name"]
            phase = p.get("status", {}).get("phase")
            chunks.append("pod %s phase=%s" % (name, phase))
            for cs in (p.get("status", {}).get("containerStatuses") or []):
                w = (cs.get("state", {}).get("waiting") or {})
                if w:
                    chunks.append("  waiting: %s %s"
                                  % (w.get("reason"), w.get("message", "")[:200]))
            rc, ev_out = kubectl(["get", "events", "-n", ns,
                                  "--field-selector", "involvedObject.name=%s" % name,
                                  "--sort-by", ".lastTimestamp"], timeout=30)
            chunks.append("  events:\n" + "\n".join(ev_out.strip().splitlines()[-8:]))
            rc, logs = kubectl(["logs", "-n", ns, name, "--tail=40"], timeout=60)
            if rc == 0 and logs.strip():
                chunks.append("  logs tail:\n" + logs.strip()[-1500:])
        diag[role] = "\n".join(chunks) or "no pods found at all"
    return False, diag


# ------------------------------------------------------------------ endpoints
CANDIDATE_PORTS = [8000, 8080, 80]


def _probe(url, timeout=4):
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status < 500
    except urllib.error.HTTPError as e:
        return e.code < 500          # 404 still proves something is listening
    except Exception:
        return False


def endpoints(roles, ns=NS, ev=None):
    """Resolve each role to http://<podIP>:<port>.

    Pod IPs are directly routable from this node (k3s flannel), which the
    preflight already demonstrated by curling a ClusterIP from the host. Going
    straight to the pod removes the KServe router from the measurement -- it is
    present for chat models and absent for a plain vllm Deployment, so leaving
    it in would put a per-arm difference inside the latency we are comparing.
    """
    out = {}
    for role in roles:
        ps = [p for p in pods_for(role, ns)
              if p.get("status", {}).get("podIP")]
        if not ps:
            raise RuntimeError("no pod with an IP for role %s" % role)
        ip = ps[0]["status"]["podIP"]

        ports = []
        for c in ps[0]["spec"].get("containers", []):
            for cp in (c.get("ports") or []):
                if cp.get("containerPort"):
                    ports.append(int(cp["containerPort"]))
        for p in CANDIDATE_PORTS:
            if p not in ports:
                ports.append(p)

        chosen = None
        for port in ports:
            base = "http://%s:%d" % (ip, port)
            if _probe(base + "/health") or _probe(base + "/v1/models"):
                chosen = base
                break
        if not chosen:
            raise RuntimeError(
                "role %s: pod %s at %s answers on none of %s"
                % (role, ps[0]["metadata"]["name"], ip, ports))
        out[role] = chosen
        if ev:
            ev.mark("endpoint", role=role, url=chosen)
    return out


def served_model(base, timeout=8):
    """Ask the engine what it calls itself.

    KServe and a plain `vllm serve` disagree: one advertises the HF repo id,
    the other whatever --served-model-name was passed. Sending the wrong name
    gets a 404 that looks exactly like a dead endpoint, so ask rather than
    assume.
    """
    try:
        with urllib.request.urlopen(base.rstrip("/") + "/v1/models",
                                    timeout=timeout) as r:
            doc = json.loads(r.read().decode("utf-8"))
        items = doc.get("data") or []
        if items and items[0].get("id"):
            return items[0]["id"]
    except Exception:
        pass
    return None


# ------------------------------------------------------------- gpu placement
def placement(roles, ns=NS):
    """Which physical device did each role actually land on?

    HAMi records its decision in the pod's own annotations. We parse any GPU
    UUID out of them rather than matching one exact key, because the key name
    has changed across HAMi versions and a missed rename would silently
    disable the check that matters most.
    """
    import re
    out = {}
    for role in roles:
        uuids = []
        for p in pods_for(role, ns):
            ann = json.dumps(p.get("metadata", {}).get("annotations", {}) or {})
            uuids += re.findall(r"GPU-[0-9a-fA-F-]{36}", ann)
        out[role] = sorted(set(uuids))
    return out


def devices_in_use(floor_mib=500):
    """Which physical devices actually hold a model right now.

    Independent of HAMi's annotation format, and therefore the check that
    cannot rot. The floor is well above the ~39 MiB the MPS server holds on an
    idle card and well below any real model.
    """
    return sorted(idx for idx, mib in gpu_used_mib().items() if mib > floor_mib)


MPS_CONTROL = "nvidia-cuda-mps-control"
MPS_PIPE = os.environ.get("CUDA_MPS_PIPE_DIRECTORY", "/tmp/nvidia-mps")
# The daemon on this node is NOT systemd's -- nvidia-mps.service has been
# failed and disabled since 2026-08-16, which is why `systemctl stop` returned
# rc=0 and changed nothing. It is a DaemonSet, and it self-heals: its
# supervisor loop restarts nvidia-cuda-mps-control within 15s of the control
# socket disappearing. Killing the process is therefore useless; the workload
# has to be moved off the node.
MPS_DS = os.environ.get("R2_MPS_DS", "qwen-benchmark-mps-daemon")
MPS_DS_NS = os.environ.get("R2_MPS_DS_NS", "llm-demo")
MPS_OFF_SELECTOR = {"r2-mps": "disabled-by-harness"}


def mps_daemon_running():
    """Can a CUDA client attach to MPS right now?

    NOT pgrep. Two different processes on this node carry the string
    "nvidia-cuda-mps-control" in their command line without being the daemon:
    a `bash -lc` wrapper around pgrep itself, and the DaemonSet's supervisor
    loop. Both matched, so the check returned True unconditionally and every
    non-MPS arm aborted -- 0/4 sub-runs across three tiers.

    The control socket is the signal the DaemonSet's own self-heal loop uses,
    and a stale socket is possible after a hard kill, so confirm by actually
    talking to the daemon. That is exactly what a CUDA client does at init,
    which makes it the right question to ask.
    """
    sock = os.path.join(MPS_PIPE, "control")
    if not os.path.exists(sock):
        return False
    rc, _ = sh(["bash", "-lc",
                "CUDA_MPS_PIPE_DIRECTORY=%s; export CUDA_MPS_PIPE_DIRECTORY; "
                "echo get_server_list | timeout 10 %s >/dev/null 2>&1"
                % (MPS_PIPE, MPS_CONTROL)], timeout=25)
    return rc == 0


def mps_client_pids():
    """PIDs nvidia-smi reports as MPS clients.

    nvidia-smi tags a process that reaches the GPU through the MPS server as
    type M+C rather than plain C. That is the only direct observable for
    "is this process actually multiplexed", and nothing in the pod spec
    substitutes for it -- a container with no MPS mount and no MPS env still
    attaches, because the runtime injects the pipe after the spec is written.
    """
    rc, out = sh(["nvidia-smi"], timeout=60)
    pids = []
    for line in out.splitlines():
        if "M+C" not in line:
            continue
        m = re.search(r"(\d+)\s+M\+C", line)
        if m:
            pids.append(int(m.group(1)))
    return sorted(set(pids))


def _patch_mps_ds(disable):
    """Park the MPS DaemonSet on a node selector nothing satisfies, or release it.

    Scaling is not available for a DaemonSet, and deleting it would lose the
    manifest. An unsatisfiable nodeSelector is the standard way to take one to
    zero, and it is reversible in one patch.
    """
    if disable:
        body = {"spec": {"template": {"spec":
                {"nodeSelector": MPS_OFF_SELECTOR}}}}
    else:
        body = {"spec": {"template": {"spec": {"nodeSelector": None}}}}
    return kubectl(["-n", MPS_DS_NS, "patch", "daemonset", MPS_DS,
                    "-p", json.dumps(body)], timeout=60)


def _clear_stale_socket():
    """Delete a control socket with nothing behind it.

    The supervisor's self-heal tests `[ ! -S /tmp/nvidia-mps/control ]` -- file
    EXISTENCE, not reachability. A daemon that dies while leaving its socket
    behind is therefore never restarted: the DaemonSet reads 1/1 ready, the
    socket is present, and MPS is dead. Observed on this node.

    Removing the socket is what fires the supervisor's own recovery, on its
    next 15s poll. Only ever called once mps_daemon_running() has already shown
    the daemon unreachable, so there is nothing live to break.
    """
    return sh(["sudo", "-n", "rm", "-f", os.path.join(MPS_PIPE, "control")],
              timeout=30)


def set_mps_daemon(want, ev=None, timeout=150):
    """Put MPS into the state this arm requires, and verify it took.

    THIS IS THE FIX FOR THE ROUND-TWO INVALIDATION. Round one stopped the host
    daemon for its non-MPS arms; round two left it running, and every T1 and T2
    pod became an MPS client regardless of arm -- one NEW CLIENT per tenant,
    60-90s before each replay window, in the dedicated and hami arms too.
    The arms were identical, so +0.0% and -0.008% compared MPS with itself.
    """
    # The SUPERVISOR is converged first and unconditionally, even when the
    # daemon already reads the way this arm wants.
    #
    # Returning "already down" on an unreachable daemon while leaving the
    # DaemonSet running is a 15-second time bomb: its loop restarts MPS the
    # moment the socket goes, mid-run, and a dedicated or hami arm silently
    # becomes an MPS arm. That is the exact defect this function exists to
    # prevent, so the fast path is not available.
    rc, out = _patch_mps_ds(disable=not want)
    if rc != 0:
        return False, ("could not patch daemonset %s/%s: %s"
                       % (MPS_DS_NS, MPS_DS, out[-400:]))

    t0 = time.time()
    nudged = False
    while time.time() - t0 < timeout:
        if mps_daemon_running() == want:
            if ev:
                ev.mark("mps_daemon", want=want,
                        after_s=round(time.time() - t0, 1), nudged=nudged)
            time.sleep(3)     # let the server settle before pods start
            return True, ("supervisor %s, MPS %s after %.0fs%s"
                          % ("parked" if not want else "released",
                             "up" if want else "down", time.time() - t0,
                             ", cleared a stale socket" if nudged else ""))
        # Bringing MPS back up on a node whose supervisor is wedged on a stale
        # socket needs exactly one nudge; its own loop takes over after that.
        if want and not nudged and time.time() - t0 > 25:
            _clear_stale_socket()
            nudged = True
        time.sleep(2)

    extra = ""
    if want:
        extra = (" A stale socket was cleared and it still did not come back, "
                 "so check that the DaemonSet's pod is actually running and "
                 "that passwordless sudo works." if nudged else "")
    return False, ("MPS did not go %s within %ds after patching daemonset "
                   "%s/%s.%s Check R2_MPS_DS if it is named differently here."
                   % ("up" if want else "down", timeout, MPS_DS_NS, MPS_DS,
                      extra))


def assert_mps_matches_arm(arm, roles, ev=None):
    """Verify MPS is on exactly in the arm that is supposed to have it.

    Parallel to assert_colocated, and earned the same way: a sharing arm whose
    tenants are on separate cards measures nothing, and an MPS arm that is
    indistinguishable from its control measures nothing either. Both produce
    clean numbers. Both are worse than a crash.
    """
    want = (arm == "hami-mps")
    running = mps_daemon_running()
    clients = mps_client_pids()
    detail = {"arm": arm, "want_mps": want, "daemon_running": running,
              "mps_client_pids": clients, "tenants": len(roles)}

    if want:
        ok = running and len(clients) >= len(roles)
        if not ok:
            detail["reason"] = (
                "arm hami-mps expects the daemon up and >=%d MPS clients; "
                "daemon_running=%s, clients=%s. The arm would be identical to "
                "its control." % (len(roles), running, clients))
    else:
        ok = (not running) and not clients
        if not ok:
            detail["reason"] = (
                "arm %s must have NO MPS: daemon_running=%s, M+C clients=%s. "
                "This is exactly what invalidated T1 and T2." % (arm, running,
                                                                 clients))
    if ev:
        ev.mark("mps_state", ok=ok, **detail)
    return ok, detail


def assert_colocated(roles, ns=NS, ev=None):
    """A sharing arm whose tenants are on different cards is not a sharing arm.

    NOT a paranoid check. Measured on this node: an unpinned single tenant was
    placed by HAMi on GPU 1 while GPU 0 sat at 39 MiB. With two tenants and no
    pin, landing on separate cards is the expected outcome, and it produces
    clean numbers that mean nothing -- worse than a crash, because a crash
    stops you.

    Two independent verdicts, because they fail differently:

      annotations  HAMi's own record of what it decided. Precise, but the key
                   name has moved between versions, so absence proves nothing.
      nvidia-smi   what the hardware says. Cannot be fooled by a renamed
                   annotation, cannot attribute a device to a role.

    The physical check is the one that can veto. Returns (ok, detail).
    """
    if len(roles) < 2:
        return True, {"devices_in_use": devices_in_use()}

    phys = devices_in_use()
    place = placement(roles, ns)
    unknown = [r for r, u in place.items() if not u]
    ann_devices = sorted({u for us in place.values() for u in us})

    detail = {"placement": place, "distinct_devices": ann_devices,
              "devices_in_use": phys}

    # Physical verdict first: more than one device holding a model means the
    # tenants are not sharing anything, whatever the annotations say.
    if len(phys) > 1:
        detail["reason"] = ("nvidia-smi shows %d devices holding models (%s); "
                            "the tenants are on separate cards"
                            % (len(phys), phys))
        if ev:
            ev.mark("placement_SPLIT", devices=len(phys), detail=json.dumps(detail))
        return False, detail

    if unknown:
        detail["warning"] = (
            "no GPU UUID in annotations for %s -- HAMi may not record placement "
            "under that key in this version. Falling back to nvidia-smi, which "
            "shows %d device(s) in use." % (", ".join(unknown), len(phys)))
        if ev:
            ev.mark("placement_annotation_missing", devices=len(phys))
        return len(phys) <= 1, detail

    ok = len(ann_devices) == 1 and len(phys) <= 1
    if ev:
        ev.mark("placement", colocated=ok, ann_devices=len(ann_devices),
                phys_devices=len(phys), detail=json.dumps(detail))
    return ok, detail


# ----------------------------------------------------------------- smoke test
def _post(url, body, timeout=120):
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST",
                                 headers={"content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read(4096).decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read(2048).decode("utf-8", "replace")
    except Exception as exc:
        return None, "%s: %s" % (type(exc).__name__, exc)


def smoke_test(endpoints, models, kinds, timeout=120, ev=None):
    """One real request per role, BEFORE the measured replay starts.

    T1's first run spent 45 minutes producing 0/885 successful LLM requests in
    every arm. The pods were Ready, the placement was verified, the trace was
    identical -- and every single generative request failed. Readiness only
    proves /health answers; it says nothing about whether the endpoint accepts
    the request shape the client will send.

    Thirty seconds here would have caught it before the first arm, instead of
    after the fourth. Returns (ok, {role: detail}).
    """
    detail, ok = {}, True
    for role, base in sorted(endpoints.items()):
        model = models.get(role, role)
        kind = kinds.get(role, "chat")
        if kind == "chat":
            paths = ["/v1/chat/completions"]
            body = {"model": model, "max_tokens": 8, "stream": False,
                    "messages": [{"role": "user", "content": "reply with: ok"}]}
        elif kind == "rerank":
            paths = ["/v1/rerank", "/rerank", "/v1/score", "/score"]
            body = {"model": model, "query": "what is MPS",
                    "documents": ["NVIDIA MPS shares a GPU", "unrelated"]}
        else:
            paths = ["/v1/embeddings"]
            body = {"model": model, "input": ["hello world"]}

        last = None
        for path in paths:
            code, text = _post(base + path, body, timeout)
            last = "%s %s -> %s %s" % (path, model, code, (text or "")[:240])
            if code == 200:
                detail[role] = "ok via %s" % path
                break
        else:
            ok = False
            detail[role] = "FAILED: %s" % last
        if ev:
            ev.mark("smoke", role=role, ok=(role in detail
                                            and detail[role].startswith("ok")))
    return ok, detail


# -------------------------------------------------------------- gpu sampling
class GpuSampler(threading.Thread):
    """1 Hz nvidia-smi sampling into a CSV, alongside Prometheus.

    Not redundant with DCGM: DCGM is scraped at 5s and goes through Prometheus,
    so a 20-second run tail is four points. Peak VRAM is a required metric and a
    5s grid can miss the peak entirely. This also survives Prometheus being
    unavailable, which matters because it is the only copy of the hardware view
    that lives inside the results directory.
    """

    def __init__(self, path, interval=1.0):
        super().__init__(daemon=True)
        self.path = path
        self.interval = interval
        # NOT self._stop: threading.Thread has a PRIVATE METHOD of that name,
        # and join() calls it via _wait_for_tstate_lock once the thread ends.
        # Shadowing it with an Event made join() raise
        #     TypeError: 'Event' object is not callable
        # in the CALLING thread -- which aborted the caller before it could
        # collect Prometheus, silently losing prom.json for every sub-run of
        # the first T1 attempt while the run otherwise looked healthy.
        self._halt = threading.Event()
        self.rows = 0

    def run(self):
        q = ("index,memory.used,memory.total,utilization.gpu,utilization.memory,"
             "temperature.gpu,power.draw,clocks.sm")
        with open(self.path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["t"] + q.split(","))
            while not self._halt.is_set():
                rc, out = sh(["nvidia-smi", "--query-gpu=" + q,
                              "--format=csv,noheader,nounits"], timeout=15)
                now = time.time()
                if rc == 0:
                    for line in out.strip().splitlines():
                        w.writerow([round(now, 3)]
                                   + [x.strip() for x in line.split(",")])
                        self.rows += 1
                    fh.flush()
                self._halt.wait(self.interval)

    def stop(self):
        self._halt.set()
        self.join(timeout=10)


# ---------------------------------------------------------------- prometheus
def prometheus_url():
    """ClusterIP of the existing kube-prometheus-stack, reachable from the node."""
    for ns_, name in (("default", "prometheus-operated"),
                      ("default", "prometheus-kube-prometheus-prometheus")):
        rc, ip = kubectl(["get", "svc", name, "-n", ns_,
                          "-o", "jsonpath={.spec.clusterIP}"], timeout=30)
        ip = ip.strip()
        if rc == 0 and ip and ip != "None":
            return "http://%s:9090" % ip
    return None


# The metric set behind requirement 1. Client-side latency comes from
# replay.requests.csv and is authoritative -- these are the SERVER-side view,
# the one on the dashboard and therefore the one on the video.
#
# On SM occupancy: DCGM_FI_PROF_* is absent on this hardware (confirmed on both
# exporters), so there is no true occupancy counter to collect. What is here
# instead is the honest substitute set -- power and energy track real SM work,
# MEM_COPY_UTIL separates memory-bound from compute-bound, and the per-tenant
# vLLM counters do the discriminating that DCGM_FI_DEV_GPU_UTIL cannot.
# Metric names move between vLLM releases. They have now been wrong twice --
# vllm:gpu_cache_usage_perc and vllm:time_per_output_token_seconds_bucket both
# vanished by 0.15.1, and each silently produced an empty panel that reads as
# "the system was idle". So names are no longer hardcoded: every alias lists
# candidates in preference order, resolve_metrics() picks whichever this
# Prometheus actually holds, and prom_collect records the choice. An alias that
# resolves to nothing is SKIPPED with a reason rather than emitting a query
# whose empty result is indistinguishable from no load.
METRIC_CANDIDATES = {
    "TTFT":      ["vllm:time_to_first_token_seconds"],
    # True per-token spacing ONLY. vllm:request_decode_time_seconds is total
    # decode time per REQUEST and is a different quantity -- accepting it as a
    # fallback would put the wrong number under an "ITL" label. It is collected
    # separately below under its own name. If neither per-token metric exists,
    # the itl_* queries are skipped and the CLIENT measurement stands, which is
    # authoritative anyway: replay.py streams and records real inter-token gaps.
    "ITL":       ["vllm:time_per_output_token_seconds",
                  "vllm:inter_token_latency_seconds"],
    "DECODE":    ["vllm:request_decode_time_seconds"],
    "PREFILL":   ["vllm:request_prefill_time_seconds"],
    "QUEUE":     ["vllm:request_queue_time_seconds"],
    "E2E":       ["vllm:e2e_request_latency_seconds"],
    "GEN":       ["vllm:generation_tokens_total"],
    "PROMPT":    ["vllm:prompt_tokens_total"],
    "RUNNING":   ["vllm:num_requests_running"],
    "WAITING":   ["vllm:num_requests_waiting"],
    "KV":        ["vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"],
    "PREEMPT":   ["vllm:num_preemptions_total"],
    "PREFIX":    ["vllm:prefix_cache_hit_rate", "vllm:gpu_prefix_cache_hit_rate"],
    "SUCCESS":   ["vllm:request_success_total"],
    "HAMICORE":  ["hami_gpu_core_allocated_ratio", "hami_core_allocated_ratio"],
    "HAMILIMIT": ["hami_gpu_core_limit_ratio", "hami_core_limit_ratio"],
    "HAMIMEM":   ["hami_gpu_memory_allocated_bytes", "hami_memory_allocated_bytes"],
}

# Histogram aliases carry no suffix; each query supplies _bucket, _sum, _count.
QUERIES = {
    "ttft_p99":      'histogram_quantile(0.99, sum by (le, r2_role) (rate({TTFT}_bucket{{namespace="%(ns)s"}}[1m])))',
    "ttft_p95":      'histogram_quantile(0.95, sum by (le, r2_role) (rate({TTFT}_bucket{{namespace="%(ns)s"}}[1m])))',
    "ttft_p50":      'histogram_quantile(0.50, sum by (le, r2_role) (rate({TTFT}_bucket{{namespace="%(ns)s"}}[1m])))',
    "ttft_mean":     'sum by (r2_role) (rate({TTFT}_sum{{namespace="%(ns)s"}}[1m])) / clamp_min(sum by (r2_role) (rate({TTFT}_count{{namespace="%(ns)s"}}[1m])), 0.001)',
    "itl_p99":       'histogram_quantile(0.99, sum by (le, r2_role) (rate({ITL}_bucket{{namespace="%(ns)s"}}[1m])))',
    "itl_p95":       'histogram_quantile(0.95, sum by (le, r2_role) (rate({ITL}_bucket{{namespace="%(ns)s"}}[1m])))',
    "e2e_p99":       'histogram_quantile(0.99, sum by (le, r2_role) (rate({E2E}_bucket{{namespace="%(ns)s"}}[1m])))',
    "e2e_p95":       'histogram_quantile(0.95, sum by (le, r2_role) (rate({E2E}_bucket{{namespace="%(ns)s"}}[1m])))',
    "gen_tok_s":     'sum by (r2_role) (rate({GEN}{{namespace="%(ns)s"}}[1m]))',
    "prompt_tok_s":  'sum by (r2_role) (rate({PROMPT}{{namespace="%(ns)s"}}[1m]))',
    "running":       'sum by (r2_role) ({RUNNING}{{namespace="%(ns)s"}})',
    "waiting":       'sum by (r2_role) ({WAITING}{{namespace="%(ns)s"}})',
    "kv_usage":      '{KV}{{namespace="%(ns)s"}}',
    "preemptions":   'sum by (r2_role) (rate({PREEMPT}{{namespace="%(ns)s"}}[1m]))',
    "prefix_hit":    '{PREFIX}{{namespace="%(ns)s"}}',
    "req_success":   'sum by (r2_role) (rate({SUCCESS}{{namespace="%(ns)s"}}[1m]))',
    # Per-request phase breakdown. Not a substitute for ITL, but it separates
    # queueing from prefill from decode, which is exactly what tells apart
    # "the GPU is slow" from "the scheduler is holding requests".
    "decode_p95":    'histogram_quantile(0.95, sum by (le, r2_role) (rate({DECODE}_bucket{{namespace="%(ns)s"}}[1m])))',
    "prefill_p95":   'histogram_quantile(0.95, sum by (le, r2_role) (rate({PREFILL}_bucket{{namespace="%(ns)s"}}[1m])))',
    "queue_p95":     'histogram_quantile(0.95, sum by (le, r2_role) (rate({QUEUE}_bucket{{namespace="%(ns)s"}}[1m])))',
    "hami_cores":    '{HAMICORE}',
    "hami_limit":    '{HAMILIMIT}',
    "hami_mem":      '{HAMIMEM}',
    "gpu_util":      'DCGM_FI_DEV_GPU_UTIL',
    "fb_used":       'DCGM_FI_DEV_FB_USED',
    "mem_copy_util": 'DCGM_FI_DEV_MEM_COPY_UTIL',
    "sm_clock":      'DCGM_FI_DEV_SM_CLOCK',
    "power":         'DCGM_FI_DEV_POWER_USAGE',
    "energy_mj":     'DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION',
    "gpu_temp":      'DCGM_FI_DEV_GPU_TEMP',
}


def metric_names(base, timeout=30):
    """Every metric name this Prometheus currently knows about."""
    url = base + "/api/v1/label/__name__/values"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            doc = json.loads(r.read().decode("utf-8"))
        return set(doc.get("data", []) or [])
    except Exception:
        return set()


def resolve_metrics(base, known=None):
    """alias -> the candidate that exists here. Unresolvable aliases omitted."""
    known = known if known is not None else metric_names(base)
    out = {}
    for alias, cands in METRIC_CANDIDATES.items():
        for c in cands:
            # A histogram counts as present if any suffixed series exists.
            if c in known or any(c + sfx in known
                                 for sfx in ("_bucket", "_sum", "_count")):
                out[alias] = c
                break
    return out


def prom_range(base, expr, t0, t1, step=5):
    q = urllib.parse.urlencode(dict(query=expr, start="%.3f" % t0,
                                    end="%.3f" % t1, step=str(step)))
    url = base + "/api/v1/query_range?" + q
    try:
        with urllib.request.urlopen(url, timeout=45) as r:
            doc = json.loads(r.read().decode("utf-8"))
        if doc.get("status") != "success":
            return {"error": doc.get("error", "non-success")}
        return doc["data"]
    except Exception as exc:
        return {"error": "%s: %s" % (type(exc).__name__, exc)}


def prom_collect(out_path, t0, t1, ns=NS, step=5, base=None, ev=None):
    """Snapshot every query over the run window into one JSON file.

    Requirement 3 is the whole reason this is a file rather than a dashboard
    link: Prometheus retention will expire, and a graph that cannot be redrawn
    six months later is not evidence. Everything needed to redraw every panel
    lands here.
    """
    base = base or prometheus_url()
    if not base:
        if ev:
            ev.mark("prom_unavailable")
        return None
    known = metric_names(base)
    res = resolve_metrics(base, known)
    doc = {"base": base, "t0": t0, "t1": t1, "step": step, "ns": ns,
           "collected_at": time.time(), "series": {},
           "resolved_metrics": res,
           "unresolved": sorted(set(METRIC_CANDIDATES) - set(res))}
    errs, skipped = 0, []
    for name, tmpl in QUERIES.items():
        need = set(re.findall(r"\{([A-Z]+)\}", tmpl))
        if not need.issubset(res):
            skipped.append(name)
            doc["series"][name] = {
                "skipped": "no metric present for %s"
                           % ",".join(sorted(need - set(res)))}
            continue
        expr = tmpl.format(**res) % {"ns": ns}
        data = prom_range(base, expr, t0 - 30, t1 + 30, step)
        if isinstance(data, dict):
            data["expr"] = expr
        if "error" in data:
            errs += 1
        doc["series"][name] = data
    doc["skipped"] = skipped
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh)
    n = sum(len(v.get("result", [])) for v in doc["series"].values()
            if isinstance(v, dict))
    if ev:
        ev.mark("prom_collected", series=n, errors=errs,
                skipped=len(skipped), path=os.path.basename(out_path))
    return doc


# ------------------------------------------------------------------ misc bits
def write_json(path, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True)


def read_json(path, default=None):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return default


def preconditions(ev=None):
    """Refuse to start a multi-hour run on a node that will evict it.

    The probe pod was already killed twice by ephemeral-storage eviction, and
    eviction is node-wide: it takes running benchmark pods with it. Losing an
    arm at minute 40 of a three-arm run is worse than not starting.
    """
    problems = []
    rc, out = sh(["df", "-Pk", "/var/lib/rancher/k3s/agent/containerd"], timeout=30)
    if rc == 0:
        try:
            avail_gib = int(out.strip().splitlines()[1].split()[3]) / 1048576.0
            if avail_gib < 6.0:
                problems.append("only %.1f GiB free on the container filesystem; "
                                "kubelet evicts at 4.84 GiB and eviction is "
                                "node-wide" % avail_gib)
        except Exception:
            pass
    used = gpu_used_mib()
    for idx, mib in sorted(used.items()):
        if mib > 500:
            problems.append("GPU %d already holds %d MiB; something else is "
                            "resident" % (idx, mib))
    if not prometheus_url():
        problems.append("Prometheus ClusterIP not resolvable; server-side "
                        "metrics would be lost")
    mem = host_mem_mib()
    if mem and mem["available"] < 4096:
        problems.append("only %d MiB of host RAM available; a single vLLM "
                        "engine needs roughly %d MiB resident"
                        % (mem["available"], HOST_RSS_DEFAULT))
    if mem and mem["swap_total"] == 0:
        # Not fatal, but it is why an overshoot becomes a kill rather than a
        # slowdown, and worth stating once at the top of a multi-hour run.
        print("  NOTE: no swap configured -- host memory overshoot goes "
              "straight to the OOM killer.")
    if ev:
        ev.mark("preconditions", ok=not problems, problems=len(problems))
    return problems
