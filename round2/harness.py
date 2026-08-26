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
import json
import os
import shlex
import signal
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
    p = subprocess.run(cmd if isinstance(cmd, list) else shlex.split(cmd),
                       capture_output=True, text=True, timeout=timeout)
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
        if not quiet:
            print("  WARNING: pods with r2-role still present after %ds" % timeout)
    return time.time() - t0


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


def assert_colocated(roles, ns=NS, ev=None):
    """A sharing arm whose tenants are on different cards is not a sharing arm.

    This node has two idle 3080 Tis and HAMi's default GPU policy is spread, so
    this is the likely failure, not a paranoid one. It produces clean numbers
    that mean nothing, which is worse than a crash. Returns (ok, detail).
    """
    if len(roles) < 2:
        return True, {}
    place = placement(roles, ns)
    unknown = [r for r, u in place.items() if not u]
    if unknown:
        if ev:
            ev.mark("placement_unknown", roles=",".join(unknown))
        return True, {"warning": "no GPU UUID in annotations for %s; HAMi may "
                                 "not be recording placement in this version. "
                                 "Cross-check nvidia-smi per-device memory."
                                 % ", ".join(unknown), "placement": place}
    devices = set()
    for u in place.values():
        devices.update(u)
    ok = len(devices) == 1
    if ev:
        ev.mark("placement", colocated=ok, devices=len(devices),
                detail=json.dumps(place))
    return ok, {"placement": place, "distinct_devices": sorted(devices)}


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
        self._stop = threading.Event()
        self.rows = 0

    def run(self):
        q = ("index,memory.used,memory.total,utilization.gpu,utilization.memory,"
             "temperature.gpu,power.draw,clocks.sm")
        with open(self.path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["t"] + q.split(","))
            while not self._stop.is_set():
                rc, out = sh(["nvidia-smi", "--query-gpu=" + q,
                              "--format=csv,noheader,nounits"], timeout=15)
                now = time.time()
                if rc == 0:
                    for line in out.strip().splitlines():
                        w.writerow([round(now, 3)]
                                   + [x.strip() for x in line.split(",")])
                        self.rows += 1
                    fh.flush()
                self._stop.wait(self.interval)

    def stop(self):
        self._stop.set()
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
QUERIES = {
    # --- vLLM, per tenant
    "ttft_p99":      'histogram_quantile(0.99, sum by (le, r2_role) (rate(vllm:time_to_first_token_seconds_bucket{namespace="%(ns)s"}[1m])))',
    "ttft_p95":      'histogram_quantile(0.95, sum by (le, r2_role) (rate(vllm:time_to_first_token_seconds_bucket{namespace="%(ns)s"}[1m])))',
    "ttft_p50":      'histogram_quantile(0.50, sum by (le, r2_role) (rate(vllm:time_to_first_token_seconds_bucket{namespace="%(ns)s"}[1m])))',
    "ttft_mean":     'sum by (r2_role) (rate(vllm:time_to_first_token_seconds_sum{namespace="%(ns)s"}[1m])) / clamp_min(sum by (r2_role) (rate(vllm:time_to_first_token_seconds_count{namespace="%(ns)s"}[1m])), 0.001)',
    "itl_p99":       'histogram_quantile(0.99, sum by (le, r2_role) (rate(vllm:time_per_output_token_seconds_bucket{namespace="%(ns)s"}[1m])))',
    "itl_p95":       'histogram_quantile(0.95, sum by (le, r2_role) (rate(vllm:time_per_output_token_seconds_bucket{namespace="%(ns)s"}[1m])))',
    "e2e_p99":       'histogram_quantile(0.99, sum by (le, r2_role) (rate(vllm:e2e_request_latency_seconds_bucket{namespace="%(ns)s"}[1m])))',
    "e2e_p95":       'histogram_quantile(0.95, sum by (le, r2_role) (rate(vllm:e2e_request_latency_seconds_bucket{namespace="%(ns)s"}[1m])))',
    "gen_tok_s":     'sum by (r2_role) (rate(vllm:generation_tokens_total{namespace="%(ns)s"}[1m]))',
    "prompt_tok_s":  'sum by (r2_role) (rate(vllm:prompt_tokens_total{namespace="%(ns)s"}[1m]))',
    "running":       'sum by (r2_role) (vllm:num_requests_running{namespace="%(ns)s"})',
    "waiting":       'sum by (r2_role) (vllm:num_requests_waiting{namespace="%(ns)s"})',
    "kv_usage":      'vllm:gpu_cache_usage_perc{namespace="%(ns)s"}',
    "preemptions":   'sum by (r2_role) (rate(vllm:num_preemptions_total{namespace="%(ns)s"}[1m]))',
    "prefix_hit":    'vllm:gpu_prefix_cache_hit_rate{namespace="%(ns)s"}',
    "req_success":   'sum by (r2_role) (rate(vllm:request_success_total{namespace="%(ns)s"}[1m]))',
    # --- HAMi, what the scheduler promised
    "hami_cores":    'hami_gpu_core_allocated_ratio',
    "hami_limit":    'hami_gpu_core_limit_ratio',
    "hami_mem":      'hami_gpu_memory_allocated_bytes',
    # --- hardware, what the card did
    "gpu_util":      'DCGM_FI_DEV_GPU_UTIL',
    "fb_used":       'DCGM_FI_DEV_FB_USED',
    "mem_copy_util": 'DCGM_FI_DEV_MEM_COPY_UTIL',
    "sm_clock":      'DCGM_FI_DEV_SM_CLOCK',
    "power":         'DCGM_FI_DEV_POWER_USAGE',
    "energy_mj":     'DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION',
    "gpu_temp":      'DCGM_FI_DEV_GPU_TEMP',
}


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
    doc = {"base": base, "t0": t0, "t1": t1, "step": step, "ns": ns,
           "collected_at": time.time(), "series": {}}
    errs = 0
    for name, tmpl in QUERIES.items():
        data = prom_range(base, tmpl % {"ns": ns}, t0 - 30, t1 + 30, step)
        if "error" in data:
            errs += 1
        doc["series"][name] = data
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh)
    n = sum(len(v.get("result", [])) for v in doc["series"].values()
            if isinstance(v, dict))
    if ev:
        ev.mark("prom_collected", series=n, errors=errs,
                path=os.path.basename(out_path))
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
    if ev:
        ev.mark("preconditions", ok=not problems, problems=len(problems))
    return problems
