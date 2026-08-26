#!/usr/bin/env python3
"""Everything that must be true before a run, checked in about a minute.

T1's first attempt cost 45 minutes and produced 0/885 successful LLM requests,
no HAMi metrics, and no prom.json. Every one of those was knowable beforehand.
This is the gate: it fails loudly and early instead.

    python 13-validate.py              # all checks, including a live smoke test
    python 13-validate.py --no-deploy  # skip the smoke test (~50s, no GPU work)

Exit code 0 means a run is worth starting.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.request

import harness as H

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
FAILS, WARNS = [], []


def head(t):
    print("\n=== %s" % t)


def ok(msg):
    print("  [ok]   %s" % msg)


def warn(msg):
    print("  [warn] %s" % msg)
    WARNS.append(msg)


def bad(msg):
    print("  [FAIL] %s" % msg)
    FAILS.append(msg)


# --------------------------------------------------------------- monitoring
def selector_matches(sel, labels):
    """prometheus-operator semantics: None selects nothing, {} selects all."""
    if sel is None:
        return False, "selector absent -> selects nothing"
    ml = sel.get("matchLabels") or {}
    me = sel.get("matchExpressions") or []
    if not ml and not me:
        return True, "empty selector -> selects all"
    for k, v in ml.items():
        if labels.get(k) != v:
            return False, "needs %s=%s, object has %s" % (k, v, labels.get(k))
    return True, "matchLabels satisfied"


def check_prometheus():
    head("Prometheus is configured to see our monitors")
    d = H.kjson(["get", "prometheus", "-A"])
    items = (d or {}).get("items", [])
    if not items:
        bad("no Prometheus CR found")
        return
    for p in items:
        ns = p["metadata"]["namespace"]
        name = p["metadata"]["name"]
        spec = p.get("spec", {})
        print("  %s/%s" % (ns, name))
        for kind, sel_key, ns_key, mon_ns in (
                ("ServiceMonitor", "serviceMonitorSelector",
                 "serviceMonitorNamespaceSelector", ["kube-system"]),
                ("PodMonitor", "podMonitorSelector",
                 "podMonitorNamespaceSelector", ["llm-demo"])):
            sel = spec.get(sel_key)
            nssel = spec.get(ns_key)
            m, why = selector_matches(sel, {"release": "prometheus"})
            print("      %-15s selector: %s" % (kind, why))
            if not m:
                bad("%s selector will not match our monitors (label "
                    "release=prometheus). Selector is: %s" % (kind, json.dumps(sel)))
            # namespace selector is the subtle one
            if nssel is None:
                bad("%sNamespaceSelector is ABSENT, so Prometheus only looks in "
                    "its OWN namespace (%s). Our monitors live in %s and are "
                    "therefore ignored -- which is why they applied cleanly and "
                    "collected nothing."
                    % (kind, ns, ", ".join(mon_ns)))
                print("           fix (either one):")
                print("             kubectl -n %s patch prometheus %s --type=merge \\"
                      % (ns, name))
                print("               -p '{\"spec\":{\"%s\":{}}}'" % ns_key)
                print("             ...or move the monitors into %s and let their" % ns)
                print("                own namespaceSelector reach out.")
            elif not (nssel.get("matchLabels") or nssel.get("matchExpressions")):
                ok("%sNamespaceSelector is {} -> all namespaces" % kind)
            else:
                warn("%sNamespaceSelector is restrictive: %s"
                     % (kind, json.dumps(nssel)))


def check_targets_and_metrics():
    head("scrape targets and metric names")
    base = H.prometheus_url()
    if not base:
        bad("Prometheus ClusterIP not resolvable")
        return
    try:
        with urllib.request.urlopen(base + "/api/v1/targets?state=active",
                                    timeout=30) as r:
            act = json.loads(r.read().decode("utf-8"))["data"]["activeTargets"]
    except Exception as exc:
        bad("cannot read targets: %s" % exc)
        return
    jobs = {}
    for t in act:
        j = t.get("labels", {}).get("job", "?")
        if re.search(r"hami|dcgm|vllm|r2-", j, re.I):
            jobs.setdefault(j, [0, 0])
            jobs[j][0] += 1
            jobs[j][1] += (t.get("health") == "up")
    for j, (n, up) in sorted(jobs.items()):
        (ok if up == n else warn)("target %-42s %d/%d up" % (j, up, n))
    if not any("hami" in j.lower() and "dcgm" not in j.lower() for j in jobs):
        bad("no HAMi scheduler/device-plugin target is being scraped: "
            "hami_gpu_core_allocated_ratio and friends will be empty")

    known = H.metric_names(base)
    res = H.resolve_metrics(base, known)
    runnable = [k for k, t in H.QUERIES.items()
                if set(re.findall(r"\{([A-Z]+)\}", t)).issubset(res)]
    skipped = sorted(set(H.QUERIES) - set(runnable))
    ok("%d/%d queries resolvable" % (len(runnable), len(H.QUERIES)))
    for a in sorted(set(H.METRIC_CANDIDATES) - set(res)):
        warn("no metric for alias %s -> %s will be skipped"
             % (a, ", ".join(k for k in skipped
                             if a in re.findall(r"\{([A-Z]+)\}", H.QUERIES[k]))))


# ------------------------------------------------------------------ cluster
def check_models():
    head("models on disk, flat and complete")
    rc, out = H.sh([os.path.join(HERE, "08-flatten-models.sh"), "--check"],
                   timeout=180)
    for line in out.strip().splitlines():
        if re.search(r"NOT BUILT|INCOMPLETE|NO CACHE|FAILED", line):
            bad(line.strip())
    if rc == 0:
        ok("every role has a complete flat directory")
    elif not FAILS:
        warn("08-flatten-models.sh --check exited %d" % rc)


def check_scheduler():
    head("HAMi scheduler policy")
    d = H.kjson(["get", "deploy", "hami-scheduler", "-n", "kube-system"])
    if not d:
        bad("hami-scheduler not found")
        return
    for c in d["spec"]["template"]["spec"]["containers"]:
        flags = [a for a in ((c.get("command") or []) + (c.get("args") or []))
                 if "scheduler-policy" in a]
        if not flags:
            continue
        gpu = [f for f in flags if f.startswith("--gpu-scheduler-policy")]
        if len(gpu) > 1:
            bad("--gpu-scheduler-policy specified %d times: %s" % (len(gpu), gpu))
        elif gpu and gpu[0].endswith("binpack"):
            ok("gpu-scheduler-policy=binpack (co-tenants will share a card)")
        elif gpu:
            bad("%s -- co-tenants will land on SEPARATE cards and the sharing "
                "arms will measure nothing. Run ./10-hami-binpack.sh" % gpu[0])


def check_node():
    head("node capacity")
    rc, out = H.sh(["df", "-Pk", "/var/lib/rancher/k3s/agent/containerd"], timeout=30)
    try:
        gib = int(out.strip().splitlines()[1].split()[3]) / 1048576.0
        margin = gib - 4.84
        (ok if margin > 6 else (warn if margin > 3 else bad))(
            "container filesystem %.1f GiB free, %.1f GiB above the eviction "
            "threshold" % (gib, margin))
    except Exception:
        warn("could not read container filesystem")
    used = H.gpu_used_mib()
    for idx, mib in sorted(used.items()):
        (ok if mib <= 300 else bad)("GPU %d holds %d MiB" % (idx, mib))
    rc, out = H.sh(["bash", "-lc",
                    "ps -eo args= | grep -c '^nvidia-cuda-mps-control' || true"],
                   timeout=30)
    if out.strip().splitlines()[-1].strip() not in ("0", ""):
        ok("MPS control daemon running (needed by the hami-mps arm)")
    else:
        bad("no MPS control daemon: the hami-mps arm cannot work")


# --------------------------------------------------------------- smoke test
def check_smoke(role="llm"):
    head("live smoke test: can %s actually serve a request?" % role)
    print("  This is the check that would have caught 0/885. Readiness only")
    print("  proves /health answers; it says nothing about the request shape.")
    out_dir = os.path.join(HERE, ".validate")
    H.teardown(quiet=True)
    H.wait_gpu_idle(timeout=120)
    subprocess.run([PY, os.path.join(HERE, "render.py"), "--scenario", "llm-embed",
                    "--arm", "dedicated", "--only", role, "--out", out_dir],
                   capture_output=True, text=True)
    H.kubectl(["apply", "-f", out_dir], timeout=120)
    try:
        good, diag = H.wait_ready([role], timeout=900)
        if not good:
            bad("%s never became Ready:\n%s" % (role, list(diag.values())[0][:1200]))
            return
        eps = H.endpoints([role])
        import render as R
        models = {role: R.MODELS[role]["repo"]}
        kinds = {role: R.MODELS[role]["kind"]}
        good, detail = H.smoke_test(eps, models, kinds, timeout=180)
        for r, d in sorted(detail.items()):
            (ok if d.startswith("ok") else bad)("%s: %s" % (r, d))
    finally:
        H.teardown(quiet=True)
        H.wait_gpu_idle(timeout=180)
        H.sh(["rm", "-rf", out_dir])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-deploy", action="store_true",
                    help="skip the live smoke test")
    ap.add_argument("--role", default="llm", help="role to smoke test")
    a = ap.parse_args()

    t0 = time.time()
    check_prometheus()
    check_targets_and_metrics()
    check_models()
    check_scheduler()
    check_node()
    if not a.no_deploy:
        check_smoke(a.role)

    print("\n" + "=" * 66)
    print("%d failure(s), %d warning(s) in %.0fs" % (len(FAILS), len(WARNS),
                                                     time.time() - t0))
    for f in FAILS:
        print("  FAIL  %s" % f.splitlines()[0][:110])
    print("=" * 66)
    if FAILS:
        print("Fix these before running. Each one silently degraded T1's first")
        print("attempt rather than stopping it.")
        return 1
    print("Clear to run:  python run.py --test T1")
    return 0


if __name__ == "__main__":
    sys.exit(main())
