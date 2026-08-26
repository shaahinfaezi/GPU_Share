#!/usr/bin/env python3
"""Build the HAMi ServiceMonitors from what the Services actually declare.

WHY
13-validate.py cleared every hypothesis except one:

    ServiceMonitor  selector: matchLabels satisfied
    ServiceMonitorNamespaceSelector is {} -> all namespaces
    [FAIL] no HAMi scheduler/device-plugin target is being scraped

The selectors are right and Prometheus is watching every namespace, so the
monitors ARE being evaluated -- and still produce no target. That leaves the
endpoint port.

60-servicemonitors.yaml says `targetPort: 31993`. 31993 is the NODEPORT.
targetPort in a ServiceMonitor refers to the POD's container port, and the
scheduler extender binds its metrics elsewhere -- its own argv says

    --metrics-bind-address=:9395

A targetPort matching no container port produces no target and no complaint,
which is the third time in this project a silently-empty selector has cost a
run. So this stops guessing: it reads each Service, probes every port for a
/metrics endpoint that actually returns HAMi series, and writes the monitor
using the port NAME the Service itself declares.

    python 14-fix-servicemonitors.py            # probe and report
    python 14-fix-servicemonitors.py --write    # ...and rewrite the manifest
    python 14-fix-servicemonitors.py --apply    # ...and kubectl apply it
"""
import argparse
import json
import os
import sys
import urllib.request

import harness as H

HERE = os.path.dirname(os.path.abspath(__file__))
MANIFEST = os.path.join(HERE, "monitoring", "60-servicemonitors.yaml")

# (monitor name, namespace, service name, what we expect to find there)
WANTED = [
    ("r2-hami-scheduler", "kube-system", "hami-scheduler", "hami_"),
    ("r2-hami-device-plugin", "kube-system", "hami-device-plugin-monitor", "hami_"),
]


def probe(url, needle, timeout=8):
    """Does this URL serve metrics containing `needle`? Returns (n_lines, ok)."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            body = r.read(200000).decode("utf-8", "replace")
    except Exception:
        return 0, False
    lines = [l for l in body.splitlines() if l and not l.startswith("#")]
    return len(lines), any(l.startswith(needle) for l in lines)


def inspect_service(ns, name, needle):
    """Every port on a Service, with whether it actually serves the metrics."""
    d = H.kjson(["get", "svc", name, "-n", ns])
    if not d:
        return None, []
    cip = d["spec"].get("clusterIP")
    rows = []
    for p in d["spec"].get("ports", []):
        port = p.get("port")
        url = "http://%s:%s/metrics" % (cip, port)
        n, hit = probe(url, needle)
        rows.append(dict(name=p.get("name"), port=port,
                         targetPort=p.get("targetPort"),
                         nodePort=p.get("nodePort"),
                         lines=n, serves=hit))
    return d, rows


def monitor_yaml(mon, ns, svc_labels, port_name, port_num):
    """A ServiceMonitor keyed on the port NAME when the Service gives one.

    Prometheus resolves `port` against the Service's port name, which is stable
    across chart versions in a way a raw number is not, and -- unlike
    targetPort -- does not require knowing what the pod called its container
    port.
    """
    sel = "\n".join("      %s: %s" % (k, v) for k, v in sorted(svc_labels.items()))
    if port_name:
        endpoint = "    - port: %s" % port_name
    else:
        endpoint = "    - targetPort: %s" % port_num
    return """---
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: %s
  namespace: %s
  labels:
    release: prometheus
spec:
  selector:
    matchLabels:
%s
  namespaceSelector:
    matchNames: [%s]
  endpoints:
%s
      path: /metrics
      interval: 5s
      scrapeTimeout: 4s
      honorLabels: true
""" % (mon, ns, sel, ns, endpoint)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true", help="rewrite the manifest")
    ap.add_argument("--apply", action="store_true", help="write, then kubectl apply")
    a = ap.parse_args()

    blocks, problems = [], []
    for mon, ns, svc, needle in WANTED:
        print("\n=== %s/%s" % (ns, svc))
        d, rows = inspect_service(ns, svc, needle)
        if d is None:
            print("   service not found")
            problems.append("%s missing" % svc)
            continue
        labels = {k: v for k, v in (d["metadata"].get("labels") or {}).items()
                  if k in ("app.kubernetes.io/name", "app.kubernetes.io/component")}
        print("   selector labels: %s" % json.dumps(labels))
        best = None
        for r in rows:
            mark = ""
            if r["serves"]:
                mark = "   <== serves %s metrics" % needle
                if best is None:
                    best = r
            elif r["lines"]:
                mark = "   (answers /metrics, but no %s series)" % needle
            print("   port name=%-14s port=%-6s targetPort=%-8s nodePort=%-7s "
                  "lines=%-4s%s" % (r["name"], r["port"], r["targetPort"],
                                    r["nodePort"], r["lines"], mark))
        if best is None:
            # Fall back to any port that answers at all, so a monitor still
            # gets written rather than silently omitted.
            answering = [r for r in rows if r["lines"]]
            if answering:
                best = answering[0]
                print("   no port served %s series; using the one that answers"
                      % needle)
            else:
                print("   NO port on this Service answers /metrics from the "
                      "cluster network")
                problems.append("%s: no usable port" % svc)
                continue
        print("   -> monitor will use %s"
              % ("port: %s" % best["name"] if best["name"]
                 else "targetPort: %s" % best["port"]))
        blocks.append(monitor_yaml(mon, ns, labels, best["name"], best["port"]))

    if not blocks:
        print("\nNothing to write.")
        return 1

    if not (a.write or a.apply):
        print("\n--- would write these (re-run with --write or --apply):")
        for b in blocks:
            print(b)
        return 0

    # Keep the vLLM PodMonitors and the DCGM monitor; replace only the two
    # HAMi ServiceMonitors, which are the ones that were never matching.
    existing = open(MANIFEST, encoding="utf-8").read()
    keep = []
    for doc in existing.split("\n---\n"):
        if "name: r2-hami-scheduler" in doc or "name: r2-hami-device-plugin" in doc:
            continue
        if doc.strip():
            keep.append(doc if doc.startswith("---") else "---\n" + doc)
    out = ("\n".join(keep).rstrip() + "\n"
           + "".join(blocks))
    open(MANIFEST, "w", encoding="utf-8").write(out)
    print("\nwrote %s" % MANIFEST)

    if a.apply:
        rc, o = H.kubectl(["apply", "-f", MANIFEST], timeout=120)
        print(o.strip())
        if rc:
            return 1
        print("\nGive Prometheus ~60s, then:  python 13-validate.py --no-deploy")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
