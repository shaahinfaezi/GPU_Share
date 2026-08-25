#!/usr/bin/env python3
"""Emit the round-2 Grafana dashboard as a ConfigMap the sidecar will pick up.

Every HAMi query below uses a metric name VERIFIED live on the node:

    curl 127.0.0.1:31993/metrics
      hami_gpu_core_allocated_ratio{device_index,device_type,device_uuid,node,zone}
      hami_gpu_core_limit_ratio{...}                                          = 100
      hami_gpu_memory_allocated_bytes{device_cores,device_index,...}

The dashboard is built around one question: for a given arm (dedicated / hami /
hami-mps), what did HAMi *promise*, what did the hardware *do*, and what did the
engine *feel*? Those three are deliberately stacked as separate rows, because
round one's whole finding was that they disagree -- gpucores is a scheduling
token, so the HAMi row can look perfectly partitioned while the DCGM row shows
an even split.
"""
import json
import os

DS = "${DS_PROMETHEUS}"


def target(expr, legend, ref="A"):
    return {"expr": expr, "legendFormat": legend, "refId": ref,
            "datasource": {"type": "prometheus", "uid": DS}}


def panel(pid, title, x, y, w, h, targets, unit="short", desc="", stack=False):
    return {
        "id": pid, "type": "timeseries", "title": title, "description": desc,
        "gridPos": {"x": x, "y": y, "w": w, "h": h},
        "datasource": {"type": "prometheus", "uid": DS},
        "targets": targets,
        "fieldConfig": {
            "defaults": {
                "unit": unit,
                "custom": {
                    "lineWidth": 2, "fillOpacity": 12 if not stack else 40,
                    "showPoints": "never", "spanNulls": True,
                    "stacking": {"mode": "normal" if stack else "none"},
                },
            },
            "overrides": [],
        },
        "options": {
            "legend": {"displayMode": "table", "placement": "bottom",
                       "calcs": ["mean", "max", "lastNotNull"]},
            "tooltip": {"mode": "multi", "sort": "desc"},
        },
    }


def row(pid, title, y):
    return {"id": pid, "type": "row", "title": title, "collapsed": False,
            "gridPos": {"x": 0, "y": y, "w": 24, "h": 1}, "panels": []}


P = []
y = 0

# ---------------------------------------------------------------- row 1: HAMi
P.append(row(100, "HAMi — what the scheduler promised", y)); y += 1
P.append(panel(
    1, "gpucores allocated per GPU", 0, y, 12, 8,
    [target('hami_gpu_core_allocated_ratio{node="$node"}',
            "GPU {{device_index}}  {{device_uuid}}"),
     target('hami_gpu_core_limit_ratio{node="$node"}',
            "limit GPU {{device_index}}", "B")],
    unit="percent",
    desc=("Sum of nvidia.com/gpucores requested by pods placed on each device, "
          "against the 100 ceiling. Round one established this is a PLACEMENT "
          "token, not a compute bound -- expect it to look neatly partitioned "
          "while the DCGM row below shows an even split.")))
P.append(panel(
    2, "gpumem allocated per GPU", 12, y, 12, 8,
    [target('hami_gpu_memory_allocated_bytes{node="$node"}',
            "GPU {{device_index}}"),
     target('12884901888', "physical 12 GiB", "B")],
    unit="bytes",
    desc=("nvidia.com/gpumem granted per device. Unlike gpucores this one is "
          "genuinely enforced -- round one measured peak VRAM inside quota "
          "under MPS.")))
y += 8

# ------------------------------------------------------------ row 2: hardware
P.append(row(200, "Hardware — what the GPU actually did", y)); y += 1
P.append(panel(
    3, "GPU utilization", 0, y, 8, 8,
    [target('DCGM_FI_DEV_GPU_UTIL{Hostname=~"$node.*"}', "GPU {{gpu}}")],
    unit="percent",
    desc=("Whether any kernel was resident. Round one found this pinned at "
          "95-99% in nearly every configuration, so it discriminates almost "
          "nothing -- kept because its flatness is itself the evidence.")))
P.append(panel(
    4, "Framebuffer used", 8, y, 8, 8,
    [target('DCGM_FI_DEV_FB_USED{Hostname=~"$node.*"} * 1024 * 1024',
            "GPU {{gpu}} used")],
    unit="bytes",
    desc="Real device memory in use, from DCGM rather than from inside a "
         "HAMi-intercepted container."))
P.append(panel(
    5, "SM clock / power", 16, y, 8, 8,
    [target('DCGM_FI_DEV_SM_CLOCK{Hostname=~"$node.*"}', "GPU {{gpu}} SM MHz"),
     target('DCGM_FI_DEV_POWER_USAGE{Hostname=~"$node.*"}', "GPU {{gpu}} W", "B")],
    desc=("Throttling control. Round one had to prove the force-policy decay "
          "was NOT thermal; holding clocks steady here is what licenses that "
          "claim.")))
y += 8

# ------------------------------------------------------------- row 3: engines
P.append(row(300, "vLLM — what each engine felt", y)); y += 1
P.append(panel(
    6, "Time to first token", 0, y, 12, 8,
    [target('histogram_quantile(0.99, sum by (le, r2_role) '
            '(rate(vllm:time_to_first_token_seconds_bucket{namespace="llm-demo"}[1m])))',
            "{{r2_role}} p99"),
     target('histogram_quantile(0.95, sum by (le, r2_role) '
            '(rate(vllm:time_to_first_token_seconds_bucket{namespace="llm-demo"}[1m])))',
            "{{r2_role}} p95", "B"),
     target('histogram_quantile(0.50, sum by (le, r2_role) '
            '(rate(vllm:time_to_first_token_seconds_bucket{namespace="llm-demo"}[1m])))',
            "{{r2_role}} p50", "C")],
    unit="s",
    desc=("TTFT is damaged by PREFILL contention. This is the panel that should "
          "move when a compute-bound embedding neighbour starts a batch.")))
P.append(panel(
    7, "Inter-token latency", 12, y, 12, 8,
    [target('histogram_quantile(0.99, sum by (le, r2_role) '
            '(rate(vllm:time_per_output_token_seconds_bucket{namespace="llm-demo"}[1m])))',
            "{{r2_role}} p99"),
     target('histogram_quantile(0.95, sum by (le, r2_role) '
            '(rate(vllm:time_per_output_token_seconds_bucket{namespace="llm-demo"}[1m])))',
            "{{r2_role}} p95", "B")],
    unit="s",
    desc=("ITL is damaged by DECODE contention, and is what a user perceives as "
          "stutter. Splitting it from TTFT is the instrumentation gap round one "
          "could not close, because it was not streaming.")))
y += 8
P.append(panel(
    8, "Throughput — generation tokens/s", 0, y, 8, 8,
    [target('sum by (r2_role) '
            '(rate(vllm:generation_tokens_total{namespace="llm-demo"}[1m]))',
            "{{r2_role}}")],
    unit="none", stack=True,
    desc="Stacked, so the aggregate across tenants is readable directly -- that "
         "aggregate is the number the three arms are compared on."))
P.append(panel(
    9, "Queue depth", 8, y, 8, 8,
    [target('sum by (r2_role) (vllm:num_requests_running{namespace="llm-demo"})',
            "{{r2_role}} running"),
     target('sum by (r2_role) (vllm:num_requests_waiting{namespace="llm-demo"})',
            "{{r2_role}} waiting", "B")],
    desc=("Separates 'the GPU is slow' from 'the scheduler is holding requests'. "
          "Only observable because the client drives an OPEN loop -- a fixed "
          "concurrency loop can never build a queue.")))
P.append(panel(
    10, "KV cache usage", 16, y, 8, 8,
    [target('vllm:gpu_cache_usage_perc{namespace="llm-demo"} * 100',
            "{{r2_role}}")],
    unit="percent",
    desc=("The gpumem quota IS the KV budget. This panel is where the indirect "
          "concurrency control becomes visible.")))
y += 8
P.append(panel(
    11, "Preemptions", 0, y, 12, 6,
    [target('sum by (r2_role) '
            '(rate(vllm:num_preemptions_total{namespace="llm-demo"}[1m]))',
            "{{r2_role}}")],
    desc=("A rising rate means the KV budget is too tight -- a quota problem, "
          "NOT a sharing problem. In throughput alone the two look identical.")))
P.append(panel(
    12, "Prefix cache hit rate", 12, y, 12, 6,
    [target('vllm:gpu_prefix_cache_hit_rate{namespace="llm-demo"} * 100',
            "{{r2_role}}")],
    unit="percent",
    desc=("Round one deliberately suppressed prefix reuse as a validity control. "
          "Round two allows it, so this must be reported alongside throughput or "
          "the comparison is not like-for-like.")))

dash = {
    "uid": "r2-gpu-sharing",
    "title": "GPU Sharing — Round 2 (HAMi / MPS / dedicated)",
    "tags": ["gpu", "hami", "mps", "vllm", "round2"],
    "timezone": "browser",
    "schemaVersion": 39,
    "version": 1,
    "refresh": "5s",
    "time": {"from": "now-30m", "to": "now"},
    "templating": {"list": [
        {"name": "DS_PROMETHEUS", "type": "datasource", "query": "prometheus",
         "current": {}, "hide": 0, "label": "Datasource"},
        {"name": "node", "type": "query", "datasource": {"type": "prometheus", "uid": DS},
         "query": "label_values(hami_gpu_core_limit_ratio, node)",
         "current": {"text": "mr-gholami", "value": "mr-gholami"},
         "refresh": 1, "label": "Node"},
    ]},
    "panels": P,
}

body = json.dumps(dash, indent=2)
cm = """# Round-2 Grafana dashboard.
#
# Delivered as a ConfigMap because your Grafana came from kube-prometheus-stack,
# whose sidecar imports any ConfigMap labelled grafana_dashboard="1". Nothing is
# installed and no Grafana settings change.
#
#   kubectl apply -f 61-dashboard.yaml
#   # appears within ~30s under Dashboards -> "GPU Sharing - Round 2"
#
# Generated by make_dashboard.py -- edit that, not this file.
apiVersion: v1
kind: ConfigMap
metadata:
  name: r2-gpu-sharing-dashboard
  namespace: default
  labels:
    grafana_dashboard: "1"
data:
  r2-gpu-sharing.json: |
""" + "\n".join("    " + ln for ln in body.split("\n")) + "\n"

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "61-dashboard.yaml")
with open(out, "w", encoding="utf-8") as fh:
    fh.write(cm)

rows = [p for p in P if p["type"] == "row"]
pans = [p for p in P if p["type"] != "row"]
print("wrote %s" % out)
print("rows   : %s" % ", ".join(r["title"] for r in rows))
print("panels : %d" % len(pans))
for p in pans:
    print("   %-32s %d target(s)" % (p["title"], len(p["targets"])))
