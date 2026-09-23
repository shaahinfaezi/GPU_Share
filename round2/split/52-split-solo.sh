#!/usr/bin/env bash
# STAGE 1 of 2. The two split tenants alone, one at a time.
#
# S1  tp  dedicated  --tensor-parallel-size 2,   alone on both cards
# S2  pp  dedicated  --pipeline-parallel-size 2, alone on both cards
#
# WHY THIS STAGE EXISTS AND IS NOT OPTIONAL
# -----------------------------------------
# Every number the sharing arms produce is a RATIO against these two. The
# saturation ratio -- combined co-tenancy tok/s divided by the sum of the solo
# tok/s -- is the single variable that decides whether MPS helps at all. Round
# two measured the curve five times:
#
#       99.6% saturated  ->  MPS +0.06%
#       86.3%            ->  MPS +8.6%
#       80.9%            ->  MPS +19.0%
#       55.9%            ->  MPS +20.8%
#
# Run the sharing arms at a rate that leaves the card unsaturated and MPS
# correctly shows nothing, and the report says "MPS does not help with split
# models" when what it measured was the rate. That is the trap this stage
# exists to close: it measures each tenant's actual capacity, then writes the
# co-tenancy rate that lands the pair in the informative band.
#
# It also measures per-rank host RSS, which is the other thing nobody knows:
# four ranks on a 15,988 MiB box with no swap is the densest this node has
# ever been asked to be.
#
#   cd /opt/GPU_Share/round2/split && chmod +x *.sh
#   nohup ./52-split-solo.sh > logs/solo.log 2>&1 &
#   tail -f logs/solo.log
set -uo pipefail
cd "$(dirname "$0")"
exec 9>.split-run.lock
flock -n 9 || { echo "Another split stage is running" >&2; exit 2; }

STAMP="$(date -u +%Y%m%d-%H%M%S)"
RUN="results/S-solo-${STAMP}"
LOG="logs/solo-${STAMP}.log"
NS="${R2_NS:-llm-demo}"
PY=python3
mkdir -p logs "${RUN}"
export PYTHONPATH="$(cd .. && pwd):${PYTHONPATH:-}"

# Deliberately ABOVE what either tenant can serve. The achieved completion
# rate under overload IS the capacity, and capacity is what stage 2 needs.
# Measuring it at a polite rate would only measure the rate.
# Per-role arrival rates, not a 50/50 split of one aggregate rate.
PROBE_RATE_TP="${PROBE_RATE_TP:-8}"
PROBE_RATE_PP="${PROBE_RATE_PP:-2}"
PROBE_SECS="${PROBE_SECS:-180}"

log() { echo "$(date -u +%FT%TZ) $*" | tee -a "${LOG}"; }

teardown() {
  kubectl delete llminferenceservice -n "${NS}" -l r2-series=split \
      --ignore-not-found --timeout=180s >>"${LOG}" 2>&1
  ${PY} - >>"${LOG}" 2>&1 <<'EOF' || true
import sys, os
sys.path.insert(0, os.path.dirname(os.getcwd()))
import harness as H
H.wait_gpu_idle(timeout=240)
EOF
  sleep 10
}

# A stage that fails must not kill the script -- the other tenant's baseline is
# still worth having, and a half-populated calibration is better than none.
CURRENT_APPLIED=0
trap 'log "INTERRUPTED"; if [ "${CURRENT_APPLIED}" = 1 ]; then teardown; fi; exit 130' INT TERM

log "=== split solo baselines, run ${RUN}"
log "=== MPS must be OFF for a dedicated arm"
${PY} - >>"${LOG}" 2>&1 <<'EOF'
import sys, os
sys.path.insert(0, os.path.dirname(os.getcwd()))
import harness as H
if H.pods_for('tp') or H.pods_for('pp'):
    raise SystemExit('Existing split pods found; inspect before starting solo')
if not H.wait_gpu_idle(timeout=30):
    raise SystemExit('GPUs are not idle')
ok, detail = H.set_mps_daemon(False)
if not ok:
    raise SystemExit('Cannot disable MPS: %s' % detail)
EOF
[ $? -eq 0 ] || exit 2

for ROLE in tp pp; do
  OUT="${RUN}/dedicated__${ROLE}"
  mkdir -p "${OUT}/rendered"
  log ""
  log "=== ${ROLE} dedicated: render + apply"
  if [ "${ROLE}" = tp ]; then PROBE_RATE="${PROBE_RATE_TP}"; else PROBE_RATE="${PROBE_RATE_PP}"; fi

  ${PY} - "${OUT}" >>"${LOG}" 2>&1 <<'EOF'
import sys, os
sys.path.insert(0, os.path.dirname(os.getcwd()))
import harness as H
if not H.wait_gpu_idle(timeout=240):
    raise SystemExit('Previous workload did not release GPUs')
H.write_json(os.path.join(sys.argv[1], 'idle-hostmem.json'), H.host_mem_mib())
EOF
  [ $? -eq 0 ] || exit 2

  ${PY} 50-render-split.py --arm dedicated --only "${ROLE}" \
        --out "${OUT}/rendered" >>"${LOG}" 2>&1
  [ $? -eq 0 ] || exit 2
  CURRENT_APPLIED=1
  kubectl apply -f "${OUT}/rendered/" >>"${LOG}" 2>&1
  if [ $? -ne 0 ]; then teardown; exit 2; fi

  # One python block does readiness, endpoint resolution, the served-model
  # name and the trace, because each step needs the previous one's answer and
  # a shell round-trip per step is where these scripts historically broke.
  ${PY} - "${ROLE}" "${OUT}" "${PROBE_RATE}" "${PROBE_SECS}" >>"${LOG}" 2>&1 <<'EOF'
import sys, os, json, subprocess
sys.path.insert(0, os.path.dirname(os.getcwd()))
import harness as H
from split_checks import ensure_fd_limit, remap_trace, replay_quality, snapshot

role, out, rate, secs = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
here = os.getcwd()
root = os.path.dirname(here)
ensure_fd_limit()
host = H.HostSampler(os.path.join(out, 'host.csv')); host.start()

ready, readiness = H.wait_ready([role], timeout=900)
with open(os.path.join(out, "readiness.json"), "w", encoding="utf-8") as fh:
    json.dump({"ok": ready, "diagnostics": readiness}, fh, indent=2)
if not ready:
    snapshot(H, os.path.join(out, 'diagnostics'))
    raise RuntimeError("%s readiness timed out; diagnostics saved to readiness.json"
                       % role)
ok, detail = H.assert_mps_matches_arm('dedicated', [role])
if not ok:
    raise RuntimeError('Dedicated MPS state mismatch: %s' % detail)
eps = H.endpoints([role])
model = H.served_model(eps[role]) or "Qwen/Qwen2.5-0.5B-Instruct"
print("endpoint %s -> %s (model %s)" % (role, eps[role], model))

# Host RSS with exactly this tenant resident. The number stage 2 needs.
mem = H.host_mem_mib()
rss = H.vllm_rss_mib()
json.dump({"available_mib": mem.get("available"), "vllm_rss_mib": rss},
          open(os.path.join(out, "hostmem.json"), "w"), indent=2)
print("host: %s MiB available, vLLM RSS %s MiB" % (mem.get("available"), rss))

# gen_trace has no tp/pp scenario, and adding one would edit a file eight
# finished tiers depend on. llm2-llm2 is the symmetric two-chat-tenant
# scenario; its two service names are remapped here instead.
trace = os.path.join(out, "trace.json")
subprocess.check_call([sys.executable,
    os.path.join(root, "workload", "gen_trace.py"),
    "--scenario", "llm2-llm2", "--rate", rate, "--duration", secs,
    "--mix", 'llm2a=1,llm2b=0' if role == 'tp' else 'llm2a=0,llm2b=1',
    "--source", "sharegpt", "--out", trace])
doc = json.load(open(trace, encoding="utf-8"))
if doc['params']['source'] != 'sharegpt':
    raise RuntimeError('ShareGPT corpus unavailable; refusing synthetic fallback')
remap_trace(doc)
json.dump(doc, open(trace, "w", encoding="utf-8"))

subprocess.check_call([sys.executable,
    os.path.join(root, "workload", "replay.py"),
    "--trace", trace, "--arm", "dedicated",
    "--endpoint", "%s=%s" % (role, eps[role]),
    "--model", "%s=%s" % (role, model),
    "--only", role, "--out-dir", out,
    "--max-inflight", "4096",
    "--note", "S-solo dedicated %s" % role])
host.stop()
quality = replay_quality(out, [role])
H.write_json(os.path.join(out, 'quality.json'), quality)
if not quality['ok']:
    raise RuntimeError('Replay cannot calibrate: %s' % quality['issues'])
H.write_json(os.path.join(out, 'completed.json'), {'ok': True})
EOF
  RC=$?
  log "=== ${ROLE} dedicated finished rc=${RC}"
  if [ "${RC}" -ne 0 ]; then
    ${PY} - "${OUT}" >>"${LOG}" 2>&1 <<'EOF'
import sys, os
sys.path.insert(0, os.path.dirname(os.getcwd()))
import harness as H
from split_checks import snapshot
snapshot(H, os.path.join(sys.argv[1], 'diagnostics'))
EOF
    teardown
    exit "${RC}"
  fi
  teardown
  CURRENT_APPLIED=0
done

# ------------------------------------------------------- write the calibration
log ""
log "=== calibration"
${PY} - "${RUN}" >>"${LOG}" 2>&1 <<'EOF'
import sys, os, json, csv
from split_checks import config_identity, replay_quality

run = sys.argv[1]
cal = {"solo": {}, "config": config_identity(), "source_run": run,
       "validated": True, "mix_policy": "capacity-weighted"}
for role in ("tp", "pp"):
    directory = os.path.join(run, 'dedicated__%s' % role)
    if not os.path.exists(os.path.join(directory, 'completed.json')):
        raise SystemExit('Solo role did not complete successfully: %s' % role)
    quality = replay_quality(directory, [role])
    if not quality['ok']:
        raise SystemExit('Invalid replay: %s' % quality['issues'])
    p = os.path.join(run, "dedicated__%s" % role, "replay.summary.json")
    if not os.path.exists(p):
        print("MISSING %s -- stage 2 cannot be calibrated from a partial run" % p)
        continue
    s = json.load(open(p, encoding="utf-8"))["per_service"][role]
    mem_path = os.path.join(run, "dedicated__%s" % role, "hostmem.json")
    mem = json.load(open(mem_path, encoding="utf-8")) if os.path.exists(mem_path) else {}
    idle = json.load(open(os.path.join(directory, 'idle-hostmem.json')))['available']
    with open(os.path.join(directory, 'host.csv'), newline='') as fh:
        minimum = min(float(row['mem_available']) for row in csv.DictReader(fh))
    if minimum < 1024 or idle <= minimum:
        raise SystemExit('Host-memory measurement lacks headroom or a positive delta: %s' % role)
    cal["solo"][role] = {
        "tokens_per_second":   s["tokens_per_second"],
        "requests_per_second": s["requests_per_second"],
        "latency_mean":        s["latency_e2e"]["mean"],
        "ttft_mean":           s["ttft"]["mean"],
        "vllm_rss_mib":        mem.get("vllm_rss_mib", [None, None])[0],
        "host_available_mib":  mem.get("available_mib"),
        "idle_available_mib": idle,
        "minimum_available_mib": minimum,
        "peak_host_delta_mib": max(0, idle - minimum),
        "quality": quality,
    }
    print("%s solo: %.1f tok/s, %.2f req/s, mean latency %.2f s"
          % (role, s["tokens_per_second"], s["requests_per_second"],
             s["latency_e2e"]["mean"]))

if len(cal["solo"]) == 2:
    cap = sum(v["requests_per_second"] for v in cal["solo"].values())
    # Offer ~95% of the pair's combined measured capacity. T17 offered a
    # comparable load and landed at 80.9% achieved saturation, the point where
    # the MPS effect is largest while both arms still finish the trace.
    cal["cotenancy_rate"] = round(0.95 * cap, 1)
    cal['mix'] = {r: cal['solo'][r]['requests_per_second'] / cap for r in ('tp', 'pp')}
    cal["solo_tokens_total"] = sum(v["tokens_per_second"] for v in cal["solo"].values())
    print("combined measured capacity %.2f req/s -> stage 2 rate %.1f req/s"
          % (cap, cal["cotenancy_rate"]))
else:
    raise SystemExit("INCOMPLETE: both solo baselines are required; refusing to invent a co-tenancy rate")

json.dump(cal, open(os.path.join(run, "calibration.json"), "w"), indent=2)
# Stage 2 reads this path; a stable name saves passing the stamp around.
json.dump(cal, open("calibration.json", "w"), indent=2)
EOF
CAL_RC=$?
if [ "${CAL_RC}" -ne 0 ]; then
  log "=== calibration failed rc=${CAL_RC}; do not start a sharing arm"
  exit "${CAL_RC}"
fi

log ""
log "=== solo stage done. Next:"
log "===   ./53-split-share.sh hami"
