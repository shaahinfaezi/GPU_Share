#!/usr/bin/env bash
# STAGE 2 of 2. One split-sharing arm per invocation.
#
# S3  hami      tp + pp co-resident, HAMi quotas, no MPS
# S4  hami-mps  identical, MPS daemon up, every client at 100%
#
# WHAT IS ACTUALLY ON THE CARDS
# -----------------------------
# Two tenants, each 2-way split, on two physical cards:
#
#       card 0:  tp rank 0  +  pp rank 0
#       card 1:  tp rank 1  +  pp rank 1
#
# Four vLLM ranks, two per card. Every card is simultaneously a SHARING target
# (two tenants on it) and a SPLITTING target (one tenant spanning off it). That
# is the combination under test, and it is the first time either half of this
# project has seen the other.
#
# WHY ONLY THESE TWO ARMS
# -----------------------
# The decision this feeds is "can we run split models on the shared fleet", and
# that has exactly two candidate answers on a HAMi node: HAMi alone, or HAMi
# with MPS. A third arm would not change what anyone does on Monday.
#
# MPS AT 100 FOR EVERY CLIENT, NOT SPLIT BETWEEN THEM
# ---------------------------------------------------
# T14 divided the budget so the clients summed to 100 and came out 6.9% BEHIND
# no-MPS, at a fairness spread of 15.20. On a contended card the choice is
# binary: MPS on at 100, or MPS off. Nothing in splitting changes that, and
# this arm does not re-litigate it.
#
# THE FAULT QUESTION IS NOT ASKED HERE, DELIBERATELY
# --------------------------------------------------
# Round two's most decisive finding is that a GLOBAL-scope CUDA fault is
# contained under HAMi (0 of 516) and PROPAGATES under MPS (307 of 515, 59.6%).
# Splitting should make that strictly worse -- a 2-rank tenant has twice the
# surface and its ranks sit on both cards, so one global fault under MPS could
# plausibly reach every tenant on the box rather than one card's worth. That is
# a prediction, not a measurement, and it is written down here so it is not
# later mistaken for one. Injecting it needs 23-failure-inject.py taught about
# multi-device roles; see the note at the end of the README.
#
#   cd /opt/GPU_Share/round2/split && chmod +x *.sh
#   nohup ./53-split-share.sh hami > logs/share-hami.log 2>&1 &
#   nohup ./53-split-share.sh hami-mps > logs/share-hami-mps.log 2>&1 &
#   tail -f logs/share.log
set -uo pipefail
cd "$(dirname "$0")"

STAMP="$(date -u +%Y%m%d-%H%M%S)"
ARM="${1:-${ARM:-}}"
case "${ARM}" in
  hami|hami-mps) ;;
  *) echo "Usage: $0 {hami|hami-mps}" >&2; exit 2 ;;
esac
SERIES="${SERIES:-S-share-${STAMP}}"
RUN="results/${SERIES}"
LOG="logs/share-${ARM}-${STAMP}.log"
NS="${R2_NS:-llm-demo}"
PY=python3
mkdir -p logs "${RUN}"
exec 9>.split-run.lock
flock -n 9 || { echo "Another split stage is running" >&2; exit 2; }
export PYTHONPATH="$(cd .. && pwd):${PYTHONPATH:-}"

SECS="${SECS:-300}"
DASHBOARD_HOLD_SECS="${DASHBOARD_HOLD_SECS:-300}"

log() { echo "$(date -u +%FT%TZ) $*" | tee -a "${LOG}"; }

if [ ! -f calibration.json ]; then
  log "FATAL: calibration.json missing. Run ./52-split-solo.sh first --"
  log "       without the solo baselines every ratio below is uninterpretable."
  exit 2
fi
RATE=$(${PY} -c "import json;print(json.load(open('calibration.json'))['cotenancy_rate'])")
log "=== calibrated co-tenancy rate: ${RATE} req/s for ${SECS}s"

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

restore_mps() {
  if [ -f "${OUT}/mps-state-touched" ]; then
    ${PY} - >>"${LOG}" 2>&1 <<'EOF'
import harness as H
ok, detail = H.set_mps_daemon(False)
if not ok:
    raise SystemExit('Failed to restore MPS off: %s' % detail)
EOF
  fi
}
trap 'log "INTERRUPTED"; if [ -f "${OUT:-}/apply-started" ]; then teardown; fi; restore_mps; exit 130' INT TERM

  OUT="${RUN}/${ARM}__tp+pp"
  [ ! -e "${OUT}" ] || { log "Refusing to overwrite ${OUT}; use a new SERIES"; exit 2; }
  mkdir -p "${OUT}/rendered"
  log ""
  log "=== arm ${ARM} starting"

  ${PY} 50-render-split.py --arm "${ARM}" --out "${OUT}/rendered" >>"${LOG}" 2>&1
  [ $? -eq 0 ] || exit 2

  # Everything for this arm in one process: the samplers have to outlive the
  # apply and the replay, and the MPS daemon state has to be asserted against
  # the arm AFTER the pods exist, not before.
  ${PY} - "${ARM}" "${OUT}" "${RATE}" "${SECS}" >>"${LOG}" 2>&1 <<'EOF'
import sys, os, json, subprocess, time, shutil
sys.path.insert(0, os.path.dirname(os.getcwd()))
import harness as H
from split_checks import (memory_budget, same_two_devices, snapshot, config_identity,
                          remap_trace, ensure_fd_limit, replay_quality)

arm, out, rate, secs = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
root = os.path.dirname(os.getcwd())
roles = ["tp", "pp"]
ev = H.Events(os.path.join(out, "events.jsonl"))
host = None; gpu = None
try:
    ev.mark("arm_start", arm=arm)
    if H.pods_for('tp') or H.pods_for('pp'):
        raise RuntimeError('Existing split pods remain; inspect and clean up before starting a new arm')
    if not H.wait_gpu_idle(timeout=30, ev=ev):
        raise RuntimeError('GPUs are not idle; inspect the resident workloads first')
    cal = json.load(open('calibration.json', encoding='utf-8'))
    if not cal.get('validated') or cal.get('config') != config_identity():
        raise RuntimeError('Rerun solo using this SPLIT_PROFILE and updated renderer; '
                           'old or unvalidated baselines cannot calibrate this run')
    series = os.path.dirname(out)
    frozen_cal = os.path.join(series, 'calibration.json')
    if os.path.exists(frozen_cal):
        if json.load(open(frozen_cal, encoding='utf-8')) != cal:
            raise RuntimeError('Calibration differs from the first arm in this SERIES')
    else:
        H.write_json(frozen_cal, cal)
    ensure_fd_limit()
    gate = memory_budget(cal, H.host_mem_mib().get('available', 0))
    H.write_json(os.path.join(out, 'admission.json'), gate)
    if not gate['ok']:
        raise RuntimeError('Host admission needs %s MiB available; found %s MiB. '
                           'Separate arms still require BOTH split tenants concurrently.'
                           % (gate['required_mib'], gate['available_mib']))

    # The daemon must be up BEFORE the pods, or the clients start without it
    # and the arm silently measures plain HAMi under an MPS label.
    open(os.path.join(out, 'mps-state-touched'), 'w').close()
    ok, detail = H.set_mps_daemon(arm == "hami-mps", ev=ev)
    if not ok:
        raise RuntimeError("could not set MPS state: %s" % detail)

    host = H.HostSampler(os.path.join(out, "host.csv")); host.start()
    ev.mark("apply")
    open(os.path.join(out, 'apply-started'), 'w').close()
    H.kubectl(["apply", "-f", os.path.join(out, "rendered")], check=True)

    ready, readiness = H.wait_ready(roles, timeout=900, ev=ev)
    with open(os.path.join(out, "readiness.json"), "w", encoding="utf-8") as fh:
        json.dump({"ok": ready, "diagnostics": readiness}, fh, indent=2)
    if not ready:
        raise RuntimeError("readiness timed out; diagnostics saved to readiness.json")
    ev.mark("ready")

    # Assert, do not assume. Round two shipped an arm whose "co-tenants" had
    # been spread onto separate cards; it looked clean and meant nothing.
    ok, detail = H.assert_mps_matches_arm(arm, roles, ev=ev)
    if not ok:
        raise RuntimeError("MPS state does not match arm: %s" % detail)
    place = H.placement(roles)
    json.dump(place, open(os.path.join(out, "placement.json"), "w"), indent=2)
    ev.mark("placement", detail=json.dumps(place))
    if not same_two_devices(place):
        raise RuntimeError('Both tenants must be allocated the same two GPU UUIDs: %s' % place)
    nodes = {p.get('spec', {}).get('nodeName') for r in roles for p in H.pods_for(r)}
    for role in roles:
        pods = H.pods_for(role)
        if len(pods) != 1 or pods[0]['metadata'].get('deletionTimestamp') or not any(
                c.get('type') == 'Ready' and c.get('status') == 'True'
                for c in pods[0].get('status', {}).get('conditions', [])):
            raise RuntimeError('Both roles must be Ready simultaneously: %s' % role)
    if len(nodes) != 1 or None in nodes:
        raise RuntimeError('Split tenants must be on one verified node: %s' % nodes)

    busy = [d for d in H.devices_in_use(floor_mib=500)]
    ev.mark("devices_in_use", n=len(busy), devices=json.dumps(busy))
    if len(busy) < 2:
        # Both tenants are 2-way split; one busy card means the split did not
        # happen and the whole premise of the arm is gone.
        raise RuntimeError("only %d card(s) in use -- expected 2 (split failed)"
                           % len(busy))

    eps = H.endpoints(roles, ev=ev)
    models = {r: (H.served_model(eps[r]) or "Qwen/Qwen2.5-0.5B-Instruct")
              for r in roles}
    smoke_ok, smoke_detail = H.smoke_test(eps, models,
                                           {r: "chat" for r in roles}, ev=ev)
    if not smoke_ok:
        raise RuntimeError("smoke test failed: %s" % smoke_detail)

    mem = H.host_mem_mib()
    rss = H.vllm_rss_mib()
    # This gate is based on the live four-rank process set, not a role-name
    # estimate. A replay under an already OOM-bound host is not a benchmark.
    live_problem = []
    if mem.get("available", 0) < H.HOST_HEADROOM_MIB:
        live_problem.append("only %d MiB host RAM available after both tenants became Ready"
                            % mem.get("available", 0))
    json.dump({"host": mem, "vllm_rss_mib": rss,
               "problems": live_problem},
              open(os.path.join(out, "hostmem.json"), "w"), indent=2)
    if live_problem:
        raise RuntimeError("host-memory admission failed: %s" % live_problem[0])

    trace = os.path.join(out, "trace.json")
    frozen_trace = os.path.join(series, 'trace.json')
    if not os.path.exists(frozen_trace):
        candidate = os.path.join(out, 'trace.candidate.json')
        subprocess.check_call([sys.executable,
            os.path.join(root, "workload", "gen_trace.py"),
            "--scenario", "llm2-llm2", "--rate", rate, "--duration", secs,
            '--mix', 'llm2a=%s,llm2b=%s' % (cal['mix']['tp'], cal['mix']['pp']),
            "--source", "sharegpt", "--out", candidate])
        doc = json.load(open(candidate, encoding='utf-8'))
        if doc['params']['source'] != 'sharegpt':
            raise RuntimeError('ShareGPT unavailable; refusing synthetic fallback')
        remap_trace(doc)
        doc['params']['split_mix'] = cal['mix']
        H.write_json(frozen_trace, doc)
    doc = json.load(open(frozen_trace, encoding='utf-8'))
    original_id = doc['trace_id']
    if (doc['params']['duration'] != float(secs) or doc['params']['rate'] != float(rate)
            or doc['params'].get('split_mix') != cal['mix']
            or remap_trace(doc)['trace_id'] != original_id):
        raise RuntimeError('Stored series trace settings or digest do not match')
    shutil.copyfile(frozen_trace, trace)

    gpu = H.GpuSampler(os.path.join(out, "gpu.csv")); gpu.start()
    t0 = ev.mark("replay_start")
    subprocess.check_call([sys.executable,
        os.path.join(root, "workload", "replay.py"),
        "--trace", trace, "--arm", arm,
        "--endpoint", "tp=%s" % eps["tp"], "--endpoint", "pp=%s" % eps["pp"],
        "--model", "tp=%s" % models["tp"], "--model", "pp=%s" % models["pp"],
        "--out-dir", out, "--max-inflight", "4096",
        "--note", "S-share %s tp+pp" % arm])
    t1 = ev.mark("replay_end")
    quality = replay_quality(out, roles)
    if host.min_available is None or host.min_available < H.HOST_HEADROOM_MIB:
        quality['ok'] = False
        quality['issues'].append('Host RAM fell below headroom during startup/replay')
    H.write_json(os.path.join(out, 'quality.json'), quality)
    if not quality['ok']:
        raise RuntimeError('Replay failed acceptance: %s' % quality['issues'])
    ok, detail = H.assert_mps_matches_arm(arm, roles, ev=ev)
    if not ok:
        raise RuntimeError('MPS state changed during replay: %s' % detail)
    H.write_json(os.path.join(out, 'dashboard-window.json'),
                 {'replay_start_epoch': t0, 'replay_end_epoch': t1,
                  'note': 'Use this historical interval for the screenshot; hold time has no load.'})

    try:
        H.prom_collect(os.path.join(out, "prom.json"), t0, t1, ev=ev)
    except Exception as e:
        ev.mark("prom_failed", error=str(e))   # never fatal; gpu.csv is the copy
    H.write_json(os.path.join(out, 'completed.json'), {'ok': True})
except Exception as e:
    ev.mark("arm_failed", error=str(e))
    try:
        snapshot(H, os.path.join(out, 'diagnostics'))
    except Exception as diagnostic_error:
        ev.mark('diagnostics_failed', error=str(diagnostic_error))
    print("ARM %s FAILED: %s" % (arm, e))
    raise
finally:
    for s in (gpu, host):
        if s:
            try: s.stop()
            except Exception: pass
    ev.mark("arm_end")
    ev.close()
EOF
  RC=$?
  if [ "${RC}" -ne 0 ]; then
    log "=== arm ${ARM} FAILED rc=${RC}; diagnostic artifacts retained in ${OUT}"
    if [ -f "${OUT}/apply-started" ]; then teardown; fi
    restore_mps
    exit "${RC}"
  fi
  log "=== arm ${ARM} finished successfully"
  log "=== dashboard capture window: ${DASHBOARD_HOLD_SECS}s (workloads remain live)"
  log "=== Grafana: kubectl -n default port-forward svc/prometheus-grafana 3000:80"
  log "=== Open the GPU Sharing - Round 2 dashboard and capture the ${ARM} arm now."
  log "=== Select the replay interval in ${OUT}/dashboard-window.json; the hold is idle."
  sleep "${DASHBOARD_HOLD_SECS}"
  teardown

restore_mps || exit 2

log ""
log "=== arm done. Collect both separately-run arms with:"
log "===   python3 54-split-collect.py ${RUN}"
