#!/usr/bin/env bash
# STAGE 2 of 2. Both split tenants on the same two cards, HAMi then HAMi+MPS.
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
#   nohup ./53-split-share.sh > logs/share.log 2>&1 &
#   tail -f logs/share.log
set -uo pipefail
cd "$(dirname "$0")"

STAMP="$(date -u +%Y%m%d-%H%M%S)"
RUN="results/S-share-${STAMP}"
LOG="logs/share-${STAMP}.log"
NS="${R2_NS:-llm-demo}"
PY=python3
mkdir -p logs "${RUN}"
export PYTHONPATH="$(cd .. && pwd):${PYTHONPATH:-}"

SECS="${SECS:-240}"

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

trap 'log "INTERRUPTED"; teardown; exit 130' INT TERM

for ARM in hami hami-mps; do
  OUT="${RUN}/${ARM}__tp+pp"
  mkdir -p "${OUT}/rendered"
  log ""
  log "=== arm ${ARM} starting"

  ${PY} 50-render-split.py --arm "${ARM}" --out "${OUT}/rendered" >>"${LOG}" 2>&1

  # Everything for this arm in one process: the samplers have to outlive the
  # apply and the replay, and the MPS daemon state has to be asserted against
  # the arm AFTER the pods exist, not before.
  ${PY} - "${ARM}" "${OUT}" "${RATE}" "${SECS}" >>"${LOG}" 2>&1 <<'EOF'
import sys, os, json, subprocess, time
sys.path.insert(0, os.path.dirname(os.getcwd()))
import harness as H

arm, out, rate, secs = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
root = os.path.dirname(os.getcwd())
roles = ["tp", "pp"]
ev = H.Events(os.path.join(out, "events.jsonl"))
host = None; gpu = None
try:
    ev.mark("arm_start", arm=arm)

    # The daemon must be up BEFORE the pods, or the clients start without it
    # and the arm silently measures plain HAMi under an MPS label.
    H.set_mps_daemon(arm == "hami-mps", ev=ev)

    host = H.HostSampler(os.path.join(out, "host.csv")); host.start()
    ev.mark("apply")
    H.kubectl(["apply", "-f", os.path.join(out, "rendered")], check=True)

    H.wait_ready(roles, timeout=900, ev=ev)
    ev.mark("ready")

    # Assert, do not assume. Round two shipped an arm whose "co-tenants" had
    # been spread onto separate cards; it looked clean and meant nothing.
    H.assert_mps_matches_arm(arm, roles, ev=ev)
    place = H.placement(roles)
    json.dump(place, open(os.path.join(out, "placement.json"), "w"), indent=2)
    ev.mark("placement", detail=json.dumps(place))

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
    H.smoke_test(eps, models, {r: "chat" for r in roles}, ev=ev)

    prob = H.host_memory_problems(roles, ev=ev)
    json.dump({"host": H.host_mem_mib(), "vllm_rss_mib": H.vllm_rss_mib(),
               "problems": prob},
              open(os.path.join(out, "hostmem.json"), "w"), indent=2)

    trace = os.path.join(out, "trace.json")
    subprocess.check_call([sys.executable,
        os.path.join(root, "workload", "gen_trace.py"),
        "--scenario", "llm2-llm2", "--rate", rate, "--duration", secs,
        "--source", "sharegpt", "--out", trace])
    doc = json.load(open(trace, encoding="utf-8"))
    ren = {"llm2a": "tp", "llm2b": "pp"}
    for r in doc["requests"]:
        r["service"] = ren.get(r["service"], r["service"])
    json.dump(doc, open(trace, "w", encoding="utf-8"))

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

    try:
        H.prom_collect(os.path.join(out, "prom.json"), t0, t1, ev=ev)
    except Exception as e:
        ev.mark("prom_failed", error=str(e))   # never fatal; gpu.csv is the copy
except Exception as e:
    ev.mark("arm_failed", error=str(e))
    print("ARM %s FAILED: %s" % (arm, e))
finally:
    for s in (gpu, host):
        if s:
            try: s.stop()
            except Exception: pass
    ev.mark("arm_end")
    ev.close()
EOF
  log "=== arm ${ARM} finished rc=$?"
  teardown
done

${PY} - >>"${LOG}" 2>&1 <<'EOF' || true
import sys, os
sys.path.insert(0, os.path.dirname(os.getcwd()))
import harness as H
H.set_mps_daemon(False)
EOF

log ""
log "=== both arms done. Collect with:"
log "===   python3 54-split-collect.py ${RUN}"
