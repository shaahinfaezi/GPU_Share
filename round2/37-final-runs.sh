#!/usr/bin/env bash
# THE LAST RUNS. Three of them, and after this nothing on this node is left.
#
# Everything else in the matrix is now in one of four states: measured,
# answered from source, impossible on this hardware, or not decision-changing.
# What remains is exactly this:
#
#   1+2. vram-hog ON A CARD WITH HEADROOM, both arms.
#        Section 15.4 proved the escape route is OPEN -- a pod requesting no
#        nvidia.com/* resource at all obtained a ~250 MiB CUDA context on the
#        tenants' card and attached itself to the MPS daemon as a third client.
#        What it could NOT do is say how much such a pod can take, because all
#        five attempts ran against two tenants holding ~11.5 GiB of a 12,288 MiB
#        card. The hog was stopped by physics, not by HAMi. Dropping to ONE
#        tenant leaves roughly 8 GiB to steal, which is the only configuration
#        in which the magnitude is measurable. This is the study's last open
#        SECURITY question and the fault story is now its headline finding.
#
#   3.   T17, llm2-llm2 at 28 req/s -- a saturated point that is not overload.
#        The throughput claim rests on T20 (+0.06% at 99.6%, no queue), T14
#        (+8.6% at 86.3%) and T21 (+20.8% at 55.9%, but 30-39 s mean latency
#        and neither arm finished the trace). The last of those is unsellable:
#        no operator runs in a regime where no SLO is meetable. 28 is the
#        geometric mean of T15's 20 and T21's 40 and brackets the knee.
#
# NOT in this series, and each for a stated reason:
#   - three tenants          impossible: 8,448 MiB measured RSS against 8,152
#                            available, with zero headroom. Needs >= 24 GiB.
#   - default policy latch   answered from source: rate_limiter() returns early
#                            on cached_util_switch == 0 before its own writer.
#   - gpucores sum > 100     answered from source: Fit() plus a 0-100 validation.
#   - deviceMemoryScaling    needs a node that can be rebuilt.
#   - MPS SM curve at load   the overbook curve already settled it: on a
#                            saturated card the choice is binary, 100 or off.
#   - depth-law crossover    a multi-run sweep, and the qualitative law is
#                            confirmed nine times. It changes no recommendation.
#
# Runs detached, one stage at a time, forced teardown between stages.
#
#   cd /opt/GPU_Share/round2 && git pull && chmod +x *.sh
#   nohup ./37-final-runs.sh > ~/final-runs.log 2>&1 &
set -uo pipefail

cd "$(dirname "$0")"
STAMP="$(date -u +%Y%m%d-%H%M%S)"
LOG="logs/final-runs-${STAMP}.log"
mkdir -p logs

PY=python3

stage() {
  local name="$1"; shift
  echo | tee -a "${LOG}"
  echo "=== ${name} starting $(date -u +%FT%TZ)" | tee -a "${LOG}"
  "$@" >>"${LOG}" 2>&1
  local rc=$?
  echo "=== ${name} finished rc=${rc} $(date -u +%FT%TZ)" | tee -a "${LOG}"
  # A fault stage that fails must not take the tier with it.
  return 0
}

teardown() {
  ${PY} - <<'EOF' >>"${LOG}" 2>&1 || true
import sys, os
sys.path.insert(0, os.getcwd())
import harness as H
H.teardown()
EOF
  sleep 10
}

{
  echo "=== final runs ${STAMP}"
  echo "=== 1) vram-hog hami     (one tenant, card has headroom)"
  echo "=== 2) vram-hog hami-mps (one tenant, card has headroom)"
  echo "=== 3) T17 llm2-llm2 @28 (saturated, not overloaded)"
  echo "=== estimated 55 min"
} 2>&1 | tee -a "${LOG}"

stage "vram-hog/hami" \
  ${PY} -u 23-failure-inject.py --fault vram-hog --arm hami \
        --roles llm --victim llm --duration 300 --at 90
teardown

stage "vram-hog/hami-mps" \
  ${PY} -u 23-failure-inject.py --fault vram-hog --arm hami-mps \
        --roles llm --victim llm --duration 300 --at 90
teardown

stage "T17" ${PY} -u run.py --test T17
teardown

{
  echo
  echo "=== all stages done $(date -u +%FT%TZ)"
  echo "=== collect with:"
  echo "===   python3 34-collect-c.py T17-$(date -u +%Y%m%d) T10-vram-hog-$(date -u +%Y%m%d)"
  echo "=== and for the fault detail:"
  echo "===   python3 32-collect-faults.py"
} 2>&1 | tee -a "${LOG}"
