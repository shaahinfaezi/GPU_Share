#!/usr/bin/env bash
# Run several tiers back to back, detached, with one log per tier.
#
#   ./21-run-series.sh T1 T2 T3
#   SKIP_VALIDATE=1 ./21-run-series.sh T1 T2 T3
#   ./21-run-series.sh T2@4 T2@5 T4      # TEST@RATE overrides the tier's rate
#
# The @RATE form exists to sweep one scenario across offered load without
# defining a new tier. T2@4 and T2@5 put the two-decoder scenario at roughly
# 107% and 130% combined demand, filling the gap between T1 (100.5%, null) and
# T2 (155%, -26.7%) that nothing else in the matrix covers.
#
# Each tier is about 35 minutes. The series keeps going if one tier fails --
# a tier that aborts on the MPS or co-location assertion should not cost the
# tiers behind it -- and the summary at the end says which produced results.
set -uo pipefail
cd "$(dirname "$0")"
[ $# -gt 0 ] || { echo "usage: 21-run-series.sh T1 [T2 ...]"; exit 2; }

if [ "${SKIP_VALIDATE:-0}" != "1" ]; then
  echo "=== gate: 13-validate.py"
  python3 13-validate.py --no-deploy || {
    echo "REFUSING TO LAUNCH. Fix, or re-run with SKIP_VALIDATE=1."; exit 1; }
fi

# Passwordless sudo is not optional: without it the harness cannot stop the
# MPS daemon for the non-MPS arms, which is the whole point of this series.
if ! sudo -n true 2>/dev/null; then
  echo "FATAL: passwordless sudo is required to control the MPS daemon."
  echo "       Without it every arm runs as an MPS client and the sharing"
  echo "       comparison measures nothing -- which is what invalidated T1/T2."
  exit 1
fi

# And sudo working is not the same as the toggle working. It was printed as a
# separate command once, failed, and the series was launched anyway -- 105
# minutes of 0/4 sub-runs. A gate that has to be run by hand is not a gate.
echo "=== gate: MPS daemon toggle"
if ! ./22-mps-check.sh; then
  echo
  echo "REFUSING TO LAUNCH. The harness cannot move the MPS daemon, so every"
  echo "arm would run as an MPS client and hami-vs-hami-mps would measure"
  echo "nothing. This is the exact defect that voided T1 and T2."
  exit 1
fi

# And none of the above notices that the scheduler cannot place a GPU pod at
# all. After T4's OOM cascade HAMi lost the node's registration; the node was
# Ready, nvidia-smi was clean, every gate above passed, and every pod sat
# Pending through a 900s ready timeout. --fix restarts the device plugin,
# which is non-destructive and does not touch running workloads.
echo "=== gate: node can schedule GPU pods"
if ! ./25-node-check.sh --fix; then
  echo
  echo "REFUSING TO LAUNCH. The scheduler cannot place a GPU pod on this node,"
  echo "so every sub-run would sit Pending until its ready timeout expired."
  exit 1
fi

mkdir -p logs
STAMP=$(date +%Y%m%d-%H%M%S)
SERIES="$(pwd)/logs/series-${STAMP}.log"
TESTS="$*"

{
  echo "=== series ${TESTS}  started $(date -Is)"
  for T in $TESTS; do
    NAME="${T%%@*}"
    RATEARG=""
    TAG="${NAME}"
    if [ "$T" != "$NAME" ]; then
      RATEARG="--rate ${T#*@}"
      TAG="${NAME}r${T#*@}"
    fi
    LOG="$(pwd)/logs/${TAG}-${STAMP}.log"
    echo
    echo "=== ${T} starting $(date -Is)  -> ${LOG}"
    python3 -u run.py --test "$NAME" $RATEARG > "$LOG" 2>&1
    RC=$?
    DIR=$(grep -m1 -oE "results/${NAME}-[0-9]{8}-[0-9]{6}" "$LOG" 2>/dev/null | head -1)
    OKN=$(grep -oE "[0-9]+/[0-9]+ sub-runs ok" "$LOG" 2>/dev/null | tail -1)
    echo "=== ${T} finished rc=${RC} $(date -Is)"
    echo "    run dir : $(pwd)/${DIR:-UNKNOWN}"
    echo "    sub-runs: ${OKN:-unknown}"
    ABORT=$(grep -m1 -oE "ABORTING the remaining sub-runs: .*" "$LOG" 2>/dev/null)
    [ -n "$ABORT" ] && echo "    ABORTED : ${ABORT#ABORTING the remaining sub-runs: }"
    echo "    mps      : $(grep -c 'mps check: OK' "$LOG" 2>/dev/null) OK, $(grep -c 'mps check: FAILED' "$LOG" 2>/dev/null) FAILED"
  done
  echo
  echo "=== series done $(date -Is)"
} > "$SERIES" 2>&1 &

PID=$!
ln -sfn "$SERIES" logs/latest-series.log
sleep 3
echo
echo "=== series launched: ${TESTS}"
echo "  pid      ${PID}"
echo "  summary  ${SERIES}"
echo "  per-tier $(pwd)/logs/<TEST>-${STAMP}.log   (TEST@RATE logs as TESTrRATE)"
echo
echo "  follow   tail -f ${SERIES}"
echo "  detail   tail -f $(pwd)/logs/$(echo $TESTS | awk '{print $1}')-${STAMP}.log"
echo "  stop     kill ${PID}"
echo
echo "Roughly $(( $# * 35 )) minutes for $# tier(s). Safe to disconnect."
echo "When it finishes, send the summary log and the run dirs."
