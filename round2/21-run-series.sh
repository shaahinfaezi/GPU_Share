#!/usr/bin/env bash
# Run several tiers back to back, detached, with one log per tier.
#
#   ./21-run-series.sh T1 T2 T3
#   SKIP_VALIDATE=1 ./21-run-series.sh T1 T2 T3
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

mkdir -p logs
STAMP=$(date +%Y%m%d-%H%M%S)
SERIES="$(pwd)/logs/series-${STAMP}.log"
TESTS="$*"

{
  echo "=== series ${TESTS}  started $(date -Is)"
  for T in $TESTS; do
    LOG="$(pwd)/logs/${T}-${STAMP}.log"
    echo
    echo "=== ${T} starting $(date -Is)  -> ${LOG}"
    python3 -u run.py --test "$T" > "$LOG" 2>&1
    RC=$?
    DIR=$(grep -m1 -oE "results/${T}-[0-9]{8}-[0-9]{6}" "$LOG" 2>/dev/null | head -1)
    OKN=$(grep -oE "[0-9]+/[0-9]+ sub-runs ok" "$LOG" 2>/dev/null | tail -1)
    echo "=== ${T} finished rc=${RC} $(date -Is)"
    echo "    run dir : $(pwd)/${DIR:-UNKNOWN}"
    echo "    sub-runs: ${OKN:-unknown}"
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
echo "  per-tier $(pwd)/logs/<TEST>-${STAMP}.log"
echo
echo "  follow   tail -f ${SERIES}"
echo "  detail   tail -f $(pwd)/logs/$(echo $TESTS | awk '{print $1}')-${STAMP}.log"
echo "  stop     kill ${PID}"
echo
echo "Roughly $(( $# * 35 )) minutes for $# tier(s). Safe to disconnect."
echo "When it finishes, send the summary log and the run dirs."
