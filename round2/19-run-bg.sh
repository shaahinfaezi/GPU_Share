#!/usr/bin/env bash
# Gate, then launch a tier detached so the SSH session can end.
#
#   ./19-run-bg.sh T2                    # validate, then run T2 in the background
#   ./19-run-bg.sh T2 --arms hami,hami-mps
#   SKIP_VALIDATE=1 ./19-run-bg.sh T3    # only when a validate just passed
#
# setsid reparents to init, so the run survives logout. python3 -u is not
# optional: without it the log stays in a 8 KiB buffer and a disconnected user
# sees an empty file for twenty minutes.
set -uo pipefail
cd "$(dirname "$0")"
T=${1:?usage: 19-run-bg.sh <T1..T8> [extra run.py args]}
shift || true

if [ "${SKIP_VALIDATE:-0}" != "1" ]; then
  echo "=== gate: 13-validate.py"
  python3 13-validate.py --no-deploy || {
    echo
    echo "REFUSING TO LAUNCH. Every check here failed a real run once before;"
    echo "fix it, or re-run with SKIP_VALIDATE=1 if you know why it is noise."
    exit 1
  }
fi

mkdir -p logs
STAMP=$(date +%Y%m%d-%H%M%S)
LOG="$(pwd)/logs/${T}-${STAMP}.log"
setsid nohup python3 -u run.py --test "$T" "$@" > "$LOG" 2>&1 &
PID=$!
ln -sfn "$LOG" "logs/latest-${T}.log"

# run.py prints "=== <test> <scenario> <run_dir>" as its first line of work.
RUN_DIR=""
for _ in $(seq 1 30); do
  sleep 2
  kill -0 "$PID" 2>/dev/null || break
  RUN_DIR=$(grep -m1 -oE "results/${T}-[0-9]{8}-[0-9]{6}" "$LOG" 2>/dev/null | head -1)
  [ -n "$RUN_DIR" ] && break
done

echo
echo "=== ${T} launched"
echo "  pid      ${PID}"
echo "  log      ${LOG}"
if [ -n "$RUN_DIR" ]; then
  echo "  run dir  $(pwd)/${RUN_DIR}"
else
  echo "  run dir  not printed yet -- grep the log:"
  echo "             grep -oE 'results/${T}-[0-9-]+' ${LOG}"
fi
echo
echo "  follow   tail -f ${LOG}"
echo "  check    tail -30 ${LOG}"
echo "  stop     kill ${PID}"
echo
echo "Safe to disconnect. When it finishes, send the tail of the log and the"
echo "contents of the run dir."
