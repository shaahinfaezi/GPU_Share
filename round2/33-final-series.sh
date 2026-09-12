#!/usr/bin/env bash
# The last run: the three questions left that only the GPU box can answer.
#
# Everything else in the matrix is either measured, or is a question research
# settled, or is a question this node physically cannot answer. What remains:
#
#   T18  three tenants. Every co-tenancy result in this study is two-body --
#        the depth law, the fairness reversal, the overbook curve. Nothing
#        says they extend to three. T18 has been refused twice by the
#        host-memory guard (9,524 MiB needed against 7,559 free), so this
#        wrapper parks the observability stack first to buy the headroom.
#        If it is STILL refused, that refusal is the finding and the
#        three-tenant question is unanswerable on this hardware.
#
#   T20  real Azure arrivals. No tier anywhere in this study replays a
#        production arrival process; they all regenerate Poisson. The
#        conversation trace measures at CV 1.057-1.094, close to Poisson, so
#        the registered prediction is that nothing much changes -- and that
#        IS the result, because it would vindicate the substitution used in
#        every other tier. If it does change, every tier inherits a caveat.
#
#   T21  a symmetric pair that actually saturates. T13 and T15 answered the
#        fairness question and both failed to answer the capacity one,
#        staying above 99.5% of solo-sum throughput. Does the MPS throughput
#        gain survive when the two tenants are identical?
#
# Runs detached, one stage at a time, with a forced teardown between stages
# and the host-RAM restore in a trap so it happens even if the series dies.
set -uo pipefail

cd "$(dirname "$0")"
STAMP="$(date -u +%Y%m%d-%H%M%S)"
LOG="logs/final-${STAMP}.log"
mkdir -p logs

restore() {
  echo | tee -a "${LOG}"
  echo "=== restoring the observability stack" | tee -a "${LOG}"
  ./31-freeram.sh up 2>&1 | tee -a "${LOG}"
}
trap restore EXIT INT TERM

{
  echo "=== final series ${STAMP}"
  echo "=== stages: T18 (three tenants), T20 (real Azure arrivals), T21 (saturated symmetric pair)"
  echo

  echo "=== step 1/2: free host RAM for the three-tenant tier"
  ./31-freeram.sh down
  echo

  echo "=== step 2/2: the tiers"
  # 28-series-b.sh BACKGROUNDS ITSELF and returns immediately. Calling it
  # bare made this wrapper finish in seconds, firing the EXIT trap and
  # restoring the observability stack 90 minutes before the series ended --
  # handing back the very memory the three-tenant tier needed. Capture the
  # pid it prints and block on it, so the trap fires when the work is done.
  ./28-series-b.sh T18 T20 T21 | tee /tmp/r2-series-launch.txt
  SERIES_PID="$(awk '/^ *pid /{print $2; exit}' /tmp/r2-series-launch.txt)"
  if [[ -z "${SERIES_PID}" ]]; then
    echo "!!! could not read the series pid; NOT restoring, do it by hand:"
    echo "!!!   ./31-freeram.sh up"
    trap - EXIT INT TERM
    exit 1
  fi
  echo "--- waiting on series pid ${SERIES_PID}"
  while kill -0 "${SERIES_PID}" 2>/dev/null; do sleep 30; done
  echo "--- series pid ${SERIES_PID} has exited"
} 2>&1 | tee -a "${LOG}"

echo
echo "=== final series done. Summary: ${LOG}"
