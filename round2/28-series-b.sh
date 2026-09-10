#!/usr/bin/env bash
# Run tiers AND fault injections back to back, detached, one log each.
#
#   ./28-series-b.sh T8 T13 T14 fault:vram-hog fault:tenant-fault
#   ./28-series-b.sh T8 T8@mps=75            # sweep the MPS ceiling
#
# 21-run-series.sh only drives run.py. The T10 faults live in a different
# script with a different CLI, so an overnight batch that wants both had to be
# hand-assembled -- which is how the last one ended up with no teardown between
# stages and killed everything queued behind it.
#
# Item forms:
#   T8                 python3 run.py --test T8
#   T8@mps=75          ... --mps-percent <role>=75 for every role in the tier
#   fault:NAME         python3 23-failure-inject.py --fault NAME, both arms
#
# Measured timings, from this project's own run dirs: a four-sub-run tier is
# 28-32 min, a three-sub-run tier 25 min, a fault run 9 min per arm.
set -uo pipefail
cd "$(dirname "$0")"
[ $# -gt 0 ] || { echo "usage: 28-series-b.sh T8 T13 fault:vram-hog ..."; exit 2; }

# ---- gates ---------------------------------------------------------------
if ! sudo -n true 2>/dev/null; then
  echo "FATAL: passwordless sudo is required to move the MPS daemon."
  echo "       Without it every arm runs as an MPS client and hami-vs-hami-mps"
  echo "       measures nothing -- the defect that voided T1 and T2."
  exit 1
fi
echo "=== gate: MPS daemon toggle"
./22-mps-check.sh || { echo "REFUSING TO LAUNCH: cannot move the MPS daemon."; exit 1; }
echo "=== gate: node can schedule GPU pods"
./25-node-check.sh --fix || { echo "REFUSING TO LAUNCH: scheduler cannot place a GPU pod."; exit 1; }

mkdir -p logs
STAMP=$(date +%Y%m%d-%H%M%S)
SERIES="$(pwd)/logs/seriesB-${STAMP}.log"
ITEMS="$*"

# Rough estimate so the operator knows whether to wait or sleep, and a
# per-stage wall-clock cap so one hung stage cannot eat the rest of the night.
# T12 is a four-hour soak, so the cap has to be per stage rather than global.
MINS=0
for I in $ITEMS; do
  case "$I" in
    fault:*) MINS=$((MINS + 18)) ;;   # two arms at ~9 min
    T7*)     MINS=$((MINS + 25)) ;;
    T12*)    MINS=$((MINS + 290)) ;;  # 4 h replay + init + teardown
    T9*)     MINS=$((MINS + 55)) ;;   # two arms, init-dominated
    *)       MINS=$((MINS + 30)) ;;
  esac
done

stage_cap() {                       # seconds, generous: a cap that fires is a bug
  case "$1" in
    T12*)    echo 23400 ;;          # 6.5 h, matches T12 and T12@arms=...
    T9*)     echo 5400 ;;           # 1.5 h
    fault:*) echo 3600 ;;           # 1 h for both arms
    *)       echo 4200 ;;           # 70 min
  esac
}
FAILED=""; DONE_OK=""

{
  echo "=== series B: ${ITEMS}"
  echo "=== started $(date -Is), estimated ${MINS} min"
  for I in $ITEMS; do
    # Pre-stage guard. Host RAM is the binding constraint on this node and
    # there is no swap, so an overshoot is an OOM kill rather than a slowdown.
    # Two co-tenants need roughly 6 GiB resident.
    AVAIL=$(awk '/^MemAvailable:/{print int($2/1024)}' /proc/meminfo)
    DISK=$(df -Pk /var/lib/rancher/k3s/agent/containerd 2>/dev/null | awk 'NR==2{print int($4/1048576)}')
    if [ "${AVAIL:-0}" -lt 7000 ]; then
      echo "=== ${I} WAITING: only ${AVAIL} MiB host RAM available, need ~7000"
      for _ in 1 2 3 4 5 6; do
        sleep 30
        AVAIL=$(awk '/^MemAvailable:/{print int($2/1024)}' /proc/meminfo)
        [ "${AVAIL:-0}" -ge 7000 ] && break
      done
    fi
    if [ "${AVAIL:-0}" -lt 6000 ]; then
      echo "=== ${I} SKIPPED: host RAM still ${AVAIL} MiB. Not burning 30 min on a run the guard will refuse."
      continue
    fi
    if [ "${DISK:-99}" -lt 6 ]; then
      echo "=== ${I} SKIPPED: only ${DISK} GiB free on the container filesystem; kubelet evicts node-wide at 4.84"
      continue
    fi

    TAG="${I//[:@=]/-}"
    LOG="$(pwd)/logs/${TAG}-${STAMP}.log"
    echo
    echo "=== ${I} starting $(date -Is)  -> ${LOG}"

    case "$I" in
      fault:*)
        F="${I#fault:}"
        # Both arms. The hami arm is the control that says whether a fault is
        # an MPS property or just a Kubernetes one -- vram-escape was contained
        # on BOTH arms, and without the control that would have read as an MPS
        # success rather than as HAMi doing its job.
        RC=0
        for ARM in hami hami-mps; do
          echo "--- ${F} arm=${ARM} $(date -Is)"
          timeout -k 60 "$(stage_cap "$I")" \
            python3 -u 23-failure-inject.py --fault "$F" --arm "$ARM" >> "$LOG" 2>&1
          ARC=$?
          echo "--- ${F} arm=${ARM} rc=${ARC} $(date -Is)"
          [ "$ARC" -ne 0 ] && RC=$ARC
        done
        ;;
      *@arms=*)
        # T12 runs four arms by default and each one is a four-hour soak.
        # Unconstrained that is a sixteen-hour stage, which is how an
        # overnight batch turns into a lost day. `T12@arms=hami-mps` runs the
        # single arm the soak question is actually about.
        T="${I%@arms=*}"; A="${I#*@arms=}"
        echo "--- ${T} with --arms ${A//,/ }"
        timeout -k 60 "$(stage_cap "$I")"           python3 -u run.py --test "$T" --arms ${A//,/ } > "$LOG" 2>&1
        RC=$?
        ;;
      *@mps=*)
        T="${I%@mps=*}"; P="${I#*@mps=}"
        # Apply the percentage to every role the tier declares, so the caller
        # does not have to know the role names.
        ROLES=$(python3 -c "import run; print(','.join('%s=$P' % r for r in run.ROLES[run.TESTS['$T']['scenario']]))" 2>/dev/null)
        if [ -z "$ROLES" ]; then
          echo "    SKIPPED : could not expand roles for ${T}"; continue
        fi
        echo "--- ${T} with --mps-percent ${ROLES}"
        timeout -k 60 "$(stage_cap "$I")" \
          python3 -u run.py --test "$T" --mps-percent "$ROLES" > "$LOG" 2>&1
        RC=$?
        ;;
      T9*)
        # 09-scaledown.py is its own driver, one arm per invocation.
        RC=0
        for ARM in hami hami-mps; do
          echo "--- T9 arm=${ARM} $(date -Is)"
          timeout -k 60 "$(stage_cap "$I")" \
            python3 -u 09-scaledown.py --arm "$ARM" >> "$LOG" 2>&1
          ARC=$?
          echo "--- T9 arm=${ARM} rc=${ARC} $(date -Is)"
          [ "$ARC" -ne 0 ] && RC=$ARC
        done
        ;;
      *)
        timeout -k 60 "$(stage_cap "$I")" \
          python3 -u run.py --test "$I" > "$LOG" 2>&1
        RC=$?
        ;;
    esac

    echo "=== ${I} finished rc=${RC} $(date -Is)"
    DIR=$(grep -m1 -oE "results/[A-Za-z0-9_-]+-[0-9]{8}-[0-9]{6}" "$LOG" 2>/dev/null | head -1)
    echo "    run dir : $(pwd)/${DIR:-see log}"
    OKN=$(grep -oE "[0-9]+/[0-9]+ sub-runs ok" "$LOG" 2>/dev/null | tail -1)
    [ -n "$OKN" ] && echo "    sub-runs: ${OKN}"
    ABORT=$(grep -m1 -oE "ABORTING the remaining sub-runs: .*" "$LOG" 2>/dev/null)
    [ -n "$ABORT" ] && echo "    ABORTED : ${ABORT#ABORTING the remaining sub-runs: }"

    # Leftover pods poison the NEXT item's preconditions. This is the exact
    # failure that cost an overnight batch: T10 had no teardown, and every
    # stage queued behind it exited rc=2 against pods that were still up.
    LEFT=$(kubectl get pods -n "${R2_NS:-llm-demo}" --no-headers 2>/dev/null | wc -l)
    if [ "${LEFT:-0}" -gt 0 ]; then
      echo "    WARNING : ${LEFT} pod(s) still in ${R2_NS:-llm-demo}; forcing teardown"
      kubectl delete llminferenceservice --all -n "${R2_NS:-llm-demo}" --wait=true --timeout=180s >/dev/null 2>&1
      sleep 20
    fi

    if [ "${RC}" -eq 124 ] || [ "${RC}" -eq 137 ]; then
      echo "    TIMEOUT : stage exceeded $(stage_cap "$I")s and was killed"
    fi
    if [ "${RC}" -eq 0 ]; then DONE_OK="${DONE_OK} ${I}"; else FAILED="${FAILED} ${I}(rc=${RC})"; fi

    # Leave the daemon in the state 13-validate.py expects, so a stage that
    # died mid-arm cannot hand the next one a half-configured node.
    ./22-mps-check.sh >/dev/null 2>&1 || echo "    WARNING : MPS daemon toggle unhealthy after ${I}"
  done
  echo
  echo "=== series done $(date -Is)"
  echo "=== ok    :${DONE_OK:- none}"
  echo "=== failed:${FAILED:- none}"
} > "$SERIES" 2>&1 &

PID=$!
# Detach from the controlling terminal. Every stage so far has survived a
# disconnect, but a twelve-hour run should not rest on the shell choosing not
# to send SIGHUP.
disown "$PID" 2>/dev/null || true
ln -sfn "$SERIES" logs/latest-seriesB.log
sleep 3
echo
echo "=== launched: ${ITEMS}"
echo "  pid      ${PID}"
echo "  summary  ${SERIES}"
echo "  follow   tail -f ${SERIES}"
echo "  stop     kill ${PID}"
echo
echo "Roughly ${MINS} minutes. Safe to disconnect."
