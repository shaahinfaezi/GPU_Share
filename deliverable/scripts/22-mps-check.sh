#!/usr/bin/env bash
# Five-second check that the harness can see and move the MPS daemon.
#
# Run this before a series -- 21-run-series.sh does it for you. It exercises
# the exact code path every sub-run depends on, so a detection bug shows up
# here instead of after 105 minutes of 0/4 sub-runs.
set -uo pipefail
cd "$(dirname "$0")"
python3 - <<'PY'
import harness as H

print("=== detection")
print("  control socket     : %s/control" % H.MPS_PIPE)
print("  daemon reachable   : %s" % H.mps_daemon_running())
print("  MPS client pids    : %s" % H.mps_client_pids())
print("  supervisor         : daemonset %s/%s" % (H.MPS_DS_NS, H.MPS_DS))

rc, out = H.kubectl(["-n", H.MPS_DS_NS, "get", "daemonset", H.MPS_DS,
                     "-o", "jsonpath={.status.numberReady}/"
                           "{.status.desiredNumberScheduled} ready"])
print("                       %s" % (out.strip() or "NOT FOUND"))
if rc != 0:
    print("")
    print("  Cannot see the DaemonSet, so the toggle cannot work.")
    print("  Set R2_MPS_DS / R2_MPS_DS_NS if it is named differently here.")
    raise SystemExit(1)

print("")
print("=== toggle (this is what every sub-run does)")
for want in (False, True):
    ok, how = H.set_mps_daemon(want)
    seen = H.mps_daemon_running()
    verdict = "OK" if (ok and seen == want) else "FAILED"
    print("  set %-4s -> %-6s  reachable=%-5s  (%s)"
          % ("down" if not want else "up", verdict, seen, how))
    if verdict != "OK":
        print("")
        print("  The run would abort here. Every arm would run as an MPS")
        print("  client and hami-vs-hami-mps would measure nothing -- the")
        print("  exact defect that voided T1 and T2. Fix before launching.")
        raise SystemExit(1)

print("")
print("=== left up, which is the state 13-validate.py expects")
raise SystemExit(0)
PY
