#!/usr/bin/env bash
# Five-second check that the harness can see and move the MPS daemon.
#
# Run this once before a series. It exercises the exact code path the run
# depends on, so a detection bug shows up here instead of after 105 minutes of
# 0/4 sub-runs.
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
print("                       %s" % (out.strip() or "NOT FOUND -- set R2_MPS_DS"))
if rc != 0:
    print("
  Cannot see the DaemonSet, so the toggle cannot work.")
    raise SystemExit(1)

print("\n=== toggle (this is what every sub-run does)")
for want in (False, True):
    ok, how = H.set_mps_daemon(want)
    seen = H.mps_daemon_running()
    verdict = "OK" if (ok and seen == want) else "FAILED"
    print("  set to %-5s -> %-7s  now running=%s   (%s)"
          % ("down" if not want else "up", verdict, seen, how))
    if verdict == "FAILED":
        print("\n  The run would abort here. Fix before launching a series.")
        raise SystemExit(1)

print("\n=== left running, which is what 13-validate.py expects")
raise SystemExit(0)
PY
