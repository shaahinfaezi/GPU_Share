#!/usr/bin/env bash
# A swapfile used as an OOM BACKSTOP, not as a resource.
#
# This node has SwapTotal 0, so every host-memory overshoot goes straight to
# the OOM killer -- which is exactly what destroyed T4, taking two vLLM
# engines and Prometheus with it. That is why the harness refuses three
# tenants rather than attempting them.
#
# Three tenants need ~8,600 MiB of host RAM raw and the node frees to about
# 8,156 MiB with the observability stack parked. The gap is small. Swap turns
# that gap from "the kernel kills a vLLM engine mid-run and the results are
# gone" into "the run gets slower and you can SEE that it did".
#
# The discipline that makes this legitimate:
#
#   Swap is a safety net, never a resource. If the run touches it in any
#   meaningful amount, the latency numbers are contaminated and the run is
#   VOID. 36-three-body.sh records swap used before and after and says so
#   plainly. A tier that only fits because it swapped has not measured
#   three-tenant co-tenancy; it has measured swapping.
#
#   ./35-swapfile.sh on     create and enable
#   ./35-swapfile.sh off    disable and delete
#   ./35-swapfile.sh status
set -uo pipefail

SWAPFILE="${SWAPFILE:-/swapfile-r2}"
SIZE_GB="${SIZE_GB:-6}"

used_mib() { awk '/SwapTotal/{t=$2}/SwapFree/{f=$2}END{printf "%d", (t-f)/1024}' /proc/meminfo; }
total_mib() { awk '/SwapTotal/{printf "%d", $2/1024}' /proc/meminfo; }

case "${1:-status}" in

on)
  if [[ "$(total_mib)" -gt 0 ]]; then
    echo "swap already enabled: $(total_mib) MiB total, $(used_mib) MiB used"
    exit 0
  fi
  if [[ ! -f "${SWAPFILE}" ]]; then
    echo "creating ${SWAPFILE} (${SIZE_GB} GiB) -- this writes to /, check space first"
    df -Ph / | tail -1 | sed 's/^/  /'
    # fallocate is instant but some filesystems refuse it for swap; dd is safe.
    if ! fallocate -l "${SIZE_GB}G" "${SWAPFILE}" 2>/dev/null; then
      dd if=/dev/zero of="${SWAPFILE}" bs=1M count=$((SIZE_GB * 1024)) status=progress
    fi
    chmod 600 "${SWAPFILE}"
    mkswap "${SWAPFILE}" >/dev/null
  fi
  swapon "${SWAPFILE}" && echo "swap on: $(total_mib) MiB"
  # Only ever page out under real pressure -- this is a backstop, and a high
  # swappiness would have the kernel paging a healthy vLLM engine for cache.
  sysctl -w vm.swappiness=1 >/dev/null && echo "vm.swappiness=1 (backstop only)"
  ;;

off)
  if [[ "$(total_mib)" -eq 0 ]]; then
    echo "swap already off"
  else
    echo "swap used at teardown: $(used_mib) MiB"
    swapoff "${SWAPFILE}" 2>/dev/null && echo "swapoff ok"
  fi
  [[ -f "${SWAPFILE}" ]] && rm -f "${SWAPFILE}" && echo "removed ${SWAPFILE}"
  ;;

status)
  echo "SwapTotal $(total_mib) MiB, SwapUsed $(used_mib) MiB"
  [[ -f "${SWAPFILE}" ]] && ls -lh "${SWAPFILE}"
  ;;

*)
  echo "usage: $0 {on|off|status}" >&2
  exit 2
  ;;
esac
