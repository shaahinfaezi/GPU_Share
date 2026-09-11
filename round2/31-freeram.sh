#!/usr/bin/env bash
# Free host RAM so the three-tenant tier can actually deploy, then put it back.
#
# T18 and T4 are the only tiers this study could not run, and both were refused
# by the host-memory guard rather than by anything about GPU sharing: three
# tenants need about 9,524 MiB and the node had 7,559 MiB free. That refusal is
# itself a reported finding -- host RAM, not VRAM, is what caps density here --
# but it leaves the single largest structural gap in the matrix unanswered,
# because every co-tenancy result in the study is two-body.
#
# This is the cheap way to get the three-body answer on this node: park the
# observability and UI workloads that are not needed for a run, measure what
# that bought, and restore them afterwards. What it does NOT touch: HAMi's
# scheduler and device plugin (the thing under test), Prometheus (the run
# collects from it), or anything in kube-system that schedules pods.
#
#   ./31-freeram.sh down     park them, report the gain
#   ./31-freeram.sh up       restore every replica count it changed
#   ./31-freeram.sh status   show what is currently parked
#
# The saved state lives in STATE so `up` restores the original counts rather
# than assuming they were 1.
set -uo pipefail

STATE="${STATE:-/opt/GPU_Share/round2/.freeram-state}"

# deployment:namespace pairs. Deliberately conservative.
TARGETS=(
  "my-hami-webui:kube-system"
  "grafana:default"
  "prometheus-grafana:default"
)
# DaemonSets cannot be scaled; they are parked with a nodeSelector instead.
DS_TARGETS=(
  "my-hami-webui-dcgm-exporter:kube-system"
)

avail() { awk '/MemAvailable/{printf "%d", $2/1024}' /proc/meminfo; }

banner() {
  echo
  echo "=== $1"
}

case "${1:-status}" in

down)
  before=$(avail)
  banner "host memory before: ${before} MiB available"
  : > "${STATE}"

  for t in "${TARGETS[@]}"; do
    dep="${t%%:*}"; ns="${t##*:}"
    n=$(kubectl -n "${ns}" get deploy "${dep}" -o jsonpath='{.spec.replicas}' 2>/dev/null || true)
    if [[ -z "${n}" ]]; then
      echo "  skip     ${ns}/${dep} (not present)"
      continue
    fi
    if [[ "${n}" == "0" ]]; then
      echo "  already  ${ns}/${dep} at 0"
      continue
    fi
    echo "deploy ${dep} ${ns} ${n}" >> "${STATE}"
    kubectl -n "${ns}" scale deploy "${dep}" --replicas=0 >/dev/null 2>&1 \
      && echo "  parked   ${ns}/${dep} (was ${n})" \
      || echo "  FAILED   ${ns}/${dep}"
  done

  for t in "${DS_TARGETS[@]}"; do
    ds="${t%%:*}"; ns="${t##*:}"
    if ! kubectl -n "${ns}" get ds "${ds}" >/dev/null 2>&1; then
      echo "  skip     ${ns}/${ds} (not present)"
      continue
    fi
    # A nodeSelector nothing matches evicts the pods without deleting the
    # DaemonSet, and `up` removes the selector again.
    echo "ds ${ds} ${ns}" >> "${STATE}"
    kubectl -n "${ns}" patch ds "${ds}" --type merge \
      -p '{"spec":{"template":{"spec":{"nodeSelector":{"r2-freeram":"parked"}}}}}' \
      >/dev/null 2>&1 \
      && echo "  parked   ${ns}/${ds} (nodeSelector)" \
      || echo "  FAILED   ${ns}/${ds}"
  done

  echo
  echo -n "  waiting for memory to be released "
  for _ in $(seq 1 30); do
    sleep 2; echo -n "."
    now=$(avail)
    [[ $(( now - before )) -ge 900 ]] && break
  done
  echo
  after=$(avail)
  banner "host memory after: ${after} MiB available  (freed $(( after - before )) MiB)"
  echo
  echo "  T18 needs about 9524 MiB for three tenants."
  if [[ "${after}" -ge 9524 ]]; then
    echo "  OK -- the guard should now admit three tenants."
  else
    echo "  STILL SHORT by $(( 9524 - after )) MiB. T18 will be refused again,"
    echo "  and that refusal remains a legitimate result: on this node the"
    echo "  three-tenant question cannot be answered without more RAM."
  fi
  ;;

up)
  if [[ ! -s "${STATE}" ]]; then
    echo "nothing recorded in ${STATE}; nothing to restore"
    exit 0
  fi
  banner "restoring"
  while read -r kind name ns n; do
    case "${kind}" in
      deploy)
        kubectl -n "${ns}" scale deploy "${name}" --replicas="${n}" >/dev/null 2>&1 \
          && echo "  restored ${ns}/${name} -> ${n}" \
          || echo "  FAILED   ${ns}/${name}"
        ;;
      ds)
        kubectl -n "${ns}" patch ds "${name}" --type json \
          -p '[{"op":"remove","path":"/spec/template/spec/nodeSelector/r2-freeram"}]' \
          >/dev/null 2>&1 \
          && echo "  restored ${ns}/${name} (nodeSelector removed)" \
          || echo "  FAILED   ${ns}/${name}"
        ;;
    esac
  done < "${STATE}"
  rm -f "${STATE}"
  echo
  echo "host memory now: $(avail) MiB available"
  ;;

status)
  banner "host memory: $(avail) MiB available"
  if [[ -s "${STATE}" ]]; then
    echo "  parked by this script:"
    sed 's/^/    /' "${STATE}"
  else
    echo "  nothing parked by this script"
  fi
  ;;

*)
  echo "usage: $0 {down|up|status}" >&2
  exit 2
  ;;
esac
