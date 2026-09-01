#!/usr/bin/env bash
# Do two co-tenants actually land on ONE card? The last thing before T1.
#
# WHY THIS IS A SCRIPT AND NOT THREE COMMANDS
# The three-command version printed in 10-hami-binpack.sh does not work:
#
#     kubectl wait --for=condition=Ready pod -l kserve.io/component=workload ...
#     error: no matching resources found
#     0, 39 MiB
#     1, 39 MiB
#
# `kubectl wait` with a label selector FAILS IMMEDIATELY when nothing matches
# yet -- it does not wait for resources to appear. KServe takes ~20 s to create
# the pods and minutes to load weights, so the wait bailed, nvidia-smi read two
# idle cards, and the teardown removed the evidence. The run looked like a
# clean failure and actually measured nothing.
#
# harness.wait_ready() polls rather than using `kubectl wait`, so run.py never
# had this bug; only the copy-paste snippet did.
#
#   ./11-verify-colocation.sh            # llm + embed, the T1 pair
#   ./11-verify-colocation.sh --scenario llm-llm
#   ./11-verify-colocation.sh --keep     # leave them up
set -uo pipefail

NS="${NS:-llm-demo}"
SCEN="llm-embed"
KEEP=0
WAIT_APPEAR="${WAIT_APPEAR:-240}"
WAIT_READY="${WAIT_READY:-900}"
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="${HERE}/.colo"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --scenario) SCEN="${2:-llm-embed}"; shift 2 ;;
    --keep) KEEP=1; shift ;;
    -h|--help) sed -n '2,24p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

cleanup() {
  if [[ "${KEEP}" -eq 1 ]]; then
    echo; echo "--- left running. Remove with:"
    echo "      kubectl delete llminferenceservice -n ${NS} -l r2-role"
    return
  fi
  echo; echo "--- tearing down"
  kubectl delete llminferenceservice,deployment,service -n "${NS}" -l r2-role \
    --ignore-not-found --wait=false >/dev/null 2>&1
  rm -rf "${OUT}"
  # Wait for the VRAM to actually come back, so a following run does not start
  # against a card that is still being released.
  for _ in $(seq 1 40); do
    mx="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
          | sort -n | tail -1)"
    [[ "${mx:-0}" -le 300 ]] && break
    sleep 3
  done
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader 2>/dev/null \
    | sed 's/^/      /'
}
trap cleanup EXIT

echo "=== scheduler policy in effect"
kubectl -n kube-system get deploy hami-scheduler -o json 2>/dev/null | python3 -c '
import json, sys
d = json.load(sys.stdin)
for c in d["spec"]["template"]["spec"]["containers"]:
    hits = [a for a in ((c.get("command") or []) + (c.get("args") or []))
            if "scheduler-policy" in a]
    if hits:
        print("  %s" % ", ".join(hits))
'

echo
echo "=== deploying ${SCEN} (arm hami)"
rm -rf "${OUT}"
python3 "${HERE}/render.py" --scenario "${SCEN}" --arm hami --out "${OUT}" || exit 1
kubectl delete llminferenceservice -n "${NS}" -l r2-role --ignore-not-found >/dev/null 2>&1
sleep 3
kubectl apply -f "${OUT}" || exit 1

WANT="$(ls -1 "${OUT}"/*.yaml | wc -l)"
echo
echo "--- waiting for ${WANT} workload pod(s) to appear (KServe takes ~20s)"
t0=$SECONDS
have=0
while (( SECONDS - t0 < WAIT_APPEAR )); do
  have="$(kubectl get pods -n "${NS}" -l kserve.io/component=workload \
          --no-headers 2>/dev/null | wc -l)"
  [[ "${have}" -ge "${WANT}" ]] && break
  sleep 5
done
echo "    ${have} pod(s) after $(( SECONDS - t0 ))s"
if [[ "${have}" -lt "${WANT}" ]]; then
  echo "    never appeared; the CRs were accepted but no workload was created"
  kubectl get events -n "${NS}" --sort-by .lastTimestamp 2>/dev/null | tail -10 | sed 's/^/      /'
  exit 1
fi

echo
echo "--- waiting for them to become Ready (weights load: several minutes)"
# Poll, do not `kubectl wait`: a selector matching nothing errors instantly.
t0=$SECONDS
ready=0
while (( SECONDS - t0 < WAIT_READY )); do
  n="$(kubectl get pods -n "${NS}" -l kserve.io/component=workload -o json 2>/dev/null \
     | python3 -c '
import json, sys
d = json.load(sys.stdin)
n = 0
for p in d.get("items", []):
    conds = {c["type"]: c["status"] for c in (p.get("status", {}).get("conditions") or [])}
    n += conds.get("Ready") == "True"
print(n)
')"
  printf '\r    %s/%s Ready after %ss   ' "${n:-0}" "${WANT}" "$(( SECONDS - t0 ))"
  if [[ "${n:-0}" -ge "${WANT}" ]]; then ready=1; break; fi
  sleep 10
done
echo
if [[ "${ready}" -ne 1 ]]; then
  echo "    not all Ready. Current state:"
  kubectl get pods -n "${NS}" -l kserve.io/component=workload -o wide 2>&1 | sed 's/^/      /'
  for pod in $(kubectl get pods -n "${NS}" -l kserve.io/component=workload \
               -o jsonpath="{.items[*].metadata.name}" 2>/dev/null); do
    echo "      --- ${pod}:"
    kubectl logs "${pod}" -n "${NS}" --tail=25 2>&1 | sed 's/^/        /'
  done
  exit 1
fi

echo
echo "=== WHERE DID THEY LAND?"
echo "--- HAMi's own record, per pod:"
kubectl get pods -n "${NS}" -l kserve.io/component=workload -o json 2>/dev/null | python3 -c '
import json, re, sys
d = json.load(sys.stdin)
devs = {}
for p in d.get("items", []):
    name = p["metadata"]["name"]
    ann = json.dumps(p["metadata"].get("annotations") or {})
    u = sorted(set(re.findall(r"GPU-[0-9a-fA-F-]{36}", ann)))
    devs[name] = u
    print("      %-44s %s" % (name[:44], ", ".join(u) or "(no device recorded)"))
allu = {x for v in devs.values() for x in v}
print("      distinct devices in annotations: %d" % len(allu))
'
echo
echo "--- and what the hardware says:"
nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader 2>/dev/null | sed 's/^/      /'

busy="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null \
        | awk -F, '$2+0 > 500 {print $1}' | tr -d ' ' | paste -sd, -)"
nbusy="$(printf '%s' "${busy}" | tr ',' '\n' | grep -c . || true)"

echo
echo "=================================================================="
if [[ "${nbusy}" -eq 1 ]]; then
  echo "  CO-LOCATED on device ${busy}. binpack works."
  echo "  The sharing arms will actually share. T1 is clear to run:"
  echo "      python3 run.py --test T1"
  rc=0
elif [[ "${nbusy}" -eq 0 ]]; then
  echo "  NO device holds a model, yet the pods report Ready. That should not"
  echo "  happen -- check whether the pods are on this node at all."
  rc=1
else
  echo "  SPLIT across devices ${busy}. binpack did not co-locate them."
  echo
  echo "  The sharing arms cannot run: two tenants on separate cards contend"
  echo "  for nothing, and would produce a clean result that measures nothing."
  echo "  run.py aborts on this, so no invalid data can reach the report."
  echo
  echo "  Fallback: render every tenant as a plain Deployment, where the pod"
  echo "  template is ours and nvidia.com/use-gpuuuid does land. That removes"
  echo "  KServe from the measured path, which is a real cost given KServe is"
  echo "  part of what is being evaluated -- so say the word and I will add it."
  rc=1
fi
echo "=================================================================="
exit "${rc}"
