#!/usr/bin/env bash
# Make HAMi pack co-tenants onto one GPU instead of spreading them.
#
# WHY THIS IS NEEDED
# The per-pod pin does not work through KServe, and cannot be made to. Measured:
#
#   pinned to      : GPU-bc62f652-... (index 0)
#   device(s) busy : 1
#   pod annotations: hami.io/vgpu-devices-allocated: GPU-fa422f59-...
#
# nvidia.com/use-gpuuuid was absent from the pod entirely. KServe drops
# annotations for the same reason it drops labels: LLMInferenceService's
# spec.template is a bare PodSpec, which has no metadata field, so there is
# nowhere for either to go. Only hami.io/* annotations appear, and HAMi wrote
# those itself.
#
# HAMi's default gpuSchedulerPolicy is `spread`: each new pod goes to the
# least-loaded device. With two idle cards and two tenants, that reliably puts
# them on SEPARATE GPUs -- which makes a "sharing" arm an uncontended arm and
# produces clean numbers that measure nothing. Three consecutive unpinned runs
# landed on GPU 1 with GPU 0 at 39 MiB.
#
# `binpack` inverts it: prefer the fullest device that still fits. Tenant A
# takes GPU 0, tenant B sees GPU 0 with 4096 MiB free against GPU 1 with 12288,
# and binpack prefers GPU 0. Co-located, with no pod annotations involved.
#
# This is also the production-correct answer, not just a test fixture. Packing
# is what you want when the point is to share a card; spreading is what you want
# when the point is isolation. Round two is measuring sharing.
#
#   ./10-hami-binpack.sh            # show current policy, then set binpack
#   ./10-hami-binpack.sh --check    # show only
#   ./10-hami-binpack.sh --revert   # back to spread
#   ./10-hami-binpack.sh --inspect  # what flags does this build take?
set -uo pipefail

NS="${NS:-kube-system}"
DEP="${DEP:-hami-scheduler}"
CHECK=0
INSPECT=0
POLICY="binpack"
for a in "$@"; do
  case "$a" in
    --check) CHECK=1 ;;
    --inspect) INSPECT=1 ;;
    --revert) POLICY="spread" ;;
    -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "unknown option: $a" >&2; exit 2 ;;
  esac
done

echo "=== current scheduler policy"
# The flags live on the vgpu-scheduler-extender container, not on whichever
# container happens to be first.
kubectl -n "${NS}" get deploy "${DEP}" -o json 2>/dev/null \
  | python3 -c '
import json, sys
d = json.load(sys.stdin)
for c in d["spec"]["template"]["spec"]["containers"]:
    args = c.get("args") or []
    hits = [a for a in args if "scheduler-policy" in a]
    print("  container %-28s %s" % (c["name"], ", ".join(hits) if hits else "(no policy flags -- HAMi default: gpu=spread)"))
' || { echo "  could not read ${DEP} in ${NS}" >&2; exit 1; }

echo
echo "--- which device each round-2 pod last landed on"
kubectl get pods -A -o json 2>/dev/null | python3 -c '
import json, re, sys
d = json.load(sys.stdin)
seen = False
for p in d.get("items", []):
    ann = json.dumps(p.get("metadata", {}).get("annotations", {}) or {})
    u = re.findall(r"GPU-[0-9a-fA-F-]{36}", ann)
    if u and p["metadata"]["name"].startswith("r2-"):
        seen = True
        print("  %-46s %s" % (p["metadata"]["name"][:46], sorted(set(u))[0]))
if not seen:
    print("  (no r2-* pods running right now)")
'

if [[ "${INSPECT:-0}" -eq 1 ]]; then
  echo
  echo "=== full extender container spec"
  kubectl -n "${NS}" get deploy "${DEP}" -o json 2>/dev/null | python3 -c '
import json, sys
d = json.load(sys.stdin)
for c in d["spec"]["template"]["spec"]["containers"]:
    if "extender" not in c["name"]:
        continue
    print("  image   %s" % c.get("image"))
    print("  command %s" % (c.get("command") or "(entrypoint)"))
    print("  args:")
    for a in (c.get("args") or []):
        print("    %s" % a)
'
  echo
  echo "=== flags this build actually accepts"
  POD="$(kubectl -n "${NS}" get pods -l app.kubernetes.io/component=hami-scheduler          -o jsonpath="{.items[0].metadata.name}" 2>/dev/null)"
  if [[ -n "${POD}" ]]; then
    for bin in /k8s-vgpu/bin/scheduler /usr/local/bin/scheduler scheduler; do
      out="$(kubectl -n "${NS}" exec "${POD}" -c vgpu-scheduler-extender --              "${bin}" --help 2>&1)"
      if grep -qi "policy" <<<"${out}"; then
        grep -iE "policy" <<<"${out}" | sed 's/^/    /'
        break
      fi
    done
    grep -qi policy <<<"${out:-}" ||       echo "    (--help not reachable; the arg list above is the evidence)"
  fi

  echo
  echo "=== rollout strategy -- why a rolling update stalls here"
  kubectl -n "${NS}" get deploy "${DEP}" -o json 2>/dev/null | python3 -c '
import json, sys
d = json.load(sys.stdin)
sp = d["spec"]
print("    strategy: %s" % json.dumps(sp.get("strategy")))
print("    replicas: %s" % sp.get("replicas"))
print("    terminationGracePeriodSeconds: %s"
      % sp["template"]["spec"].get("terminationGracePeriodSeconds"))
'
  echo
  echo "    A surge-based rolling update needs the OLD replica to terminate."
  echo "    HAMi elects a leader, and the first attempt here stalled with the"
  echo "    new pod healthy and logging \"patched follower label\" while the old"
  echo "    one refused to leave. This script now scales 0 -> 1 instead."
  exit 0
fi

if [[ "${CHECK}" -eq 1 ]]; then
  echo
  echo "Run without --check to set gpuSchedulerPolicy=${POLICY}."
  exit 0
fi

echo
echo "=== setting gpuSchedulerPolicy=${POLICY}"
# Patch by container NAME so this cannot silently edit the wrong container if
# the chart reorders them.
PATCHED="$(kubectl -n "${NS}" get deploy "${DEP}" -o json 2>/dev/null \
  | POLICY="${POLICY}" python3 -c '
import json, os, sys
pol = os.environ["POLICY"]
d = json.load(sys.stdin)
# ONLY the extender. An earlier version matched on "scheduler" in the
# container name and broke on the first hit, which is kube-scheduler --
# upstream kube-scheduler does not know --gpu-scheduler-policy, refused to
# start, and the rollout hung with the old replica pending termination.
# The HAMi flags live only on the vgpu-scheduler-extender container.
# (No apostrophes in here: this whole block is a single-quoted shell string,
# and one apostrophe ends it mid-script.)
target = None
for c in d["spec"]["template"]["spec"]["containers"]:
    if "extender" in c["name"]:
        target = c
        break
if target is None:
    sys.exit("no vgpu-scheduler-extender container in this deployment; "
             "inspect it by hand before patching anything")
args = [a for a in (target.get("args") or [])
        if not a.startswith("--gpu-scheduler-policy")
        and not a.startswith("--node-scheduler-policy")]
args += ["--gpu-scheduler-policy=" + pol, "--node-scheduler-policy=binpack"]
target["args"] = args
print(json.dumps({"spec": {"template": {"spec": {"containers": [
    {"name": target["name"], "args": args}]}}}}))
')"
if [[ -z "${PATCHED}" ]]; then
  echo "  could not build the patch" >&2
  exit 1
fi
kubectl -n "${NS}" patch deploy "${DEP}" --type=strategic -p "${PATCHED}" || exit 1

echo
echo "--- restarting the scheduler (scale 0 -> 1, not a rolling update)"
# A rolling update stalls here. The first attempt sat on "1 old replicas are
# pending termination" while the NEW pod came up healthy and logged
# "Successfully patched follower label" -- so the extender accepted the flags
# and the blockage was the old replica refusing to leave, a leader-election
# interaction rather than a bad argument.
#
# Scaling to zero and back sidesteps the surge entirely. It costs a few seconds
# with no HAMi scheduler, during which GPU pods stay Pending rather than being
# mis-scheduled, which is the safe direction.
kubectl -n "${NS}" scale deploy/"${DEP}" --replicas=0 >/dev/null 2>&1
for _ in $(seq 1 30); do
  n="$(kubectl -n "${NS}" get pods -l app.kubernetes.io/component=hami-scheduler        --no-headers 2>/dev/null | wc -l)"
  [[ "${n}" -eq 0 ]] && break
  sleep 2
done
kubectl -n "${NS}" scale deploy/"${DEP}" --replicas=1 >/dev/null 2>&1

echo "--- waiting for it to come back"
ready=0
for _ in $(seq 1 60); do
  # Count Ready containers via -o json rather than a jsonpath filter: the
  # filter needs nested quotes, and every attempt at escaping them through a
  # patch produced a subtly wrong expression that matched nothing -- which
  # would make this loop always time out and always revert, even on success.
  if kubectl -n "${NS}" get pods -l app.kubernetes.io/component=hami-scheduler \
       -o json 2>/dev/null | python3 -c '
import json, sys
d = json.load(sys.stdin)
for p in d.get("items", []):
    conds = {c["type"]: c["status"] for c in (p.get("status", {}).get("conditions") or [])}
    if conds.get("Ready") == "True":
        print("READY"); break
' | grep -q READY; then ready=1; break; fi
  sleep 3
done

if [[ "${ready}" -ne 1 ]]; then
  # Diagnostics BEFORE the revert. The previous version reverted first and then
  # dumped logs, which captured the healthy old pod and explained nothing.
  echo
  echo "SCHEDULER DID NOT COME BACK. Capturing evidence before reverting:"
  kubectl -n "${NS}" get pods -l app.kubernetes.io/component=hami-scheduler     -o wide 2>&1 | sed 's/^/      /'
  BAD="$(kubectl -n "${NS}" get pods -l app.kubernetes.io/component=hami-scheduler          -o jsonpath="{.items[0].metadata.name}" 2>/dev/null)"
  if [[ -n "${BAD}" ]]; then
    kubectl -n "${NS}" describe pod "${BAD}" 2>&1       | sed -n "/Containers:/,/Events:/p" | head -40 | sed 's/^/      /'
    echo "      --- extender logs:"
    kubectl -n "${NS}" logs "${BAD}" -c vgpu-scheduler-extender --tail=40 2>&1       | sed 's/^/      /'
    kubectl -n "${NS}" logs "${BAD}" -c vgpu-scheduler-extender --previous       --tail=20 2>&1 | sed 's/^/      prev: /'
  fi
  echo
  echo "REVERTING so the cluster can schedule GPU pods again."
  kubectl -n "${NS}" rollout undo deploy/"${DEP}" >/dev/null 2>&1
  kubectl -n "${NS}" scale deploy/"${DEP}" --replicas=1 >/dev/null 2>&1
  sleep 10
  kubectl -n "${NS}" get pods -l app.kubernetes.io/component=hami-scheduler 2>&1     | sed 's/^/      /'
  exit 1
fi
echo "  scheduler is Ready"

echo
echo "=== after"
kubectl -n "${NS}" get deploy "${DEP}" -o json 2>/dev/null \
  | python3 -c '
import json, sys
d = json.load(sys.stdin)
for c in d["spec"]["template"]["spec"]["containers"]:
    hits = [a for a in (c.get("args") or []) if "scheduler-policy" in a]
    if hits:
        print("  container %-28s %s" % (c["name"], ", ".join(hits)))
'

cat <<'EOF'

VERIFY IT ACTUALLY CHANGED PLACEMENT -- the flag being present is not the
same as HAMi honouring it. Deploy the two T1 tenants and look at the cards:

    python3 render.py --scenario llm-embed --arm hami --out .binpack
    kubectl apply -f .binpack
    kubectl wait --for=condition=Ready pod -l kserve.io/component=workload \
      -n llm-demo --timeout=900s
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

One device holding both models means co-location works and T1 can run.
Two devices holding one each means binpack did not take, and the fallback is
to render every tenant as a plain Deployment, where the pod template is ours
and nvidia.com/use-gpuuuid definitely lands.

    kubectl delete -f .binpack && rm -rf .binpack
EOF
