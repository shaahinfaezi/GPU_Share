#!/usr/bin/env bash
# Does deviceCoreScaling do what we think, and what does it break?
#
#   ./29-corescaling.sh            # probe at scaling=2, then revert
#   ./29-corescaling.sh 3          # probe at scaling=3
#   ./29-corescaling.sh --revert   # put the node back and exit
#
# WHY THIS EXISTS
#
# `nvidia.com/gpucores` is a scheduler admission token. HAMi's Fit() rejects on
# `dev.Totalcore - dev.Usedcores < k.Coresreq`. Under the default policy it is
# never enforced at runtime, and under MPS it is void entirely -- both measured.
# So on a shared card the scheduler believes the card is full at sum(gpucores)
# = 100 while the runtime lets every tenant use everything. Our own tiers ran
# gpucores 50 + MPS 100: a 2x compute overbook that nothing in the manifest
# declares.
#
# deviceCoreScaling is the knob meant to fix that -- it raises the ADVERTISED
# core budget so the scheduler's model can match a deliberately overbooked
# runtime. register.go computes `devcore = DeviceCoreScaling * 100`.
#
# THE TRAP THIS SCRIPT IS REALLY FOR
#
# HAMi's exclusive-card path is gated on the unscaled total:
#
#     if dev.Totalcore == 100 && k.Coresreq == 100 && dev.Used > 0 {
#             reason[common.ExclusiveDeviceAllocateConflict]++
#
# With scaling > 1, Totalcore is 200 (or 300), that branch can never fire, and
# `gpucores: 100` silently stops meaning "give me the whole card". A pod that
# asked for exclusivity would quietly get a half share with a neighbour. That is
# a source reading and it needs a measurement, because if it is right it is a
# correctness bug for anyone using scaling and exclusivity in the same cluster.
#
# WHAT IT DOES
#   1. records the current ConfigMap and node registration
#   2. sets deviceCoreScaling, restarts the device plugin, waits for the node to
#      re-register, and PROVES the advertised total actually changed
#   3. probe A -- can two gpucores:100 pods now co-exist on one card?
#   4. probe B -- how many gpucores:50 pods fit, and do they land on one card?
#   5. reverts, and re-proves the revert took
#
# Nothing here runs a benchmark. It answers scheduling questions only, so it is
# cheap: no model loads, tiny pause containers.
set -uo pipefail
cd "$(dirname "$0")"

NS="${R2_NS:-llm-demo}"
CM_NS="${HAMI_NS:-kube-system}"
CM="${HAMI_CM:-hami-device-plugin}"
[ -n "${HAMI_CM:-}" ] && HAMI_CM_PINNED=1
NODE="${R2_NODE:-mr-gholami}"
DS="${HAMI_DS:-hami-device-plugin}"
SCALE="${1:-2}"
OUT="$(pwd)/results/corescaling-$(date +%Y%m%d-%H%M%S)"

say()  { printf '\n\033[1m=== %s\033[0m\n' "$1"; }
ok()   { printf '  \033[32mOK\033[0m    %s\n' "$1"; }
bad()  { printf '  \033[31mFAIL\033[0m  %s\n' "$1"; }
note() { printf '  %-22s %s\n' "$1" "$2"; }

# Advertised nvidia.com/gpu and the per-device devcore HAMi registered.
advertised() {
  kubectl get node "$NODE" -o jsonpath='{.status.allocatable.nvidia\.com/gpu}' 2>/dev/null
}
devcores() {
  kubectl get node "$NODE" \
    -o jsonpath='{.metadata.annotations.hami\.io/node-nvidia-register}' 2>/dev/null \
    | grep -oE '"devcore":[0-9]+' | head -4 | tr '\n' ' '
}

revert() {
  say "revert: deviceCoreScaling -> 1"
  kubectl -n "$CM_NS" get cm "$CM" -o yaml \
    | sed -E 's/"devicecorescaling"[[:space:]]*:[[:space:]]*[0-9]+/"devicecorescaling": 1/I' \
    | kubectl apply -f - >/dev/null 2>&1
  kubectl -n "$CM_NS" rollout restart ds "$DS" >/dev/null 2>&1
  kubectl -n "$CM_NS" rollout status ds "$DS" --timeout=180s >/dev/null 2>&1
  sleep 20
  note "devcore now" "$(devcores)"
  kubectl delete pod -n "$NS" -l r2-probe=corescaling --ignore-not-found >/dev/null 2>&1
}

if [ "${1:-}" = "--revert" ]; then revert; exit 0; fi

mkdir -p "$OUT"
trap 'echo; echo "interrupted -- reverting"; revert' INT TERM

say "before"
BEFORE_GPU=$(advertised); BEFORE_CORE=$(devcores)
note "nvidia.com/gpu" "${BEFORE_GPU:-?}"
note "registered devcore" "${BEFORE_CORE:-?}"
kubectl -n "$CM_NS" get cm "$CM" -o yaml > "$OUT/configmap-before.yaml" 2>/dev/null
kubectl get node "$NODE" -o json > "$OUT/node-before.json" 2>/dev/null
if ! echo "$BEFORE_CORE" | grep -q '"devcore":100'; then
  bad "node is not at the stock devcore of 100 -- refusing to change it further"
  note "found" "$BEFORE_CORE"
  exit 1
fi

# The previous attempt failed silently: the ConfigMap edit was piped into
# `kubectl apply` with stderr discarded, so a rejected patch looked exactly
# like a successful one and the only symptom was the node not re-registering.
# Show what is actually in there before touching it.
# Which ConfigMap does the plugin ACTUALLY read? The DaemonSet mounts more
# than one. register.go reads plugin.schedulerConfig.DeviceCoreScaling -- the
# SCHEDULER device config, not the plugin's own -- so patching
# hami-device-plugin changed a real key in the wrong object and the node never
# re-registered. Pick the mounted ConfigMap that carries the key, preferring
# the scheduler one, unless the caller pinned HAMI_CM.
if [ -z "${HAMI_CM_PINNED:-}" ]; then
  MOUNTED=$(kubectl -n "$CM_NS" get ds "$DS" -o jsonpath='{.spec.template.spec.volumes[*].configMap.name}' 2>/dev/null)
  for CAND in $(echo "$MOUNTED" | tr ' ' '
' | grep -i scheduler) $(echo "$MOUNTED" | tr ' ' '
' | grep -vi scheduler); do
    if kubectl -n "$CM_NS" get cm "$CAND" -o yaml 2>/dev/null | grep -qi devicecorescaling; then
      CM="$CAND"; break
    fi
  done
fi
note "using configmap" "$CM"
note "all mounted" "${MOUNTED:-?}"

say "what the device plugin is actually reading"
note "scaling keys" "$(kubectl -n "$CM_NS" get cm "$CM" -o yaml 2>/dev/null | grep -oiE 'device(memory|core)scaling[^,}]*' | tr -d '
' | paste -sd' ' - | head -c 160)"
note "plugin args" "$(kubectl -n "$CM_NS" get ds "$DS" -o jsonpath='{.spec.template.spec.containers[*].args}' 2>/dev/null | head -c 200)"
note "mounted cms" "$(kubectl -n "$CM_NS" get ds "$DS" -o jsonpath='{.spec.template.spec.volumes[*].configMap.name}' 2>/dev/null)"
echo "  If the plugin reads a DIFFERENT ConfigMap or a --config-file flag,"
echo "  set HAMI_CM to that name and re-run. Editing the wrong object is"
echo "  exactly how the last attempt produced a silent no-op."

say "applying deviceCoreScaling=$SCALE"
CUR=$(kubectl -n "$CM_NS" get cm "$CM" -o yaml)
if echo "$CUR" | grep -qi '"devicecorescaling"'; then
  NEWCM=$(echo "$CUR" | sed -E "s/\"devicecorescaling\"[[:space:]]*:[[:space:]]*[0-9.]+/\"devicecorescaling\": $SCALE/I")
  HOW="replaced existing key"
else
  NEWCM=$(echo "$CUR" | sed -E "s/(\"devicesplitcount\"[[:space:]]*:[[:space:]]*[0-9]+,)/\1 \"devicecorescaling\": $SCALE,/I")
  HOW="inserted new key"
fi
APPLY_OUT=$(echo "$NEWCM" | kubectl apply -f - 2>&1)
note "$HOW" "$APPLY_OUT"
# Prove the OBJECT changed before blaming the plugin for not reacting.
if kubectl -n "$CM_NS" get cm "$CM" -o yaml | grep -qiE "devicecorescaling\"?[[:space:]]*:[[:space:]]*$SCALE"; then
  ok "the ConfigMap now contains devicecorescaling=$SCALE"
else
  bad "the ConfigMap does NOT contain devicecorescaling=$SCALE."
  bad "The edit did not take, so nothing below would mean anything."
  note "current" "$(kubectl -n "$CM_NS" get cm "$CM" -o yaml | grep -oiE 'devicecorescaling[^,}]*' | head -1)"
  revert; exit 1
fi
kubectl -n "$CM_NS" rollout restart ds "$DS" >/dev/null 2>&1
kubectl -n "$CM_NS" rollout status ds "$DS" --timeout=180s >/dev/null 2>&1
sleep 25

AFTER_CORE=$(devcores)
note "registered devcore" "${AFTER_CORE:-?}"
WANT=$((SCALE * 100))
if echo "$AFTER_CORE" | grep -q "\"devcore\":$WANT"; then
  ok "the node re-registered at devcore=$WANT -- scaling is live"
else
  bad "node did NOT re-register at devcore=$WANT; every probe below is void"
  note "found" "$AFTER_CORE"
  revert; exit 1
fi
kubectl get node "$NODE" -o json > "$OUT/node-after.json" 2>/dev/null

mkprobe() {  # name, cores
  cat <<YAML | kubectl apply -f - >/dev/null 2>&1
apiVersion: v1
kind: Pod
metadata:
  name: $1
  namespace: $NS
  labels: {r2-probe: corescaling}
spec:
  restartPolicy: Never
  containers:
    - name: p
      image: $(kubectl get pods -n "$NS" -o jsonpath='{.items[0].spec.containers[0].image}' 2>/dev/null || echo busybox)
      command: ["sh","-c","sleep 900"]
      resources:
        limits:
          nvidia.com/gpu: "1"
          nvidia.com/gpumem: "1024"
          nvidia.com/gpucores: "$2"
YAML
}

phase() { kubectl get pod -n "$NS" "$1" -o jsonpath='{.status.phase}' 2>/dev/null; }
dev_of() {
  kubectl get pod -n "$NS" "$1" -o jsonpath='{.metadata.annotations}' 2>/dev/null \
    | grep -oE 'GPU-[0-9a-f-]{36}' | head -1
}

# ---- probe A: does gpucores:100 still mean EXCLUSIVE? --------------------
say "probe A -- two gpucores:100 pods"
echo "  The exclusivity branch is gated on Totalcore == 100. At scaling=$SCALE"
echo "  it cannot fire, so the SECOND pod is predicted to schedule -- which"
echo "  would mean gpucores:100 no longer reserves a card."
mkprobe r2-excl-a 100; sleep 12
mkprobe r2-excl-b 100; sleep 25
PA=$(phase r2-excl-a); PB=$(phase r2-excl-b)
DA=$(dev_of r2-excl-a); DB=$(dev_of r2-excl-b)
note "excl-a" "${PA:-?}  ${DA:-no device}"
note "excl-b" "${PB:-?}  ${DB:-no device}"
if [ "$PB" = "Pending" ]; then
  ok "exclusivity SURVIVES scaling -- the second pod was refused"
  VERDICT_A="exclusivity-holds"
elif [ -n "$DA" ] && [ "$DA" = "$DB" ]; then
  bad "EXCLUSIVITY IS BROKEN -- two gpucores:100 pods share $DA"
  VERDICT_A="exclusivity-broken-same-card"
else
  note "verdict" "both ran but on different cards -- inconclusive on this node"
  VERDICT_A="inconclusive-different-cards"
fi
kubectl describe pod -n "$NS" r2-excl-b > "$OUT/probe-a-excl-b.txt" 2>&1
kubectl delete pod -n "$NS" -l r2-probe=corescaling --ignore-not-found >/dev/null 2>&1
sleep 15

# ---- probe B: how many gpucores:50 pods fit on one card? -----------------
say "probe B -- four gpucores:50 pods"
echo "  At scaling=1 a card takes two. At scaling=$SCALE the budget is $WANT,"
echo "  so $((WANT / 50)) should fit per card if the scaling is honoured by Fit()."
for i in 1 2 3 4; do mkprobe "r2-half-$i" 50; sleep 8; done
sleep 25
RUNNING=0; PENDING=0
for i in 1 2 3 4; do
  P=$(phase "r2-half-$i"); D=$(dev_of "r2-half-$i")
  note "half-$i" "${P:-?}  ${D:-no device}"
  [ "$P" = "Running" ] && RUNNING=$((RUNNING+1))
  [ "$P" = "Pending" ] && PENDING=$((PENDING+1))
done
note "scheduled" "$RUNNING running, $PENDING pending"
kubectl get pods -n "$NS" -l r2-probe=corescaling -o json > "$OUT/probe-b-pods.json" 2>&1

{
  echo "{"
  echo "  \"scaling\": $SCALE,"
  echo "  \"devcore_before\": \"$BEFORE_CORE\","
  echo "  \"devcore_after\": \"$AFTER_CORE\","
  echo "  \"probe_a_verdict\": \"$VERDICT_A\","
  echo "  \"probe_a_phases\": {\"excl_a\": \"$PA\", \"excl_b\": \"$PB\"},"
  echo "  \"probe_a_devices\": {\"excl_a\": \"$DA\", \"excl_b\": \"$DB\"},"
  echo "  \"probe_b_running\": $RUNNING,"
  echo "  \"probe_b_pending\": $PENDING"
  echo "}"
} > "$OUT/corescaling.json"

revert
say "done"
note "results" "$OUT"
echo
echo "  Send $OUT/corescaling.json plus the probe files. The two answers that"
echo "  matter: whether gpucores:100 still reserves a card under scaling, and"
echo "  whether the raised budget actually packs more tenants onto one device."
