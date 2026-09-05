#!/usr/bin/env bash
# Does HAMi refuse a gpucores sum greater than 100?
#
#   chmod +x 26-gpucores-probe.sh
#   ./26-gpucores-probe.sh            # default: llm=80 + llm2=80 = 160
#   ./26-gpucores-probe.sh 60 60      # any pair you like
#
# WHY THIS EXISTS
#
# Every scenario in the matrix sets gpucores to 100/len(scenario), so the
# co-tenants always sum to exactly 100 and HAMi has never been offered more.
# hami_gpu_core_allocated_ratio duly reads 100 in every shared arm and 0 in
# every dedicated one. So we know what HAMi reports at the boundary and nothing
# about what it does past it.
#
# That matters because T5 established what gpucores is NOT. With gpucores=50
# the 7B still produced 626.27 tok/s against 728.79 solo -- 85.9%. A value that
# reserved or capped half the card could not do that. So gpucores does not
# bound compute at runtime on this build, and its only remaining job is
# admission: how many pods HAMi will place on one device.
#
# Which leaves exactly one question, and it decides how a fleet is configured:
#
#   REFUSED   gpucores is a hard admission capacity. Tenants per card is
#             100/value, you tune density by LOWERING the per-pod number, and
#             the operator has a real placement control.
#
#   ADMITTED  gpucores is advisory in both directions -- it neither bounds
#             compute nor limits placement. Density is then controlled only by
#             gpumem, and the number is decoration. This is the worse answer
#             and it changes the deployment guidance.
#
# The probe watches SCHEDULING, not readiness. Whether a pod eventually serves
# takes minutes of weight loading and is irrelevant here; whether the scheduler
# will place it is decided in seconds.
set -uo pipefail
cd "$(dirname "$0")"

A="${1:-80}"
B="${2:-80}"
SUM=$((A + B))
NS="${R2_NS:-llm-demo}"
OUT="$(pwd)/results/gpucores-probe-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUT"

say()  { printf '\n\033[1m=== %s\033[0m\n' "$1"; }
ok()   { printf '  \033[32mOK\033[0m    %s\n' "$1"; }
bad()  { printf '  \033[31m--\033[0m    %s\n' "$1"; }

say "probe: llm=${A} + llm2=${B} = ${SUM}"
echo "  out $OUT"
if [ "$SUM" -le 100 ]; then
  echo "  NOTE: ${SUM} <= 100, so this is a control run, not the real question."
fi

say "1. clean slate"
kubectl -n "$NS" delete llminferenceservice --all --ignore-not-found --wait=false >/dev/null 2>&1
for i in $(seq 1 40); do
  N=$(kubectl -n "$NS" get pods -l r2-role -o name 2>/dev/null | wc -l)
  [ "$N" = "0" ] && break
  sleep 3
done
USED=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null | tr '\n' ' ')
ok "gpu memory.used: ${USED:-unknown}"

say "2. render the hami arm with the sum we want to test"
python3 render.py --scenario llm-llm --arm hami --out "$OUT/rendered" \
        --gpucores "llm=${A},llm2=${B}" 2>&1 | sed 's/^/  /'
grep -h "gpucores" "$OUT"/rendered/*.yaml | sort -u | sed 's/^/  /'

say "3. apply"
kubectl apply -f "$OUT/rendered" 2>&1 | sed 's/^/  /'

say "4. did the scheduler place them?"
# Pods appear a little after the LLMInferenceService; give them time to exist
# before deciding anything about whether they were scheduled.
PODS=""
for i in $(seq 1 30); do
  PODS=$(kubectl -n "$NS" get pods -l r2-role -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}' 2>/dev/null)
  [ "$(echo "$PODS" | grep -c .)" -ge 2 ] && break
  sleep 3
done
echo "  pods: $(echo "$PODS" | tr '\n' ' ')"

SCHEDULED=0; PENDING=0
for i in $(seq 1 40); do
  SCHEDULED=0; PENDING=0
  for P in $PODS; do
    [ -z "$P" ] && continue
    NODE=$(kubectl -n "$NS" get pod "$P" -o jsonpath='{.spec.nodeName}' 2>/dev/null)
    if [ -n "$NODE" ]; then SCHEDULED=$((SCHEDULED + 1)); else PENDING=$((PENDING + 1)); fi
  done
  [ "$PENDING" = "0" ] && break
  sleep 3
done
kubectl -n "$NS" get pods -l r2-role -o wide 2>&1 | sed 's/^/  /'

say "5. what the scheduler said"
kubectl -n "$NS" get events --sort-by=.lastTimestamp 2>/dev/null \
  | grep -Ei "FailedScheduling|FilteringFailed|Insufficient|unregistered" \
  | tail -12 | sed 's/^/  /' || echo "  (no scheduling failures recorded)"

ALLOC=$(kubectl get node -o jsonpath='{.items[0].metadata.annotations.hami\.io/node-nvidia-register}' 2>/dev/null | head -c 200)
[ -n "$ALLOC" ] && echo "  hami registration (first 200 bytes): $ALLOC"

say "VERDICT  (llm=${A} + llm2=${B} = ${SUM})"
{
  echo "{"
  echo "  \"a\": ${A}, \"b\": ${B}, \"sum\": ${SUM},"
  echo "  \"scheduled\": ${SCHEDULED}, \"pending\": ${PENDING},"
  echo "  \"admitted\": $([ "$PENDING" = "0" ] && echo true || echo false)"
  echo "}"
} > "$OUT/verdict.json"

if [ "$PENDING" = "0" ] && [ "$SCHEDULED" -ge 2 ]; then
  bad "ADMITTED -- both pods scheduled with a gpucores sum of ${SUM}."
  echo "        HAMi did not treat gpucores as an admission capacity. Combined"
  echo "        with T5 (gpucores=50 returned 85.9% of solo throughput), the"
  echo "        value bounds NOTHING on this build -- not compute, not"
  echo "        placement. Density is controlled by gpumem alone, and the"
  echo "        deployment guidance has to say so."
else
  ok "REFUSED -- ${PENDING} pod(s) unscheduled at a sum of ${SUM}."
  echo "        gpucores IS a hard admission capacity. Tenants per card is"
  echo "        100/value: 50 -> 2 tenants, 25 -> 4, 10 -> 10. Density is tuned"
  echo "        by LOWERING the per-pod value, never by oversubscribing the sum."
  echo "        Check the events above to confirm the refusal names cores and"
  echo "        not memory -- at ${A}/${B} the VRAM (8192+3840=12032) fits, so"
  echo "        a memory refusal would mean something else is resident."
fi

say "6. teardown"
kubectl -n "$NS" delete llminferenceservice --all --ignore-not-found --wait=false >/dev/null 2>&1
echo "  verdict saved: $OUT/verdict.json"
