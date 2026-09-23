#!/usr/bin/env bash
# GATES. Run this to completion before starting the series.
#
# The series below is ~2.5 hours. Six things can make it produce nothing, and
# five of them are checkable in fifteen minutes. This script checks them and
# refuses to pass if any fails, so the failure costs minutes instead of an
# evening. It measures; it does not guess.
#
#   1. two cards, both idle
#   2. HAMi present, binpack -- with spread, "co-tenants" land on separate
#      cards and the sharing arm becomes an uncontended arm producing a clean
#      and entirely meaningless result. Measured three times in round two.
#   3. the PVC and the image exist
#   4. HAMi honours nvidia.com/gpu:2 WITH a gpumem quota, applying the quota
#      PER DEVICE. Assumed everywhere below. Never measured on this node --
#      every round-two tenant asked for exactly one card.
#   5. NCCL COMPLETES UNDER MPS. This is the real risk. Tensor parallelism
#      all-reduces across ranks every layer; under MPS both ranks are clients
#      of one daemon. If that deadlocks, arm 4 hangs for hours and reports
#      nothing. Round one left a 91-nccl-tp-test.yaml behind precisely because
#      this was already suspected.
#   6. host RAM. 15,988 MiB, no swap, and round two measured ~3.1 GiB of RSS
#      per vLLM tenant that did NOT scale with model size. Four ranks is the
#      first time this box has been asked for four. The solo stage measures
#      the real per-rank figure; this gate only refuses an already-full box.
#
# Foreground on purpose -- you want to read it.
#
#   cd /opt/GPU_Share/round2/split && chmod +x *.sh
#   ./51-split-preflight.sh 2>&1 | tee logs/preflight-$(date -u +%Y%m%d-%H%M%S).log
set -uo pipefail
cd "$(dirname "$0")"
exec 9>.split-run.lock
flock -n 9 || { echo "Another split stage is running" >&2; exit 2; }

NS="${R2_NS:-llm-demo}"
PY=python3
FAIL=0
export PYTHONPATH="$(cd .. && pwd):${PYTHONPATH:-}"
mkdir -p logs
# Do not let the EXIT cleanup own a deployment started elsewhere.
${PY} - <<'EOF'
import harness as H
if H.pods_for('tp') or H.pods_for('pp'):
    raise SystemExit('Existing split pods found; inspect before preflight')
if not H.wait_gpu_idle(timeout=30):
    raise SystemExit('GPUs are not idle')
ok, detail = H.set_mps_daemon(False)
if not ok:
    raise SystemExit('Cannot disable MPS: %s' % detail)
EOF
[ $? -eq 0 ] || exit 2

ok()   { echo "  PASS  $*"; }
bad()  { echo "  FAIL  $*"; FAIL=1; }
note() { echo "  ..    $*"; }
head_() { echo; echo "=== $* "; }

cleanup() {
  kubectl delete llminferenceservice -n "${NS}" -l r2-series=split \
      --ignore-not-found --timeout=120s >/dev/null 2>&1
  ${PY} - <<'EOF' >/dev/null 2>&1 || true
import sys, os
sys.path.insert(0, os.path.dirname(os.getcwd()))
import harness as H
H.wait_gpu_idle(timeout=180)
EOF
}
finish() {
  cleanup
  ${PY} - <<'EOF'
import harness as H
ok, detail = H.set_mps_daemon(False)
if not ok:
    raise SystemExit('Cannot restore MPS off: %s' % detail)
EOF
}
trap finish EXIT

# ------------------------------------------------------------------ 1. cards
head_ "1. cards"
CARDS=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
[ "${CARDS}" -ge 2 ] && ok "${CARDS} cards visible" \
                     || bad "need 2 cards, found ${CARDS}"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader \
  | while IFS=, read -r i used total; do
      u=$(echo "${used}" | tr -dc '0-9')
      if [ "${u:-0}" -gt 500 ]; then
        echo "  WARN  card ${i} already holds ${used} -- teardown first"
      else
        echo "  PASS  card ${i} idle (${used} of ${total})"
      fi
    done

# ------------------------------------------------------------------- 2. HAMi
head_ "2. HAMi scheduler"
if kubectl get pods -A 2>/dev/null | grep -q hami; then
  ok "HAMi pods present"
else
  bad "no HAMi pods -- this whole series is a HAMi test"
fi
POL=$(kubectl get cm -n kube-system hami-scheduler-device -o yaml 2>/dev/null \
      | grep -i "gpuSchedulerPolicy" | head -1)
case "${POL}" in
  *binpack*) ok "gpuSchedulerPolicy=binpack" ;;
  "")        echo "  WARN  could not read gpuSchedulerPolicy; run ../10-hami-binpack.sh" ;;
  *)         bad "policy is not binpack (${POL}) -- co-tenants will land on separate cards" ;;
esac

# --------------------------------------------------------- 3. PVC and image
head_ "3. model and image"
if kubectl get pvc -n "${NS}" qwen-local-pv-claim >/dev/null 2>&1; then
  ok "PVC qwen-local-pv-claim exists"
else
  bad "PVC qwen-local-pv-claim missing -- both manifests mount it"
fi
if curl -sf http://127.0.0.1:5000/v2/kserve/huggingfaceserver/tags/list \
     | grep -q "v0.17.0-gpu"; then
  ok "huggingfaceserver:v0.17.0-gpu in the local registry"
else
  echo "  WARN  could not confirm the image tag; if the pods ImagePullBackOff, that is why"
fi

# ------------------------------------- 4. does HAMi honour a 2-GPU request?
[ "${FAIL}" -eq 0 ] || exit 1
head_ "4. HAMi multi-device: nvidia.com/gpu=2 with a gpumem quota"
note "applying the tp tenant alone on the hami arm"
rm -rf rendered-preflight && mkdir -p rendered-preflight
${PY} 50-render-split.py --arm hami --only tp --out rendered-preflight >/dev/null
kubectl apply -f rendered-preflight/ >/dev/null 2>&1

READY=$(${PY} - <<'EOF'
import sys, os
sys.path.insert(0, os.path.dirname(os.getcwd()))
import harness as H
ready, diag = H.wait_ready(["tp"], timeout=600)
print("yes" if ready else "no: %s" % diag)
EOF
)
case "${READY}" in
  yes) ok "tp tenant Ready with nvidia.com/gpu=2 + gpumem=4096" ;;
  *)   bad "tp tenant never became Ready -- ${READY}" ;;
esac

if [ "${READY}" = "yes" ]; then
  echo "  --    per-card VRAM with ONE 2-way-split tenant resident:"
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader \
    | sed 's/^/        card /'
  USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -rn | head -1)
  ON=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
       | awk '$1>500' | wc -l)
  [ "${ON}" -eq 2 ] && ok "the tenant is resident on BOTH cards (split confirmed)" \
                    || bad "tenant occupies ${ON} card(s), expected 2 -- TP did not split"
  if [ "${USED:-0}" -le 5200 ]; then
    ok "peak card usage ${USED} MiB <= the 4096 quota + context"
  else
    bad "peak card usage ${USED} MiB -- the gpumem quota is NOT being applied per device"
  fi
fi
cleanup

# ------------------------------------------------ 5. NCCL under MPS (the risk)
head_ "5. NCCL all-reduce under MPS"
note "starting the MPS daemon, then the same tenant on the hami-mps arm"
${PY} - <<'EOF'
import sys, os
sys.path.insert(0, os.path.dirname(os.getcwd()))
import harness as H
ok, detail = H.set_mps_daemon(True)
if not ok:
    raise SystemExit('Cannot enable MPS: %s' % detail)
EOF
[ $? -eq 0 ] || exit 2
rm -rf rendered-preflight && mkdir -p rendered-preflight
${PY} 50-render-split.py --arm hami-mps --only tp --out rendered-preflight >/dev/null
kubectl apply -f rendered-preflight/ >/dev/null 2>&1

MPSREADY=$(${PY} - <<'EOF'
import sys, os
sys.path.insert(0, os.path.dirname(os.getcwd()))
import harness as H
try:
    ready, diag = H.wait_ready(["tp"], timeout=600)
    if not ready:
        raise RuntimeError(diag)
    eps = H.endpoints(["tp"])
    m = H.served_model(eps["tp"]) or "Qwen/Qwen2.5-0.5B-Instruct"
    ok, detail = H.assert_mps_matches_arm('hami-mps', ['tp'])
    if not ok:
        raise RuntimeError(detail)
    ok, detail = H.smoke_test(eps, {"tp": m}, {"tp": "chat"}, timeout=120)
    if not ok:
        raise RuntimeError(detail)
    print("yes")
except Exception as e:
    print("no: %s" % e)
EOF
)
case "${MPSREADY}" in
  yes) ok "tensor-parallel vLLM served a request with MPS in the path"
       ok "single-tenant MPS smoke passed; four-rank memory admission remains separate" ;;
  *)   bad "TP under MPS did not serve -- ${MPSREADY}"
       echo "        THIS IS THE ONE THAT MATTERS. If it failed, do NOT run"
       echo "        53-split-share.sh's hami-mps stage until the cause is resolved."
       echo "        Save pod/controller events and logs before attributing the cause."
       echo "        A startup failure alone does not establish NCCL incompatibility." ;;
esac
cleanup
${PY} - <<'EOF'
import sys, os
sys.path.insert(0, os.path.dirname(os.getcwd()))
import harness as H
H.set_mps_daemon(False)
EOF

# -------------------------------------------------------------- 6. host RAM
head_ "6. host memory headroom"
${PY} - <<'EOF'
import sys, os
sys.path.insert(0, os.path.dirname(os.getcwd()))
import harness as H
m = H.host_mem_mib()
avail = m.get("available") or m.get("MemAvailable") or 0
print("  ..    %d MiB available" % avail)
# Four ranks. Round two measured ~3.1 GiB per vLLM tenant, framework-dominated
# and NOT proportional to model size, so 0.5B buys less relief than it looks
# like it should. Ranks fork and share pages, so 4x is the pessimistic bound.
need = 4 * 2400
if avail >= need:
    print("  PASS  %d MiB >= %d MiB estimated for 4 ranks" % (avail, need))
else:
    print("  WARN  %d MiB < %d MiB rough estimate for four ranks" % (avail, need))
    print("        Solo is permitted to MEASURE one tenant at a time. Sharing is")
    print("        not admitted here; stage 2 checks the measured peak host deltas.")
    if avail < 1024:
        print('  FAIL  Less than 1024 MiB host headroom')
        sys.exit(1)
EOF
[ $? -ne 0 ] && FAIL=1

# ------------------------------------------------------------------ verdict
echo
if [ "${FAIL}" -eq 0 ]; then
  echo "=== PREFLIGHT PASSED. Start the series:"
  echo "===   nohup ./52-split-solo.sh  > logs/solo.log  2>&1 &"
else
  echo "=== PREFLIGHT FAILED. Fix the FAIL lines above before running the series."
fi
exit "${FAIL}"
