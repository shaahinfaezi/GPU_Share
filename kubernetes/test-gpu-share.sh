#!/bin/bash
# Test script to verify MPS and VRAM Shim are working on Kubernetes
# Usage: ./test-gpu-share.sh

set -e

NAMESPACE="gpu-share"

echo "=============================================="
echo " GPU Share Test Suite"
echo "=============================================="

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

pass() { echo -e "${GREEN}✓ PASS${NC}: $1"; }
fail() { echo -e "${RED}✗ FAIL${NC}: $1"; }
warn() { echo -e "${YELLOW}⚠ WARN${NC}: $1"; }

echo ""
echo "=== Test 1: Check VRAM Shim Installation ==="
SHIM_POD=$(kubectl get pod -n $NAMESPACE -l app=vram-shim-installer -o name 2>/dev/null | head -1)
if [ -z "$SHIM_POD" ]; then
    fail "No vram-shim-installer pod found"
else
    # Check if shim file exists
    if kubectl exec -n $NAMESPACE $SHIM_POD -- ls -la /opt/vram-shim/vram_limiter_runtime.so 2>/dev/null; then
        pass "VRAM shim .so file exists"
    else
        fail "VRAM shim .so file NOT found"
    fi
fi

echo ""
echo "=== Test 2: Check MPS Daemon Status ==="
MPS_POD=$(kubectl get pod -n $NAMESPACE -l app=nvidia-mps-daemon -o name 2>/dev/null | head -1)
if [ -z "$MPS_POD" ]; then
    fail "No nvidia-mps-daemon pod found"
else
    # Check MPS daemon logs
    MPS_LOGS=$(kubectl logs -n $NAMESPACE $MPS_POD --tail=20 2>/dev/null)
    if echo "$MPS_LOGS" | grep -q "MPS control daemon"; then
        pass "MPS control daemon is running"
    else
        warn "MPS daemon may not be fully started. Check logs:"
        echo "$MPS_LOGS"
    fi
    
    # Check if MPS pipe directory exists
    if kubectl exec -n $NAMESPACE $MPS_POD -- ls -la /tmp/nvidia-mps 2>/dev/null; then
        pass "MPS pipe directory exists"
    else
        fail "MPS pipe directory NOT found"
    fi
fi

echo ""
echo "=== Test 3: Check GPU Compute Mode ==="
GPU_MODE=$(kubectl exec -n $NAMESPACE $MPS_POD -- nvidia-smi --query-gpu=compute_mode --format=csv,noheader 2>/dev/null | head -1)
if [ "$GPU_MODE" = "Exclusive_Process" ]; then
    pass "GPU is in EXCLUSIVE_PROCESS mode (required for MPS)"
else
    warn "GPU compute mode is '$GPU_MODE', should be 'Exclusive_Process' for MPS"
fi

echo ""
echo "=== Test 4: Deploy Test Pod with Shim + MPS ==="
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: gpu-share-test
  namespace: $NAMESPACE
spec:
  restartPolicy: Never
  containers:
  - name: test
    image: nvidia/cuda:12.1.0-base-ubuntu22.04
    command: ["sleep", "300"]
    env:
    - name: LD_PRELOAD
      value: "/opt/vram-shim/vram_limiter_runtime.so"
    - name: VRAM_LIMIT_MB
      value: "1000"
    - name: CUDA_MPS_PIPE_DIRECTORY
      value: "/tmp/nvidia-mps"
    volumeMounts:
    - name: vram-shim
      mountPath: /opt/vram-shim
      readOnly: true
    - name: mps-pipe
      mountPath: /tmp/nvidia-mps
    resources:
      limits:
        nvidia.com/gpu: "1"
  volumes:
  - name: vram-shim
    hostPath:
      path: /opt/vram-shim
  - name: mps-pipe
    hostPath:
      path: /tmp/nvidia-mps
  nodeSelector:
    nvidia.com/gpu.present: "true"
EOF

echo "Waiting for test pod to be ready..."
kubectl wait --for=condition=Ready pod/gpu-share-test -n $NAMESPACE --timeout=60s || true

echo ""
echo "=== Test 5: Verify Shim is Loaded in Test Pod ==="
# Check if LD_PRELOAD is set correctly
LD_CHECK=$(kubectl exec -n $NAMESPACE gpu-share-test -- bash -c 'echo $LD_PRELOAD' 2>/dev/null)
if echo "$LD_CHECK" | grep -q "vram_limiter"; then
    pass "LD_PRELOAD is set correctly: $LD_CHECK"
else
    fail "LD_PRELOAD is NOT set correctly"
fi

# Check if shim file is accessible
if kubectl exec -n $NAMESPACE gpu-share-test -- ls -la /opt/vram-shim/vram_limiter_runtime.so 2>/dev/null; then
    pass "Shim file is accessible from test pod"
else
    fail "Shim file NOT accessible from test pod"
fi

echo ""
echo "=== Test 6: Verify MPS Connection in Test Pod ==="
if kubectl exec -n $NAMESPACE gpu-share-test -- ls -la /tmp/nvidia-mps 2>/dev/null; then
    pass "MPS pipe directory is accessible from test pod"
else
    fail "MPS pipe directory NOT accessible from test pod"
fi

# Check MPS env var
MPS_CHECK=$(kubectl exec -n $NAMESPACE gpu-share-test -- bash -c 'echo $CUDA_MPS_PIPE_DIRECTORY' 2>/dev/null)
if [ "$MPS_CHECK" = "/tmp/nvidia-mps" ]; then
    pass "CUDA_MPS_PIPE_DIRECTORY is set correctly"
else
    fail "CUDA_MPS_PIPE_DIRECTORY is NOT set correctly"
fi

echo ""
echo "=== Test 7: GPU Memory Test (optional - requires Python) ==="
kubectl exec -n $NAMESPACE gpu-share-test -- bash -c '
if command -v python3 &>/dev/null; then
    python3 -c "
import os
print(f\"VRAM_LIMIT_MB: {os.environ.get('VRAM_LIMIT_MB', 'NOT SET')}\")
print(f\"LD_PRELOAD: {os.environ.get('LD_PRELOAD', 'NOT SET')}\")
print(f\"CUDA_MPS_PIPE_DIRECTORY: {os.environ.get('CUDA_MPS_PIPE_DIRECTORY', 'NOT SET')}\")
"
else
    echo "Python not available in test image"
fi
' 2>/dev/null || true

echo ""
echo "=== Cleanup ==="
read -p "Delete test pod? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    kubectl delete pod gpu-share-test -n $NAMESPACE --force --grace-period=0 2>/dev/null || true
    pass "Test pod deleted"
fi

echo ""
echo "=============================================="
echo " Test Suite Complete"
echo "=============================================="
echo ""
echo "If all tests passed, you can deploy KServe services:"
echo "  kubectl apply -f 10-kserve-inferenceservice.yaml"
echo "  kubectl apply -f 11-multi-model-sharing.yaml"
