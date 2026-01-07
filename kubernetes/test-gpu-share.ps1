# Test script to verify MPS and VRAM Shim are working on Kubernetes
# Usage: .\test-gpu-share.ps1

$ErrorActionPreference = "Continue"
$Namespace = "gpu-share"

Write-Host "==============================================" -ForegroundColor Cyan
Write-Host " GPU Share Test Suite" -ForegroundColor Cyan
Write-Host "==============================================" -ForegroundColor Cyan

function Pass { param($msg) Write-Host "✓ PASS: $msg" -ForegroundColor Green }
function Fail { param($msg) Write-Host "✗ FAIL: $msg" -ForegroundColor Red }
function Warn { param($msg) Write-Host "⚠ WARN: $msg" -ForegroundColor Yellow }

Write-Host ""
Write-Host "=== Test 1: Check VRAM Shim Installation ===" -ForegroundColor Cyan
$ShimPod = kubectl get pod -n $Namespace -l app=vram-shim-installer -o name 2>$null | Select-Object -First 1
if (-not $ShimPod) {
    Fail "No vram-shim-installer pod found"
} else {
    $ShimCheck = kubectl exec -n $Namespace $ShimPod -- ls -la /opt/vram-shim/vram_limiter_runtime.so 2>&1
    if ($LASTEXITCODE -eq 0) {
        Pass "VRAM shim .so file exists"
        Write-Host $ShimCheck -ForegroundColor Gray
    } else {
        Fail "VRAM shim .so file NOT found"
    }
}

Write-Host ""
Write-Host "=== Test 2: Check MPS Daemon Status ===" -ForegroundColor Cyan
$MpsPod = kubectl get pod -n $Namespace -l app=nvidia-mps-daemon -o name 2>$null | Select-Object -First 1
if (-not $MpsPod) {
    Fail "No nvidia-mps-daemon pod found"
} else {
    $MpsLogs = kubectl logs -n $Namespace $MpsPod --tail=20 2>&1
    if ($MpsLogs -match "MPS") {
        Pass "MPS daemon is running"
    } else {
        Warn "MPS daemon may not be fully started"
        Write-Host $MpsLogs -ForegroundColor Gray
    }
}

Write-Host ""
Write-Host "=== Test 3: Check GPU Compute Mode ===" -ForegroundColor Cyan
if ($MpsPod) {
    $GpuMode = kubectl exec -n $Namespace $MpsPod -- nvidia-smi --query-gpu=compute_mode --format=csv,noheader 2>&1
    if ($GpuMode -match "Exclusive") {
        Pass "GPU is in EXCLUSIVE_PROCESS mode (required for MPS)"
    } else {
        Warn "GPU compute mode is '$GpuMode', should be 'Exclusive_Process' for MPS"
    }
}

Write-Host ""
Write-Host "=== Test 4: Deploy Test Pod with Shim + MPS ===" -ForegroundColor Cyan

$TestPodYaml = @"
apiVersion: v1
kind: Pod
metadata:
  name: gpu-share-test
  namespace: $Namespace
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
"@

$TestPodYaml | kubectl apply -f - 2>&1

Write-Host "Waiting for test pod to be ready..."
kubectl wait --for=condition=Ready pod/gpu-share-test -n $Namespace --timeout=60s 2>$null

Write-Host ""
Write-Host "=== Test 5: Verify Shim is Loaded in Test Pod ===" -ForegroundColor Cyan
$LdCheck = kubectl exec -n $Namespace gpu-share-test -- bash -c 'echo $LD_PRELOAD' 2>&1
if ($LdCheck -match "vram_limiter") {
    Pass "LD_PRELOAD is set correctly: $LdCheck"
} else {
    Fail "LD_PRELOAD is NOT set correctly"
}

$ShimAccess = kubectl exec -n $Namespace gpu-share-test -- ls -la /opt/vram-shim/vram_limiter_runtime.so 2>&1
if ($LASTEXITCODE -eq 0) {
    Pass "Shim file is accessible from test pod"
} else {
    Fail "Shim file NOT accessible from test pod"
}

Write-Host ""
Write-Host "=== Test 6: Verify MPS Connection in Test Pod ===" -ForegroundColor Cyan
$MpsPipeCheck = kubectl exec -n $Namespace gpu-share-test -- ls -la /tmp/nvidia-mps 2>&1
if ($LASTEXITCODE -eq 0) {
    Pass "MPS pipe directory is accessible from test pod"
} else {
    Fail "MPS pipe directory NOT accessible from test pod"
}

$MpsEnvCheck = kubectl exec -n $Namespace gpu-share-test -- bash -c 'echo $CUDA_MPS_PIPE_DIRECTORY' 2>&1
if ($MpsEnvCheck -eq "/tmp/nvidia-mps") {
    Pass "CUDA_MPS_PIPE_DIRECTORY is set correctly"
} else {
    Fail "CUDA_MPS_PIPE_DIRECTORY is NOT set correctly (got: $MpsEnvCheck)"
}

Write-Host ""
Write-Host "=== Cleanup ===" -ForegroundColor Cyan
$Response = Read-Host "Delete test pod? (y/N)"
if ($Response -eq "y" -or $Response -eq "Y") {
    kubectl delete pod gpu-share-test -n $Namespace --force --grace-period=0 2>$null
    Pass "Test pod deleted"
}

Write-Host ""
Write-Host "==============================================" -ForegroundColor Cyan
Write-Host " Test Suite Complete" -ForegroundColor Cyan
Write-Host "==============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "If all tests passed, you can deploy KServe services:" -ForegroundColor White
Write-Host "  kubectl apply -f 10-kserve-inferenceservice.yaml" -ForegroundColor Gray
Write-Host "  kubectl apply -f 11-multi-model-sharing.yaml" -ForegroundColor Gray
