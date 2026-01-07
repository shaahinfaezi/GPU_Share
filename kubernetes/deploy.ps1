# Deploy GPU Share components to Kubernetes cluster

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Namespace = "gpu-share"

Write-Host "==============================================" -ForegroundColor Cyan
Write-Host " GPU Share Kubernetes Deployment" -ForegroundColor Cyan
Write-Host "==============================================" -ForegroundColor Cyan

# Check kubectl is available
if (-not (Get-Command kubectl -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: kubectl not found. Please install kubectl first." -ForegroundColor Red
    exit 1
}

# Check cluster connectivity
Write-Host "Checking cluster connectivity..."
try {
    kubectl cluster-info | Out-Null
    Write-Host "✓ Connected to cluster" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Cannot connect to Kubernetes cluster." -ForegroundColor Red
    exit 1
}

# Check for GPU nodes
Write-Host "Checking for GPU nodes..."
$GpuNodes = kubectl get nodes -l nvidia.com/gpu.present=true -o name 2>$null
if (-not $GpuNodes) {
    Write-Host "WARNING: No nodes with label 'nvidia.com/gpu.present=true' found." -ForegroundColor Yellow
    Write-Host "You may need to label your GPU nodes:" -ForegroundColor Yellow
    Write-Host "  kubectl label nodes <node-name> nvidia.com/gpu.present=true" -ForegroundColor Yellow
}

# Deploy in order
Write-Host ""
Write-Host "Step 1: Creating namespaces..." -ForegroundColor Cyan
kubectl apply -f "$ScriptDir\00-namespace.yaml"
Write-Host "✓ Namespaces created" -ForegroundColor Green

Write-Host ""
Write-Host "Step 2: Deploying VRAM Shim Installer DaemonSet..." -ForegroundColor Cyan
kubectl apply -f "$ScriptDir\01-vram-shim-daemonset.yaml"
Write-Host "✓ VRAM Shim DaemonSet deployed" -ForegroundColor Green

Write-Host ""
Write-Host "Step 3: Deploying MPS Daemon DaemonSet..." -ForegroundColor Cyan
kubectl apply -f "$ScriptDir\02-mps-daemon-daemonset.yaml"
Write-Host "✓ MPS Daemon DaemonSet deployed" -ForegroundColor Green

Write-Host ""
Write-Host "Step 4: Deploying MPS Device Plugin..." -ForegroundColor Cyan
kubectl apply -f "$ScriptDir\03-mps-device-plugin.yaml"
Write-Host "✓ MPS Device Plugin deployed" -ForegroundColor Green

Write-Host ""
Write-Host "Step 5: Applying ConfigMaps..." -ForegroundColor Cyan
kubectl apply -f "$ScriptDir\04-configmap.yaml"
Write-Host "✓ ConfigMaps applied" -ForegroundColor Green

# Wait for DaemonSets to be ready
Write-Host ""
Write-Host "Waiting for DaemonSets to be ready..." -ForegroundColor Cyan
kubectl rollout status daemonset/vram-shim-installer -n $Namespace --timeout=120s 2>$null
kubectl rollout status daemonset/nvidia-mps-daemon -n $Namespace --timeout=120s 2>$null
kubectl rollout status daemonset/mps-device-plugin -n $Namespace --timeout=120s 2>$null

Write-Host ""
Write-Host "==============================================" -ForegroundColor Cyan
Write-Host " Deployment Complete!" -ForegroundColor Cyan
Write-Host "==============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor White
Write-Host "1. Verify shim installation:" -ForegroundColor White
Write-Host "   kubectl exec -it (kubectl get pod -n $Namespace -l app=vram-shim-installer -o name | Select-Object -First 1) -n $Namespace -- ls -la /opt/vram-shim/" -ForegroundColor Gray
Write-Host ""
Write-Host "2. Verify MPS daemon:" -ForegroundColor White
Write-Host "   kubectl logs -n $Namespace -l app=nvidia-mps-daemon" -ForegroundColor Gray
Write-Host ""
Write-Host "3. Deploy KServe InferenceService:" -ForegroundColor White
Write-Host "   kubectl apply -f $ScriptDir\10-kserve-inferenceservice.yaml" -ForegroundColor Gray
Write-Host ""
Write-Host "4. For multi-model sharing:" -ForegroundColor White
Write-Host "   kubectl apply -f $ScriptDir\11-multi-model-sharing.yaml" -ForegroundColor Gray
