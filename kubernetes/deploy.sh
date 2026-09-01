#!/bin/bash
# Deploy GPU Share components to Kubernetes cluster

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAMESPACE="gpu-share"

echo "=============================================="
echo " GPU Share Kubernetes Deployment"
echo "=============================================="

# Check kubectl is available
if ! command -v kubectl &> /dev/null; then
    echo "ERROR: kubectl not found. Please install kubectl first."
    exit 1
fi

# Check cluster connectivity
echo "Checking cluster connectivity..."
if ! kubectl cluster-info &> /dev/null; then
    echo "ERROR: Cannot connect to Kubernetes cluster."
    exit 1
fi
echo "✓ Connected to cluster"

# Check for GPU nodes
echo "Checking for GPU nodes..."
GPU_NODES=$(kubectl get nodes -l nvidia.com/gpu.present=true -o name 2>/dev/null || true)
if [ -z "$GPU_NODES" ]; then
    echo "WARNING: No nodes with label 'nvidia.com/gpu.present=true' found."
    echo "You may need to label your GPU nodes:"
    echo "  kubectl label nodes <node-name> nvidia.com/gpu.present=true"
fi

# Deploy in order
echo ""
echo "Step 1: Creating namespaces..."
kubectl apply -f "$SCRIPT_DIR/00-namespace.yaml"
echo "✓ Namespaces created"

echo ""
echo "Step 2: Deploying VRAM Shim Installer DaemonSet..."
kubectl apply -f "$SCRIPT_DIR/01-vram-shim-daemonset.yaml"
echo "✓ VRAM Shim DaemonSet deployed"

echo ""
echo "Step 3: Deploying MPS Daemon DaemonSet..."
kubectl apply -f "$SCRIPT_DIR/02-mps-daemon-daemonset.yaml"
echo "✓ MPS Daemon DaemonSet deployed"

echo ""
echo "Step 4: Deploying MPS Device Plugin..."
kubectl apply -f "$SCRIPT_DIR/03-mps-device-plugin.yaml"
echo "✓ MPS Device Plugin deployed"

echo ""
echo "Step 5: Applying ConfigMaps..."
kubectl apply -f "$SCRIPT_DIR/04-configmap.yaml"
echo "✓ ConfigMaps applied"

# Wait for DaemonSets to be ready
echo ""
echo "Waiting for DaemonSets to be ready..."
kubectl rollout status daemonset/vram-shim-installer -n $NAMESPACE --timeout=120s || true
kubectl rollout status daemonset/nvidia-mps-daemon -n $NAMESPACE --timeout=120s || true
kubectl rollout status daemonset/mps-device-plugin -n $NAMESPACE --timeout=120s || true

echo ""
echo "=============================================="
echo " Deployment Complete!"
echo "=============================================="
echo ""
echo "Next steps:"
echo "1. Verify shim installation:"
echo "   kubectl exec -it \$(kubectl get pod -n $NAMESPACE -l app=vram-shim-installer -o name | head -1) -n $NAMESPACE -- ls -la /opt/vram-shim/"
echo ""
echo "2. Verify MPS daemon:"
echo "   kubectl logs -n $NAMESPACE -l app=nvidia-mps-daemon"
echo ""
echo "3. Deploy KServe InferenceService:"
echo "   kubectl apply -f $SCRIPT_DIR/10-kserve-inferenceservice.yaml"
echo ""
echo "4. For multi-model sharing:"
echo "   kubectl apply -f $SCRIPT_DIR/11-multi-model-sharing.yaml"
