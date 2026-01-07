# GPU Share Kubernetes Deployment

This folder contains all Kubernetes manifests needed to deploy MPS-based GPU sharing with VRAM isolation for KServe inference.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Kubernetes Cluster                        │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │  KServe Pod 1   │  │  KServe Pod 2   │  │  KServe Pod 3   │  │
│  │  LD_PRELOAD=    │  │  LD_PRELOAD=    │  │  LD_PRELOAD=    │  │
│  │  vram_shim.so   │  │  vram_shim.so   │  │  vram_shim.so   │  │
│  │  VRAM: 3000MB   │  │  VRAM: 3000MB   │  │  VRAM: 3000MB   │  │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘  │
│           │                    │                    │           │
│           └────────────────────┼────────────────────┘           │
│                                │                                │
│                    ┌───────────▼───────────┐                    │
│                    │   MPS Control Daemon   │                    │
│                    │   /tmp/nvidia-mps      │                    │
│                    └───────────┬───────────┘                    │
│                                │                                │
│                    ┌───────────▼───────────┐                    │
│                    │      Physical GPU      │                    │
│                    │   EXCLUSIVE_PROCESS    │                    │
│                    └───────────────────────┘                    │
└─────────────────────────────────────────────────────────────────┘
```

## Components

| Component | File | Purpose |
|-----------|------|---------|
| Namespace | `00-namespace.yaml` | Create gpu-share namespace |
| VRAM Shim | `01-vram-shim-daemonset.yaml` | Install shim on all GPU nodes |
| MPS Daemon | `02-mps-daemon-daemonset.yaml` | Run MPS control daemon |
| MPS Device Plugin | `03-mps-device-plugin.yaml` | Expose virtual MPS GPUs |
| ConfigMap | `04-configmap.yaml` | Shared configuration |
| KServe Example | `10-kserve-inferenceservice.yaml` | Example LLM deployment |

## Quick Deployment

Use the deployment scripts for automated setup:

```bash
# Linux/Mac
./deploy.sh

# Windows PowerShell
.\deploy.ps1
```

## Manual Deployment Order

```bash
# 1. Create namespace
kubectl apply -f 00-namespace.yaml

# 2. Deploy VRAM shim installer (runs on each GPU node)
kubectl apply -f 01-vram-shim-daemonset.yaml

# 3. Deploy MPS daemon (runs on each GPU node)
kubectl apply -f 02-mps-daemon-daemonset.yaml

# 4. Deploy MPS device plugin (exposes nvidia.com/mps-gpu)
kubectl apply -f 03-mps-device-plugin.yaml

# 5. Apply shared configuration
kubectl apply -f 04-configmap.yaml

# 6. Deploy KServe InferenceService
kubectl apply -f 10-kserve-inferenceservice.yaml
```

## How It Works

### 1. VRAM Shim Injection
- DaemonSet installs `vram_limiter_runtime.so` to `/opt/vram-shim/` on each node
- Pods mount this via hostPath and set `LD_PRELOAD`
- Shim intercepts `cudaMalloc` to enforce per-pod VRAM limits

### 2. MPS Sharing
- MPS daemon runs on each GPU node in EXCLUSIVE_PROCESS mode
- Multiple pods share the same GPU via MPS context
- `NVIDIA_MPS_ACTIVE_THREAD_PERCENTAGE` controls SM fairness

### 3. KServe Integration
- InferenceService requests `nvidia.com/mps-gpu: 1` (virtual GPU)
- Init container or sidecar can inject shim if not using hostPath
- Model serves requests while respecting VRAM limits

## Configuration

### VRAM Limit per Pod
```yaml
env:
- name: GPU_VRAM_LIMIT_MB
  value: "3000"  # 3GB per pod
```

### MPS Thread Percentage
```yaml
env:
- name: NVIDIA_MPS_ACTIVE_THREAD_PERCENTAGE
  value: "25"  # 4 pods can share equally
```

## Verification

```bash
# Check shim installation
kubectl exec -it <pod> -- ls -la /opt/vram-shim/

# Check MPS status
kubectl exec -it <mps-daemon-pod> -- nvidia-smi

# Check pod VRAM usage
kubectl exec -it <inference-pod> -- cat /tmp/vram_limiter.log
```
