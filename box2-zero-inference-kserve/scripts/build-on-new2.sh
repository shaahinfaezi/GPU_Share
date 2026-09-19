#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-box2.local/deepspeed-zero-kserve:qwen-test}"
BASE_IMAGE="${BASE_IMAGE:-nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04}"
TORCH_VERSION="${TORCH_VERSION:-2.5.1}"
TORCH_CUDA="${TORCH_CUDA:-cu121}"
CONTAINERD_RUNTIME="${CONTAINERD_RUNTIME:-k3s}"

command -v nvidia-smi >/dev/null
nvidia-smi

if command -v docker >/dev/null; then
  BUILD=(docker build)
  SAVE=(docker save "${IMAGE}")
  RUN=(docker run --rm --entrypoint bash "${IMAGE}")
elif command -v nerdctl >/dev/null; then
  BUILD=(nerdctl build)
  SAVE=(nerdctl save "${IMAGE}")
  RUN=(nerdctl run --rm --entrypoint bash "${IMAGE}")
else
  echo "Docker or nerdctl is required on new2 to build the image." >&2
  exit 2
fi

"${BUILD[@]}" \
  --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
  --build-arg "TORCH_VERSION=${TORCH_VERSION}" \
  --build-arg "TORCH_CUDA=${TORCH_CUDA}" \
  -t "${IMAGE}" .

case "${CONTAINERD_RUNTIME}" in
  k3s)
    "${SAVE[@]}" | sudo k3s ctr images import -
    ;;
  ctr)
    "${SAVE[@]}" | sudo ctr -n k8s.io images import -
    ;;
  *)
    echo "CONTAINERD_RUNTIME must be k3s or ctr." >&2
    exit 3
    ;;
esac
echo "Imported ${IMAGE} into the worker's Kubernetes containerd image store."

# Verify the extension before KServe hides the build failure behind a pod restart.
report_file="$(mktemp)"
trap 'rm -f "${report_file}"' EXIT
"${RUN[@]}" -lc 'ds_report' | tee "${report_file}"
if ! grep -Eq '^async_io.*\[OKAY\]' "${report_file}"; then
  echo "DeepSpeed async_io is unavailable. Do not deploy an NVMe-offload image without it." >&2
  exit 3
fi
if command -v k3s >/dev/null; then
  sudo k3s ctr images ls | grep -F "${IMAGE%%:*}" || true
fi

echo "DeepSpeed async_io is available in ${IMAGE}."
