#!/usr/bin/env bash
set -euo pipefail

ZERO_ROOT="${ZERO_ROOT:-/srv/gpu-share/zero}"
MIN_FREE_GIB="${MIN_FREE_GIB:-80}"

if [[ "${EUID}" -ne 0 ]]; then
  echo "Run through sudo so directory ownership can be set safely." >&2
  exit 2
fi

mkdir -p "${ZERO_ROOT}/models" "${ZERO_ROOT}/scratch"
mount_source="$(findmnt -no SOURCE -T "${ZERO_ROOT}")"
mount_target="$(findmnt -no TARGET -T "${ZERO_ROOT}")"
available_gib="$(df -BG --output=avail "${ZERO_ROOT}" | tail -n1 | tr -dc '0-9')"

echo "ZERO_ROOT=${ZERO_ROOT}"
echo "mount=${mount_source} -> ${mount_target}"
echo "available=${available_gib} GiB"

if [[ "${mount_target}" == "/" ]]; then
  echo "Refusing to use the root filesystem. Mount an NVMe filesystem and choose its path." >&2
  exit 3
fi
if (( available_gib < MIN_FREE_GIB )); then
  echo "Need at least ${MIN_FREE_GIB} GiB free for this pilot." >&2
  exit 4
fi

chmod 0755 "${ZERO_ROOT}/models"
chmod 0777 "${ZERO_ROOT}/scratch"
nvidia-smi
echo "Prepared local PV directories on ${mount_target}."
