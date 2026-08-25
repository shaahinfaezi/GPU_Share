#!/usr/bin/env bash
# Pre-pull the serving image, deliberately and once.
#
# WHY THIS IS ITS OWN STEP
# The preflight probe was evicted mid-pull with "node was low on resource:
# ephemeral-storage". Eviction is node-wide, so if that pull happens while a
# benchmark is running it can take the benchmark pods down with it and destroy
# the run. Pulling now, with nothing else on the node, converts a random
# mid-run failure into a controlled step you can watch.
#
# It also front-loads the cost: once the image is in containerd's store, every
# later pod starts from it instead of re-pulling.
#
#   ./04-prepull-image.sh            # pull and verify
#   ./04-prepull-image.sh --check    # just report whether it is resident
#   ./04-prepull-image.sh --from-tar # import /data2/huggingface.tar instead
set -uo pipefail

IMG="${IMG:-127.0.0.1:5000/kserve/huggingfaceserver:v0.17.0-gpu}"
TAR="${TAR:-/data2/huggingface.tar}"
MIN_FREE_GIB="${MIN_FREE_GIB:-20}"
CHECK=0
FROM_TAR=0

for a in "$@"; do
  case "$a" in
    --check) CHECK=1 ;;
    --from-tar) FROM_TAR=1 ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown option: $a" >&2; exit 2 ;;
  esac
done

CRICTL="crictl"
command -v k3s >/dev/null 2>&1 && CRICTL="k3s crictl"
CTR="ctr"
command -v k3s >/dev/null 2>&1 && CTR="k3s ctr"

store_avail_gib() {
  df -Pk /var/lib/rancher/k3s/agent/containerd 2>/dev/null \
    | awk 'NR==2{printf "%.1f", $4/1048576}'
}

resident() { ${CRICTL} images 2>/dev/null | grep -q "huggingfaceserver"; }

echo "image  : ${IMG}"
echo "store  : /var/lib/rancher/k3s/agent/containerd  ($(store_avail_gib) GiB free)"
echo

if resident; then
  echo "ALREADY RESIDENT:"
  ${CRICTL} images 2>/dev/null | grep huggingfaceserver | sed 's/^/  /'
  [[ "${CHECK}" -eq 1 ]] && exit 0
  echo
  echo "Nothing to do. Delete it first if you want a fresh pull:"
  echo "  ${CRICTL} rmi ${IMG}"
  exit 0
fi

echo "NOT resident."
if [[ "${CHECK}" -eq 1 ]]; then
  echo "Run without --check to pull it."
  exit 1
fi

avail="$(store_avail_gib)"
if awk -v a="${avail}" -v m="${MIN_FREE_GIB}" 'BEGIN{exit !(a < m)}'; then
  echo
  echo "REFUSING: only ${avail} GiB free, want ${MIN_FREE_GIB}."
  echo "A vLLM GPU image unpacks to roughly 10-15 GiB, and crossing kubelet's"
  echo "eviction threshold mid-pull is what killed the probe pod."
  echo "Run ./03-disk-triage.sh --clean first."
  exit 1
fi

if [[ "${FROM_TAR}" -eq 1 ]]; then
  # The archive on /data2 is v0.16.0-gpu, NOT the v0.17.0-gpu the manifests
  # reference. Importing it gives a working server but a different vLLM
  # version, which would silently change what is being measured. Only use this
  # if the registry cannot serve v0.17.0.
  [[ -f "${TAR}" ]] || { echo "no such archive: ${TAR}" >&2; exit 1; }
  echo "importing ${TAR} (this is v0.16.0-gpu — a DIFFERENT version)"
  echo "  contained image:"
  tar -xOf "${TAR}" index.json 2>/dev/null \
    | grep -o '"io.containerd.image.name":"[^"]*"' | sed 's/^/    /'
  echo
  read -r -p "import anyway? [y/N] " ans
  [[ "${ans}" == "y" || "${ans}" == "Y" ]] || { echo "aborted"; exit 1; }
  time ${CTR} images import "${TAR}"
else
  echo "pulling from the local registry (no external network involved —"
  echo "127.0.0.1:5000 is backed by /data2/registry_volume)"
  echo
  t0=$SECONDS
  if ! ${CRICTL} pull "${IMG}"; then
    echo
    echo "PULL FAILED." >&2
    echo "Check the registry actually has this tag:" >&2
    echo "  curl -s http://127.0.0.1:5000/v2/kserve/huggingfaceserver/tags/list | tr ',' '\\n'" >&2
    echo "If v0.17.0-gpu is absent, the archive on /data2 has v0.16.0-gpu:" >&2
    echo "  ./04-prepull-image.sh --from-tar" >&2
    exit 1
  fi
  echo
  echo "pull took $(( SECONDS - t0 ))s"
fi

echo
echo "--- verifying"
if resident; then
  ${CRICTL} images 2>/dev/null | grep huggingfaceserver | sed 's/^/  /'
else
  echo "  image still not listed — something went wrong" >&2
  exit 1
fi
echo
echo "store free after: $(store_avail_gib) GiB"
kubectl describe node "$(hostname)" 2>/dev/null | grep -E "DiskPressure" | head -2 | sed 's/^/  /'
echo
echo "Next: ./00-preflight.sh  — section 4 can finally answer the pooling"
echo "question now that the image is local and the probe pod will start."
