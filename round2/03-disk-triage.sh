#!/usr/bin/env bash
# Disk triage — the round-2 blocker.
#
# The preflight probe pod was EVICTED, twice, while pulling the serving image:
#
#   The node was low on resource: ephemeral-storage.
#   Threshold quantity: 5193265229, available: 4373084Ki   -> then 869716Ki
#
# That matters far beyond one probe. Every round-2 pod uses that same image, so
# until this is resolved nothing can be deployed at all. Worse, eviction is
# node-wide: a pull that crosses the threshold can take *running* benchmark
# pods down with it, which would destroy a run mid-flight.
#
# The confusing part is that `df /` reports ~23 GiB free while kubelet reports
# ~4 GiB available. kubelet measures the filesystem holding the container
# runtime's image store, and applies its own eviction threshold on top. This
# script shows both numbers side by side and finds what is actually consuming.
#
#   ./03-disk-triage.sh              # diagnose only
#   ./03-disk-triage.sh --clean      # safe reclaim: unused images, logs, apt
#   ./03-disk-triage.sh --plan-move  # print the containerd-to-/data2 procedure
set -uo pipefail

CLEAN=0
PLAN=0
for a in "$@"; do
  case "$a" in
    --clean) CLEAN=1 ;;
    --plan-move) PLAN=1 ;;
    -h|--help) sed -n '2,25p' "$0"; exit 0 ;;
    *) echo "unknown option: $a" >&2; exit 2 ;;
  esac
done

have() { command -v "$1" >/dev/null 2>&1; }
CRICTL="crictl"
have k3s && CRICTL="k3s crictl"

hr() { printf '\n=== %s\n' "$1"; }

hr "1. what kubelet thinks vs what df thinks"
kubectl describe node "$(hostname)" 2>/dev/null \
  | grep -E "DiskPressure|ephemeral-storage" | head -6
echo "--- df, real filesystems:"
df -h | grep -Ev "tmpfs|overlay|shm" | head -10
echo
echo "--- the filesystem kubelet actually watches:"
for d in /var/lib/rancher/k3s/agent/containerd /var/lib/kubelet /var/lib/rancher/k3s; do
  if [[ -d "$d" ]]; then
    fs="$(df -P "$d" | awk 'NR==2{print $6}')"
    avail="$(df -Ph "$d" | awk 'NR==2{print $4}')"
    printf '    %-46s -> %-12s avail %s\n' "$d" "$fs" "$avail"
  fi
done
echo
echo "    kubelet's default eviction threshold is 10% of that filesystem, or"
echo "    the explicit --eviction-hard value. 5193265229 bytes = 4.84 GiB,"
echo "    which is what the probe pod hit."

hr "2. where the space went"
echo "--- biggest directories under /var/lib/rancher:"
du -sh /var/lib/rancher/k3s/agent/containerd/* 2>/dev/null | sort -h -r | head -6
echo "--- other usual suspects:"
for d in /var/log /var/lib/containerd /var/lib/docker /root /tmp /var/cache/apt; do
  [[ -d "$d" ]] && printf '    %-26s %s\n' "$d" "$(du -sh "$d" 2>/dev/null | cut -f1)"
done
echo "--- journal:"; journalctl --disk-usage 2>/dev/null | sed 's/^/    /'

hr "3. container images"
${CRICTL} images 2>/dev/null | head -20
echo
echo "--- total image store:"
du -sh /var/lib/rancher/k3s/agent/containerd/io.containerd.content.v1.content 2>/dev/null \
  | sed 's/^/    /'
echo "--- is the benchmark image resident?"
if ${CRICTL} images 2>/dev/null | grep -qi huggingfaceserver; then
  ${CRICTL} images 2>/dev/null | grep -i huggingfaceserver | sed 's/^/    /'
else
  echo "    NOT PRESENT."
  echo "    It must be pulled for every round-2 pod, and that pull is what"
  echo "    crossed the eviction threshold. Fix the disk before deploying."
fi
echo "--- local registry backing store (this is on /data2, which is fine):"
du -sh /data2/registry_volume 2>/dev/null | sed 's/^/    /'

hr "4. what /data2 has spare"
df -h /data2 | tail -1
echo "--- /data2/huggingface.tar is an OCI image archive, not a model cache:"
echo "    (blobs/sha256 + index.json + manifest.json is the OCI layout)"
ls -lh /data2/huggingface.tar 2>/dev/null | sed 's/^/    /'
echo "    If it is the serving image, importing it is FASTER and SAFER than"
echo "    pulling, but it still unpacks into the store on / — so it does not"
echo "    avoid the disk problem by itself. Inspect with:"
echo "      tar -xOf /data2/huggingface.tar index.json | head -c 400"

if [[ "${CLEAN}" -eq 1 ]]; then
  hr "5. reclaiming (safe operations only)"
  echo "--- pruning unused images"
  ${CRICTL} rmi --prune 2>&1 | tail -5
  echo "--- vacuuming journal to 200M"
  journalctl --vacuum-size=200M 2>&1 | tail -3
  echo "--- apt cache"
  apt-get clean 2>/dev/null && echo "    apt-get clean done"
  echo "--- removing evicted/completed pods cluster-wide"
  kubectl get pods -A --field-selector 'status.phase!=Running,status.phase!=Pending' \
    -o custom-columns=NS:.metadata.namespace,N:.metadata.name --no-headers 2>/dev/null \
    | while read -r ns n; do kubectl delete pod -n "$ns" "$n" --wait=false >/dev/null 2>&1; done
  echo "    done"
  echo
  echo "--- after:"
  df -h | grep -Ev "tmpfs|overlay|shm" | head -6
fi

if [[ "${PLAN}" -eq 1 ]]; then
  hr "5. moving containerd's image store to /data2 (the durable fix)"
  cat <<'PLANEOF'
/ has ~97 GiB total and the vLLM GPU image is large enough that one copy is a
material fraction of it. /data2 has 78 GiB spare and is already where the
registry volume lives, so the image store belongs there too.

This restarts k3s. Do it while nothing is being benchmarked.

  # 1. stop k3s
  sudo systemctl stop k3s

  # 2. move the agent directory (contains the image store)
  sudo mkdir -p /data2/k3s-agent
  sudo rsync -aHAX --info=progress2 /var/lib/rancher/k3s/agent/ /data2/k3s-agent/
  sudo mv /var/lib/rancher/k3s/agent /var/lib/rancher/k3s/agent.bak

  # 3. bind-mount it back, and make that survive reboot
  sudo mkdir -p /var/lib/rancher/k3s/agent
  echo '/data2/k3s-agent /var/lib/rancher/k3s/agent none bind 0 0' \
    | sudo tee -a /etc/fstab
  sudo mount -a

  # 4. start k3s and confirm
  sudo systemctl start k3s
  sudo k3s crictl images | head
  df -h /var/lib/rancher/k3s/agent/containerd

  # 5. once healthy, reclaim the old copy
  sudo rm -rf /var/lib/rancher/k3s/agent.bak

Lighter alternative if you would rather not touch k3s: raise the eviction
threshold so a big pull cannot trip it, and keep / clean by hand.

  sudo mkdir -p /etc/rancher/k3s
  cat <<'EOF' | sudo tee /etc/rancher/k3s/config.yaml
kubelet-arg:
  - "eviction-hard=nodefs.available<2Gi,imagefs.available<2Gi"
  - "image-gc-high-threshold=80"
  - "image-gc-low-threshold=70"
EOF
  sudo systemctl restart k3s

That trades safety margin for headroom: it lets the disk run closer to full
before evicting, which is acceptable here only because nothing else shares
this node.
PLANEOF
fi

hr "verdict"
# Measure against kubelet's ACTUAL eviction threshold, not a round number.
# An earlier version of this script called 33 GiB "enough headroom" and was
# wrong: the 9.5 GB serving image unpacks to roughly 29 GiB in overlayfs, which
# took the node from 34 GiB free to 5.3 GiB -- about 460 MB above the threshold.
THRESH_BYTES="$(kubectl get --raw /api/v1/nodes/$(hostname)/proxy/configz 2>/dev/null   | grep -o '"nodefs.available":"[^"]*"' | head -1 | cut -d'"' -f4)"
THRESH_GIB="4.84"     # observed on this node: 5193265229 bytes
avail_k="$(df -Pk /var/lib/rancher/k3s/agent/containerd 2>/dev/null | awk 'NR==2{print $4}')"
avail_g="$(awk -v k="${avail_k:-0}" 'BEGIN{printf "%.1f", k/1048576}')"
margin="$(awk -v a="${avail_g}" -v t="${THRESH_GIB}" 'BEGIN{printf "%.2f", a-t}')"

echo "container-runtime filesystem : ${avail_g} GiB available"
echo "kubelet eviction threshold   : ${THRESH_GIB} GiB${THRESH_BYTES:+  (configz: ${THRESH_BYTES})}"
echo "margin before eviction       : ${margin} GiB"
echo

img_resident=no
(k3s crictl images 2>/dev/null || crictl images 2>/dev/null)   | grep -qi huggingfaceserver && img_resident=yes
echo "serving image resident       : ${img_resident}"
if [[ "${img_resident}" == "no" ]]; then
  echo
  echo "  Pulling it will cost roughly 29 GiB unpacked (9.5 GB compressed)."
  echo "  You need that much ON TOP of the margin below."
fi
echo

if awk -v m="${margin}" 'BEGIN{exit !(m < 3)}'; then
  echo "STATUS: CRITICAL — under 3 GiB above the eviction threshold."
  echo "  Pod logs alone can cross this. Eviction is node-wide and will kill"
  echo "  running benchmark pods mid-run."
  echo "  Reclaim now:"
  echo "    systemctl is-active containerd docker   # if inactive, /var/lib/containerd is stale"
  echo "    ./03-disk-triage.sh --clean"
  echo "    ./03-disk-triage.sh --plan-move         # durable: move the store to /data2"
  exit 2
fi
if awk -v m="${margin}" -v r="${img_resident}" 'BEGIN{exit !(m < 12 && r == "no")}'; then
  echo "STATUS: TOO TIGHT to pull the serving image."
  echo "  Free space first, or the pull will evict pods on its way in."
  exit 1
fi
if awk -v m="${margin}" 'BEGIN{exit !(m < 8)}'; then
  echo "STATUS: WORKABLE but thin. Watch it during the first run."
  exit 0
fi
echo "STATUS: healthy."
exit 0
