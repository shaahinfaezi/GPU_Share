#!/usr/bin/env bash
# Round-2 model download — standalone, survives disconnect.
#
# Builds the same HuggingFace cache layout that Qwen already sits in, so once
# the directory is in place the LLMInferenceService resolves these models the
# way it resolves Qwen today: HF_HOME points at the cache root and the server
# looks the repo up by its `model.name`. Only the backing disk differs.
#
# TYPICAL USE ON THE BOX
#   sudo mkdir -p /data2/hf-cache && sudo chmod 0777 /data2/hf-cache
#   ./download-models.sh --tiers 1 --background     # detaches, safe to log out
#   ./download-models.sh --status                   # progress, any time
#   ./verify-models.sh                              # are they usable?
#
# Re-running is safe: complete repos are skipped, partial ones resume.
set -uo pipefail

DEST="/data2/hf-cache"
TIERS="1"
MIN_FREE_GIB=12
DRY_RUN=0
LIST=0
BACKGROUND=0
STATUS=0
LOG=""

usage() {
  cat <<'USAGE'
Usage: download-models.sh [options]

  --dest DIR         cache root; becomes HF_HOME   (default /data2/hf-cache)
  --tiers 1,2,3      which tiers to fetch          (default 1)
  --min-free GIB     refuse to start a repo below this free space (default 12)
  --background       detach and keep running after you log out
  --status           show progress of a running/finished download, then exit
  --log FILE         log path (default DEST/download.log)
  --list             print the model table and exit
  --dry-run          show what would be fetched, download nothing
  -h, --help         this text

TIERS
  1  ~11 GiB  Qwen2.5-1.5B-Instruct, bge-m3, Qwen2.5-7B-Instruct-AWQ
              enough for tests T1, T2 and T3
  2  ~10 GiB  bge-reranker-v2-m3, Qwen2.5-3B-Instruct, bge-large-en-v1.5
  3  ~13 GiB  Qwen2-VL-2B-Instruct, Llama-3.2-3B-Instruct, whisper-large-v3-turbo

DISCONNECT-SAFE
  --background re-executes under setsid+nohup, so the download is reparented to
  init and survives the SSH session ending. The pid is written to
  DEST/download.pid. Progress bars keep streaming into the log:
      tail -f /data2/hf-cache/download.log
      ./download-models.sh --status
  Stop it with:  kill $(cat /data2/hf-cache/download.pid)

GATED REPOS
  Llama-3.2-3B-Instruct needs an accepted licence and a token:
      export HF_TOKEN=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
  Without it that one repo is reported FAILED and the rest still land.
USAGE
}

# tier | repo | extra ignore globs ("-" for none) | approx GiB after ignores
MODELS=(
  "1|Qwen/Qwen2.5-1.5B-Instruct|-|3.1"
  "1|BAAI/bge-m3|onnx/*,*.onnx,imgs/*|2.3"
  "1|Qwen/Qwen2.5-7B-Instruct-AWQ|-|5.6"
  "2|BAAI/bge-reranker-v2-m3|onnx/*,*.onnx,imgs/*|2.3"
  "2|Qwen/Qwen2.5-3B-Instruct|-|6.2"
  "2|BAAI/bge-large-en-v1.5|onnx/*,*.onnx,openvino/*|1.3"
  "3|Qwen/Qwen2-VL-2B-Instruct|-|4.4"
  "3|meta-llama/Llama-3.2-3B-Instruct|original/*|6.5"
  "3|openai/whisper-large-v3-turbo|-|1.6"
)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dest)       DEST="${2:-}"; shift 2 ;;
    --tiers)      TIERS="${2:-}"; shift 2 ;;
    --min-free)   MIN_FREE_GIB="${2:-}"; shift 2 ;;
    --background) BACKGROUND=1; shift ;;
    --status)     STATUS=1; shift ;;
    --log)        LOG="${2:-}"; shift 2 ;;
    --dry-run)    DRY_RUN=1; shift ;;
    --list)       LIST=1; shift ;;
    -h|--help)    usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

repo_dir() { echo "models--$(echo "$1" | sed 's|/|--|g')"; }

if [[ "${LIST}" -eq 1 ]]; then
  printf '%-5s %-40s %6s  %s\n' TIER REPO GiB IGNORED
  for row in "${MODELS[@]}"; do
    IFS='|' read -r t r ig sz <<<"${row}"
    printf '%-5s %-40s %6s  %s\n' "${t}" "${r}" "${sz}" "${ig}"
  done
  exit 0
fi

LOG="${LOG:-${DEST}/download.log}"
PIDFILE="${DEST}/download.pid"

# ------------------------------------------------------------------ status
if [[ "${STATUS}" -eq 1 ]]; then
  echo "cache : ${DEST}"
  if [[ -f "${PIDFILE}" ]] && kill -0 "$(cat "${PIDFILE}" 2>/dev/null)" 2>/dev/null; then
    echo "state : RUNNING (pid $(cat "${PIDFILE}"))"
  elif [[ -f "${LOG}" ]]; then
    echo "state : not running (log present)"
  else
    echo "state : never started"
  fi
  echo
  printf '%-40s %8s %8s  %s\n' REPO HAVE EXPECT STATUS
  total_have=0
  for row in "${MODELS[@]}"; do
    IFS='|' read -r t r ig sz <<<"${row}"
    [[ ",${TIERS}," == *",${t},"* ]] || continue
    d="${DEST}/hub/$(repo_dir "${r}")"
    if [[ -d "${d}" ]]; then
      have="$(du -sm "${d}" 2>/dev/null | cut -f1)"
      have_gib="$(awk -v m="${have:-0}" 'BEGIN{printf "%.1f", m/1024}')"
      total_have=$(( total_have + ${have:-0} ))
      if compgen -G "${d}/blobs/*.incomplete" >/dev/null 2>&1; then
        st="downloading"
      elif awk -v h="${have_gib}" -v e="${sz}" 'BEGIN{exit !(h >= e*0.92)}'; then
        st="complete"
      else
        st="PARTIAL"
      fi
      printf '%-40s %7sG %7sG  %s\n' "${r}" "${have_gib}" "${sz}" "${st}"
    else
      printf '%-40s %8s %7sG  %s\n' "${r}" "-" "${sz}" "not started"
    fi
  done
  echo
  echo "total on disk: $(awk -v m="${total_have}" 'BEGIN{printf "%.1f", m/1024}') GiB"
  [[ -d "${DEST}" ]] && df -h "${DEST}" | tail -1
  if [[ -f "${LOG}" ]]; then
    echo
    echo "last log lines:"
    tr '\r' '\n' < "${LOG}" | grep -v '^[[:space:]]*$' | tail -6 | sed 's/^/  /'
    echo
    echo "follow with: tail -f ${LOG}"
  fi
  exit 0
fi

mkdir -p "${DEST}" || { echo "ERROR: cannot create ${DEST}" >&2; exit 1; }
DEST="$(cd "${DEST}" && pwd)"
LOG="${LOG:-${DEST}/download.log}"
PIDFILE="${DEST}/download.pid"

# -------------------------------------------------------------------- detach
if [[ "${BACKGROUND}" -eq 1 ]]; then
  if [[ -f "${PIDFILE}" ]] && kill -0 "$(cat "${PIDFILE}" 2>/dev/null)" 2>/dev/null; then
    echo "already running (pid $(cat "${PIDFILE}")). Use --status." >&2; exit 1
  fi
  # setsid detaches from the controlling terminal, so SIGHUP on logout is never
  # delivered; nohup covers hosts without setsid.
  self="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
  args=(--dest "${DEST}" --tiers "${TIERS}" --min-free "${MIN_FREE_GIB}" --log "${LOG}")
  if command -v setsid >/dev/null 2>&1; then
    setsid nohup bash "${self}" "${args[@]}" >>"${LOG}" 2>&1 < /dev/null &
  else
    nohup bash "${self}" "${args[@]}" >>"${LOG}" 2>&1 < /dev/null &
  fi
  echo $! > "${PIDFILE}"
  sleep 1
  echo "detached : pid $(cat "${PIDFILE}")"
  echo "log      : ${LOG}"
  echo
  echo "  tail -f ${LOG}"
  echo "  $(basename "$0") --status"
  echo "  kill \$(cat ${PIDFILE})"
  exit 0
fi

# ------------------------------------------------------------------ preflight
PY_BIN=""
for cand in python3 python; do
  if command -v "${cand}" >/dev/null 2>&1 && "${cand}" -c "import sys" >/dev/null 2>&1; then
    PY_BIN="${cand}"; break
  fi
done
[[ -n "${PY_BIN}" ]] || { echo "ERROR: no working python found (tried python3, python)" >&2; exit 1; }

if ! "${PY_BIN}" -c "import huggingface_hub" 2>/dev/null; then
  echo "huggingface_hub not installed; installing into the current environment"
  "${PY_BIN}" -m pip install --quiet --upgrade "huggingface_hub>=0.23" || {
    echo "ERROR: could not install huggingface_hub. Try:" >&2
    echo "  ${PY_BIN} -m venv .venv && . .venv/bin/activate && pip install huggingface_hub" >&2
    exit 1; }
fi

# --- token sanity. An INVALID token is worse than none: huggingface.co answers
# 401 to every request that carries one, including public repos, so a typo or a
# copied placeholder breaks the entire download rather than just gated models.
if [[ -n "${HF_TOKEN:-}" ]]; then
  case "${HF_TOKEN}" in
    hf_xxxxxxxx*|hf_XXXX*|"<"*|"your"*|"YOUR"*)
      echo "ERROR: HF_TOKEN is the placeholder value: ${HF_TOKEN}" >&2
      echo "       Either export a real token or unset it:" >&2
      echo "         unset HF_TOKEN" >&2
      exit 2 ;;
  esac
  if command -v curl >/dev/null 2>&1; then
    code="$(curl -s -o /dev/null -w '%{http_code}' -m 15       -H "Authorization: Bearer ${HF_TOKEN}"       https://huggingface.co/api/whoami-v2 2>/dev/null)"
    if [[ "${code}" == "200" ]]; then
      who="$(curl -s -m 15 -H "Authorization: Bearer ${HF_TOKEN}"         https://huggingface.co/api/whoami-v2 2>/dev/null         | grep -o '"name":"[^"]*"' | head -1 | cut -d'"' -f4)"
      echo "token      : valid (user ${who:-?})"
    else
      echo "ERROR: HF_TOKEN rejected by huggingface.co (HTTP ${code})." >&2
      echo "       A bad token 401s PUBLIC repos too, so this would fail" >&2
      echo "       everything, not just gated models. Fix or unset it:" >&2
      echo "         unset HF_TOKEN" >&2
      exit 2
    fi
  fi
else
  echo "token      : none (gated repos will fail; public ones are fine)"
fi

export HF_HOME="${DEST}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
# Keep tqdm on: the progress bars are the whole point when tailing the log.
unset HF_HUB_DISABLE_PROGRESS_BARS 2>/dev/null || true

free_gib() { df -Pk "${DEST}" | awk 'NR==2 {printf "%.1f", $4/1048576}'; }

echo "======================================================================"
echo "started    : $(date -Is 2>/dev/null || date)"
echo "python     : ${PY_BIN} ($(${PY_BIN} -c 'import sys;print(sys.version.split()[0])'))"
echo "cache root : ${DEST}   (HF_HOME)"
echo "tiers      : ${TIERS}"
echo "free space : $(free_gib) GiB"
echo "======================================================================"

ok=(); skipped=(); failed=()
start_all=$SECONDS

for row in "${MODELS[@]}"; do
  IFS='|' read -r tier repo ignore approx <<<"${row}"

  if [[ ",${TIERS}," != *",${tier},"* ]]; then
    skipped+=("${repo} (tier ${tier} not selected)"); continue
  fi

  avail="$(free_gib)"
  if awk -v a="${avail}" -v m="${MIN_FREE_GIB}" 'BEGIN{exit !(a < m)}'; then
    skipped+=("${repo} (only ${avail} GiB free, need ${MIN_FREE_GIB})"); continue
  fi

  if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "would fetch  ${repo}  (tier ${tier}, ~${approx} GiB, ignore: ${ignore})"
    ok+=("${repo} [dry-run]"); continue
  fi

  echo
  echo "=== ${repo}   tier ${tier}   ~${approx} GiB   ${avail} GiB free   $(date -Is 2>/dev/null || date)"
  t0=$SECONDS
  if REPO="${repo}" IGNORE="${ignore}" "${PY_BIN}" - <<'PY'
import os, sys
from huggingface_hub import snapshot_download

repo = os.environ["REPO"]
extra = [g for g in os.environ.get("IGNORE", "-").split(",") if g and g != "-"]
# Weights only. Most repos ship duplicate .bin/.pth copies beside the
# safetensors; skipping them roughly halves the transfer.
ignore = ["*.pth", "*.bin", "*.h5", "*.msgpack", "*.ot", "*.tflite"] + extra
try:
    path = snapshot_download(repo_id=repo, ignore_patterns=ignore, max_workers=4)
except Exception as exc:
    print("    %s: %s" % (type(exc).__name__, str(exc)[:300]), file=sys.stderr)
    sys.exit(1)

# A snapshot directory contains ONLY symlinks into ../../blobs, so skipping
# symlinks (as this did) reported 0.00 GiB for every model. Follow them, and
# dedupe by real path so a blob shared by two files is counted once.
seen, total = set(), 0
for root, _, files in os.walk(path):
    for f in files:
        rp = os.path.realpath(os.path.join(root, f))
        if rp in seen:
            continue
        seen.add(rp)
        try:
            total += os.path.getsize(rp)
        except OSError:
            pass
print("    ok  %.2f GiB in %d files  ->  %s" % (total / 2**30, len(seen), path))
PY
  then
    echo "    took $(( SECONDS - t0 ))s"
    ok+=("${repo}")
  else
    failed+=("${repo}")
  fi
done

echo
echo "================= SUMMARY  $(date -Is 2>/dev/null || date) ================="
for r in "${ok[@]:-}";      do [[ -n "${r}" ]] && echo "  ok       ${r}"; done
for r in "${skipped[@]:-}"; do [[ -n "${r}" ]] && echo "  skipped  ${r}"; done
for r in "${failed[@]:-}";  do [[ -n "${r}" ]] && echo "  FAILED   ${r}"; done
echo
echo "elapsed    : $(( SECONDS - start_all ))s"
echo "cache root : ${DEST}"
echo "size       : $(du -sh "${DEST}" 2>/dev/null | cut -f1)"
echo "free       : $(free_gib) GiB"
rm -f "${PIDFILE}"
echo
echo "Next:"
echo "  ./verify-models.sh --dest ${DEST}          # complete and loadable?"
echo "  kubectl apply -f 01-models-storage.yaml    # publish ${DEST} as r2-models-pvc"

[[ ${#failed[@]} -eq 0 ]] || exit 1
exit 0
