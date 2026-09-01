#!/usr/bin/env bash
# Give each model a flat directory that KServe can mount directly.
#
# THE PROBLEM, straight from the crashloop log:
#
#   non-default args: {'model': '/mnt/models', 'served_model_name': ['BAAI/bge-m3'],
#                      'runner': 'pooling', 'convert': 'embed', 'dtype': 'half'}
#   ValidationError: Invalid repository ID or local directory specified: '/mnt/models'
#   - For Hugging Face models: ensure the presence of a 'config.json'.
#
# KServe passes spec.model.uri as the MOUNT and sets vLLM's --model to the mount
# point, using spec.model.name only as --served-model-name. So whatever the uri
# points at must itself look like a model directory: config.json at the top.
#
# Our PVC root is /data2/hf-cache, an HF CACHE tree:
#     hub/models--BAAI--bge-m3/snapshots/<sha>/config.json
# There is no config.json at the root, so vLLM refuses. Round one worked only
# because its PVC happened to point at a directory holding the model files
# directly.
#
# WHY HARDLINKS AND NOT SYMLINKS
# The snapshot directories are already symlink farms pointing at ../../blobs/.
# KServe mounts a pvc:// subpath with subPath, so a relative symlink escaping
# the subtree resolves to nothing inside the container. Hardlinks are real
# directory entries for the same inode: no extra disk (same filesystem), and
# nothing to resolve at read time.
#
#   ./08-flatten-models.sh            # build /data2/hf-cache/models/<role>
#   ./08-flatten-models.sh --check    # report only
#   ./08-flatten-models.sh --force    # rebuild even if present
set -uo pipefail

HF_ROOT="${HF_ROOT:-/data2/hf-cache}"
DEST_ROOT="${DEST_ROOT:-${HF_ROOT}/models}"
CHECK=0
FORCE=0
for a in "$@"; do
  case "$a" in
    --check) CHECK=1 ;;
    --force) FORCE=1 ;;
    -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "unknown option: $a" >&2; exit 2 ;;
  esac
done

# role|repo   -- must match render.py's MODELS table
ROLES=(
  "llm|Qwen/Qwen2.5-7B-Instruct-AWQ"
  "llm2|Qwen/Qwen2.5-1.5B-Instruct-AWQ"
  "llm-mid|Qwen/Qwen2.5-3B-Instruct"
  "embed|BAAI/bge-m3"
  "rerank|BAAI/bge-reranker-v2-m3"
  "embed-small|BAAI/bge-large-en-v1.5"
  "vl|Qwen/Qwen2-VL-2B-Instruct"
)

echo "cache  ${HF_ROOT}"
echo "dest   ${DEST_ROOT}"
df -Ph "${HF_ROOT}" 2>/dev/null | tail -1 | sed 's/^/free   /'
echo
printf '%-13s %-34s %8s %6s  %s\n' ROLE REPO SIZE FILES STATUS
printf '%s\n' "----------------------------------------------------------------------------------"

rc=0
for entry in "${ROLES[@]}"; do
  role="${entry%%|*}"
  repo="${entry##*|}"
  org="${repo%%/*}"
  name="${repo##*/}"
  snapdir="${HF_ROOT}/hub/models--${org}--${name}/snapshots"
  dest="${DEST_ROOT}/${role}"

  if [[ ! -d "${snapdir}" ]]; then
    printf '%-13s %-34s %8s %6s  %s\n' "${role}" "${repo}" "-" "-" "NO CACHE"
    rc=1; continue
  fi
  # Newest snapshot, in case a repo was re-downloaded at a different revision.
  snap="$(ls -1dt "${snapdir}"/*/ 2>/dev/null | head -1)"
  if [[ -z "${snap}" || ! -f "${snap}/config.json" ]]; then
    printf '%-13s %-34s %8s %6s  %s\n' "${role}" "${repo}" "-" "-" "NO config.json"
    rc=1; continue
  fi

  # A flat directory is keyed on the ROLE, not the repo, so changing a role's
  # repo in the table leaves the OLD model sitting at the same path. Skipping
  # it because config.json exists is how llm2 kept serving Qwen2.5-1.5B-Instruct
  # after the table moved to the -AWQ build: vLLM read a config.json with no
  # quantization_config and died with
  #     Cannot find the config file for awq_marlin
  # which reads like a missing file and is really a stale directory. The marker
  # makes the mismatch visible, and rebuilding is automatic.
  built_from=""
  [[ -f "${dest}/.source-repo" ]] && built_from="$(cat "${dest}/.source-repo")"
  if [[ -f "${dest}/config.json" && "${FORCE}" -eq 0 && "${built_from}" == "${repo}" ]]; then
    sz="$(du -sh --dereference "${dest}" 2>/dev/null | cut -f1)"
    n="$(find "${dest}" -type f -not -name .source-repo | wc -l)"
    printf '%-13s %-34s %8s %6s  %s\n' "${role}" "${repo}" "${sz}" "${n}" "present"
    continue
  fi
  if [[ -f "${dest}/config.json" && "${FORCE}" -eq 0 ]]; then
    if [[ -n "${built_from}" ]]; then
      why="STALE, holds ${built_from}"
    else
      why="no marker, rebuilding to verify"
    fi
    printf '%-13s %-34s %8s %6s  %s\n' "${role}" "${repo}" "-" "-" "${why}"
  fi
  if [[ "${CHECK}" -eq 1 ]]; then
    printf '%-13s %-34s %8s %6s  %s\n' "${role}" "${repo}" "-" "-" "NOT BUILT"
    rc=1; continue
  fi

  rm -rf "${dest}"
  mkdir -p "${dest}"
  linked=0
  copied=0
  # RECURSIVE, and it has to be. An earlier version took only the top level,
  # which is fine for the Qwen models but silently broke all three
  # sentence-transformers ones: modules.json survived while the 1_Pooling/
  # directory it points at did not. vLLM then read modules.json, failed to find
  # the pooling config on disk, and fell back to fetching it from the Hub using
  # the local path as a repo id:
  #
  #   Repo id must be in the form 'repo_name' or 'namespace/repo_name':
  #   '/mnt/models'
  #
  # It cost exactly one file per model -- bge-m3 15 vs 14, reranker 9 vs 8,
  # bge-large 12 vs 11 -- which is why the counts are now checked below.
  #
  # Recursing is safe: the onnx/openvino variants were excluded at download
  # time by ignore_patterns, so they are not in the snapshot to begin with.
  while IFS= read -r f; do
    rel="${f#"${snap}"}"
    rel="${rel#/}"
    real="$(readlink -f "${f}")"
    [[ -f "${real}" ]] || continue
    mkdir -p "${dest}/$(dirname "${rel}")"
    if ln "${real}" "${dest}/${rel}" 2>/dev/null; then
      linked=$((linked+1))
    elif cp -a "${real}" "${dest}/${rel}" 2>/dev/null; then
      copied=$((copied+1))
    fi
  done < <(find "${snap}" -mindepth 1 \( -type f -o -type l \))

  # A count mismatch means a file did not make it across. That failure mode is
  # invisible until vLLM tries to read the missing file minutes later, so it is
  # worth one find per model here.
  want="$(find "${snap}" -mindepth 1 \( -type f -o -type l \) | wc -l)"
  got="$(find "${dest}" -type f | wc -l)"
  if [[ "${want}" -ne "${got}" ]]; then
    printf '%-13s %-34s %8s %6s  %s\n' "${role}" "${repo}" "-" \
      "${got}/${want}" "INCOMPLETE"
    rc=1; continue
  fi

  if [[ -f "${dest}/config.json" ]]; then
    # Written AFTER the count check, so the marker cannot skew want vs got.
    printf '%s' "${repo}" > "${dest}/.source-repo"
    sz="$(du -sh --dereference "${dest}" 2>/dev/null | cut -f1)"
    note="${linked} hardlinked"
    [[ "${copied}" -gt 0 ]] && note="${note}, ${copied} COPIED (cross-device)"
    printf '%-13s %-34s %8s %6s  %s\n' "${role}" "${repo}" "${sz}" \
      "$((linked+copied))" "${note}"
  else
    printf '%-13s %-34s %8s %6s  %s\n' "${role}" "${repo}" "-" "-" "FAILED"
    rc=1
  fi
done

echo
echo "--- disk after (hardlinks cost nothing; copies would show here)"
df -Ph "${HF_ROOT}" 2>/dev/null | tail -1 | sed 's/^/    /'
echo
if [[ "${rc}" -eq 0 ]]; then
  echo "All roles have a flat directory. render.py points spec.model.uri at"
  echo "pvc://r2-models-pvc/models/<role>, so KServe mounts a real model tree"
  echo "with config.json at its root, which is what vLLM demands."
else
  echo "Some roles are not ready. Build them with:  ./08-flatten-models.sh"
fi
exit "${rc}"
