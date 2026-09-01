#!/usr/bin/env bash
# Stage the prompt corpora BEFORE a run, onto /data2.
#
# TWO TRAPS THIS CLOSES
#
# 1. --source sharegpt calls hf_hub_download at trace-generation time. The
#    ShareGPT file is roughly 650 MB and lands in HF_HOME, which defaults to
#    ~/.cache/huggingface on `/`. `/` currently has ~13 GiB free against a
#    kubelet eviction threshold of 4.84 GiB, and eviction is node-wide: it
#    would take running benchmark pods with it. Staged here, it goes to
#    /data2/hf-cache next to the models, where there is 45 GiB.
#
# 2. --source azure needs AZURE_TRACE_CSV to point at a real file. If it does
#    not, gen_trace.py prints one warning line and FALLS BACK TO SYNTHETIC.
#    T6 exists specifically to replay production arrival shapes, so a silent
#    fallback would leave T6 measuring exactly what T1 measures while claiming
#    otherwise. run.py now aborts on that mismatch, but the real fix is to
#    have the file present.
#
#   ./07-prefetch-traces.sh              # fetch both
#   ./07-prefetch-traces.sh --check      # report what is staged, download nothing
set -uo pipefail

HF_ROOT="${HF_ROOT:-/data2/hf-cache}"
TRACE_DIR="${TRACE_DIR:-/data2/hf-cache/traces}"
CHECK=0
for a in "$@"; do
  case "$a" in
    --check) CHECK=1 ;;
    -h|--help) sed -n '2,22p' "$0"; exit 0 ;;
    *) echo "unknown option: $a" >&2; exit 2 ;;
  esac
done

export HF_HOME="${HF_ROOT}"
AZ_BASE="https://raw.githubusercontent.com/Azure/AzurePublicDataset/master/data"
AZ_FILES=("AzureLLMInferenceTrace_conv.csv" "AzureLLMInferenceTrace_code.csv")

echo "HF_HOME    ${HF_HOME}"
echo "traces     ${TRACE_DIR}"
df -Ph "${HF_ROOT}" 2>/dev/null | tail -1 | sed 's/^/free       /'
echo

# ---------------------------------------------------------------- sharegpt
SG_REPO="anon8231489123/ShareGPT_Vicuna_unfiltered"
SG_FILE="ShareGPT_V3_unfiltered_cleaned_split.json"
echo "=== ShareGPT (${SG_FILE})"
sg_path="$(python3 - <<PY 2>/dev/null
import os
os.environ["HF_HOME"] = "${HF_ROOT}"
try:
    from huggingface_hub import try_to_load_from_cache
    p = try_to_load_from_cache("${SG_REPO}", "${SG_FILE}", repo_type="dataset")
    print(p if isinstance(p, str) else "")
except Exception:
    print("")
PY
)"
if [[ -n "${sg_path}" && -f "${sg_path}" ]]; then
  echo "  cached: ${sg_path}"
  echo "  size  : $(du -h "${sg_path}" 2>/dev/null | cut -f1)"
elif [[ "${CHECK}" -eq 1 ]]; then
  echo "  NOT staged. Run without --check."
else
  echo "  downloading (~650 MB) into ${HF_ROOT} ..."
  python3 - <<PY
import os
os.environ["HF_HOME"] = "${HF_ROOT}"
from huggingface_hub import hf_hub_download
p = hf_hub_download("${SG_REPO}", "${SG_FILE}", repo_type="dataset")
print("  ->", p, "%.1f MB" % (os.path.getsize(p) / 1048576.0))
PY
fi

# ------------------------------------------------------------------- azure
echo
echo "=== Azure production inference traces"
mkdir -p "${TRACE_DIR}" 2>/dev/null
for f in "${AZ_FILES[@]}"; do
  dest="${TRACE_DIR}/${f}"
  if [[ -s "${dest}" ]]; then
    echo "  present: ${f}  ($(du -h "${dest}" | cut -f1), $(( $(wc -l < "${dest}") - 1 )) rows)"
  elif [[ "${CHECK}" -eq 1 ]]; then
    echo "  MISSING: ${f}"
  else
    echo "  fetching ${f}"
    if curl -sfL --max-time 300 -o "${dest}.part" "${AZ_BASE}/${f}"; then
      mv "${dest}.part" "${dest}"
      echo "    -> $(du -h "${dest}" | cut -f1), $(( $(wc -l < "${dest}") - 1 )) rows"
    else
      rm -f "${dest}.part"
      echo "    FAILED. T6 will fall back to synthetic lengths, and run.py"
      echo "    will refuse the run rather than let it pretend otherwise."
    fi
  fi
done

echo
echo "=== export these for a run"
echo "  export HF_HOME=${HF_ROOT}"
if [[ -s "${TRACE_DIR}/${AZ_FILES[0]}" ]]; then
  echo "  export AZURE_TRACE_CSV=${TRACE_DIR}/${AZ_FILES[0]}"
else
  echo "  # AZURE_TRACE_CSV unset -- T6 is not runnable until the CSV is staged"
fi
echo
echo "run.py exports both automatically when they are not already set,"
echo "so this only has to be right once."
