#!/usr/bin/env bash
# Verify downloaded models — three depths, cheapest first.
#
#   ./verify-models.sh                  # structural: files present and consistent
#   ./verify-models.sh --deep           # + parse config, count params, check dtype
#   ./verify-models.sh --serve embed    # + actually load it in vLLM and query it
#
# The structural pass is the one that catches the common failure: a download
# that stopped part-way leaves a plausible-looking directory whose shard index
# references files that are not there. vLLM only discovers that minutes into a
# deployment, so it is worth ten seconds here.
set -uo pipefail

DEST="/data2/hf-cache"
DEEP=0
SERVE=""
NS="${NS:-llm-demo}"
IMG="${IMG:-127.0.0.1:5000/kserve/huggingfaceserver:v0.17.0-gpu}"

usage() {
  cat <<'USAGE'
Usage: verify-models.sh [options]

  --dest DIR      cache root (default /data2/hf-cache)
  --deep          parse configs, sum parameters, report dtype and KV geometry
  --serve ROLE    load one model in vLLM inside the cluster and send a real
                  request. ROLE is one of: llm, llm2, embed, rerank, vl
  -h, --help      this text

EXIT
  0  everything checked is usable
  1  at least one model is incomplete or unloadable
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dest)  DEST="${2:-}"; shift 2 ;;
    --deep)  DEEP=1; shift ;;
    --serve) SERVE="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

PY_BIN=""
for cand in python3 python; do
  if command -v "${cand}" >/dev/null 2>&1 && "${cand}" -c "import sys" >/dev/null 2>&1; then
    PY_BIN="${cand}"; break
  fi
done
[[ -n "${PY_BIN}" ]] || { echo "ERROR: no working python found" >&2; exit 1; }

# ---------------------------------------------------------------- structural
DEST="${DEST}" DEEP="${DEEP}" "${PY_BIN}" - <<'PY'
import json, os, sys

DEST = os.environ["DEST"]
DEEP = os.environ.get("DEEP") == "1"
HUB = os.path.join(DEST, "hub")

# role -> repo, mirroring render.py so the two cannot disagree
EXPECT = {
    "llm":         "Qwen/Qwen2.5-7B-Instruct-AWQ",
    "llm2":        "Qwen/Qwen2.5-1.5B-Instruct",
    "llm-small":   "Qwen/Qwen2.5-0.5B-Instruct",
    "llm-mid":     "Qwen/Qwen2.5-3B-Instruct",
    "embed":       "BAAI/bge-m3",
    "rerank":      "BAAI/bge-reranker-v2-m3",
    "embed-small": "BAAI/bge-large-en-v1.5",
    "vl":          "Qwen/Qwen2-VL-2B-Instruct",
    "whisper":     "openai/whisper-large-v3-turbo",
    "llama":       "meta-llama/Llama-3.2-3B-Instruct",
}

# Gated and deliberately optional: reported but never counted as a failure.
OPTIONAL = {"llama"}


def snapshot(repo):
    d = os.path.join(HUB, "models--" + repo.replace("/", "--"), "snapshots")
    if not os.path.isdir(d):
        return None
    subs = [os.path.join(d, x) for x in os.listdir(d)]
    subs = [x for x in subs if os.path.isdir(x)]
    return max(subs, key=os.path.getmtime) if subs else None


def real(p):
    """Resolve the symlink into blobs/ and report the true size."""
    try:
        return os.path.getsize(os.path.realpath(p))
    except OSError:
        return 0


def check(repo):
    snap = snapshot(repo)
    if not snap:
        return "MISSING", "not downloaded", {}
    files = os.listdir(snap)
    problems = []

    cfgp = os.path.join(snap, "config.json")
    cfg = {}
    if not os.path.exists(cfgp):
        problems.append("no config.json")
    else:
        try:
            cfg = json.load(open(cfgp, encoding="utf-8"))
        except Exception as e:
            problems.append("config.json unparsable: %s" % e)

    # Shard index must reference files that actually exist. This is the check
    # that catches a truncated download.
    # Weights may legitimately be .bin rather than .safetensors -- BAAI/bge-m3
    # ships only pytorch_model.bin. Checking for safetensors alone reported a
    # perfectly good repo as BROKEN.
    idx = [f for f in files if f.endswith(".safetensors.index.json")
                            or f.endswith(".bin.index.json")]
    weights = [f for f in files
               if f.endswith(".safetensors") or f.endswith(".bin")]
    if idx:
        try:
            wm = json.load(open(os.path.join(snap, idx[0]), encoding="utf-8"))
            need = sorted(set(wm.get("weight_map", {}).values()))
            missing = [n for n in need if not os.path.exists(os.path.join(snap, n))]
            empty = [n for n in need if real(os.path.join(snap, n)) == 0]
            if missing:
                problems.append("%d/%d shards missing (%s)"
                                % (len(missing), len(need), missing[0]))
            if empty:
                problems.append("%d shards are 0 bytes" % len(empty))
        except Exception as e:
            problems.append("shard index unreadable: %s" % e)
    elif not weights:
        problems.append("no weights found (.safetensors or .bin)")
    else:
        if any(real(os.path.join(snap, w)) == 0 for w in weights):
            problems.append("a weights file is 0 bytes")

    has_tok = any(f in files for f in
                  ("tokenizer.json", "tokenizer_config.json", "tokenizer.model",
                   "vocab.json", "spiece.model", "preprocessor_config.json"))
    if not has_tok:
        problems.append("no tokenizer/preprocessor files")

    total = sum(real(os.path.join(snap, f)) for f in files)
    info = dict(size_gib=total / 2**30, files=len(files), snap=snap)

    if DEEP and cfg:
        info["arch"] = (cfg.get("architectures") or ["?"])[0]
        info["dtype"] = cfg.get("torch_dtype", "?")
        q = cfg.get("quantization_config") or {}
        if q:
            info["quant"] = "%s-%sbit" % (q.get("quant_method", "?"), q.get("bits", "?"))
        layers = cfg.get("num_hidden_layers") or cfg.get("num_layers")
        kvh = cfg.get("num_key_value_heads") or cfg.get("num_attention_heads")
        heads = cfg.get("num_attention_heads")
        hid = cfg.get("hidden_size")
        arch = info.get("arch", "")
        pooling = any(k in arch for k in
                      ("Embedding", "SequenceClassification", "BertModel",
                       "XLMRoberta", "Pooling"))
        if pooling:
            # No KV cache exists for an encoder/pooling model. Printing one
            # invites exactly the wrong capacity planning.
            info["kv_note"] = "encoder/pooling - no KV cache"
        elif layers and kvh and heads and hid:
            head_dim = cfg.get("head_dim") or (hid // heads)
            # 2 (K and V) * layers * kv_heads * head_dim * 2 bytes, in KiB
            info["kv_kib_per_tok"] = 2 * layers * kvh * head_dim * 2 / 1024
        info["max_pos"] = cfg.get("max_position_embeddings")

    return ("OK" if not problems else "BROKEN"), "; ".join(problems), info


print("cache: %s\n" % DEST)
hdr = "%-14s %-34s %7s %6s  %s" % ("ROLE", "REPO", "GiB", "FILES", "STATUS")
print(hdr)
print("-" * len(hdr))
bad = 0
present = 0
for role, repo in EXPECT.items():
    state, why, info = check(repo)
    if state == "MISSING":
        continue                      # not every tier is expected to be here
    present += 1
    if state != "OK":
        if role in OPTIONAL:
            print("%-14s %-34s %7s %6s  SKIPPED (gated, optional)"
                  % (role, repo, "-", "-"))
            present -= 1
            continue
        bad += 1
    print("%-14s %-34s %7.2f %6d  %s%s"
          % (role, repo, info.get("size_gib", 0), info.get("files", 0),
             state, (" — " + why) if why else ""))
    if DEEP and state == "OK":
        bits = []
        for k in ("arch", "dtype", "quant", "max_pos"):
            if info.get(k) is not None and k in info:
                bits.append("%s=%s" % (k, info[k]))
        if "kv_kib_per_tok" in info:
            bits.append("KV=%.0f KiB/tok" % info["kv_kib_per_tok"])
        elif info.get("kv_note"):
            bits.append(info["kv_note"])
        if bits:
            print("%-14s %s" % ("", "  ".join(bits)))

print()
if present == 0:
    print("nothing downloaded yet under %s" % DEST)
    sys.exit(1)
print("%d model(s) present, %d broken" % (present, bad))
sys.exit(1 if bad else 0)
PY
structural=$?

# ------------------------------------------------------------------- serving
if [[ -n "${SERVE}" ]]; then
  echo
  echo "=== live serving check: role ${SERVE}"
  command -v kubectl >/dev/null 2>&1 || { echo "kubectl not found" >&2; exit 1; }

  case "${SERVE}" in
    llm)    REPO="Qwen/Qwen2.5-7B-Instruct-AWQ"; MODE=chat;  EXTRA="--quantization=awq_marlin" ;;
    llm2)   REPO="Qwen/Qwen2.5-1.5B-Instruct";   MODE=chat;  EXTRA="" ;;
    vl)     REPO="Qwen/Qwen2-VL-2B-Instruct";    MODE=chat;  EXTRA="" ;;
    embed)  REPO="BAAI/bge-m3";                  MODE=embed; EXTRA="--runner=pooling --convert=embed --dtype=half" ;;
    rerank) REPO="BAAI/bge-reranker-v2-m3";      MODE=score; EXTRA="--runner=pooling --convert=classify --dtype=half" ;;
    *) echo "unknown role: ${SERVE}" >&2; exit 2 ;;
  esac

  POD="r2-verify-${SERVE}"
  kubectl delete pod "${POD}" -n "${NS}" --ignore-not-found >/dev/null 2>&1
  echo "starting ${POD} with ${REPO} (this loads real weights; give it 3-5 min)"

  # No nvidia.com/gpumem here on purpose: this is a functional check, not a
  # measurement, so it should not contend with anything being benchmarked.
  kubectl run "${POD}" -n "${NS}" --restart=Never --image="${IMG}" \
    --overrides="$(cat <<JSON
{"spec":{"containers":[{
  "name":"main","image":"${IMG}","command":["/bin/bash","-lc"],
  "args":["vllm serve ${REPO} --served-model-name=t --host=0.0.0.0 --port=8000 --gpu-memory-utilization=0.45 --max-model-len=2048 ${EXTRA} & sleep 260; echo READY-WINDOW-OVER"],
  "env":[{"name":"HF_HOME","value":"/mnt/models"},
         {"name":"HF_HUB_OFFLINE","value":"1"},
         {"name":"TRANSFORMERS_OFFLINE","value":"1"}],
  "resources":{"limits":{"nvidia.com/gpu":"1"}},
  "volumeMounts":[{"name":"models","mountPath":"/mnt/models"}]
}],
"volumes":[{"name":"models","persistentVolumeClaim":{"claimName":"r2-models-pvc"}}],
"restartPolicy":"Never"}}
JSON
)" >/dev/null

  echo -n "waiting for the server to answer /health "
  ready=0
  for _ in $(seq 1 60); do
    if kubectl exec -n "${NS}" "${POD}" -- bash -lc \
         'curl -sf http://127.0.0.1:8000/health >/dev/null' 2>/dev/null; then
      ready=1; break
    fi
    echo -n "."; sleep 5
  done
  echo

  if [[ "${ready}" -ne 1 ]]; then
    echo "server never became ready. Last 30 log lines:"
    kubectl logs -n "${NS}" "${POD}" --tail=30 2>&1 | sed 's/^/  /'
    kubectl delete pod "${POD}" -n "${NS}" --ignore-not-found >/dev/null 2>&1
    exit 1
  fi

  echo "server is up; sending a real ${MODE} request"
  case "${MODE}" in
    chat)
      kubectl exec -n "${NS}" "${POD}" -- bash -lc \
        'curl -s http://127.0.0.1:8000/v1/chat/completions -H "content-type: application/json" -d "{\"model\":\"t\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with the single word: ok\"}],\"max_tokens\":8}"' \
        | head -c 600 ;;
    embed)
      kubectl exec -n "${NS}" "${POD}" -- bash -lc \
        'curl -s http://127.0.0.1:8000/v1/embeddings -H "content-type: application/json" -d "{\"model\":\"t\",\"input\":[\"hello world\",\"second document\"]}" | python -c "import json,sys; d=json.load(sys.stdin); print(\"vectors:\", len(d.get(\"data\",[])), \"dim:\", len((d.get(\"data\") or [{}])[0].get(\"embedding\",[])), \"usage:\", d.get(\"usage\"))"' ;;
    score)
      kubectl exec -n "${NS}" "${POD}" -- bash -lc \
        'for p in /v1/rerank /rerank /v1/score /score; do echo "--- $p"; curl -s -o /tmp/o -w "%{http_code}\n" http://127.0.0.1:8000$p -H "content-type: application/json" -d "{\"model\":\"t\",\"query\":\"what is MPS\",\"documents\":[\"NVIDIA MPS shares a GPU\",\"unrelated text\"]}"; head -c 300 /tmp/o; echo; done' ;;
  esac
  echo
  echo
  echo "cleaning up"
  kubectl delete pod "${POD}" -n "${NS}" --ignore-not-found >/dev/null 2>&1
  echo "If the request above returned real content, this model is servable."
fi

exit "${structural}"
