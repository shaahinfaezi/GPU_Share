#!/usr/bin/env python3
"""Ask vLLM what it can do, by interrogating objects rather than text.

Preflight section 4 reported "routes: none found" and printed nothing for the
serve flags. Neither was vLLM's answer -- both were the probe being wrong:

  * the flag check shelled out to `vllm serve --help` with stderr sent to
    /dev/null, so if the console script is not on PATH in the KServe image the
    failure was invisible and read as "these flags do not exist";

  * the route check regex-scanned the SOURCE TEXT of api_server.py for the
    literal "/v1/embeddings". Modern vLLM registers those paths through
    decorators and helper modules, so the literal is not in that one file.
    Absence of a string is not absence of a route.

This file builds the real argparse parser and walks the real APIRouter, so it
cannot fail in that direction. Run it inside the serving image:

    kubectl exec -i -n llm-demo r2-probe -- python - < probe_vllm.py
"""
import importlib
import sys


def head(t):
    print("\n--- %s" % t)


head("versions")
print("  python  %s" % sys.version.split()[0])
try:
    import vllm
    print("  vllm    %s" % vllm.__version__)
except Exception as exc:
    print("  vllm    IMPORT FAILED %r" % exc)
    raise SystemExit(1)

# 1 ----------------------------------------------- the real argparse parser
head("serve flags (from vllm's own parser, not from --help text)")
WANT = ["--task", "--runner", "--convert", "--kv-cache-dtype", "--dtype",
        "--quantization", "--gpu-memory-utilization", "--max-model-len",
        "--enable-prefix-caching", "--max-num-seqs", "--served-model-name",
        "--enable-chunked-prefill", "--disable-log-requests", "--port"]
try:
    from vllm.entrypoints.openai.cli_args import make_arg_parser
    try:
        from vllm.utils import FlexibleArgumentParser
    except Exception:
        from vllm.utils.argparse_utils import FlexibleArgumentParser
    parser = make_arg_parser(FlexibleArgumentParser())
    opts = {}
    for act in parser._actions:
        for s in act.option_strings:
            opts[s] = act.choices
    for flag in WANT:
        if flag in opts:
            ch = opts[flag]
            ch = ",".join(str(c) for c in ch) if ch else "-"
            print("  %-28s PRESENT   choices: %s" % (flag, ch[:88]))
        else:
            print("  %-28s absent" % flag)
except Exception as exc:
    print("  parser probe failed: %r" % exc)

# 2 ------------------------------------------ the real FastAPI route table
head("registered HTTP routes (walked from the APIRouter objects)")
POOLY = ("embed", "rerank", "score", "pooling", "classify")
try:
    from vllm.entrypoints.openai import api_server as A
    paths = set()
    for name in dir(A):
        try:
            obj = getattr(A, name)
        except Exception:
            continue
        for r in getattr(obj, "routes", []) or []:
            p = getattr(r, "path", None)
            if isinstance(p, str):
                paths.add(p)
    if paths:
        for p in sorted(paths):
            mark = "   <== POOLING" if any(k in p for k in POOLY) else ""
            print("  %-36s%s" % (p, mark))
    else:
        print("  no routes on any module-level router; see the class check below")
except Exception as exc:
    print("  route enumeration failed: %r" % exc)

# 3 ------------------------------------------ the handler classes themselves
head("pooling handler classes")
CAND = [
    ("vllm.entrypoints.openai.serving_embedding", "OpenAIServingEmbedding", "embeddings"),
    ("vllm.entrypoints.openai.serving_pooling", "OpenAIServingPooling", "raw pooling"),
    ("vllm.entrypoints.openai.serving_score", "ServingScores", "rerank / score"),
    ("vllm.entrypoints.openai.serving_classification", "ServingClassification", "classify"),
]
for mod, cls, what in CAND:
    short = mod.split(".")[-1]
    try:
        m = importlib.import_module(mod)
        if hasattr(m, cls):
            print("  OK          %s.%s  (%s)" % (short, cls, what))
        else:
            print("  NO CLASS    %s lacks %s  (%s)" % (short, cls, what))
    except Exception as exc:
        print("  NO MODULE   %s  (%s: %s)" % (short, type(exc).__name__, what))

# 4 --------------------------------------------- runner / convert vocabulary
head("runner / convert accepted values")
try:
    from vllm.config import ModelConfig
    ann = getattr(ModelConfig, "__annotations__", {})
    found = False
    for f in ("runner", "convert", "task"):
        if f in ann:
            print("  ModelConfig.%-8s %s" % (f, str(ann[f])[:100]))
            found = True
    if not found:
        print("  (none of runner/convert/task on ModelConfig; rely on the")
        print("   parser choices printed above)")
except Exception as exc:
    print("  (%s)" % type(exc).__name__)

# 5 ------------------------------------------ KServe's argument passthrough
head("KServe huggingfaceserver -- does it forward unknown args to vLLM?")
MARKERS = ("parse_known_args", "AsyncEngineArgs", "make_arg_parser",
           "build_vllm_engine_args", "maybe_add_vllm")
try:
    import glob
    import os
    import huggingfaceserver
    d = os.path.dirname(huggingfaceserver.__file__)
    print("  at %s" % d)
    seen = set()
    for f in glob.glob(os.path.join(d, "**", "*.py"), recursive=True):
        try:
            s = open(f, encoding="utf-8", errors="ignore").read()
        except Exception:
            continue
        for pat in MARKERS:
            if pat in s:
                key = (os.path.relpath(f, d), pat)
                if key not in seen:
                    seen.add(key)
                    print("    %-42s %s" % key)
    if seen:
        pats = {p for _, p in seen}
        if "parse_known_args" in pats:
            print("  parse_known_args present => unrecognised flags ARE passed through")
        else:
            print("  engine-args markers present but no parse_known_args:")
            print("  KServe may build its own arg list, i.e. a whitelist")
    else:
        print("    no passthrough markers found -- KServe likely whitelists args,")
        print("    in which case pooling models need a plain vllm Deployment")
except Exception as exc:
    print("  huggingfaceserver probe failed: %r" % exc)

print("")
