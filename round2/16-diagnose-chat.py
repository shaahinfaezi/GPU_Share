#!/usr/bin/env python3
"""Why does every generative request fail while the smoke test passes?

THE EVIDENCE
  smoke test          200, ok        payload has "stream": False
  replay, 0.5 req/s   0/40 ok        payload has "stream": True
  embeddings          868/868 ok     non-streaming
  rate is irrelevant  0/40 at 0.5/s, 0/102 at 1/s, 0/152 at 1.5/s

Rate independence rules out overload, which was my hypothesis and was wrong.
The only difference between the passing and failing paths is streaming.

One detail makes this fail SILENTLY. do_chat ends with

    rec["ok"] = rec["t_first"] is not None

so a response that streams but never yields choices[0].delta.content leaves
ok=False with HTTP 200 and an empty error field -- which is why the first error
dump showed nothing at all.

    python 16-diagnose-chat.py                 # read results, then live A/B
    python 16-diagnose-chat.py --from-results  # read results only, instant
"""
import argparse
import csv
import collections
import glob
import json
import os
import subprocess
import sys
import time

import harness as H
import render as R

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable


def from_results():
    print("=== what the failed requests recorded")
    files = (sorted(glob.glob(os.path.join(HERE, "calibration", "llm-*", "r*",
                                           "replay.requests.csv")))
             + sorted(glob.glob(os.path.join(HERE, "results", "*", "*llm*",
                                             "replay.requests.csv"))))
    if not files:
        print("   no replay.requests.csv found")
        return
    for f in files[:3]:
        rows = list(csv.DictReader(open(f, encoding="utf-8")))
        rel = os.path.relpath(f, HERE)
        print("\n   %s  (%d rows)" % (rel, len(rows)))
        c = collections.Counter(
            (r.get("status"), r.get("ok"), r.get("chunks"),
             (r.get("error") or "")[:160]) for r in rows)
        for (st, ok, ch, err), n in c.most_common(5):
            print("     status=%-5s ok=%-6s chunks=%-4s x%-4d %s"
                  % (st, ok, ch, n, err or "(no error text)"))
        if rows:
            print("     first row:")
            for k, v in rows[0].items():
                if k in ("prompt", "response_text"):
                    v = str(v)[:60] + "..."
                print("       %-15s %s" % (k, str(v)[:150]))


def live_ab(endpoint, model):
    """Send both payload shapes and print exactly what comes back."""
    import requests
    print("\n=== live A/B against %s" % endpoint)
    prompt = "Count from one to five."

    base = {"model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 32, "temperature": 0.0}

    print("\n--- A: non-streaming (what the smoke test sends)")
    try:
        r = requests.post(endpoint + "/v1/chat/completions",
                          json=dict(base, stream=False), timeout=120)
        print("    HTTP %s" % r.status_code)
        print("    %s" % r.text[:400])
    except Exception as exc:
        print("    EXCEPTION %r" % exc)

    for label, extra in (("B: streaming, with stream_options",
                          {"stream": True,
                           "stream_options": {"include_usage": True}}),
                         ("C: streaming, WITHOUT stream_options",
                          {"stream": True})):
        print("\n--- %s (what replay.py sends)" % label)
        try:
            with requests.post(endpoint + "/v1/chat/completions",
                               json=dict(base, **extra), timeout=120,
                               stream=True) as r:
                print("    HTTP %s   content-type=%s"
                      % (r.status_code, r.headers.get("content-type")))
                if not r.ok:
                    print("    body: %s" % r.text[:400])
                    continue
                lines, deltas, first = 0, 0, None
                t0 = time.perf_counter()
                for raw in r.iter_lines(decode_unicode=True):
                    if raw is None:
                        continue
                    lines += 1
                    if lines <= 3:
                        print("    raw[%d]: %s" % (lines, str(raw)[:200]))
                    if not raw.startswith("data:"):
                        continue
                    data = raw[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        obj = json.loads(data)
                    except ValueError:
                        continue
                    ch = obj.get("choices") or []
                    if ch and (ch[0].get("delta") or {}).get("content"):
                        deltas += 1
                        if first is None:
                            first = time.perf_counter() - t0
                print("    -> %d line(s), %d content delta(s), first at %s"
                      % (lines, deltas,
                         ("%.3fs" % first) if first else "never"))
                if deltas == 0:
                    print("    *** no content deltas: do_chat would record")
                    print("        ok=False with status 200 and no error ***")
        except Exception as exc:
            print("    EXCEPTION %r" % exc)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from-results", action="store_true")
    ap.add_argument("--role", default="llm")
    a = ap.parse_args()

    from_results()
    if a.from_results:
        return 0

    role = a.role
    m = R.MODELS[role]
    # Reuse a running pod if there is one; otherwise deploy.
    try:
        eps = H.endpoints([role])
        print("\n   reusing running %s at %s" % (role, eps[role]))
        deployed = False
    except Exception:
        print("\n   deploying %s (about 3 minutes)" % role)
        H.teardown(quiet=True)
        H.wait_gpu_idle(timeout=240)
        rdir = os.path.join(HERE, ".diagchat")
        subprocess.run([PY, os.path.join(HERE, "render.py"),
                        "--scenario", "%s-only" % role, "--arm", "dedicated",
                        "--out", rdir], capture_output=True, text=True)
        H.kubectl(["apply", "-f", rdir], timeout=180)
        good, diag = H.wait_ready([role], timeout=1200)
        if not good:
            print("   never became Ready:\n%s" % list(diag.values())[0][:1500])
            return 1
        eps = H.endpoints([role])
        deployed = True

    try:
        live_ab(eps[role], m["repo"])
    finally:
        if deployed:
            H.teardown(quiet=True)
            H.wait_gpu_idle(timeout=240)
            H.sh(["rm", "-rf", os.path.join(HERE, ".diagchat")])
    return 0


if __name__ == "__main__":
    sys.exit(main())
