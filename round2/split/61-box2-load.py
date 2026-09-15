#!/usr/bin/env python3
"""Open-loop load driver for the box-2 pair. Speaks both protocols.

WHY NOT workload/replay.py
--------------------------
replay.py issues STREAMING OpenAI chat requests, because TTFT and inter-token
latency do not exist in a non-streaming response. llama.cpp's server is
OpenAI-compatible and works with it. The custom runtime is not: it serves
POST /v1/models/{name}:predict, returns one JSON blob, and has no streaming
path at all. Pointing replay.py at it yields a 404 per request and a clean
report of nothing.

So this driver measures what BOTH can actually produce -- end-to-end latency
and completed requests per second -- and does not pretend to a TTFT it cannot
observe on one side. That is a real limit of the custom runtime as written,
not of the measurement, and it belongs in the report as such.

OPEN LOOP, like replay.py. Requests go out on a fixed schedule regardless of
what the server is doing, so a queue can form and be seen. A closed loop would
silently reduce the offered rate as the server slowed, which is exactly the
degradation under test.

    python3 61-box2-load.py --arm hami \
        --endpoint llamacpp=http://10.42.0.31:8080 \
        --endpoint custom=http://10.42.0.32:8080 \
        --rate 6 --duration 240 --out-dir results/box2-hami
"""
import argparse
import json
import os
import random
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests

SESSION = requests.Session()
PROMPTS = [
    "Summarise the tradeoffs between tensor and pipeline parallelism.",
    "Explain what a KV cache is and why its size bounds concurrency.",
    "Write a short paragraph about scheduling GPUs in Kubernetes.",
    "What does it mean for a GPU fault to be global in scope?",
    "Describe how partial CPU offload changes inference latency.",
]


def build(role, model, prompt, max_tokens):
    """Each runtime gets the body IT accepts. Same prompt, same token budget."""
    if role == "custom":
        return ("/v1/models/%s:predict" % model,
                {"instances": [{"prompt": prompt,
                                "max_new_tokens": max_tokens,
                                "do_sample": False}]})
    return ("/v1/chat/completions",
            {"model": model, "max_tokens": max_tokens, "temperature": 0.0,
             "messages": [{"role": "user", "content": prompt}]})


def tokens_out(role, body):
    if role == "custom":
        p = (body.get("predictions") or [{}])[0]
        return int(p.get("tokens_generated") or 0)
    u = body.get("usage") or {}
    return int(u.get("completion_tokens") or 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--endpoint", action="append", default=[], metavar="ROLE=URL")
    ap.add_argument("--model", action="append", default=[], metavar="ROLE=NAME",
                    help="default llamacpp=qwen, custom=qwen2-custom")
    ap.add_argument("--rate", type=float, default=6.0, help="requests/s TOTAL")
    ap.add_argument("--duration", type=float, default=240.0)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=20260914)
    a = ap.parse_args()

    eps = dict(kv.split("=", 1) for kv in a.endpoint)
    models = {"llamacpp": "qwen", "custom": "qwen2-custom"}
    models.update(dict(kv.split("=", 1) for kv in a.model))
    roles = sorted(eps)
    if not roles:
        raise SystemExit("need at least one --endpoint ROLE=URL")

    os.makedirs(a.out_dir, exist_ok=True)
    rnd = random.Random(a.seed)

    # The schedule is built BEFORE the clock starts, so a slow server cannot
    # change what was offered. Poisson arrivals, split evenly across roles.
    sched, t = [], 0.0
    while t < a.duration:
        t += rnd.expovariate(a.rate)
        if t >= a.duration:
            break
        sched.append((t, roles[len(sched) % len(roles)],
                      rnd.choice(PROMPTS)))

    SESSION.mount("http://", requests.adapters.HTTPAdapter(
        pool_connections=64, pool_maxsize=1024, max_retries=0))

    rows, lock = [], threading.Lock()
    t0 = time.time()

    def fire(when, role, prompt):
        delay = when - (time.time() - t0)
        if delay > 0:
            time.sleep(delay)
        path, body = build(role, models[role], prompt, a.max_tokens)
        sent = time.time()
        rec = {"role": role, "t_sched": when, "t_sent": sent - t0}
        try:
            r = SESSION.post(eps[role] + path, json=body, timeout=a.timeout)
            done = time.time()
            rec["status"] = r.status_code
            rec["latency"] = done - sent
            rec["tokens"] = tokens_out(role, r.json()) if r.ok else 0
        except Exception as e:
            rec["status"] = 0
            rec["latency"] = time.time() - sent
            rec["tokens"] = 0
            rec["error"] = str(e)[:200]
        with lock:
            rows.append(rec)

    with ThreadPoolExecutor(max_workers=1024) as pool:
        for when, role, prompt in sched:
            pool.submit(fire, when, role, prompt)
    wall = time.time() - t0

    out = {"arm": a.arm, "endpoints": eps, "models": models,
           "rate": a.rate, "duration": a.duration, "wall_seconds": wall,
           "requests_total": len(rows), "per_service": {}}
    for role in roles:
        rs = [r for r in rows if r["role"] == role]
        ok = [r for r in rs if r.get("status") == 200]
        lat = sorted(r["latency"] for r in ok)
        tok = sum(r["tokens"] for r in ok)
        out["per_service"][role] = {
            "requests": len(ok),
            "failed": len(rs) - len(ok),
            "output_tokens": tok,
            "tokens_per_second": tok / wall if wall else 0.0,
            "requests_per_second": len(ok) / wall if wall else 0.0,
            "latency_e2e": {
                "mean": statistics.fmean(lat) if lat else 0.0,
                "p50": lat[int(0.50 * (len(lat) - 1))] if lat else 0.0,
                "p95": lat[int(0.95 * (len(lat) - 1))] if lat else 0.0,
                "p99": lat[int(0.99 * (len(lat) - 1))] if lat else 0.0,
                "max": lat[-1] if lat else 0.0,
            },
            # No TTFT: the custom runtime does not stream, so there is no first
            # token to time. Recorded as absent rather than as zero.
            "ttft": None,
        }
        s = out["per_service"][role]
        print("%-9s %5d ok / %3d failed | %7.1f tok/s | %6.2f req/s | mean %.2fs"
              % (role, s["requests"], s["failed"], s["tokens_per_second"],
                 s["requests_per_second"], s["latency_e2e"]["mean"]))

    with open(os.path.join(a.out_dir, "load.summary.json"), "w",
              encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    with open(os.path.join(a.out_dir, "load.requests.json"), "w",
              encoding="utf-8") as fh:
        json.dump(rows, fh)


if __name__ == "__main__":
    main()
