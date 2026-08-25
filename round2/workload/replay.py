#!/usr/bin/env python3
"""Replay a generated trace against one or more services and record everything.

Two things make this different from round one's client.py, and both are
required by the metric list:

1. STREAMING. Round one issued non-streaming requests, so it could only ever
   measure end-to-end latency. TTFT and inter-token latency do not exist in a
   non-streaming response. Chat requests here use `stream: true` and the
   arrival time of the first content chunk is recorded directly.

2. OPEN LOOP. Round one drove a fixed concurrency, which is a closed loop: a
   new request is only issued when an old one finishes, so the offered load
   silently drops when the server slows down and a queue can never form.
   Requests here are issued at the wall-clock offset the trace assigns them,
   whatever the server is doing. Queueing delay is therefore observable, and
   it is the thing sharing actually damages.

Per-request timings recorded (seconds from run origin):
    t_sched     when the trace says it should go out
    t_sent      when it actually went out       -> client_delay = sent - sched
    t_first     first content byte              -> TTFT = first - sent
    t_last      final byte                      -> TTLT = last - sent
    itl_*       inter-token gaps between chunks
"""
import argparse
import csv
import json
import os
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests

DEFAULT_TIMEOUT = 600.0


# --------------------------------------------------------------- endpoints
def resolve_chat(base, model, timeout, insecure):
    """KServe fronts vLLM at a few different prefixes depending on the route."""
    for path in ("/openai/v1/chat/completions", "/v1/chat/completions"):
        url = base.rstrip("/") + path
        try:
            r = requests.post(url, json={"model": model, "messages": [{"role": "user", "content": "ping"}],
                                         "max_tokens": 1},
                              timeout=15, verify=not insecure)
            if r.status_code < 500:
                return url
        except requests.RequestException:
            continue
    return base.rstrip("/") + "/v1/chat/completions"


def resolve_simple(base, suffixes):
    return [base.rstrip("/") + s for s in suffixes]


# --------------------------------------------------------------- one request
def do_chat(url, model, row, timeout, insecure, origin, sched):
    sent = time.perf_counter()
    rec = dict(t_sched=sched, t_sent=sent - origin, t_first=None, t_last=None,
               prompt_tokens=0, output_tokens=0, chunks=0, itl=[], ok=False,
               status=0, error="")
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": row["prompt"]}],
        "max_tokens": int(row.get("max_tokens", 128)),
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if row.get("ignore_eos"):
        payload["ignore_eos"] = True
    try:
        with requests.post(url, json=payload, timeout=timeout, stream=True,
                           verify=not insecure) as resp:
            rec["status"] = resp.status_code
            if not resp.ok:
                rec["error"] = resp.text[:300]
                rec["t_last"] = time.perf_counter() - origin
                return rec
            prev = None
            for raw in resp.iter_lines(decode_unicode=True):
                if not raw or not raw.startswith("data:"):
                    continue
                data = raw[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except ValueError:
                    continue
                usage = obj.get("usage")
                if usage:
                    rec["prompt_tokens"] = usage.get("prompt_tokens", 0) or 0
                    rec["output_tokens"] = usage.get("completion_tokens", 0) or 0
                choices = obj.get("choices") or []
                if not choices:
                    continue
                delta = (choices[0].get("delta") or {}).get("content")
                if not delta:
                    continue
                now = time.perf_counter()
                if rec["t_first"] is None:
                    rec["t_first"] = now - origin
                else:
                    rec["itl"].append(now - prev)
                prev = now
                rec["chunks"] += 1
            rec["t_last"] = time.perf_counter() - origin
            rec["ok"] = rec["t_first"] is not None
            if not rec["output_tokens"]:
                # Server did not return usage; chunk count is the best proxy.
                rec["output_tokens"] = rec["chunks"]
    except Exception as exc:
        rec["t_last"] = time.perf_counter() - origin
        rec["error"] = repr(exc)[:300]
    return rec


def do_pooling(urls, model, row, timeout, insecure, origin, sched, kind):
    """Embedding / rerank: one round trip, so TTFT and TTLT coincide."""
    sent = time.perf_counter()
    rec = dict(t_sched=sched, t_sent=sent - origin, t_first=None, t_last=None,
               prompt_tokens=0, output_tokens=0, chunks=0, itl=[], ok=False,
               status=0, error="")
    if kind == "embed":
        payload = {"model": model, "input": row["inputs"]}
    else:
        payload = {"model": model, "query": row["query"], "documents": row["documents"]}
    last = ""
    for url in urls:
        try:
            r = requests.post(url, json=payload, timeout=timeout, verify=not insecure)
            rec["status"] = r.status_code
            if r.status_code == 404:
                last = "404 " + url
                continue
            now = time.perf_counter() - origin
            rec["t_first"] = rec["t_last"] = now
            if r.ok:
                try:
                    body = r.json()
                except ValueError:
                    body = {}
                usage = body.get("usage") or {}
                rec["prompt_tokens"] = usage.get("prompt_tokens", 0) or 0
                n = len(body.get("data") or body.get("results") or [])
                rec["output_tokens"] = 0      # pooling models emit no tokens
                rec["chunks"] = n
                rec["ok"] = True
            else:
                rec["error"] = r.text[:300]
            return rec
        except Exception as exc:
            last = repr(exc)[:300]
    rec["t_last"] = time.perf_counter() - origin
    rec["error"] = last or "no endpoint matched"
    return rec


# --------------------------------------------------------------- statistics
def pct(vals, p):
    if not vals:
        return None
    s = sorted(vals)
    k = (len(s) - 1) * p / 100.0
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def stat_block(vals):
    if not vals:
        return None
    return dict(n=len(vals), mean=statistics.fmean(vals), p50=pct(vals, 50),
                p90=pct(vals, 90), p95=pct(vals, 95), p99=pct(vals, 99),
                min=min(vals), max=max(vals))


def contended_window(rows):
    """The interval in which EVERY service had a request in flight.

    Carried over from round one unchanged: a throttled tenant finishes later and
    runs alone in the tail, which flatters its throughput. Only this window is
    comparable across arms.
    """
    per = {}
    for r in rows:
        if not r["ok"]:
            continue
        per.setdefault(r["service"], []).append((r["t_sent"], r["t_last"]))
    if len(per) < 2:
        return None
    start = max(min(a for a, _ in v) for v in per.values())
    end = min(max(b for _, b in v) for v in per.values())
    if end <= start:
        return None
    return start, end


def tokens_in(rows, service, start, end):
    """Output tokens delivered inside [start,end], prorated for straddlers."""
    total = 0.0
    for r in rows:
        if r["service"] != service or not r["ok"] or not r["output_tokens"]:
            continue
        a, b = r["t_sent"], r["t_last"]
        if b <= a:
            continue
        ov = min(b, end) - max(a, start)
        if ov > 0:
            total += r["output_tokens"] * ov / (b - a)
    return total


def summarize(rows, meta, wall):
    live = [r for r in rows if not r["warmup"]]
    ok = [r for r in live if r["ok"]]
    out = dict(meta)
    out["wall_seconds"] = wall
    out["requests_total"] = len(live)
    out["requests_ok"] = len(ok)
    out["requests_failed"] = len(live) - len(ok)

    services = sorted(set(r["service"] for r in live))
    out["model_count"] = len(services)
    per = {}
    for svc in services:
        rs = [r for r in ok if r["service"] == svc]
        if not rs:
            per[svc] = dict(requests=0)
            continue
        e2e = [r["t_last"] - r["t_sent"] for r in rs]
        ttft = [r["t_first"] - r["t_sent"] for r in rs if r["t_first"] is not None]
        ttlt = e2e                                    # last token == end of stream
        itl = [g for r in rs for g in r["itl"]]
        delay = [r["t_sent"] - r["t_sched"] for r in rs]
        toks = sum(r["output_tokens"] for r in rs)
        ptoks = sum(r["prompt_tokens"] for r in rs)
        span = max(r["t_last"] for r in rs) - min(r["t_sent"] for r in rs)
        per[svc] = dict(
            kind=rs[0]["kind"], requests=len(rs),
            failed=len([r for r in live if r["service"] == svc and not r["ok"]]),
            output_tokens=toks, prompt_tokens=ptoks,
            tokens_per_second=(toks / span if span else 0),
            requests_per_second=(len(rs) / span if span else 0),
            latency_e2e=stat_block(e2e), ttft=stat_block(ttft),
            ttlt=stat_block(ttlt), itl=stat_block(itl),
            client_delay=stat_block(delay),
        )
    out["per_service"] = per

    all_e2e = [r["t_last"] - r["t_sent"] for r in ok]
    all_ttft = [r["t_first"] - r["t_sent"] for r in ok if r["t_first"] is not None]
    all_itl = [g for r in ok for g in r["itl"]]
    out["aggregate"] = dict(
        output_tokens=sum(r["output_tokens"] for r in ok),
        prompt_tokens=sum(r["prompt_tokens"] for r in ok),
        tokens_per_second=sum(r["output_tokens"] for r in ok) / wall if wall else 0,
        requests_per_second=len(ok) / wall if wall else 0,
        latency_e2e=stat_block(all_e2e), ttft=stat_block(all_ttft),
        ttlt=stat_block(all_e2e), itl=stat_block(all_itl),
        client_delay=stat_block([r["t_sent"] - r["t_sched"] for r in ok]),
    )

    win = contended_window(live)
    if win:
        start, end = win
        span = end - start
        cts = dict((s, tokens_in(live, s, start, end)) for s in services)
        gen = dict((s, v) for s, v in cts.items() if v > 0)
        out["contended"] = dict(
            start=start, end=end, seconds=span,
            per_service_tokens=cts,
            per_service_tokens_per_second=dict((s, v / span) for s, v in cts.items()),
            aggregate_tokens_per_second=sum(cts.values()) / span,
            ratio=(max(gen.values()) / min(gen.values())
                   if len(gen) > 1 and min(gen.values()) > 0 else None),
        )
    else:
        out["contended"] = None
    return out


# --------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trace", required=True)
    ap.add_argument("--arm", required=True, help="dedicated | hami | hami-mps")
    ap.add_argument("--endpoint", action="append", default=[],
                    metavar="ROLE=URL", help="repeatable, e.g. llm=http://1.2.3.4:80")
    ap.add_argument("--model", action="append", default=[], metavar="ROLE=NAME",
                    help="model name sent in the request body, per role")
    ap.add_argument("--only", action="append", default=[],
                    help="replay only these roles (dedicated baselines)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    ap.add_argument("--insecure", action="store_true")
    ap.add_argument("--max-inflight", type=int, default=512)
    ap.add_argument("--note", default="")
    a = ap.parse_args()

    with open(a.trace, encoding="utf-8") as fh:
        trace = json.load(fh)
    eps = dict(kv.split("=", 1) for kv in a.endpoint)
    models = dict(kv.split("=", 1) for kv in a.model)
    only = set(a.only) or None

    rows_in = [r for r in trace["requests"]
               if (only is None or r["service"] in only)]
    if not rows_in:
        sys.exit("no requests match --only %s" % sorted(only or []))
    missing = sorted(set(r["service"] for r in rows_in) - set(eps))
    if missing:
        sys.exit("no --endpoint for role(s): %s" % ", ".join(missing))

    # Resolve chat URLs once, before the clock starts.
    chat_url, pool_url = {}, {}
    for role in sorted(set(r["service"] for r in rows_in)):
        base = eps[role]
        kind = next(r["kind"] for r in rows_in if r["service"] == role)
        mdl = models.get(role, "model")
        if kind == "chat":
            chat_url[role] = resolve_chat(base, mdl, a.timeout, a.insecure)
        elif kind == "embed":
            pool_url[role] = resolve_simple(base, ("/openai/v1/embeddings", "/v1/embeddings"))
        else:
            pool_url[role] = resolve_simple(
                base, ("/openai/v1/rerank", "/v1/rerank", "/rerank", "/v1/score", "/score"))

    os.makedirs(a.out_dir, exist_ok=True)
    results, lock = [], threading.Lock()
    origin = time.perf_counter()

    def fire(row):
        role, kind = row["service"], row["kind"]
        mdl = models.get(role, "model")
        sched = row["t"] if not row["warmup"] else 0.0
        if kind == "chat":
            rec = do_chat(chat_url[role], mdl, row, a.timeout, a.insecure, origin, sched)
        else:
            rec = do_pooling(pool_url[role], mdl, row, a.timeout, a.insecure,
                             origin, sched, kind)
        rec.update(service=role, kind=kind, warmup=row["warmup"])
        with lock:
            results.append(rec)

    pool = ThreadPoolExecutor(max_workers=a.max_inflight)

    # Warm-up: closed loop, excluded from every metric.
    warm = [r for r in rows_in if r["warmup"]]
    if warm:
        print("warm-up: %d requests" % len(warm), flush=True)
        list(pool.map(fire, warm))

    live = sorted((r for r in rows_in if not r["warmup"]), key=lambda r: r["t"])
    print("replaying %d requests over %.0fs (open loop, trace %s)"
          % (len(live), trace["params"]["duration"], trace["trace_id"]), flush=True)

    origin = time.perf_counter()     # restart the clock after warm-up
    for row in live:
        due = origin + row["t"]
        gap = due - time.perf_counter()
        if gap > 0:
            time.sleep(gap)
        pool.submit(fire, row)
    pool.shutdown(wait=True)
    wall = time.perf_counter() - origin

    meta = dict(arm=a.arm, note=a.note, trace_id=trace["trace_id"],
                trace_params=trace["params"],
                roles=sorted(set(r["service"] for r in rows_in)),
                endpoints=eps, models=models,
                only=sorted(only) if only else None)
    summary = summarize(results, meta, wall)

    base = os.path.join(a.out_dir, "replay")
    with open(base + ".summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    cols = ["service", "kind", "warmup", "ok", "status", "t_sched", "t_sent",
            "t_first", "t_last", "prompt_tokens", "output_tokens", "chunks", "error"]
    with open(base + ".requests.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in sorted(results, key=lambda x: x["t_sent"]):
            w.writerow(r)
    # Raw inter-token gaps, kept separately so ITL can be re-analysed later.
    with open(base + ".itl.json", "w", encoding="utf-8") as fh:
        json.dump([dict(service=r["service"], t_sent=r["t_sent"], itl=r["itl"])
                   for r in results if r["itl"]], fh)

    ag = summary["aggregate"]
    print("\narm %s   ok %d/%d   %.1f tok/s   %.2f req/s"
          % (a.arm, summary["requests_ok"], summary["requests_total"],
             ag["tokens_per_second"], ag["requests_per_second"]))
    for name in ("ttft", "latency_e2e", "itl"):
        b = ag.get(name)
        if b:
            print("  %-12s p50 %.3f  p95 %.3f  p99 %.3f  mean %.3f"
                  % (name, b["p50"], b["p95"], b["p99"], b["mean"]))
    if summary["contended"]:
        print("  contended    %.1f s, %.1f tok/s aggregate"
              % (summary["contended"]["seconds"],
                 summary["contended"]["aggregate_tokens_per_second"]))
    print("wrote %s.{summary.json,requests.csv,itl.json}" % base)


if __name__ == "__main__":
    main()
