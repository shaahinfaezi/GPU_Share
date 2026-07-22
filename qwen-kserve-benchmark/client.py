#!/usr/bin/env python3
import argparse
import csv
import json
import os
import statistics
import sys
import time
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

try:
    from requests.packages.urllib3.exceptions import InsecureRequestWarning
except Exception:  # pragma: no cover - fallback for alternative requests builds
    InsecureRequestWarning = None


PROMPT = "Explain Kubernetes GPU sharing in exactly 64 words."
REQUEST = {
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "messages": [{"role": "user", "content": PROMPT}],
    "max_tokens": 64,
    "temperature": 0,
    "top_p": 1,
    "stream": False,
}


def endpoint_url(base):
    base = base.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/openai/v1"):
        return base + "/chat/completions"
    return base + "/openai/v1/chat/completions"


def output_tokens(payload):
    usage = payload.get("usage", {}) if isinstance(payload, dict) else {}
    return int(usage.get("completion_tokens", 0) or 0)


def request_once(service, base, index, timeout, insecure):
    url = endpoint_url(base)
    started = time.perf_counter()
    try:
        response = requests.post(url, json=REQUEST, timeout=timeout, verify=not insecure)
        elapsed = time.perf_counter() - started
        try:
            payload = response.json()
        except ValueError:
            payload = {"raw": response.text[:1000]}
        tokens = output_tokens(payload)
        return {
            "service": service,
            "request": index,
            "ok": response.ok,
            "status": response.status_code,
            "latency_seconds": elapsed,
            "output_tokens": tokens,
            "output_tokens_per_second": tokens / elapsed if elapsed and tokens else 0,
            "error": "" if response.ok else json.dumps(payload),
        }
    except Exception as exc:
        return {
            "service": service,
            "request": index,
            "ok": False,
            "status": 0,
            "latency_seconds": time.perf_counter() - started,
            "output_tokens": 0,
            "output_tokens_per_second": 0,
            "error": repr(exc),
        }


def run_requests(endpoints, mode, requests_per_service, timeout, sequential, insecure):
    warmup = []
    for service, base in endpoints.items():
        warmup.append(request_once(service, base, 0, timeout, insecure))

    results = []
    started = time.perf_counter()
    if sequential:
        for service, base in endpoints.items():
            for index in range(1, requests_per_service + 1):
                results.append(request_once(service, base, index, timeout, insecure))
    else:
        with ThreadPoolExecutor(max_workers=len(endpoints)) as pool:
            futures = [
                pool.submit(request_once, service, base, index, timeout, insecure)
                for service, base in endpoints.items()
                for index in range(1, requests_per_service + 1)
            ]
            for future in as_completed(futures):
                results.append(future.result())
    wall = time.perf_counter() - started

    for row in results:
        row["mode"] = mode
    summary = summarize(results, wall)
    summary["warmup_requests"] = len(warmup)
    return {"mode": mode, "warmup": warmup, "results": results, "summary": summary}


def percentile(values, p):
    if not values:
        return 0
    values = sorted(values)
    index = min(len(values) - 1, max(0, round((p / 100) * (len(values) - 1))))
    return values[index]


def summarize(results, wall):
    successful = [row for row in results if row["ok"]]
    latencies = [row["latency_seconds"] for row in successful]
    tokens = sum(row["output_tokens"] for row in successful)
    by_service = defaultdict(list)
    for row in results:
        by_service[row["service"]].append(row)

    service_summaries = {}
    for service, rows in sorted(by_service.items()):
        service_successful = [row for row in rows if row["ok"]]
        service_latencies = [row["latency_seconds"] for row in service_successful]
        service_tokens = sum(row["output_tokens"] for row in service_successful)
        service_summaries[service] = {
            "total_requests": len(rows),
            "successful_requests": len(service_successful),
            "failed_requests": len(rows) - len(service_successful),
            "requests_per_second": len(service_successful) / wall if wall else 0,
            "output_tokens": service_tokens,
            "output_tokens_per_second": service_tokens / wall if wall else 0,
            "latency_mean_seconds": statistics.mean(service_latencies) if service_latencies else 0,
            "latency_p50_seconds": percentile(service_latencies, 50),
            "latency_p95_seconds": percentile(service_latencies, 95),
            "latency_p99_seconds": percentile(service_latencies, 99),
        }

    return {
        "total_requests": len(results),
        "successful_requests": len(successful),
        "failed_requests": len(results) - len(successful),
        "wall_seconds": wall,
        "requests_per_second": len(successful) / wall if wall else 0,
        "output_tokens": tokens,
        "output_tokens_per_second": tokens / wall if wall else 0,
        "latency_mean_seconds": statistics.mean(latencies) if latencies else 0,
        "latency_p50_seconds": percentile(latencies, 50),
        "latency_p95_seconds": percentile(latencies, 95),
        "latency_p99_seconds": percentile(latencies, 99),
        "services": service_summaries,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True)
    parser.add_argument("--endpoint", action="append", required=True, help="service=url")
    parser.add_argument("--output", required=True)
    parser.add_argument("--requests-per-service", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--sequential", action="store_true")
    parser.add_argument("--insecure", action="store_true")
    args = parser.parse_args()

    endpoints = {}
    for item in args.endpoint:
        service, separator, url = item.partition("=")
        if not separator or not service or not url:
            parser.error("--endpoint must be service=url")
        endpoints[service] = url

    if args.insecure and InsecureRequestWarning is not None:
        warnings.simplefilter("ignore", InsecureRequestWarning)

    result = run_requests(endpoints, args.mode, args.requests_per_service, args.timeout, args.sequential, args.insecure)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    with output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        rows = result["results"]
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys() if rows else ["mode"])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(result["summary"], indent=2))
    return 0 if result["summary"]["failed_requests"] == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
