#!/usr/bin/env python3
"""Load client for the two-container Qwen KServe GPU-sharing benchmark.

Two run styles:

  * concurrent (default) - one worker thread per service, each issuing its
    requests back to back. This models two tenants hitting a shared GPU at the
    same time. The previous version queued all 20 requests into a 2-worker pool,
    which let one service's queue drain into the other's worker and made the
    per-service numbers meaningless.

  * --sequential - kept for a single-process run, but note that true sequential
    isolation is enforced by the runner (only one service is deployed at a time),
    not by this flag.

Use --merge to combine the per-phase JSON files the runner produces in
sequential mode into one comparable result document.
"""
import argparse
import csv
import json
import statistics
import sys
import time
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

try:
    from requests.packages.urllib3.exceptions import InsecureRequestWarning
except Exception:  # pragma: no cover - fallback for alternative requests builds
    InsecureRequestWarning = None


PROMPT = "Explain Kubernetes GPU sharing in exactly 64 words."
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

# The LLMInferenceService gateway URL is path-based
# (http://<gw>/<namespace>/<service>) and which OpenAI prefix sits behind it
# depends on the router/route template and the server image. Rather than
# hardcoding one guess, probe these in order and lock onto the first that
# answers. The old hardcoded /openai/v1 guess is kept in the list.
CANDIDATE_SUFFIXES = (
    "/v1/chat/completions",
    "/openai/v1/chat/completions",
    "/chat/completions",
)


def build_request(model, max_tokens=64):
    return {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "top_p": 1,
        "stream": False,
    }


def candidate_urls(base):
    base = base.rstrip("/")
    for suffix in CANDIDATE_SUFFIXES:
        if base.endswith(suffix):
            return [base]
    # Tolerate a base that already carries a partial prefix.
    for prefix in ("/openai/v1", "/v1"):
        if base.endswith(prefix):
            return [base + "/chat/completions"]
    return [base + suffix for suffix in CANDIDATE_SUFFIXES]


def resolve_url(base, model, timeout, insecure):
    """Return (url, probe_log). Probes each candidate once with a tiny request."""
    candidates = candidate_urls(base)
    probe_log = []
    if len(candidates) == 1:
        return candidates[0], probe_log

    payload = build_request(model, max_tokens=1)
    for url in candidates:
        try:
            response = requests.post(url, json=payload, timeout=timeout, verify=not insecure)
            probe_log.append({"url": url, "status": response.status_code, "ok": response.ok})
            if response.ok:
                return url, probe_log
        except Exception as exc:
            probe_log.append({"url": url, "status": 0, "ok": False, "error": repr(exc)})

    # Nothing answered. Fall back to the first candidate so the failure is
    # recorded per request instead of aborting the whole mode.
    return candidates[0], probe_log


def output_tokens(payload):
    usage = payload.get("usage", {}) if isinstance(payload, dict) else {}
    return int(usage.get("completion_tokens", 0) or 0)


def request_once(service, url, index, payload, timeout, insecure, origin):
    started = time.perf_counter()
    try:
        response = requests.post(url, json=payload, timeout=timeout, verify=not insecure)
        elapsed = time.perf_counter() - started
        try:
            body = response.json()
        except ValueError:
            body = {"raw": response.text[:1000]}
        tokens = output_tokens(body)
        return {
            "service": service,
            "request": index,
            "ok": response.ok,
            "status": response.status_code,
            "start_offset_seconds": started - origin,
            "latency_seconds": elapsed,
            "output_tokens": tokens,
            "output_tokens_per_second": tokens / elapsed if elapsed and tokens else 0,
            "error": "" if response.ok else json.dumps(body)[:1000],
        }
    except Exception as exc:
        return {
            "service": service,
            "request": index,
            "ok": False,
            "status": 0,
            "start_offset_seconds": started - origin,
            "latency_seconds": time.perf_counter() - started,
            "output_tokens": 0,
            "output_tokens_per_second": 0,
            "error": repr(exc),
        }


def run_requests(endpoints, mode, requests_per_service, timeout, sequential, insecure, model, warmup_requests):
    resolved = {}
    probes = {}
    for service, base in endpoints.items():
        url, probe_log = resolve_url(base, model, timeout, insecure)
        resolved[service] = url
        probes[service] = probe_log

    payload = build_request(model)

    # Warm-up requests are excluded from every reported metric. They pay for
    # CUDA graph capture, the first KV-cache allocations and any lazy weight
    # paging, which would otherwise land entirely in the first measured latency.
    warm_origin = time.perf_counter()
    warmup = []
    for service, url in resolved.items():
        for index in range(warmup_requests):
            warmup.append(request_once(service, url, -(index + 1), payload, timeout, insecure, warm_origin))

    results = []
    origin = time.perf_counter()
    started = origin

    def drive(service):
        url = resolved[service]
        rows = []
        for index in range(1, requests_per_service + 1):
            rows.append(request_once(service, url, index, payload, timeout, insecure, origin))
        return rows

    if sequential:
        for service in resolved:
            results.extend(drive(service))
    else:
        with ThreadPoolExecutor(max_workers=max(1, len(resolved))) as pool:
            for rows in pool.map(drive, list(resolved)):
                results.extend(rows)

    wall = time.perf_counter() - started

    for row in results:
        row["mode"] = mode
    summary = summarize(results, wall)
    summary["warmup_requests"] = len(warmup)
    summary["warmup_requests_per_service"] = warmup_requests
    return {
        "mode": mode,
        "model": model,
        "sequential": sequential,
        "endpoints": endpoints,
        "resolved_urls": resolved,
        "endpoint_probes": probes,
        "warmup": warmup,
        "results": results,
        "summary": summary,
    }


def percentile(values, p):
    if not values:
        return 0
    values = sorted(values)
    index = min(len(values) - 1, max(0, round((p / 100) * (len(values) - 1))))
    return values[index]


def service_window(rows):
    """Wall time from the first request start to the last request end."""
    active = [row for row in rows if "start_offset_seconds" in row]
    if not active:
        return 0
    first = min(row["start_offset_seconds"] for row in active)
    last = max(row["start_offset_seconds"] + row["latency_seconds"] for row in active)
    return max(0.0, last - first)


def summarize(results, wall):
    successful = [row for row in results if row["ok"]]
    latencies = [row["latency_seconds"] for row in successful]
    tokens = sum(row["output_tokens"] for row in successful)
    by_service = defaultdict(list)
    for row in results:
        by_service[row["service"]].append(row)

    service_summaries = {}
    for service, rows in sorted(by_service.items()):
        ok_rows = [row for row in rows if row["ok"]]
        service_latencies = [row["latency_seconds"] for row in ok_rows]
        service_tokens = sum(row["output_tokens"] for row in ok_rows)
        # Per-service rates use that service's own active window, so a service
        # that finished early is not penalised by the other one still running.
        window = service_window(rows) or wall
        service_summaries[service] = {
            "total_requests": len(rows),
            "successful_requests": len(ok_rows),
            "failed_requests": len(rows) - len(ok_rows),
            "active_window_seconds": window,
            "requests_per_second": len(ok_rows) / window if window else 0,
            "output_tokens": service_tokens,
            "output_tokens_per_second": service_tokens / window if window else 0,
            "latency_mean_seconds": statistics.mean(service_latencies) if service_latencies else 0,
            "latency_p50_seconds": percentile(service_latencies, 50),
            "latency_p95_seconds": percentile(service_latencies, 95),
            "latency_p99_seconds": percentile(service_latencies, 99),
        }

    # Fairness across tenants: 1.0 means both services got identical throughput.
    # This is the number that actually answers "is the shared GPU fair?", which
    # is the QoS question in the project goals.
    throughputs = [entry["output_tokens_per_second"] for entry in service_summaries.values()]
    if len(throughputs) >= 2 and max(throughputs) > 0:
        fairness = min(throughputs) / max(throughputs)
    else:
        fairness = None

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
        "fairness_ratio": fairness,
        "services": service_summaries,
    }


def merge_documents(paths, mode):
    """Combine per-phase sequential runs into one document.

    Wall time is the SUM of the phases, not their max: in sequential mode the
    phases genuinely ran one after the other, so the aggregate request rate must
    reflect the full serialized duration.
    """
    documents = [json.loads(Path(path).read_text(encoding="utf-8")) for path in paths]
    results = []
    warmup = []
    endpoints = {}
    resolved = {}
    probes = {}
    wall = 0.0
    warmup_per_service = 0
    for document in documents:
        results.extend(document.get("results", []))
        warmup.extend(document.get("warmup", []))
        endpoints.update(document.get("endpoints", {}))
        resolved.update(document.get("resolved_urls", {}))
        probes.update(document.get("endpoint_probes", {}))
        wall += float(document.get("summary", {}).get("wall_seconds", 0) or 0)
        warmup_per_service = max(
            warmup_per_service,
            int(document.get("summary", {}).get("warmup_requests_per_service", 0) or 0),
        )

    for row in results:
        row["mode"] = mode

    summary = summarize(results, wall)
    summary["warmup_requests"] = len(warmup)
    summary["warmup_requests_per_service"] = warmup_per_service
    # Two services that never overlapped cannot be compared for fairness.
    summary["fairness_ratio"] = None
    return {
        "mode": mode,
        "model": documents[0].get("model", DEFAULT_MODEL) if documents else DEFAULT_MODEL,
        "sequential": True,
        "endpoints": endpoints,
        "resolved_urls": resolved,
        "endpoint_probes": probes,
        "phase_wall_seconds": [
            document.get("summary", {}).get("wall_seconds", 0) for document in documents
        ],
        "warmup": warmup,
        "results": results,
        "summary": summary,
    }


def write_output(result, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    rows = result["results"]
    fieldnames = list(rows[0].keys()) if rows else ["mode", "service", "request", "ok"]
    with output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True)
    parser.add_argument("--endpoint", action="append", default=[], help="service=url")
    parser.add_argument("--output", required=True)
    parser.add_argument("--requests-per-service", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=1, help="warm-up requests per service, excluded from metrics")
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--sequential", action="store_true")
    parser.add_argument("--insecure", action="store_true")
    parser.add_argument("--merge", nargs="+", default=None, help="merge these result JSON files instead of sending traffic")
    args = parser.parse_args()

    if args.insecure and InsecureRequestWarning is not None:
        warnings.simplefilter("ignore", InsecureRequestWarning)

    if args.merge:
        result = merge_documents(args.merge, args.mode)
    else:
        if not args.endpoint:
            parser.error("--endpoint is required unless --merge is used")
        endpoints = {}
        for item in args.endpoint:
            service, separator, url = item.partition("=")
            if not separator or not service or not url:
                parser.error("--endpoint must be service=url")
            endpoints[service] = url
        result = run_requests(
            endpoints,
            args.mode,
            args.requests_per_service,
            args.timeout,
            args.sequential,
            args.insecure,
            args.model,
            args.warmup,
        )

    write_output(result, args.output)
    print(json.dumps(result["summary"], indent=2))
    return 0 if result["summary"]["failed_requests"] == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
