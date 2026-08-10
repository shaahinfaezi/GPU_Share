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
import hashlib
import json
import statistics
import sys
import threading
import time
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

try:  # POSIX only; the client runs on the Linux node, but keep imports safe.
    import resource as _resource
except ImportError:  # pragma: no cover
    _resource = None


def _cpu_seconds():
    """CPU seconds burned by this client process (user + sys, incl. threads).

    Recorded so a run can be checked for the client itself being the
    bottleneck. If this approaches wall_seconds x thread_count, the measured
    "GPU" throughput is really a client limit and the mode comparison is void.
    """
    if _resource is None:
        return 0.0
    usage = _resource.getrusage(_resource.RUSAGE_SELF)
    return float(usage.ru_utime + usage.ru_stime)

try:
    from requests.packages.urllib3.exceptions import InsecureRequestWarning
except Exception:  # pragma: no cover - fallback for alternative requests builds
    InsecureRequestWarning = None


PROMPT = "Explain Kubernetes GPU sharing in exactly 64 words."
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

# Every request originally sent the SAME prompt string. Even with
# --no-enable-prefix-caching set on the server, identical prompts are not a
# neutral workload: the tokenizer output is identical, the prefill shape is
# identical every time, and any caching layer anywhere in the path (server
# prefix cache if it is ever re-enabled, router, HTTP layer) can serve repeat
# work that a real tenant would have to compute. The measured differences
# between modes were small enough that this mattered.
#
# Prompts are now DETERMINISTIC PER REQUEST INDEX. That is the important
# property: index 7 produces the same prompt in every mode, on every repeat and
# for both tenants, so no mode is handed easier work than another, while no two
# requests within a run are alike. The leading reference tag differs from the
# second token onward, so a prefix cache has nothing to reuse.
#
# Warm-up requests use negative indices and therefore a disjoint set of prompts,
# so warm-up cannot pre-populate a cache that the measured requests then hit.
_PROMPT_ANGLES = ("Explain", "Summarize", "Describe", "Outline")

_PROMPT_TOPICS = (
    "Kubernetes GPU sharing across tenants",
    "CUDA stream priority scheduling",
    "NUMA affinity for inference servers",
    "container image layer deduplication",
    "Kubernetes admission webhook ordering",
    "GPU memory fragmentation under load",
    "service mesh sidecar overhead",
    "etcd compaction and defragmentation",
    "Linux cgroup v2 CPU throttling",
    "PCIe bandwidth contention between GPUs",
    "Kubernetes pod topology spread constraints",
    "continuous batching in inference servers",
    "NVLink versus PCIe peer transfers",
    "container runtime snapshotter choices",
    "horizontal pod autoscaling signals",
    "GPU driver persistence mode tradeoffs",
    "distributed tracing sampling strategies",
    "Kubernetes network policy enforcement",
    "page cache pressure on inference nodes",
    "TLS session resumption in gateways",
    "Kubernetes device plugin registration",
    "kernel launch overhead in tight loops",
    "object storage consistency for checkpoints",
    "Kubernetes graceful pod termination",
)


def build_prompt(index, prompt_mode="varied"):
    """Deterministic prompt for a request index.

    `fixed` reproduces the original single-prompt behaviour, kept so a run can
    be compared directly against the earlier published results.
    """
    if prompt_mode == "fixed":
        return PROMPT
    # blake2s over the index: stable across runs, machines and Python versions,
    # unlike hash(), which is randomised per process by PYTHONHASHSEED.
    tag = hashlib.blake2s(str(index).encode("utf-8"), digest_size=3).hexdigest()
    angle = _PROMPT_ANGLES[abs(index) % len(_PROMPT_ANGLES)]
    topic = _PROMPT_TOPICS[abs(index) % len(_PROMPT_TOPICS)]
    return f"Ref {tag}. {angle} {topic} in exactly 64 words."

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


def build_request(model, max_tokens=64, index=0, prompt_mode="varied"):
    """Every request must produce EXACTLY max_tokens tokens.

    Greedy decoding alone does not guarantee this. A measured run showed MPS
    emitting 42 tokens per request while sequential and HAMi emitted 64: under
    concurrency vLLM batches differently, floating-point reduction order
    changes, and greedy argmax can pick EOS at a different step. The modes were
    therefore doing different amounts of work, and output_tokens_per_second was
    comparing unequal workloads.

    ignore_eos + min_tokens pin the generated length so every mode performs
    identical work. This is the standard way to make LLM serving benchmarks
    comparable, and it matters more here than the exact wording of the reply.
    """
    return {
        "model": model,
        "messages": [{"role": "user", "content": build_prompt(index, prompt_mode)}],
        "max_tokens": max_tokens,
        "min_tokens": max_tokens,
        "ignore_eos": True,
        "temperature": 0,
        "top_p": 1,
        "seed": 0,
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


def prompt_tokens(payload):
    usage = payload.get("usage", {}) if isinstance(payload, dict) else {}
    return int(usage.get("prompt_tokens", 0) or 0)


def response_text(payload):
    """The model's actual reply, so results are auditable rather than just counts."""
    if not isinstance(payload, dict):
        return ""
    choices = payload.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    return (message.get("content") or "").strip()


def request_once(service, url, index, payload, timeout, insecure, origin):
    prompt = payload.get("messages", [{}])[0].get("content", "")
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
            "end_offset_seconds": (started - origin) + elapsed,
            "latency_seconds": elapsed,
            "prompt": prompt,
            "prompt_tokens": prompt_tokens(body),
            "output_tokens": tokens,
            "output_tokens_per_second": tokens / elapsed if elapsed and tokens else 0,
            "response_text": response_text(body) if response.ok else "",
            "error": "" if response.ok else json.dumps(body)[:1000],
        }
    except Exception as exc:
        elapsed = time.perf_counter() - started
        return {
            "service": service,
            "request": index,
            "ok": False,
            "status": 0,
            "start_offset_seconds": started - origin,
            "end_offset_seconds": (started - origin) + elapsed,
            "latency_seconds": elapsed,
            "prompt": prompt,
            "prompt_tokens": 0,
            "output_tokens": 0,
            "output_tokens_per_second": 0,
            "response_text": "",
            "error": repr(exc),
        }


def run_requests(endpoints, mode, requests_per_service, timeout, sequential, insecure, model,
                 warmup_requests, concurrency=1, max_tokens=64, duration_seconds=0,
                 prompt_mode="varied", prompt_offset=0):
    resolved = {}
    probes = {}
    for service, base in endpoints.items():
        url, probe_log = resolve_url(base, model, timeout, insecure)
        resolved[service] = url
        probes[service] = probe_log

    # Built per request, not once: each index gets its own prompt so no two
    # requests in a run are identical, while index N is identical across modes.
    # prompt_offset shifts the whole prompt sequence. In-place repeats reuse the
    # SAME pods, so without an offset pass 2 would replay pass 1's prompts
    # against an already-warm server. The runner passes the repeat number, so
    # every pass sees fresh prompts - while all modes at pass N still share an
    # identical sequence, keeping the mode comparison clean.
    def payload_for(index):
        shifted = index + prompt_offset if index > 0 else index - prompt_offset
        return build_request(model, max_tokens=max_tokens, index=shifted,
                             prompt_mode=prompt_mode)

    services = list(resolved)

    # Warm-up requests are excluded from every reported metric. They pay for
    # CUDA graph capture, the first KV-cache allocations and any lazy weight
    # paging, which would otherwise land entirely in the first measured latency.
    #
    # Warm-up now runs both services CONCURRENTLY. Warming them one after the
    # other let each tenant capture CUDA graphs on an otherwise idle GPU, which
    # is not the state it is measured in, and under MPS it made the first tenant
    # alone pay the MPS server's start-up cost.
    warm_origin = time.perf_counter()
    warmup = []
    if warmup_requests > 0:
        def warm(service):
            url = resolved[service]
            return [request_once(service, url, -(index + 1), payload_for(-(index + 1)),
                                 timeout, insecure, warm_origin)
                    for index in range(warmup_requests)]

        with ThreadPoolExecutor(max_workers=max(1, len(services))) as warm_pool:
            for rows in warm_pool.map(warm, services):
                warmup.extend(rows)

    results = []
    origin_state = {"perf": time.perf_counter(), "epoch": time.time()}

    def stamp_origin():
        origin_state["perf"] = time.perf_counter()
        origin_state["epoch"] = time.time()

    # Both tenants must begin measured traffic at the same instant. Without a
    # barrier, whichever driver thread the pool happened to schedule first got a
    # head start on an uncontended GPU. That inflates its throughput, corrupts
    # the fairness ratio, and does so by a different amount in every mode.
    #
    # Barrier(action=...) runs the callback in one thread once all parties have
    # arrived and BEFORE any is released, so it is the correct place to stamp
    # the measurement origin.
    barrier = None
    if not sequential and len(services) > 1:
        barrier = threading.Barrier(len(services), action=stamp_origin)

    def drive(service):
        """Issue this service's requests, keeping `concurrency` in flight.

        concurrency=1 is a strictly serial tenant. Higher values let a single
        service saturate the GPU on its own, which is what makes HAMi's
        gpucores limit and MPS's thread percentage observable: if one service
        can only ever use ~30% of the device, a 50% cap never binds and every
        sharing mode looks identical.
        """
        if barrier is not None:
            barrier.wait()
        origin = origin_state["perf"]
        url = resolved[service]
        rows = []

        if duration_seconds > 0:
            # Fixed-duration tenants: every tenant stops at the same instant, so
            # neither gets a stretch of GPU to itself after the other has
            # exhausted a fixed request quota. That uncontended tail is the
            # largest measurement error in a fixed-count run, and it is larger
            # in exactly the modes that split throughput unevenly.
            deadline = origin + duration_seconds

            def worker(slot):
                out = []
                counter = 0
                while time.perf_counter() < deadline:
                    counter += 1
                    index = slot * 1000000 + counter
                    out.append(request_once(service, url, index, payload_for(index),
                                            timeout, insecure, origin))
                return out

            with ThreadPoolExecutor(max_workers=max(1, concurrency)) as inner:
                for out in inner.map(worker, range(max(1, concurrency))):
                    rows.extend(out)
            return rows

        if concurrency <= 1:
            for index in range(1, requests_per_service + 1):
                rows.append(request_once(service, url, index, payload_for(index),
                                         timeout, insecure, origin))
            return rows
        with ThreadPoolExecutor(max_workers=concurrency) as inner:
            futures = [
                inner.submit(request_once, service, url, index, payload_for(index),
                             timeout, insecure, origin)
                for index in range(1, requests_per_service + 1)
            ]
            for future in futures:
                rows.append(future.result())
        return rows

    cpu_before = _cpu_seconds()
    if sequential:
        stamp_origin()
        for service in services:
            results.extend(drive(service))
    else:
        with ThreadPoolExecutor(max_workers=max(1, len(services))) as pool:
            for rows in pool.map(drive, services):
                results.extend(rows)

    wall = time.perf_counter() - origin_state["perf"]
    measure_end_epoch = time.time()
    client_cpu_seconds = _cpu_seconds() - cpu_before

    for row in results:
        row["mode"] = mode
    summary = summarize(results, wall)
    summary["warmup_requests"] = len(warmup)
    summary["warmup_requests_per_service"] = warmup_requests
    summary["concurrency_per_service"] = concurrency
    summary["max_tokens"] = max_tokens
    summary["duration_seconds_requested"] = duration_seconds
    summary["prompt_mode"] = prompt_mode
    summary["prompt_offset"] = prompt_offset

    # Wall-clock bounds of the measured window. The runner uses these to clip
    # the GPU and CPU sample timelines: samplers start before the client does,
    # so an unclipped mean utilization also averages in URL probing, warm-up and
    # idle time, and those differ in length between modes.
    summary["measure_start_epoch"] = origin_state["epoch"]
    summary["measure_end_epoch"] = measure_end_epoch

    # Client-side saturation guard. threads = in-flight requests the client can
    # have outstanding. If the client burns ~1 CPU core per thread, it is the
    # bottleneck and the "GPU" comparison is measuring Python, not the GPU.
    threads = max(1, concurrency) * max(1, len(services))
    summary["client_cpu_seconds"] = client_cpu_seconds
    summary["client_cpu_cores_used"] = client_cpu_seconds / wall if wall else 0
    summary["client_saturation_ratio"] = (
        client_cpu_seconds / wall / threads if wall and threads else 0)
    summary["client_bound_suspected"] = bool(summary["client_saturation_ratio"] > 0.7)

    return {
        "mode": mode,
        "model": model,
        "sequential": sequential,
        "concurrency_per_service": concurrency,
        "max_tokens": max_tokens,
        "duration_seconds": duration_seconds,
        "prompt_mode": prompt_mode,
        "measure_start_epoch": origin_state["epoch"],
        "measure_end_epoch": measure_end_epoch,
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


def overlap_window(by_service):
    """The interval during which EVERY service had traffic in flight.

    A fixed-count run ends ragged: the faster tenant finishes its quota, and the
    slower one then runs ALONE on the whole GPU. Tokens produced in that tail are
    uncontended, so crediting them to "two tenants sharing a GPU" overstates
    sharing throughput - and overstates it most in exactly the modes that split
    throughput least evenly, which is the comparison being made. Metrics computed
    over this window describe only the genuinely contended part of the run.

    Returns (start, end) as offsets in seconds, or None if there was no overlap.
    """
    bounds = []
    for rows in by_service.values():
        active = [row for row in rows if "start_offset_seconds" in row]
        if not active:
            return None
        bounds.append((
            min(row["start_offset_seconds"] for row in active),
            max(row["start_offset_seconds"] + row["latency_seconds"] for row in active),
        ))
    if len(bounds) < 2:
        return None
    start = max(bound[0] for bound in bounds)
    end = min(bound[1] for bound in bounds)
    return (start, end) if end > start else None


def tokens_in_window(rows, start, end):
    """Tokens a service produced inside [start, end], prorated by time.

    A request straddling a boundary contributes the fraction of its tokens
    matching the fraction of its duration inside the window. Decode emits tokens
    at a near-constant rate, so time-proration is a good estimate and far less
    biased than either dropping or wholly counting a straddling request.
    """
    total = 0.0
    for row in rows:
        if not row.get("ok"):
            continue
        duration = row.get("latency_seconds") or 0
        if duration <= 0:
            continue
        row_start = row["start_offset_seconds"]
        inside = min(row_start + duration, end) - max(row_start, start)
        if inside > 0:
            total += row.get("output_tokens", 0) * (inside / duration)
    return total


def jain_index(values):
    """Jain's fairness index: 1.0 is perfectly equal, 1/n is one tenant taking all.

    min/max is dominated by a single outlier and says little beyond two tenants.
    Jain's index is the standard measure in the resource-sharing literature, so
    both are reported.
    """
    values = [value for value in values if value is not None]
    if not values or sum(values) <= 0:
        return None
    return (sum(values) ** 2) / (len(values) * sum(value * value for value in values))


def classify_error(row):
    """Bucket a failed request so error rate is diagnosable, not just a count.

    A mode that "failed 12 requests" means something completely different if
    those were 503s from an overloaded router than if they were client-side
    read timeouts caused by the GPU being starved.
    """
    if row.get("ok"):
        return None
    status = row.get("status") or 0
    if status:
        return f"http_{status}"
    error = (row.get("error") or "").lower()
    if "timeout" in error or "timed out" in error:
        return "timeout"
    if "connection" in error or "refused" in error or "reset" in error:
        return "connection"
    return "other"


def error_breakdown(rows):
    counts = defaultdict(int)
    for row in rows:
        bucket = classify_error(row)
        if bucket:
            counts[bucket] += 1
    return dict(sorted(counts.items()))


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
            "error_rate": (len(rows) - len(ok_rows)) / len(rows) if rows else 0,
            "errors": error_breakdown(rows),
            "active_window_seconds": window,
            "requests_per_second": len(ok_rows) / window if window else 0,
            "output_tokens": service_tokens,
            "output_tokens_per_second": service_tokens / window if window else 0,
            "latency_mean_seconds": statistics.mean(service_latencies) if service_latencies else 0,
            "latency_p50_seconds": percentile(service_latencies, 50),
            "latency_p95_seconds": percentile(service_latencies, 95),
            "latency_p99_seconds": percentile(service_latencies, 99),
        }

    # Contended-window metrics. These, not the whole-run numbers, are what
    # should be compared across modes: they exclude the uncontended tail where
    # one tenant had already finished and the other had the GPU to itself.
    contended = None
    window = overlap_window(by_service)
    if window:
        window_start, window_end = window
        window_seconds = window_end - window_start
        rates = {}
        for service, rows in sorted(by_service.items()):
            service_tokens = tokens_in_window(rows, window_start, window_end)
            rate = service_tokens / window_seconds if window_seconds else 0
            rates[service] = rate
            service_summaries[service]["contended_output_tokens"] = service_tokens
            service_summaries[service]["contended_tokens_per_second"] = rate
        values = list(rates.values())
        contended = {
            "window_start_offset_seconds": window_start,
            "window_end_offset_seconds": window_end,
            "window_seconds": window_seconds,
            "fraction_of_wall": window_seconds / wall if wall else 0,
            "output_tokens_per_second": sum(values),
            "per_service_tokens_per_second": rates,
            "fairness_ratio": (min(values) / max(values)) if values and max(values) > 0 else None,
            "jain_fairness_index": jain_index(values),
        }

    # Fairness across tenants: 1.0 means both services got identical throughput.
    # This is the number that actually answers "is the shared GPU fair?", which
    # is the QoS question in the project goals.
    throughputs = [entry["output_tokens_per_second"] for entry in service_summaries.values()]
    if len(throughputs) >= 2 and max(throughputs) > 0:
        fairness = min(throughputs) / max(throughputs)
    else:
        fairness = None

    # Equal-work guard. Every request should emit exactly the same number of
    # tokens (ignore_eos + min_tokens). If it does not, tokens/sec is comparing
    # different workloads across modes and the comparison is invalid.
    token_counts = sorted({row["output_tokens"] for row in successful})
    equal_work = len(token_counts) <= 1

    # Prompt-side audit. Prompts are now unique per request index, so prefill
    # work varies slightly between requests. It varies IDENTICALLY across modes
    # (index N always carries the same prompt), so it cannot bias a comparison -
    # but record the spread so that claim is checkable rather than asserted.
    prompt_lengths = sorted({row.get("prompt_tokens", 0) for row in successful})
    distinct_prompts = len({row.get("prompt", "") for row in results})

    return {
        "total_requests": len(results),
        "successful_requests": len(successful),
        "failed_requests": len(results) - len(successful),
        # Error rate is a first-class result, not a footnote: a mode that wins on
        # throughput while dropping requests has not won. Buckets separate an
        # overloaded router (http_503) from a GPU-starved server (timeout).
        "error_rate": (len(results) - len(successful)) / len(results) if results else 0,
        "errors": error_breakdown(results),
        "wall_seconds": wall,
        "output_tokens_per_request": token_counts,
        "equal_work_across_requests": equal_work,
        "distinct_prompts": distinct_prompts,
        "prompt_tokens_distinct": prompt_lengths,
        "prompt_tokens_mean": (
            statistics.mean([row.get("prompt_tokens", 0) for row in successful])
            if successful else 0),
        "requests_per_second": len(successful) / wall if wall else 0,
        "output_tokens": tokens,
        "output_tokens_per_second": tokens / wall if wall else 0,
        "latency_mean_seconds": statistics.mean(latencies) if latencies else 0,
        "latency_p50_seconds": percentile(latencies, 50),
        "latency_p95_seconds": percentile(latencies, 95),
        "latency_p99_seconds": percentile(latencies, 99),
        "fairness_ratio": fairness,
        "jain_fairness_index": jain_index(throughputs) if len(throughputs) >= 2 else None,
        "contended": contended,
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
    # Two services that never overlapped cannot be compared for fairness, and
    # have no contended window: the phases ran one after the other by design.
    summary["fairness_ratio"] = None
    summary["jain_fairness_index"] = None
    summary["contended"] = None
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

    # Human-readable transcript of what Qwen actually replied. The JSON/CSV hold
    # the same text, but this is the file to open when checking that the model
    # produced sensible output rather than 64 tokens of garbage.
    summary = result.get("summary", {})
    lines = [
        f"mode: {result.get('mode')}",
        f"model: {result.get('model')}",
        f"prompt mode: {result.get('prompt_mode', 'fixed')} "
        f"({summary.get('distinct_prompts', '?')} distinct prompts, "
        f"{summary.get('prompt_tokens_distinct', '?')} prompt-token lengths)",
        "",
    ]
    for row in rows:
        lines.append(f"--- {row['service']} request {row['request']} "
                     f"({row['output_tokens']} tokens, {row['latency_seconds']:.3f}s) ---")
        lines.append(f"PROMPT: {row.get('prompt', '')}")
        lines.append(row.get("response_text") or f"[FAILED] {row.get('error', '')}")
        lines.append("")
    output.with_name(output.stem + ".responses.txt").write_text(
        "\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True)
    parser.add_argument("--endpoint", action="append", default=[], help="service=url")
    parser.add_argument("--output", required=True)
    parser.add_argument("--requests-per-service", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=1, help="warm-up requests per service, excluded from metrics")
    parser.add_argument("--concurrency", type=int, default=1,
                        help="in-flight requests per service (1 = strictly serial tenant)")
    parser.add_argument("--max-tokens", type=int, default=64,
                        help="tokens generated per request; exact, via ignore_eos + min_tokens")
    parser.add_argument("--prompt-mode", choices=("varied", "fixed"), default="varied",
                        help="varied (default): a unique deterministic prompt per request "
                             "index, identical across modes, so nothing can be served from "
                             "cache. fixed: the original single repeated prompt, kept to "
                             "reproduce the earlier published run.")
    parser.add_argument("--prompt-offset", type=int, default=0,
                        help="shift the prompt sequence; the runner passes repeat*100000 so "
                             "in-place repeats never replay prompts against a warm server")
    parser.add_argument("--duration-seconds", type=float, default=0,
                        help="if >0, ignore --requests-per-service and have every tenant "
                             "issue requests until this deadline, so all tenants stop "
                             "together and no tenant gets an uncontended tail")
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
            args.concurrency,
            args.max_tokens,
            args.duration_seconds,
            args.prompt_mode,
            args.prompt_offset,
        )

    write_output(result, args.output)
    print(json.dumps(result["summary"], indent=2))
    return 0 if result["summary"]["failed_requests"] == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
