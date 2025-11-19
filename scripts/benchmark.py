#!/usr/bin/env python3
"""
benchmark.py — simple concurrency generator for llm-mock endpoints.
Usage: python3 benchmark.py --targets 3 --rps 20 --duration 60
"""
import argparse
import asyncio
import aiohttp
import time
import csv
from statistics import mean, median

async def worker(session, url, delay):
    await asyncio.sleep(delay)
    t0 = time.time()
    try:
        async with session.post(url, json={"work_ms": 30}, timeout=10) as resp:
            await resp.text()
            return (time.time() - t0) * 1000.0
    except Exception as e:
        return None

async def run_workload(target_ports, rps, duration, out_csv):
    tasks = []
    start = time.time()
    async with aiohttp.ClientSession() as session:
        seq = 0
        while time.time() - start < duration:
            # schedule rps requests spread across second
            for i in range(rps):
                port = target_ports[seq % len(target_ports)]
                url = f"http://localhost:{port}/infer"
                # schedule tasks with tiny jitter
                task = asyncio.ensure_future(worker(session, url, 0))
                tasks.append((time.time(), port, task))
                seq += 1
            # wait one second
            await asyncio.sleep(1)
        # gather
        results = []
        for ts, port, t in tasks:
            lat = await t
            results.append((ts, port, lat))
    # write csv
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "port", "lat_ms"])
        for r in results:
            writer.writerow(r)
    # simple stats
    latencies = [r[2] for r in results if r[2] is not None]
    print("requests:", len(results))
    print("success:", len(latencies))
    if latencies:
        print("p50:", median(latencies))
        print("mean:", mean(latencies))
        print("p95:", sorted(latencies)[int(len(latencies)*0.95)])
        print("p99:", sorted(latencies)[int(len(latencies)*0.99) if len(latencies)>100 else -1])

def parse_ports(min_c, max_c, base=5000):
    # produce ports for up to max containers
    ports = [base + i for i in range(1, max_c+1)]
    return ports

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--targets", type=int, default=2, help="num targets (containers)")
    p.add_argument("--rps", type=int, default=10, help="requests per second")
    p.add_argument("--duration", type=int, default=60, help="seconds")
    p.add_argument("--out", type=str, default="bench_out.csv", help="csv output")
    args = p.parse_args()
    ports = parse_ports(args.targets, args.targets)
    asyncio.run(run_workload(ports, args.rps, args.duration, args.out))
