#!/usr/bin/env python3
"""
MPS Benchmark Script
Tests inference performance with 1 vs 2 containers per GPU
"""

import subprocess
import requests
import time
import json
import threading
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# Configuration
LONG_PROMPT = """The history of artificial intelligence began in antiquity, with myths, stories and rumors of artificial beings endowed with intelligence or consciousness by master craftsmen. The seeds of modern AI were planted by classical philosophers who attempted to describe the process of human thinking as the mechanical manipulation of symbols. This work culminated in the invention of the programmable digital computer in the 1940s, a machine based on the abstract essence of mathematical reasoning. This device and the ideas behind it inspired a handful of scientists to begin seriously discussing the possibility of building an electronic brain. The field of AI research was founded at a workshop held on the campus of Dartmouth College during the summer of 1956. Those who attended would become the leaders of AI research for decades. Many of them predicted that a machine as intelligent as a human being would exist in no more than a generation, and they were given millions of dollars to make this vision come true. Eventually, it became obvious that they had grossly underestimated the difficulty of the project. In 1973, in response to the criticism from James Lighthill and ongoing pressure from congress, the U.S. and British Governments stopped funding undirected research into artificial intelligence, and the difficult years that followed would later be known as an AI winter. Seven years later, a visionary initiative by the Japanese Government inspired governments and industry to provide AI with billions of dollars, but by the late 1980s the investors became disillusioned and withdrew funding again."""

MAX_NEW_TOKENS = 50  # Generate enough tokens to take measurable time

def get_gpu_stats():
    """Get current GPU memory usage"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,memory.used,memory.total,utilization.gpu', 
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, check=True
        )
        stats = []
        for line in result.stdout.strip().split('\n'):
            parts = [p.strip() for p in line.split(',')]
            stats.append({
                'gpu_id': int(parts[0]),
                'used_mb': int(parts[1]),
                'total_mb': int(parts[2]),
                'util_pct': int(parts[3])
            })
        return stats
    except Exception as e:
        print(f"Error getting GPU stats: {e}")
        return []

def send_inference_request(port, prompt, max_tokens, request_id):
    """Send a single inference request and return timing info"""
    url = f"http://localhost:{port}/infer"
    payload = {"prompt": prompt, "max_new_tokens": max_tokens}
    
    start_time = time.time()
    try:
        response = requests.post(url, json=payload, timeout=120)
        end_time = time.time()
        
        if response.status_code == 200:
            data = response.json()
            return {
                'request_id': request_id,
                'port': port,
                'success': True,
                'latency_ms': data.get('latency_ms', (end_time - start_time) * 1000),
                'total_time_ms': (end_time - start_time) * 1000,
                'output_preview': data.get('text', '')[:100] + '...'
            }
        else:
            return {
                'request_id': request_id,
                'port': port,
                'success': False,
                'error': f"HTTP {response.status_code}"
            }
    except Exception as e:
        return {
            'request_id': request_id,
            'port': port,
            'success': False,
            'error': str(e)
        }

def check_container_health(port):
    """Check if container is healthy"""
    try:
        response = requests.get(f"http://localhost:{port}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def monitor_gpu_during_test(duration_sec, interval=0.5):
    """Monitor GPU usage during test, returns list of snapshots"""
    snapshots = []
    start = time.time()
    while time.time() - start < duration_sec:
        stats = get_gpu_stats()
        snapshots.append({
            'time': time.time() - start,
            'gpus': stats
        })
        time.sleep(interval)
    return snapshots

def run_parallel_requests(ports, num_requests_per_port, prompt, max_tokens):
    """Run inference requests in parallel across multiple ports"""
    results = []
    
    with ThreadPoolExecutor(max_workers=len(ports) * num_requests_per_port) as executor:
        futures = []
        request_id = 0
        
        # Submit all requests
        for port in ports:
            for i in range(num_requests_per_port):
                future = executor.submit(send_inference_request, port, prompt, max_tokens, request_id)
                futures.append(future)
                request_id += 1
        
        # Collect results
        for future in as_completed(futures):
            results.append(future.result())
    
    return results

def print_ascii_bar(label, value, max_value, width=40):
    """Print an ASCII bar chart"""
    filled = int((value / max_value) * width) if max_value > 0 else 0
    bar = '█' * filled + '░' * (width - filled)
    print(f"  {label}: [{bar}] {value:.0f}/{max_value:.0f} MB")

def print_gpu_stats_visual(stats, title="GPU Memory Usage"):
    """Print GPU stats with visual bars"""
    print(f"\n{'='*60}")
    print(f" {title}")
    print('='*60)
    for gpu in stats:
        print(f"\n  GPU {gpu['gpu_id']}:")
        print_ascii_bar("VRAM", gpu['used_mb'], gpu['total_mb'])
        print(f"  Utilization: {gpu['util_pct']}%")

def run_benchmark(test_name, ports, num_requests, prompt, max_tokens):
    """Run a single benchmark test"""
    print(f"\n{'#'*60}")
    print(f"# {test_name}")
    print(f"# Ports: {ports}")
    print(f"# Requests per port: {num_requests}")
    print('#'*60)
    
    # Check health first
    print("\nChecking container health...")
    for port in ports:
        healthy = check_container_health(port)
        status = "✓ Healthy" if healthy else "✗ Unhealthy"
        print(f"  Port {port}: {status}")
        if not healthy:
            print(f"  WARNING: Container on port {port} is not healthy!")
            return None
    
    # Show initial GPU state
    print_gpu_stats_visual(get_gpu_stats(), "Initial GPU State")
    
    # Start GPU monitoring in background
    monitor_duration = 60  # Monitor for up to 60 seconds
    gpu_snapshots = []
    monitor_thread = threading.Thread(
        target=lambda: gpu_snapshots.extend(monitor_gpu_during_test(monitor_duration, 0.5))
    )
    
    print(f"\nStarting {len(ports) * num_requests} parallel inference requests...")
    
    # Run the actual benchmark
    monitor_thread.start()
    start_time = time.time()
    results = run_parallel_requests(ports, num_requests, prompt, max_tokens)
    end_time = time.time()
    total_time = end_time - start_time
    
    # Wait a bit for monitor to capture final state
    time.sleep(1)
    
    # Show final GPU state
    print_gpu_stats_visual(get_gpu_stats(), "Final GPU State")
    
    # Analyze results
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    print(f"\n{'='*60}")
    print(" RESULTS")
    print('='*60)
    print(f"  Total requests: {len(results)}")
    print(f"  Successful: {len(successful)}")
    print(f"  Failed: {len(failed)}")
    print(f"  Total wall-clock time: {total_time*1000:.2f} ms")
    
    if successful:
        latencies = [r['latency_ms'] for r in successful]
        print(f"\n  Latency Statistics (successful requests):")
        print(f"    Min: {min(latencies):.2f} ms")
        print(f"    Max: {max(latencies):.2f} ms")
        print(f"    Avg: {sum(latencies)/len(latencies):.2f} ms")
        print(f"    Throughput: {len(successful)/total_time:.2f} requests/sec")
    
    if failed:
        print(f"\n  Failed requests:")
        for r in failed[:5]:  # Show first 5 failures
            print(f"    Request {r['request_id']} (port {r['port']}): {r.get('error', 'Unknown')}")
    
    # Show peak VRAM during test
    if gpu_snapshots:
        print(f"\n  Peak VRAM during test:")
        for gpu_id in range(2):  # Assuming 2 GPUs
            peak = max(
                (s['gpus'][gpu_id]['used_mb'] for s in gpu_snapshots if len(s['gpus']) > gpu_id),
                default=0
            )
            print(f"    GPU {gpu_id}: {peak} MB")
    
    return {
        'test_name': test_name,
        'total_time_ms': total_time * 1000,
        'successful': len(successful),
        'failed': len(failed),
        'avg_latency_ms': sum(r['latency_ms'] for r in successful) / len(successful) if successful else 0,
        'throughput_rps': len(successful) / total_time if successful else 0
    }

def check_mps_status():
    """Check if MPS daemon is running and show diagnostic info"""
    print("\n" + "="*60)
    print(" MPS DIAGNOSTIC CHECK")
    print("="*60)
    
    # Check MPS daemon process
    try:
        result = subprocess.run(['pgrep', '-x', 'nvidia-cuda-mps'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"  ✓ MPS daemon is running (PID: {result.stdout.strip()})")
        else:
            print("  ✗ MPS daemon is NOT running!")
    except Exception as e:
        print(f"  ? Could not check MPS daemon: {e}")
    
    # Check MPS pipe directory
    import os
    mps_dir = "/tmp/nvidia-mps"
    if os.path.exists(mps_dir):
        contents = os.listdir(mps_dir)
        print(f"  ✓ MPS pipe directory exists: {mps_dir}")
        print(f"    Contents: {contents}")
    else:
        print(f"  ✗ MPS pipe directory does not exist: {mps_dir}")
    
    # Check GPU compute mode
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,compute_mode', '--format=csv,noheader'],
            capture_output=True, text=True, check=True
        )
        print(f"  GPU Compute Modes:")
        for line in result.stdout.strip().split('\n'):
            parts = [p.strip() for p in line.split(',')]
            mode = parts[1] if len(parts) > 1 else 'Unknown'
            print(f"    GPU {parts[0]}: {mode}")
            if 'Exclusive' not in mode:
                print(f"      ⚠ Warning: GPU should be in EXCLUSIVE_PROCESS mode for MPS")
    except Exception as e:
        print(f"  ? Could not check compute mode: {e}")
    
    # Check active compute processes
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-compute-apps=pid,name,gpu_uuid', '--format=csv,noheader'],
            capture_output=True, text=True, check=True
        )
        if result.stdout.strip():
            print(f"  Active compute processes:")
            for line in result.stdout.strip().split('\n'):
                print(f"    {line}")
        else:
            print("  No active compute processes on GPUs")
    except Exception as e:
        print(f"  ? Could not check compute processes: {e}")
    
    print("="*60)

def main():
    print("="*60)
    print(" MPS BENCHMARK - GPU Sharing Performance Test")
    print("="*60)
    print(f" Started at: {datetime.now().isoformat()}")
    print(f" Prompt length: {len(LONG_PROMPT)} characters")
    print(f" Max new tokens: {MAX_NEW_TOKENS}")
    
    # Run MPS diagnostic check first
    check_mps_status()
    
    # Detect available containers
    print("\n Detecting containers...")
    available_ports = []
    for port in [6000, 6001, 6002, 6003]:
        if check_container_health(port):
            print(f"  ✓ Port {port} is available")
            available_ports.append(port)
        else:
            print(f"  ✗ Port {port} is not available")
    
    if len(available_ports) < 2:
        print("\n ERROR: Need at least 2 healthy containers to run benchmark")
        print(" Please ensure containers are running and healthy.")
        sys.exit(1)
    
    results = []
    
    # Test 1: Single container per GPU (baseline)
    # Using ports 6000 (GPU 0) and 6002 (GPU 1) if available
    gpu0_port = 6000 if 6000 in available_ports else None
    gpu1_port = 6002 if 6002 in available_ports else (6003 if 6003 in available_ports else None)
    
    if gpu0_port and gpu1_port:
        print("\n" + "="*60)
        print(" TEST 1: One container per GPU (Baseline)")
        print("="*60)
        result = run_benchmark(
            "1 Container/GPU - 10 requests each",
            [gpu0_port, gpu1_port],
            10,
            LONG_PROMPT,
            MAX_NEW_TOKENS
        )
        if result:
            results.append(result)
    
    # Test 2: Two containers per GPU (MPS benefit test)
    # Using all 4 ports if available
    if len(available_ports) >= 3:
        print("\n" + "="*60)
        print(" TEST 2: Two containers per GPU (MPS Test)")
        print("="*60)
        result = run_benchmark(
            "2 Containers/GPU - 10 requests each",
            available_ports[:4],
            10,
            LONG_PROMPT,
            MAX_NEW_TOKENS
        )
        if result:
            results.append(result)
    
    # Summary
    if results:
        print("\n" + "="*60)
        print(" BENCHMARK SUMMARY")
        print("="*60)
        print(f"\n  {'Test':<40} {'Requests':<10} {'Time (ms)':<12} {'Throughput':<12}")
        print("  " + "-"*74)
        for r in results:
            total_requests = r.get('successful', 0) + r.get('failed', 0)
            print(f"  {r['test_name']:<40} {total_requests:<10} {r['total_time_ms']:<12.2f} {r['throughput_rps']:<12.2f} req/s")
        
        if len(results) >= 2:
            # Calculate throughput improvement (the correct metric)
            throughput_speedup = results[1]['throughput_rps'] / results[0]['throughput_rps'] if results[0]['throughput_rps'] > 0 else 0
            
            # Old misleading metric (time ratio with different request counts)
            time_ratio = results[0]['total_time_ms'] / results[1]['total_time_ms'] if results[1]['total_time_ms'] > 0 else 0
            
            print(f"\n  Analysis:")
            print(f"    Throughput Improvement (MPS vs Baseline): {throughput_speedup:.2f}x")
            print(f"    (Test 2 processes {results[1].get('successful', 0) + results[1].get('failed', 0)} requests vs {results[0].get('successful', 0) + results[0].get('failed', 0)} in Test 1)")
            
            if throughput_speedup > 1.0:
                print(f"\n  ✓ MPS is working! {(throughput_speedup - 1) * 100:.1f}% more throughput with 4 containers vs 2")
            elif throughput_speedup > 0.9:
                print(f"\n  ~ MPS is working but minimal benefit (workload may be CPU-bound)")
            else:
                print(f"\n  ✗ MPS may not be configured correctly")
    
    print(f"\n Benchmark completed at: {datetime.now().isoformat()}")

if __name__ == "__main__":
    main()
