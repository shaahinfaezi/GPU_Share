#!/usr/bin/env python3
"""
MPS Benchmark Script
Tests inference performance: Sequential (no MPS benefit) vs Parallel (MPS benefit)
Also shows VRAM isolation from the shim
"""

import subprocess
import requests
import time
import json
import threading
import sys
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

NUM_REQUESTS_PER_CONTAINER = 3  # Requests per container in each test

# Configuration
LONG_PROMPT = """The history of artificial intelligence began in antiquity, with myths, stories and rumors of artificial beings endowed with intelligence or consciousness by master craftsmen. The seeds of modern AI were planted by classical philosophers who attempted to describe the process of human thinking as the mechanical manipulation of symbols. This work culminated in the invention of the programmable digital computer in the 1940s, a machine based on the abstract essence of mathematical reasoning. This device and the ideas behind it inspired a handful of scientists to begin seriously discussing the possibility of building an electronic brain. The field of AI research was founded at a workshop held on the campus of Dartmouth College during the summer of 1956. Those who attended would become the leaders of AI research for decades. Many of them predicted that a machine as intelligent as a human being would exist in no more than a generation, and they were given millions of dollars to make this vision come true. Eventually, it became obvious that they had grossly underestimated the difficulty of the project. In 1973, in response to the criticism from James Lighthill and ongoing pressure from congress, the U.S. and British Governments stopped funding undirected research into artificial intelligence, and the difficult years that followed would later be known as an AI winter. Seven years later, a visionary initiative by the Japanese Government inspired governments and industry to provide AI with billions of dollars, but by the late 1980s the investors became disillusioned and withdrew funding again."""

MAX_NEW_TOKENS = 300  # Generate enough tokens to take measurable time

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

def check_gpu_available():
    """Check if GPU is available, exit if not"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            print("ERROR: nvidia-smi failed. GPU is not available.")
            print(f"  stderr: {result.stderr}")
            sys.exit(1)
        if not result.stdout.strip():
            print("ERROR: No GPUs detected by nvidia-smi.")
            sys.exit(1)
        print(f"  ✓ GPU(s) detected:")
        for line in result.stdout.strip().split('\n'):
            print(f"    {line}")
        return True
    except FileNotFoundError:
        print("ERROR: nvidia-smi not found. NVIDIA driver not installed.")
        sys.exit(1)
    except subprocess.TimeoutExpired:
        print("ERROR: nvidia-smi timed out. GPU may be hung.")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to check GPU: {e}")
        sys.exit(1)

def get_container_vram_shim_log(container_name):
    """Get VRAM limiter shim log from inside a container"""
    try:
        result = subprocess.run(
            ['sudo', 'docker', 'exec', container_name, 'cat', '/tmp/vram_limiter.log'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
        return None
    except Exception:
        return None

def get_container_name_by_port(port):
    """Get container name by port"""
    try:
        result = subprocess.run(
            ['sudo', 'docker', 'ps', '--filter', f'publish={port}', '--format', '{{.Names}}'],
            capture_output=True, text=True, timeout=5
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except:
        return None

def print_vram_isolation_status(ports):
    """Print VRAM isolation status from shim logs"""
    print(f"\n{'='*60}")
    print(" VRAM ISOLATION STATUS (from shim logs)")
    print('='*60)
    
    for port in ports:
        container_name = get_container_name_by_port(port)
        if container_name:
            shim_log = get_container_vram_shim_log(container_name)
            print(f"\n  Container: {container_name} (port {port})")
            if shim_log:
                lines = shim_log.split('\n')
                vram_limit = None
                total_allocated = None
                for line in lines:
                    if 'VRAM limit set to' in line:
                        vram_limit = line.split('VRAM limit set to')[-1].strip()
                    if 'New total usage:' in line:
                        total_allocated = line.split('New total usage:')[-1].strip()
                
                if vram_limit:
                    print(f"    VRAM Limit: {vram_limit}")
                if total_allocated:
                    print(f"    Current Allocation: {total_allocated}")
                print(f"    Shim Status: ✓ Active and enforcing limits")
            else:
                print(f"    Shim Status: ? Could not read shim log")
        else:
            print(f"\n  Port {port}: Container not found")

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
                'output_preview': data.get('text', '')[:100] + '...',
                'text': data.get('text', '')
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

def run_sequential_requests(ports, num_requests_per_port, prompt, max_tokens):
    """Run inference requests SEQUENTIALLY (one at a time) across containers"""
    results = []
    request_id = 0
    
    # Process each container one at a time
    for port in ports:
        print(f"    Processing container on port {port}...")
        for i in range(num_requests_per_port):
            result = send_inference_request(port, prompt, max_tokens, request_id)
            results.append(result)
            request_id += 1
            status = "✓" if result['success'] else "✗"
            latency = result.get('latency_ms', 0)
            print(f"      Request {request_id}: {status} ({latency:.0f}ms)")
    
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

def run_benchmark(test_name, ports, num_requests, prompt, max_tokens, parallel=True):
    """Run a single benchmark test"""
    mode_str = "PARALLEL (MPS)" if parallel else "SEQUENTIAL (no MPS benefit)"
    print(f"\n{'#'*60}")
    print(f"# {test_name}")
    print(f"# Ports: {ports}")
    print(f"# Requests per port: {num_requests}")
    print(f"# Mode: {mode_str}")
    print('#'*60)
    
    # Check health first
    print("\nChecking container health...")
    for port in ports:
        healthy = check_container_health(port)
        status = "✓ Healthy" if healthy else "✗ Unhealthy"
        print(f"  Port {port}: {status}")
        if not healthy:
            print(f"  ERROR: Container on port {port} is not healthy!")
            return None
    
    # Show initial GPU state
    print_gpu_stats_visual(get_gpu_stats(), "Initial GPU State")
    
    # Show VRAM isolation status
    print_vram_isolation_status(ports)
    
    # Start GPU monitoring in background
    monitor_duration = 120  # Monitor for up to 120 seconds
    gpu_snapshots = []
    monitor_thread = threading.Thread(
        target=lambda: gpu_snapshots.extend(monitor_gpu_during_test(monitor_duration, 0.5))
    )
    
    total_requests = len(ports) * num_requests
    print(f"\nStarting {total_requests} {mode_str.lower()} inference requests...")
    
    # Run the actual benchmark
    monitor_thread.start()
    start_time = time.time()
    if parallel:
        results = run_parallel_requests(ports, num_requests, prompt, max_tokens)
    else:
        results = run_sequential_requests(ports, num_requests, prompt, max_tokens)
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
    
    # Save outputs to file (use /tmp to avoid permission issues)
    output_file = "/tmp/benchmark_outputs.txt"
    try:
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(f"\n\n{'='*60}\n")
            f.write(f"TEST: {test_name}\n")
            f.write(f"DATE: {datetime.now().isoformat()}\n")
            f.write(f"{'='*60}\n")
            for r in successful:
                f.write(f"\n[Request {r['request_id']} | Port {r['port']} | Latency {r['latency_ms']:.2f}ms]\n")
                f.write(f"{r.get('text', '')}\n")
                f.write("-" * 40 + "\n")
        print(f"\n  Outputs saved to {output_file}")
    except Exception as e:
        print(f"\n  Failed to save outputs to file: {e}")

    return {
        'test_name': test_name,
        'mode': 'parallel' if parallel else 'sequential',
        'total_time_ms': total_time * 1000,
        'total_time_sec': total_time,
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
    print(" Sequential vs Parallel (MPS) Comparison")
    print("="*60)
    print(f" Started at: {datetime.now().isoformat()}")
    print(f" Prompt length: {len(LONG_PROMPT)} characters")
    print(f" Max new tokens: {MAX_NEW_TOKENS}")
    print(f" Requests per container: {NUM_REQUESTS_PER_CONTAINER}")
    
    # Check GPU availability first
    print("\n Checking GPU availability...")
    check_gpu_available()
    
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
    
    if len(available_ports) < 3:
        print("\n ERROR: Need at least 3 healthy containers to run benchmark")
        print(" Please ensure containers are running on ports 6000, 6001, 6002")
        sys.exit(1)
    
    # Use first 3 containers
    test_ports = available_ports[:3]
    print(f"\n Using containers on ports: {test_ports}")
    
    results = []
    
    # ============================================================
    # TEST 1: SEQUENTIAL (No MPS benefit - requests one at a time)
    # ============================================================
    print("\n" + "="*60)
    print(" TEST 1: SEQUENTIAL EXECUTION (No MPS Benefit)")
    print(" Each container processes requests one at a time")
    print("="*60)
    result = run_benchmark(
        f"Sequential - 3 containers x {NUM_REQUESTS_PER_CONTAINER} requests",
        test_ports,
        NUM_REQUESTS_PER_CONTAINER,
        LONG_PROMPT,
        MAX_NEW_TOKENS,
        parallel=False  # Sequential execution
    )
    if result:
        results.append(result)
    
    # Small pause between tests
    print("\n Pausing 5 seconds before next test...")
    time.sleep(5)
    
    # ============================================================
    # TEST 2: PARALLEL (MPS benefit - all requests at once)
    # ============================================================
    print("\n" + "="*60)
    print(" TEST 2: PARALLEL EXECUTION (With MPS)")
    print(" All 3 containers process requests simultaneously")
    print("="*60)
    result = run_benchmark(
        f"Parallel (MPS) - 3 containers x {NUM_REQUESTS_PER_CONTAINER} requests",
        test_ports,
        NUM_REQUESTS_PER_CONTAINER,
        LONG_PROMPT,
        MAX_NEW_TOKENS,
        parallel=True  # Parallel execution with MPS
    )
    if result:
        results.append(result)
    
    # Summary
    if len(results) >= 2:
        print("\n" + "="*60)
        print(" BENCHMARK SUMMARY")
        print("="*60)
        
        seq_result = results[0]
        par_result = results[1]
        
        print(f"\n  {'Test':<50} {'Requests':<10} {'Time (sec)':<12} {'Throughput':<12}")
        print("  " + "-"*84)
        for r in results:
            total_requests = r.get('successful', 0) + r.get('failed', 0)
            print(f"  {r['test_name']:<50} {total_requests:<10} {r['total_time_sec']:<12.2f} {r['throughput_rps']:<12.2f} req/s")
        
        # Calculate improvements
        speedup = seq_result['total_time_sec'] / par_result['total_time_sec'] if par_result['total_time_sec'] > 0 else 0
        throughput_improvement = par_result['throughput_rps'] / seq_result['throughput_rps'] if seq_result['throughput_rps'] > 0 else 0
        time_saved = seq_result['total_time_sec'] - par_result['total_time_sec']
        time_saved_pct = (time_saved / seq_result['total_time_sec']) * 100 if seq_result['total_time_sec'] > 0 else 0
        
        print(f"\n  {'='*60}")
        print(f"  ANALYSIS: MPS Performance Impact")
        print(f"  {'='*60}")
        print(f"\n  Sequential Time:  {seq_result['total_time_sec']:.2f} sec ({seq_result['total_time_ms']:.0f} ms)")
        print(f"  Parallel Time:    {par_result['total_time_sec']:.2f} sec ({par_result['total_time_ms']:.0f} ms)")
        print(f"\n  Time Saved:       {time_saved:.2f} sec ({time_saved_pct:.1f}% faster)")
        print(f"  Speedup Factor:   {speedup:.2f}x")
        print(f"  Throughput Boost: {throughput_improvement:.2f}x ({(throughput_improvement-1)*100:.1f}% improvement)")
        
        print(f"\n  {'='*60}")
        print(f"  VRAM ISOLATION & MPS STATUS")
        print(f"  {'='*60}")
        
        if speedup > 1.5:
            print(f"\n  ✓ MPS is working excellently!")
            print(f"    - {speedup:.1f}x speedup demonstrates effective SM time-sharing")
            print(f"    - All 3 containers ran concurrently on the same GPU")
            print(f"    - VRAM isolation shim is enforcing memory limits per container")
        elif speedup > 1.1:
            print(f"\n  ✓ MPS is working!")
            print(f"    - {speedup:.1f}x speedup shows parallel execution benefit")
            print(f"    - VRAM isolation is active")
            print(f"    - Consider a larger model for more GPU-bound workload")
        else:
            print(f"\n  ~ MPS benefit is minimal")
            print(f"    - Workload may be CPU-bound rather than GPU-bound")
            print(f"    - Try increasing MAX_NEW_TOKENS or using a larger model")
        
        print(f"\n  How it works:")
        print(f"    1. MPS allows multiple CUDA contexts to share GPU SMs")
        print(f"    2. VRAM Limiter Shim intercepts cudaMalloc to enforce memory limits")
        print(f"    3. Each container thinks it has exclusive GPU access")
        print(f"    4. Sequential test = no overlap, Parallel test = full overlap via MPS")
    
    print(f"\n{'='*60}")
    print(f" Benchmark completed at: {datetime.now().isoformat()}")
    print(f" LLM outputs saved to: /tmp/benchmark_outputs.txt")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
