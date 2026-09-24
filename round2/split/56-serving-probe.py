#!/usr/bin/env python3
"""Sequential, single-tenant serving probes on Box1; never writes calibration.

Requires the existing round2 harness/replay and Linux root for pod PSS. Every
profile replays one identical, small synthetic Poisson trace. This is a serving
compatibility/memory probe, NOT a capacity baseline or shared admission test.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import threading
import time
import uuid

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import harness as H
from split_checks import replay_quality


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2), encoding='utf-8')


def make_trace(role):
    rng, rows, t = random.Random(20260924), [], 0.0
    prompt = 'Explain why measuring memory matters when serving machine learning models.'
    for _ in range(2):
        rows.append(dict(t=0, service=role, kind='chat', warmup=True,
                         prompt=prompt, max_tokens=32))
    while True:
        t += rng.expovariate(0.25)
        if t >= 60:
            break
        rows.append(dict(t=round(t, 6), service=role, kind='chat', warmup=False,
                         prompt=prompt, max_tokens=64))
    digest = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()[:16]
    return dict(requests=rows, services=[role], trace_id=digest,
                params=dict(duration=60, rate=0.25, source='synthetic', seed=20260924,
                            purpose='serving-memory-probe-not-capacity'))


def pod_pss(uid, proc_root=Path('/proc')):
    """Select all pod processes by cgroup UID, not fragile process-name matching."""
    rows = []
    missing = 0
    for proc in proc_root.glob('[0-9]*'):
        try:
            group = (proc / 'cgroup').read_text()
        except OSError:
            continue
        if uid not in group and uid.replace('-', '_') not in group:
            continue
        try:
            fields = {}
            for line in (proc / 'smaps_rollup').read_text().splitlines():
                key, _, value = line.partition(':')
                if key in ('Pss', 'Rss', 'Private_Dirty', 'Private_Clean',
                           'Shared_Dirty', 'Shared_Clean', 'Anonymous'):
                    fields[key + '_mib'] = round(float(value.split()[0]) / 1024, 2)
            rows.append(dict(pid=int(proc.name), name=(proc / 'comm').read_text().strip(),
                             **fields))
        except (OSError, ValueError, IndexError):
            if proc.exists():
                missing += 1
    return dict(processes=rows, pss_mib=sum(r.get('Pss_mib', 0) for r in rows),
                unreadable_or_exited_processes=missing,
                complete=bool(rows) and missing == 0)


class Monitor(threading.Thread):
    def __init__(self, out, reserve=1536):
        super().__init__(daemon=True)
        self.out, self.reserve = out, reserve
        self.halt, self.abort = threading.Event(), threading.Event()
        self.minimum, self.reason = None, None

    def sample(self):
        available = H.host_mem_mib().get('available')
        if available is None:
            self.reason = 'Cannot read MemAvailable'
        else:
            self.minimum = available if self.minimum is None else min(self.minimum, available)
            if available < self.reserve:
                self.reason = 'MemAvailable %s MiB below probe reserve %s MiB' % (
                    available, self.reserve)
        if self.reason:
            self.abort.set()
        return available

    def run(self):
        with (self.out / 'host.csv').open('w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['t', 'mem_available'])
            while not self.halt.is_set():
                writer.writerow([time.time(), self.sample()])
                handle.flush()
                self.halt.wait(0.5)


def command(args, timeout=60, monitor=None):
    """Interrupt a waiting CLI/replay promptly when the memory guard fires."""
    deadline = time.monotonic() + timeout
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        while True:
            if monitor and monitor.abort.is_set():
                raise RuntimeError(monitor.reason)
            if time.monotonic() >= deadline:
                raise RuntimeError('Command timed out: ' + ' '.join(args[:4]))
            try:
                output, _ = proc.communicate(timeout=0.5)
                if proc.returncode:
                    raise RuntimeError('Command failed: %s\n%s' % (args[:4], output[-4000:]))
                return output
            except subprocess.TimeoutExpired:
                pass
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.communicate()


def kube(args, timeout=60, monitor=None):
    return command(['kubectl', '-n', H.NS] + args, timeout, monitor)


def own_object(doc, token):
    return bool(doc and doc.get('metadata', {}).get('labels', {}).get('r2-probe') == token)


def cleanup(name, token, role):
    raw = kube(['get', 'llminferenceservice', name, '--ignore-not-found', '-o', 'json'])
    obj = json.loads(raw) if raw.strip() else None
    if obj is not None:
        if not own_object(obj, token):
            raise RuntimeError('Refusing to delete an object not owned by this probe: ' + name)
        kube(['delete', 'llminferenceservice', name, '--wait=true', '--timeout=120s'], 140)
    deadline = time.monotonic() + 180
    while H.pods_for(role):
        if time.monotonic() >= deadline:
            raise RuntimeError('Probe pods still exist after deletion; do not start another profile')
        time.sleep(2)
    if not H.wait_gpu_idle(timeout=120):
        raise RuntimeError('GPU memory was not released; do not start another profile')


def stable_idle():
    """Sample after verified teardown; reject an unstable baseline instead of dropping caches."""
    samples = []
    for _ in range(12):
        value = H.host_mem_mib().get('available')
        if value is None:
            raise RuntimeError('Cannot sample host memory')
        samples.append(value)
        if len(samples) >= 5 and max(samples[-5:]) - min(samples[-5:]) <= 256:
            baseline = min(samples[-5:])
            if baseline < 6656:
                raise RuntimeError('Less than 6656 MiB idle headroom; inspect node before probe')
            return baseline, samples
        time.sleep(2)
    raise RuntimeError('Idle memory is not stable; inspect node activity before probing')


def run_profile(renderer, role, profile, root, trace):
    out = root / profile
    out.mkdir()
    token, name = uuid.uuid4().hex, 'r2-' + role
    body = renderer.render(role, 'dedicated', '100', profile)
    body = body.replace('    r2-series: split\n', '    r2-series: split\n    r2-probe: %s\n' % token)
    body = body.replace('  template:\n', '  template:\n    nodeSelector: {kubernetes.io/hostname: mr-gholami}\n')
    manifest = out / 'tenant.yaml'
    manifest.write_text(body, encoding='utf-8')
    write(out / 'trace.json', trace)
    idle, samples = stable_idle()
    write(out / 'idle-hostmem.json', dict(available=idle, samples=samples))
    monitor = Monitor(out)
    result = dict(profile=profile, role=role, ok=False, calibration=False,
                  idle_available_mib=idle, trace_id=trace['trace_id'])
    if profile == 'lowmem-fork':
        result['requested_worker_method'] = 'fork'
    monitor.start()  # Before kubectl create, including storage/startup peaks.
    try:
        print('Starting %s %s alone; idle %s MiB' % (role, profile, idle), flush=True)
        kube(['create', '-f', str(manifest)], monitor=monitor)
        deadline = time.monotonic() + 900
        pod = None
        while time.monotonic() < deadline:
            if monitor.abort.is_set():
                raise RuntimeError(monitor.reason)
            pods = json.loads(kube(['get', 'pods', '-l', H.KSERVE_SEL % role,
                                    '-o', 'json'], monitor=monitor))['items']
            for p in pods:
                if p.get('spec', {}).get('nodeName') not in (None, '', 'mr-gholami'):
                    raise RuntimeError('Probe placed on unexpected node')
                statuses = p.get('status', {}).get('containerStatuses', [])
                if any(s.get('restartCount', 0) for s in statuses):
                    raise RuntimeError('Container restarted during probe startup')
                if any(c['type'] == 'Ready' and c['status'] == 'True'
                       for c in p.get('status', {}).get('conditions', [])):
                    pod = p
            if pod:
                break
            time.sleep(2)
        if pod is None:
            raise RuntimeError('Readiness timed out')
        write(out / 'pod-ready.json', pod)
        write(out / 'pss-ready.json', pod_pss(pod['metadata']['uid']))
        versions = kube(['exec', pod['metadata']['name'], '-c', 'main', '--', 'python', '-c',
                        'import importlib.metadata as m; print({k:m.version(k) for k in ("vllm","torch")})'],
                        monitor=monitor)
        (out / 'versions.txt').write_text(versions, encoding='utf-8')
        ok, detail = H.assert_mps_matches_arm('dedicated', [role])
        if not ok:
            raise RuntimeError('MPS state mismatch: %s' % detail)
        base = H.endpoints([role])[role]
        model = H.served_model(base)
        if model != renderer.ROLES[role]['model']:
            raise RuntimeError('Unexpected served model: %s' % model)
        gpu = H.gpu_used_mib()
        write(out / 'gpu-ready.json', dict(used_mib=gpu, placement=H.placement([role])))
        if len([v for v in gpu.values() if v > 500]) != 2:
            raise RuntimeError('Expected model residency on both GPUs')
        print('Ready. Replaying the bounded 60-second probe (not calibration).', flush=True)
        log = command([sys.executable, str(HERE.parent / 'workload' / 'replay.py'),
                       '--trace', str(out / 'trace.json'), '--arm', 'dedicated',
                       '--endpoint', role + '=' + base, '--model', role + '=' + model,
                       '--only', role, '--out-dir', str(out), '--max-inflight', '16',
                       '--timeout', '60', '--note', 'serving memory probe; not calibration'],
                      timeout=240, monitor=monitor)
        (out / 'replay.log').write_text(log, encoding='utf-8')
        pss = pod_pss(pod['metadata']['uid'])
        write(out / 'pss-after-replay.json', pss)
        if not pss['complete']:
            raise RuntimeError('Pod process PSS is incomplete; run on Box1 as root')
        current = json.loads(kube(['get', 'pod', pod['metadata']['name'], '-o', 'json'],
                                 monitor=monitor))
        if current['metadata']['uid'] != pod['metadata']['uid'] or any(
                s.get('restartCount', 0) for s in current['status'].get('containerStatuses', [])):
            raise RuntimeError('Pod changed or restarted during replay')
        quality = replay_quality(out, [role])
        write(out / 'quality.json', quality)
        if not quality['ok']:
            raise RuntimeError('Replay quality failed: %s' % quality['issues'])
        startup_log = kube(['logs', pod['metadata']['name'], '-c', 'main', '--tail=500'],
                           monitor=monitor)
        (out / 'server.log').write_text(startup_log, encoding='utf-8')
        if profile == 'lowmem-fork':
            overridden = any('Overriding VLLM_WORKER_MULTIPROC_METHOD' in line and
                             'spawn' in line for line in startup_log.splitlines())
            result['spawn_override_detected'] = overridden
            if overridden:
                raise RuntimeError('vLLM forced spawn; the requested fork candidate was not exercised')
        monitor.sample()
        if monitor.abort.is_set():
            raise RuntimeError(monitor.reason)
        result.update(ok=True, pod_pss_after_replay_mib=pss['pss_mib'])
    except BaseException as exc:
        result['error'] = str(exc) or type(exc).__name__
        raise
    finally:
        monitor.halt.set()
        monitor.join(timeout=5)
        result.update(minimum_available_mib=monitor.minimum,
                      peak_host_delta_mib=None if monitor.minimum is None else idle - monitor.minimum,
                      memory_guard=monitor.reason)
        # Preserve evidence before removing only the object created by this pass.
        # On memory pressure, prioritize teardown over expensive diagnostics.
        if not monitor.abort.is_set():
            try:
                for p in H.pods_for(role):
                    rc, logs = H.kubectl(['logs', '-n', H.NS, p['metadata']['name'],
                                         '--all-containers=true', '--tail=500'], timeout=20)
                    (out / 'server.log').write_text(logs, encoding='utf-8')
                rc, events = H.kubectl(['get', 'events', '-n', H.NS,
                                       '--sort-by=.lastTimestamp'], timeout=20)
                (out / 'events.txt').write_text(events, encoding='utf-8')
            except Exception as exc:
                result['diagnostic_error'] = str(exc)
        try:
            cleanup(name, token, role)
        except Exception as exc:
            result.update(ok=False, cleanup_error=str(exc))
            raise
        finally:
            write(out / 'probe.json', result)
            print(json.dumps(result, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--role', choices=['tp', 'pp'], default='tp')
    parser.add_argument('--profiles', nargs='+', choices=['lowmem-eager', 'lowmem-serve', 'lowmem-fork'],
                        default=['lowmem-eager', 'lowmem-serve'])
    args = parser.parse_args()
    if len(set(args.profiles)) != len(args.profiles):
        parser.error('Profiles must be unique')
    if sys.platform != 'linux' or os.geteuid() != 0:
        parser.error('Run on Box1 as Linux root; this tool starts a single GPU workload')
    import fcntl
    os.chdir(HERE)
    with open('.split-run.lock', 'w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('Another split stage is running')
        spec = importlib.util.spec_from_file_location('split_renderer', HERE / '50-render-split.py')
        renderer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(renderer)
        # Validate the updated renderer before making any cluster change.
        for profile in args.profiles:
            renderer.render(args.role, 'dedicated', '100', profile)
        objects = json.loads(kube(['get', 'llminferenceservice', '-o', 'json']))
        if any(x['metadata']['name'] in ('r2-tp', 'r2-pp', 'r2-embed-small')
               for x in objects['items']) or H.all_r2_pods():
            raise SystemExit('Existing inference workloads found; inspect before probing')
        if not H.wait_gpu_idle(timeout=30):
            raise SystemExit('GPUs are not idle')
        # Match the dedicated arm, using the existing benchmark MPS controller.
        ok, detail = H.set_mps_daemon(False)
        if not ok:
            raise SystemExit('Cannot disable MPS: %s' % detail)
        def interrupt(signum, frame):
            raise KeyboardInterrupt('Probe interrupted')
        signal.signal(signal.SIGTERM, interrupt)
        root = HERE / 'results' / ('S-serving-probe-' + time.strftime('%Y%m%d-%H%M%S', time.gmtime()))
        root.mkdir(parents=True, exist_ok=False)
        print('Probe directory: %s' % root, flush=True)
        trace = make_trace(args.role)
        results = []
        for profile in args.profiles:
            results.append(run_profile(renderer, args.role, profile, root, trace))
        write(root / 'comparison.json', dict(probes=results, calibration=False,
              shared_admission=False, note='Short light-load comparison; full-load peak remains unknown'))
        print('Probe complete. Share these probe.json files and server.log; do not start sharing.', flush=True)


if __name__ == '__main__':
    main()
