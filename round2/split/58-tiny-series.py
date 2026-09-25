#!/usr/bin/env python3
"""Box1 Pythia-70M TP2+TP2 controlled-load series. Run with nohup; no SSH.

Two solo baselines, then separate 300s HAMi and HAMi+MPS arms. Fixed, seeded
Poisson traffic; this measures a specified load, not maximum capacity.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import sys
import time
import uuid

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import importlib.util
spec = importlib.util.spec_from_file_location('tiny_model', HERE / '57-tiny-tp-probe.py')
tiny = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tiny)
P, H = tiny.probe, tiny.probe.H
from split_checks import replay_quality, ensure_fd_limit

ROLES = ('tp-a', 'tp-b')
RESERVE = 1024
NODE = 'mr-gholami'


def log(message):
    print(time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), message, flush=True)


def digest(rows):
    return hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()[:16]


def make_trace(rate, duration, output_tokens, seed):
    rows = []
    kind = getattr(tiny, 'TRACE_KIND', 'completion')
    prompts = (
        'A research team measured how two computers process data. Their report explains',
        'The experiment compared the time required to finish identical tasks. The method was',
        'When several applications share a computer, the available resources determine',
        'The laboratory recorded response times at regular intervals. The observations were',
    )
    for index, role in enumerate(ROLES):
        rng = random.Random(seed + index)
        for warm in range(2):
            rows.append(dict(t=0.0, service=role, kind=kind, warmup=True,
                             prompt=prompts[warm], max_tokens=32, ignore_eos=True))
        t, number = 0.0, 0
        while True:
            t += rng.expovariate(rate)
            if t >= duration:
                break
            # Unique identifiers avoid a trace consisting entirely of one
            # repeated prefix; actual server token counts remain in the CSV.
            prompt = 'Record %s-%d. ' % (role, number) + prompts[number % len(prompts)]
            rows.append(dict(t=round(t, 6), service=role, kind=kind, warmup=False,
                             prompt=prompt, max_tokens=output_tokens, ignore_eos=True))
            number += 1
    rows.sort(key=lambda row: (not row['warmup'], row['t'], row['service']))
    return dict(requests=rows, services=list(ROLES), trace_id=digest(rows),
                params=dict(duration=duration, rate=2 * rate, rate_per_tenant=rate,
                            seed=seed, source='synthetic', model=tiny.MODEL,
                            model_revision=tiny.REVISION, output_tokens=output_tokens,
                            ignore_eos=True, max_model_len=getattr(tiny, 'CONTEXT', 2048), api=kind,
                            purpose='controlled-load ' + getattr(tiny, 'DESIGN', 'TP2+TP2') + '; not capacity calibration'))


def selected_trace(trace, roles):
    rows = [dict(row) for row in trace['requests'] if row['service'] in roles]
    return dict(requests=rows, services=list(roles), trace_id=digest(rows),
                params=dict(trace['params'], parent_trace_id=trace['trace_id'],
                            rate=trace['params']['rate_per_tenant'] * len(roles)))


def admission(solos, available, budget_mib=None):
    deltas = [solos[role]['peak_host_delta_mib'] for role in ROLES]
    if any(not isinstance(x, (int, float)) or not math.isfinite(x) or x <= 0 for x in deltas):
        raise ValueError('Both solo peak host-memory measurements must be positive and finite')
    required = sum(deltas) + RESERVE
    effective = min(available, budget_mib) if budget_mib is not None else available
    return dict(ok=effective >= required, available_mib=available, budget_mib=budget_mib,
                effective_available_mib=effective, required_mib=required,
                headroom_mib=RESERVE, solo_peak_delta_mib=deltas,
                method='sum of separate host MemAvailable drops; includes other node activity')


def render(role, arm, token=None):
    body = tiny.render_tenant(role, arm)
    body = tiny.replace_once(body, '    r2-series: split\n',
                             '    r2-series: %s\n' % getattr(tiny, 'SERIES_LABEL', 'tiny-tp2') +
                             ('    r2-probe: %s\n' % token if token else ''))
    return tiny.replace_once(body, '  template:\n',
                             '  template:\n    nodeSelector: {kubernetes.io/hostname: mr-gholami}\n')


def get_pods(role, monitor=None):
    doc = json.loads(P.kube(['get', 'pods', '-l', H.KSERVE_SEL % role, '-o', 'json'],
                           monitor=monitor))
    return doc['items']


def validate_pod(pod, expected_uid=None):
    if pod['spec'].get('nodeName') != NODE:
        raise RuntimeError('Tenant must run on ' + NODE)
    if expected_uid and pod['metadata']['uid'] != expected_uid:
        raise RuntimeError('Tenant was replaced during the arm')
    if pod['metadata'].get('deletionTimestamp'):
        raise RuntimeError('Tenant is terminating')
    for status in pod.get('status', {}).get('containerStatuses', []):
        if status.get('restartCount', 0) or status.get('state', {}).get('terminated'):
            raise RuntimeError('Container restarted or terminated')
    return any(c.get('type') == 'Ready' and c.get('status') == 'True'
               for c in pod.get('status', {}).get('conditions', []))


def wait_ready(role, monitor):
    deadline = time.monotonic() + 900
    while time.monotonic() < deadline:
        if monitor.abort.is_set():
            raise RuntimeError(monitor.reason)
        pods = get_pods(role, monitor)
        if len(pods) > 1:
            raise RuntimeError('Multiple pods found for ' + role)
        for pod in pods:
            # Pending pods may not yet have a node assignment.
            if pod['spec'].get('nodeName') and validate_pod(pod):
                return pod
        time.sleep(2)
    raise RuntimeError('Readiness timed out for ' + role)


def current_pss(pods, out, suffix):
    measurements = {role: P.pod_pss(pod['metadata']['uid']) for role, pod in pods.items()}
    P.write(out / ('pss-' + suffix + '.json'), measurements)
    if not all(value['complete'] for value in measurements.values()):
        raise RuntimeError('Pod process PSS could not be completely read')
    return measurements


def mps_check(arm, pss, out, suffix):
    ok, detail = H.assert_mps_matches_arm(arm, list(pss))
    detail['workers_by_role'] = {}
    for role, measurement in pss.items():
        workers = [row['pid'] for row in measurement['processes'] if 'Worker' in row['name']]
        detail['workers_by_role'][role] = workers
        if len(workers) != 2:
            ok = False
        if arm == 'hami-mps' and not set(workers).issubset(detail['mps_client_pids']):
            ok = False
    detail['ok'] = ok
    P.write(out / ('mps-' + suffix + '.json'), detail)
    if not ok:
        raise RuntimeError('Expected two GPU workers per tenant with the correct MPS client state')


def collect_logs(pods, out, monitor=None):
    for role, pod in pods.items():
        text = P.kube(['logs', pod['metadata']['name'], '-c', 'main', '--tail=4000'],
                      monitor=monitor)
        (out / (role + '-server.log')).write_text(text, encoding='utf-8')
        if 'Overriding VLLM_WORKER_MULTIPROC_METHOD' in text and 'spawn' in text:
            raise RuntimeError('vLLM overrode fork; preserve this result as a different configuration')
        tp, pp = getattr(tiny, 'PARALLELISM', {}).get(role, (2, 1))
        if not all(term in text for term in ('tensor_parallel_size=%d' % tp, 'pipeline_parallel_size=%d' % pp,
                                              'world_size=2 rank=0', 'world_size=2 rank=1')):
            raise RuntimeError('Startup log does not confirm TP%d/PP%d and both NCCL ranks for %s' % (tp, pp, role))


def cleanup_roles(roles, token):
    errors = []
    # Delete both owned controllers before waiting for GPU memory to disappear.
    for role in roles:
        try:
            raw = P.kube(['get', 'llminferenceservice', 'r2-' + role,
                          '--ignore-not-found', '-o', 'json'])
            obj = json.loads(raw) if raw.strip() else None
            if obj:
                if not P.own_object(obj, token):
                    raise RuntimeError('Refusing to delete unowned r2-' + role)
                P.kube(['delete', 'llminferenceservice', 'r2-' + role,
                        '--wait=true', '--timeout=120s'], timeout=140)
        except Exception as exc:
            errors.append(str(exc))
    if errors:
        raise RuntimeError('; '.join(errors))
    deadline = time.monotonic() + 180
    while any(get_pods(role) for role in roles):
        if time.monotonic() >= deadline:
            raise RuntimeError('Series pods remain after deletion')
        time.sleep(2)
    if not H.wait_gpu_idle(timeout=120):
        raise RuntimeError('GPUs did not become idle after teardown')


def check_placement(roles, out, monitor):
    placement = H.placement(roles)
    gpu = H.gpu_used_mib()
    P.write(out / 'placement.json', dict(roles=placement, gpu_used_mib=gpu))
    sets = [set(placement.get(role, [])) for role in roles]
    if any(len(devices) != 2 for devices in sets) or any(devices != sets[0] for devices in sets):
        raise RuntimeError('Each tenant must occupy the same two GPU UUIDs')
    if len([used for used in gpu.values() if used > 500]) != 2:
        raise RuntimeError('Expected physical residency on both GPUs')
    monitor.sample()
    if monitor.abort.is_set():
        raise RuntimeError(monitor.reason)


def wait_with_guard(seconds, monitor):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if monitor.abort.wait(min(1, max(0, deadline - time.monotonic()))):
            raise RuntimeError(monitor.reason)


def run_arm(root, name, arm, roles, trace, args, solos=None, identity=None):
    out = root / name
    out.mkdir()
    (out / 'rendered').mkdir()
    token, attempted, pods = uuid.uuid4().hex, [], {}
    result = dict(ok=False, arm=arm, roles=list(roles), model=tiny.MODEL,
                  model_revision=tiny.REVISION,
                  profile=getattr(tiny, 'PROFILE', 'lowmem-fork'), calibration=False)
    monitor, gpu = None, None
    try:
        if any(get_pods(role) for role in ROLES) or not H.wait_gpu_idle(timeout=30):
            raise RuntimeError('Existing tenants or busy GPUs; refusing a new arm')
        idle, samples = P.stable_idle()
        result['idle_available_mib'] = idle
        P.write(out / 'idle-hostmem.json', dict(available=idle, samples=samples))
        if solos is not None:
            gate = admission(solos, idle, args.host_budget_mib)
            P.write(out / 'admission.json', gate)
            if not gate['ok']:
                raise RuntimeError('Pair needs %.0f MiB including reserve; only %.0f MiB permitted by available RAM/budget' %
                                   (gate['required_mib'], gate['effective_available_mib']))
        ok, detail = H.set_mps_daemon(arm == 'hami-mps')
        if not ok:
            raise RuntimeError('Could not set MPS state: ' + str(detail))
        # MPS itself costs host memory; recheck admission after its state changes.
        if solos is not None:
            gate = admission(solos, H.host_mem_mib().get('available', 0), args.host_budget_mib)
            P.write(out / 'admission-after-mps.json', gate)
            if not gate['ok']:
                raise RuntimeError('Pair no longer passes memory admission after changing MPS state')
        floor = max(RESERVE, idle - args.host_budget_mib + RESERVE) if args.host_budget_mib else RESERVE
        P.write(out / 'memory-budget.json', dict(idle_available_mib=idle,
                budget_mib=args.host_budget_mib, stop_below_available_mib=floor,
                reserve_mib=RESERVE, enforcement='sampled host guard, not a hard cgroup cap'))
        monitor = P.Monitor(out, reserve=floor)
        monitor.start()  # Includes first storage/startup allocations.
        gpu = H.GpuSampler(str(out / 'gpu.csv'))
        gpu.start()
        for role in roles:
            path = out / 'rendered' / (role + '.yaml')
            path.write_text(render(role, arm, token), encoding='utf-8')
            attempted.append(role)
            P.kube(['create', '-f', str(path)], monitor=monitor)
            # Stagger initialization to avoid overlapping deserialization peaks;
            # BOTH instances remain resident and replay concurrently afterward.
            pods[role] = wait_ready(role, monitor)
            P.write(out / (role + '-pod-ready.json'), pods[role])
            version_text = P.kube(['exec', pods[role]['metadata']['name'], '-c', 'main', '--',
                                  'python', '-c', 'import json,importlib.metadata as m; '
                                  'print(json.dumps({k:m.version(k) for k in ("vllm","torch")}))'],
                                 monitor=monitor)
            versions = json.loads(version_text.strip().splitlines()[-1])
            image = next(s['imageID'] for s in pods[role]['status']['containerStatuses']
                         if s['name'] == 'main')
            runtime = dict(versions=versions, image_id=image)
            P.write(out / (role + '-runtime.json'), runtime)
            if versions['vllm'] != '0.15.1' or not image:
                raise RuntimeError('This series was prepared for vLLM 0.15.1; runtime differs')
            if identity is None:
                identity = runtime
            elif runtime != identity:
                raise RuntimeError('Image or installed versions changed between comparable arms')
        for role in roles:
            current = get_pods(role, monitor)
            if len(current) != 1 or not validate_pod(current[0], pods[role]['metadata']['uid']):
                raise RuntimeError('Tenants are not simultaneously Ready')
        check_placement(roles, out, monitor)
        collect_logs(pods, out, monitor)
        pss = current_pss(pods, out, 'ready')
        mps_check(arm, pss, out, 'ready')
        endpoints = {role: 'http://' + pod['status']['podIP'] + ':8000' for role, pod in pods.items()}
        for role, url in endpoints.items():
            if H.served_model(url) != tiny.MODEL:
                raise RuntimeError('Unexpected served model for ' + role)
        local_trace = selected_trace(trace, roles)
        P.write(out / 'trace.json', local_trace)
        log('%s: replay %.1f req/s per tenant for %ds' %
            (name, args.rate_per_tenant, args.seconds))
        cmd = [sys.executable, str(HERE.parent / 'workload' / 'replay.py'),
               '--trace', str(out / 'trace.json'), '--arm', arm, '--out-dir', str(out),
               '--max-inflight', '128', '--timeout', '120',
               '--note', tiny.MODEL + ' ' + getattr(tiny, 'DESIGN', 'TP2+TP2') + ' fixed synthetic Poisson load; not capacity calibration']
        for role, url in endpoints.items():
            cmd += ['--endpoint', role + '=' + url, '--model', role + '=' + tiny.MODEL]
        try:
            text = P.command(cmd, timeout=args.seconds + 420, monitor=monitor)
        except BaseException as exc:
            (out / 'replay.log').write_text(str(exc), encoding='utf-8')
            raise
        (out / 'replay.log').write_text(text, encoding='utf-8')
        quality = replay_quality(out, roles)
        P.write(out / 'quality.json', quality)
        summary = json.loads((out / 'replay.summary.json').read_text())
        write_intervals(out, roles)
        if not quality['ok']:
            raise RuntimeError('Replay failed acceptance: ' + str(quality['issues']))
        if len(roles) > 1 and not summary.get('contended'):
            raise RuntimeError('No overlapping service intervals were measured')
        for role in roles:
            current = get_pods(role, monitor)
            if len(current) != 1 or not validate_pod(current[0], pods[role]['metadata']['uid']):
                raise RuntimeError('A tenant changed, restarted or stopped during replay')
        pss = current_pss(pods, out, 'after-replay')
        mps_check(arm, pss, out, 'after-replay')
        collect_logs(pods, out, monitor)
        monitor.sample()
        if monitor.abort.is_set():
            raise RuntimeError(monitor.reason)
        window = dict(replay_start_epoch=summary['epoch_start'], replay_end_epoch=summary['epoch_end'],
                      url='http://127.0.0.1:3000/d/r2-gpu-sharing?from=%d&to=%d&var-node=%s' %
                      (1000 * summary['epoch_start'], 1000 * summary['epoch_end'], NODE),
                      note='Historical loaded interval. The screenshot hold has no generated load.')
        without_prometheus = getattr(args, 'without_prometheus', False)
        result['prometheus_enabled'] = not without_prometheus
        if without_prometheus:
            window.update(url=None, available=False,
                          note='Prometheus explicitly disabled; no server metrics or fresh dashboard data.')
            P.write(out / 'prom-skipped.json', dict(reason=window['note']))
        else:
            try:
                H.prom_collect(str(out / 'prom.json'), summary['epoch_start'], summary['epoch_end'])
            except Exception as exc:
                P.write(out / 'prom-warning.json', dict(error=str(exc)))
        P.write(out / 'dashboard-window.json', window)
        result.update(runtime=identity, peak_host_delta_mib=idle - monitor.minimum,
                      minimum_available_mib=monitor.minimum,
                      pss_after_replay_mib={role: value['pss_mib'] for role, value in pss.items()},
                      requests_ok=summary['requests_ok'], requests_total=summary['requests_total'],
                      aggregate=summary['aggregate'], per_service=summary['per_service'])
        if without_prometheus:
            log('%s: replay passed. Prometheus disabled; client CSV/intervals and host/GPU samples saved.' % name)
        else:
            log('%s: replay passed. Dashboard: %s' % (name, window['url']))
            log('Screenshot hold %ds; select the saved replay interval. Pods remain resident.' % args.hold_seconds)
            wait_with_guard(args.hold_seconds, monitor)
        result['ok'] = True
    except BaseException as exc:
        result['error'] = str(exc) or type(exc).__name__
        raise
    finally:
        if monitor:
            monitor.halt.set()
            monitor.join(timeout=5)
            result['minimum_available_mib'] = monitor.minimum
            result['memory_guard'] = monitor.reason
            if monitor.minimum is not None:
                result['peak_host_delta_mib'] = result['idle_available_mib'] - monitor.minimum
        if gpu:
            gpu.stop()
        # On low-memory failure, prioritize deletion over expensive diagnostics.
        if not monitor or not monitor.abort.is_set():
            for role in attempted:
                try:
                    for pod in get_pods(role):
                        text = P.kube(['logs', pod['metadata']['name'], '--all-containers=true', '--tail=4000'])
                        (out / (role + '-server.log')).write_text(text, encoding='utf-8')
                    text = P.kube(['get', 'events', '--sort-by=.lastTimestamp'])
                    (out / 'cluster-events.txt').write_text(text, encoding='utf-8')
                except Exception as exc:
                    result['diagnostic_error'] = str(exc)
        try:
            if attempted:
                cleanup_roles(attempted, token)
        except Exception as exc:
            result.update(ok=False, cleanup_error=str(exc))
            raise
        finally:
            P.write(out / 'arm.json', result)
    return result


def quantile(values, fraction):
    if not values:
        return ''
    values = sorted(values)
    pos = (len(values) - 1) * fraction
    lower, upper = math.floor(pos), math.ceil(pos)
    return values[lower] + (values[upper] - values[lower]) * (pos - lower)


def write_comparison(root, results):
    comparisons = {}
    for arm in ('hami', 'hami-mps'):
        comparisons[arm] = {}
        for role in ROLES:
            solo = results['dedicated__' + role]['per_service'][role]
            shared = results[arm]['per_service'][role]
            comparisons[arm][role] = {
                'tokens_per_second_ratio_to_own_solo': shared['tokens_per_second'] / solo['tokens_per_second'],
                'latency_p95_ratio_to_own_solo': shared['latency_e2e']['p95'] / solo['latency_e2e']['p95'],
                'latency_p99_ratio_to_own_solo': shared['latency_e2e']['p99'] / solo['latency_e2e']['p99'],
            }
    P.write(root / 'comparison.json', dict(
        model=tiny.MODEL, controlled_load=True, measured_capacity=False,
        per_tenant=comparisons,
        mps_to_hami_tokens_per_second_ratio=results['hami-mps']['aggregate']['tokens_per_second'] /
                                              results['hami']['aggregate']['tokens_per_second'],
        note='Within-model, same-profile ratios at the same offered load. Not ratios to other models/profiles or proof of a saturation threshold.'))


def write_intervals(out, roles, width=5):
    """Same time bins for arrivals, completion latency and achieved throughput."""
    trace = json.loads((out / 'trace.json').read_text())
    with (out / 'replay.requests.csv').open(newline='') as handle:
        rows = [r for r in csv.DictReader(handle) if r['warmup'].lower() not in ('true', '1')]
    end = max([trace['params']['duration']] + [float(r['t_last']) for r in rows if r['t_last']])
    fields = ['service', 'start_s', 'end_s', 'scheduled_arrivals', 'sent_requests',
              'completed_ok', 'completed_failed', 'latency_p95_s', 'latency_p99_s',
              'completion_requests_per_s', 'output_tokens_per_s']
    with (out / 'intervals.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for role in roles:
            own = [r for r in rows if r['service'] == role]
            scheduled = [r for r in trace['requests'] if r['service'] == role and not r['warmup']]
            for i in range(math.floor(end / width) + 1):
                start, stop = i * width, (i + 1) * width
                completed = [r for r in own if r['t_last'] and start <= float(r['t_last']) < stop]
                ok = [r for r in completed if r['ok'].lower() in ('true', '1')]
                latency = [float(r['t_last']) - float(r['t_sent']) for r in ok]
                writer.writerow(dict(service=role, start_s=start, end_s=stop,
                                     scheduled_arrivals=sum(start <= r['t'] < stop for r in scheduled),
                                     sent_requests=sum(start <= float(r['t_sent']) < stop for r in own),
                                     completed_ok=len(ok), completed_failed=len(completed) - len(ok),
                                     latency_p95_s=quantile(latency, .95), latency_p99_s=quantile(latency, .99),
                                     completion_requests_per_s=len(ok) / width,
                                     output_tokens_per_s=sum(int(r['output_tokens']) for r in ok) / width))


def main(argv=None, model=None):
    global tiny, ROLES
    if model is not None:
        tiny = model
        ROLES = getattr(model, 'ROLES', ('tp-a', 'tp-b'))
    parser = argparse.ArgumentParser(description=getattr(tiny, 'DESCRIPTION', __doc__))
    parser.add_argument('--rate-per-tenant', type=float, default=getattr(tiny, 'DEFAULT_RATE', 2.0))
    parser.add_argument('--seconds', type=int, default=300)
    parser.add_argument('--output-tokens', type=int, default=getattr(tiny, 'DEFAULT_OUTPUT', 128))
    parser.add_argument('--hold-seconds', type=int, default=300)
    parser.add_argument('--without-prometheus', action='store_true',
                        help='Skip Prometheus checks/collection and screenshot holds; record missing server metrics')
    parser.add_argument('--host-budget-mib', type=int, default=None,
                        help='Total host envelope including 1024 MiB reserve; sampled guard, not a hard cap')
    parser.add_argument('--seed', type=int, default=20260924)
    parser.add_argument('--render-only', type=Path, help='Write reviewable YAML and trace; no cluster access')
    args = parser.parse_args(argv)
    if not math.isfinite(args.rate_per_tenant) or not 0 < args.rate_per_tenant <= 10:
        parser.error('rate-per-tenant must be >0 and <=10 for this bounded series')
    if not 60 <= args.seconds <= 900 or not 1 <= args.output_tokens <= 256 or not 0 <= args.hold_seconds <= 3600:
        parser.error('seconds 60..900, output-tokens 1..256, hold-seconds 0..3600')
    if args.host_budget_mib is not None and args.host_budget_mib <= RESERVE:
        parser.error('host-budget-mib must exceed the 1024 MiB reserve')
    if args.without_prometheus:
        args.hold_seconds = 0
    trace = make_trace(args.rate_per_tenant, args.seconds, args.output_tokens, args.seed)
    if args.render_only:
        args.render_only.mkdir(parents=True, exist_ok=False)
        for arm in ('dedicated', 'hami', 'hami-mps'):
            for role in ROLES:
                (args.render_only / (arm + '-' + role + '.yaml')).write_text(render(role, arm), encoding='utf-8')
        P.write(args.render_only / 'trace.json', trace)
        return
    if sys.platform != 'linux' or os.geteuid() != 0:
        parser.error('Run on Box1 as Linux root')
    import fcntl
    os.chdir(HERE)
    with open('.split-run.lock', 'w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('Another split stage is running')
        ensure_fd_limit(128)
        objects = json.loads(P.kube(['get', 'llminferenceservice', '-o', 'json']))
        names = {'r2-' + role for role in (*ROLES, 'tp', 'pp', 'embed-small')}
        if any(item['metadata']['name'] in names for item in objects['items']) or H.all_r2_pods():
            raise SystemExit('Existing inference workload found; inspect before starting')
        if not H.wait_gpu_idle(timeout=30):
            raise SystemExit('GPUs are not idle')
        problems = H.preconditions(require_prometheus=not args.without_prometheus)
        if problems:
            raise SystemExit('Preconditions failed: ' + '; '.join(problems))
        def interrupt(signum, frame):
            raise KeyboardInterrupt('Series interrupted')
        signal.signal(signal.SIGTERM, interrupt)
        root = HERE / 'results' / (getattr(tiny, 'RESULT_PREFIX', 'S-tiny-tp2-') +
                                 time.strftime('%Y%m%d-%H%M%S', time.gmtime()))
        root.mkdir(parents=True, exist_ok=False)
        log('SERIES DIRECTORY: ' + str(root))
        (HERE / 'logs').mkdir(exist_ok=True)
        (HERE / 'logs' / getattr(tiny, 'LATEST_FILE', 'tiny-series-latest.txt')).write_text(str(root), encoding='utf-8')
        P.write(root / 'trace.json', trace)
        P.write(root / 'settings.json', {**vars(args), 'render_only': None,
                'model': tiny.MODEL, 'revision': tiny.REVISION, 'node': NODE,
                'reserve_mib': RESERVE, 'capacity_calibration': False,
                'files_sha256': {str(p.relative_to(HERE.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in [Path(__file__), HERE / '57-tiny-tp-probe.py',
                                            HERE / '56-serving-probe.py', HERE / '50-render-split.py',
                                            HERE.parent / 'workload/replay.py',
                                            Path(getattr(tiny, 'CANDIDATE_FILE', HERE / '57-tiny-tp-probe.py'))]}})
        results, identity = {}, None
        status = dict(ok=False, model=tiny.MODEL, results=results)
        try:
            provenance = tiny.prepare_model()
            if provenance is not None:
                P.write(root / 'model-provenance.json', provenance)
            if getattr(tiny, 'SCREEN_FIRST', False):
                # One light-load rejection screen, not permission to skip full
                # same-profile baselines. A failing screen ends this attempt.
                screen_args = argparse.Namespace(rate_per_tenant=.25, seconds=60, hold_seconds=0,
                                                 host_budget_mib=args.host_budget_mib,
                                                 without_prometheus=args.without_prometheus)
                screen_trace = make_trace(.25, 60, 64, args.seed)
                screens = {}
                for role in ROLES:
                    log('START short memory screen: ' + role)
                    screens[role] = run_arm(root, 'screen__' + role, 'dedicated', [role],
                                           screen_trace, screen_args, identity=identity)
                    results['screen__' + role] = screens[role]
                    identity = screens[role]['runtime']
                available, _ = P.stable_idle()
                screen_gate = admission(screens, available, args.host_budget_mib)
                screen_gate['method'] = 'sum of separate light-load role deltas plus reserve; screening only'
                P.write(root / 'screen-admission.json', screen_gate)
                if not screen_gate['ok']:
                    raise RuntimeError('Memory screen failed: estimated %.0f MiB needed, %.0f permitted; '
                                       'full series was not started' % (screen_gate['required_mib'],
                                                                      screen_gate['effective_available_mib']))
            for role in ROLES:
                name = 'dedicated__' + role
                log('START ' + name)
                results[name] = run_arm(root, name, 'dedicated', [role], trace, args, identity=identity)
                identity = results[name]['runtime']
            solos = {role: results['dedicated__' + role] for role in ROLES}
            P.write(root / 'baselines.json', dict(solo=solos, calibrated_capacity=False,
                    trace_id=trace['trace_id'], offered_rate_per_tenant=args.rate_per_tenant))
            for arm in ('hami', 'hami-mps'):
                log('START ' + arm)
                results[arm] = run_arm(root, arm, arm, ROLES, trace, args, solos, identity)
            write_comparison(root, results)
            status['ok'] = True
        except BaseException as exc:
            status['error'] = str(exc) or type(exc).__name__
            log('SERIES STOPPED: ' + status['error'])
            raise
        finally:
            # Runs deliberately finish with the benchmark MPS daemon disabled.
            # Do not change MPS while failed cleanup left owned clients resident.
            try:
                if not any(get_pods(role) for role in ROLES):
                    ok, detail = H.set_mps_daemon(False)
                    status['mps_restored_off'] = ok
                    if not ok:
                        status.update(ok=False, mps_restore_error=str(detail))
                else:
                    status.update(ok=False, mps_restore_error='Tenants remain; inspect cleanup failure')
            except Exception as exc:
                status.update(ok=False, mps_restore_error=str(exc))
            finally:
                P.write(root / 'series.json', status)
        if not status['ok']:
            raise SystemExit('Series cleanup did not complete; inspect series.json')
        log('SERIES COMPLETE: ' + str(root))


if __name__ == '__main__':
    main()
