"""Split-specific admission and evidence checks; importing never touches a cluster."""
import json
import csv
import hashlib
import math
import os
from pathlib import Path


def config_identity():
    profile = os.environ.get('SPLIT_PROFILE', 'original')
    renderer = Path(__file__).with_name('50-render-split.py').read_bytes()
    context = profile + ':' + os.environ.get('R2_NS', 'llm-demo')
    return {'profile': profile,
            'sha256': hashlib.sha256(renderer + context.encode()).hexdigest()}


def remap_trace(doc):
    """Update metadata and digest as well as payload when renaming services."""
    for row in doc['requests']:
        row['service'] = {'llm2a': 'tp', 'llm2b': 'pp'}.get(row['service'], row['service'])
    doc['services'] = sorted({r['service'] for r in doc['requests']})
    doc['trace_id'] = hashlib.sha256(json.dumps(
        doc['requests'], ensure_ascii=False, sort_keys=True).encode('utf-8')).hexdigest()[:16]
    return doc


def ensure_fd_limit(workers=4096):
    """Linux children inherit this limit; never alter the host-wide limit."""
    import resource
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    required = 2 * workers + 256
    if hard != resource.RLIM_INFINITY and hard < required:
        raise RuntimeError('Open-file hard limit %s is below %s; raise it before replay'
                           % (hard, required))
    if soft != resource.RLIM_INFINITY and soft < required:
        resource.setrlimit(resource.RLIMIT_NOFILE, (required, hard))


def replay_quality(directory, roles, max_client_delay_p99=0.5):
    """Audit all scheduled live rows, including errors and missing completions.

    Zero errors is a strict calibration/comparison acceptance rule, not a claim
    that failed requests have no analytical value. Keep the raw artifacts.
    """
    directory = Path(directory)
    doc = json.loads((directory / 'trace.json').read_text(encoding='utf-8'))
    expected = {r: sum(x['service'] == r and not x['warmup']
                       for x in doc['requests']) for r in roles}
    with (directory / 'replay.requests.csv').open(encoding='utf-8', newline='') as fh:
        all_rows = list(csv.DictReader(fh))
    live = [x for x in all_rows if x['warmup'].lower() not in ('true', '1')]
    issues, detail = [], {}
    if any(x['ok'].lower() not in ('true', '1') for x in all_rows
           if x['warmup'].lower() in ('true', '1')):
        issues.append('warm-up contains failed requests')
    for role in roles:
        rows = [x for x in live if x['service'] == role]
        failed = [x for x in rows if x['ok'].lower() not in ('true', '1')]
        delays = sorted(float(x['t_sent']) - float(x['t_sched']) for x in rows)
        p99 = delays[max(0, math.ceil(.99 * len(delays)) - 1)] if delays else None
        detail[role] = {'expected': expected[role], 'recorded': len(rows),
                        'failed': len(failed), 'client_delay_p99_s': p99,
                        'errors': sorted({x['error'] for x in failed})[:10]}
        if not expected[role] or len(rows) != expected[role]:
            issues.append('%s: missing or unexpected request records' % role)
        if failed:
            issues.append('%s: %s request errors' % (role, len(failed)))
        if p99 is None or not math.isfinite(p99) or p99 > max_client_delay_p99:
            issues.append('%s: client scheduling delay exceeds acceptance limit' % role)
    return {'ok': not issues, 'issues': issues, 'per_service': detail,
            'max_client_delay_p99_s': max_client_delay_p99}


def memory_budget(calibration, available_mib, headroom_mib=1024):
    deltas = [calibration.get('solo', {}).get(r, {}).get('peak_host_delta_mib')
              for r in ('tp', 'pp')]
    if all(isinstance(v, (int, float)) and math.isfinite(v) and v > 0 for v in deltas):
        required = sum(deltas) + headroom_mib
        return {'available_mib': available_mib, 'required_mib': required,
                'solo_peak_delta_mib': deltas, 'headroom_mib': headroom_mib,
                'ok': available_mib >= required,
                'method': 'sum of idle-to-peak host MemAvailable deltas plus headroom; '
                          'includes other host activity, not exact pod ownership'}
    measured = [calibration.get('solo', {}).get(r, {}).get('vllm_rss_mib')
                for r in ('tp', 'pp')]
    if any(not isinstance(v, (int, float)) or v <= 0 for v in measured):
        raise ValueError('Calibration needs measured vllm_rss_mib for TP and PP; '
                         'use the latest solo calibration and hostmem.json files.')
    required = sum(measured) + headroom_mib
    return {'available_mib': available_mib, 'required_mib': required,
            'solo_rss_mib': measured, 'headroom_mib': headroom_mib,
            'ok': available_mib >= required,
            'method': 'conservative sum of solo RSS plus host headroom'}


def same_two_devices(placement):
    tp, pp = set(placement.get('tp', [])), set(placement.get('pp', []))
    return len(tp) == 2 and tp == pp


def snapshot(harness, directory):
    """Include controller events even when no workload pod was created."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, args in (
        ('pods', ['get', 'pods', '-o', 'wide']),
        ('controllers', ['get', 'llminferenceservice,deploy,rs', '-o', 'yaml']),
        ('events', ['get', 'events', '--sort-by=.lastTimestamp']),
    ):
        rc, output = harness.kubectl(args + ['-n', harness.NS], timeout=30)
        (directory / (name + '.txt')).write_text(
            'exit_code=%s\n%s' % (rc, output), encoding='utf-8')
    for role in ('tp', 'pp'):
        for pod in harness.pods_for(role):
            name = pod['metadata']['name']
            for previous in (False, True):
                args = ['logs', '-n', harness.NS, name, '--all-containers=true', '--tail=150']
                if previous:
                    args.append('--previous')
                rc, output = harness.kubectl(args, timeout=30)
                suffix = '-previous' if previous else ''
                (directory / (name + suffix + '.log')).write_text(
                    'exit_code=%s\n%s' % (rc, output), encoding='utf-8')
