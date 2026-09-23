"""Split-specific admission and evidence checks; importing never touches a cluster."""
import json
from pathlib import Path


def memory_budget(calibration, available_mib, headroom_mib=1024):
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
