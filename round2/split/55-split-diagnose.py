#!/usr/bin/env python3
"""Read-only Box1 evidence collection. Run as root on the benchmark node.

No deployment, daemon, host, or calibration settings are changed. Redirect
stdout to a log if desired. --run should identify the actual solo directory.
"""
import argparse
import json
import subprocess
from pathlib import Path
from split_checks import replay_quality


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', required=True, type=Path)
    a = ap.parse_args()
    for role in ('tp', 'pp'):
        directory = a.run / ('dedicated__' + role)
        try:
            print(json.dumps({'role': role, 'quality': replay_quality(directory, [role])}, indent=2))
        except (OSError, ValueError, KeyError) as e:
            print(json.dumps({'role': role, 'audit_error': str(e)}))
    for filename in ('calibration.json',):
        p = a.run / filename
        if p.exists():
            print(str(p))
            print(p.read_text(encoding='utf-8'))
    # PSS apportions shared resident pages; RSS can count them repeatedly.
    # Print only process names, never command-line arguments or environment.
    processes = []
    unreadable = 0
    for p in Path('/proc').glob('[0-9]*'):
        try:
            mem = {}
            for line in (p / 'smaps_rollup').read_text().splitlines():
                fields = line.split()
                if fields[0] in ('Pss:', 'Rss:'):
                    mem[fields[0][:-1]] = round(int(fields[1]) / 1024, 1)
            processes.append({'pid': int(p.name), 'name': (p / 'comm').read_text().strip(),
                              'pss_mib': mem.get('Pss', 0), 'rss_mib': mem.get('Rss', 0)})
        except (OSError, ValueError, IndexError):
            unreadable += 1
    print(json.dumps({'top_processes_by_pss': sorted(processes, key=lambda x: x['pss_mib'],
                                                   reverse=True)[:25],
                      'unreadable_or_exited_processes': unreadable}, indent=2))
    for cmd in (
        ['free', '-m'],
        ['kubectl', 'top', 'pods', '-A', '--sort-by=memory'],
        ['kubectl', '-n', 'llm-demo', 'get', 'llminferenceservice,pods', '-o', 'wide'],
        ['kubectl', 'describe', 'node', 'mr-gholami'],
    ):
        print('\n$ ' + ' '.join(cmd), flush=True)
        try:
            result = subprocess.run(cmd, timeout=40, check=False)
            print('exit_code=%s' % result.returncode, flush=True)
        except (OSError, subprocess.TimeoutExpired) as e:
            print(str(e), flush=True)


if __name__ == '__main__':
    main()
