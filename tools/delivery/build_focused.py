"""Assemble the local delivery from public-repository sources; no cluster access."""
from pathlib import Path
import hashlib
import json
import shutil
import build_pools

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / 'deliverable'


def main():
    build_pools.main()
    rows = json.loads(Path(__file__).with_name('sources.json').read_text(encoding='utf-8'))
    for item in rows:
        source = REPO / item['repo_source']
        assert source.resolve().is_relative_to(REPO), source
        assert hashlib.sha256(source.read_bytes()).hexdigest() == item['sha256'], 'Source changed; review provenance: ' + str(source)
        destination = OUT / item['file']
        assert destination.resolve().is_relative_to(OUT.resolve()), destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    for name in ('OPERATIONS_FA.md', 'README.md', 'CONFIGURATIONS_FA.md'):
        shutil.copyfile(Path(__file__).with_name(name), OUT / name)
    derived = json.loads((REPO / 'deployment/pool-provenance.json').read_text())
    for item in derived:
        dest = OUT / item['file']
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / 'deployment' / item['file'], dest)
    shutil.copyfile(REPO / 'deployment/pool-provenance.json', OUT / 'pool-provenance.json')
    shutil.copyfile(REPO / 'deployment/hami-values.yaml', OUT / 'infrastructure/hami-values.yaml')
    for name in ('hami-scrape-fixes.yaml', 'hami-webui-values.yaml'):
        shutil.copyfile(REPO / 'deployment/monitoring' / name, OUT / 'monitoring' / name)
    (OUT / 'provenance.json').write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding='utf-8')
    print('Copied', len(rows), 'source files without modifying their content')


if __name__ == '__main__':
    main()
