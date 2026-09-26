"""Read-only offline checks for the focused handoff; no deployment."""
from pathlib import Path
import ast
import hashlib
import json
import shutil
import subprocess
import re
import yaml
from jsonschema import Draft4Validator

REPO = Path(__file__).resolve().parents[2]
WORK = REPO.parent
PACKAGE = REPO / 'deliverable'


def main():
    rows = json.loads((PACKAGE / 'provenance.json').read_text(encoding='utf-8'))
    git_bash = Path('C:/Program Files/Git/bin/bash.exe')
    bash = str(git_bash) if git_bash.exists() else shutil.which('bash')
    if not bash:
        raise RuntimeError('Bash is required for shell syntax checks')
    crd = yaml.safe_load((PACKAGE / 'validation/kserve-0.17.0-llminferenceservice-crd.yaml').read_text(encoding='utf-8'))
    schema = next(v['schema']['openAPIV3Schema'] for v in crd['spec']['versions'] if v['name'] == 'v1alpha1')
    validator = Draft4Validator(schema)
    counts = {}
    for item in rows:
        p = PACKAGE / item['file']
        assert hashlib.sha256(p.read_bytes()).hexdigest() == item['sha256'], p
        assert p.read_bytes() == (REPO / item['repo_source']).read_bytes(), p
        if p.suffix == '.py':
            ast.parse(p.read_text(encoding='utf-8'))
        elif p.suffix == '.sh':
            subprocess.run([bash, '-n', str(p)], check=True, capture_output=True)
        elif p.suffix == '.yaml':
            for doc in yaml.safe_load_all(p.read_text(encoding='utf-8')):
                if doc is None:
                    continue
                assert all(k in doc for k in ('apiVersion', 'kind', 'metadata')), p
                counts[doc['kind']] = counts.get(doc['kind'], 0) + 1
                if doc['kind'] == 'LLMInferenceService':
                    validator.validate(doc)
                    main_container = doc['spec']['template']['containers'][0]
                    resource = main_container['resources']
                    assert resource['requests']['nvidia.com/gpu'] == resource['limits']['nvidia.com/gpu']
                    if '/split-tp2-pp2/' in p.as_posix():
                        role = p.stem
                        args = main_container['args']
                        for flag, value in (('--tensor-parallel-size', '2' if role == 'tp' else '1'),
                                            ('--pipeline-parallel-size', '1' if role == 'tp' else '2')):
                            assert args[args.index(flag) + 1] == value
                        assert int(resource['limits']['nvidia.com/gpu']) == 2
                        assert int(resource['limits']['nvidia.com/gpumem']) == 5632
                        env = {v['name']: v.get('value') for v in main_container['env']}
                        assert ('CUDA_MPS_PIPE_DIRECTORY' in env) == (p.parent.name == 'hami-mps')
    derived = json.loads((PACKAGE / 'pool-provenance.json').read_text())
    for item in derived:
        p = PACKAGE / item['file']
        assert hashlib.sha256(p.read_bytes()).hexdigest() == item['sha256'], p
        source = REPO / item['source']
        assert hashlib.sha256(source.read_bytes()).hexdigest() == item['source_sha256'], source
        assert p.read_bytes() == (REPO / 'deployment' / item['file']).read_bytes(), p
        doc = yaml.safe_load(p.read_text())
        if doc['kind'] == 'LLMInferenceService':
            validator.validate(doc)
            pool = p.relative_to(PACKAGE).parts[1]
            assert doc['spec']['template']['nodeSelector']['gpu-share.io/pool'] == pool
            assert doc['spec']['template']['runtimeClassName'] == 'nvidia'
            original = yaml.safe_load(source.read_text())
            actual = doc['spec']['template']['containers']
            expected = original['spec']['template']['containers']
            if pool == 'dedicated':
                for section in ('requests', 'limits'):
                    expected[0]['resources'][section]['nvidia.com/gpucores'] = '100'
            assert actual == expected, ('Runtime changed', p)
            uri = doc['spec']['model']['uri']
            if item['file'].startswith('box1/'):
                assert doc['spec']['template']['nodeSelector']['kubernetes.io/hostname'] == 'mr-gholami'
                assert uri == original['spec']['model']['uri']
            else:
                assert uri.startswith('pvc://pool-' + pool + '-models/')
        elif doc['kind'] == 'List':
            pv, pvc = doc['items']
            assert pv['metadata']['name'] == pvc['spec']['volumeName']
            assert pvc['spec']['selector']['matchLabels'] == pv['metadata']['labels']
        elif doc['kind'] == 'DaemonSet':
            assert doc['spec']['template']['spec']['nodeSelector'] == {
                'gpu-share.io/pool':'hami-mps', 'gpu-share.io/mps-enabled':'true'}
            assert doc['spec']['template']['spec']['runtimeClassName'] == 'nvidia'
        elif doc['kind'] == 'PodMonitor':
            assert doc['metadata']['name'] == 'r2-vllm-kserve'
            pattern = doc['spec']['podMetricsEndpoints'][0]['relabelings'][0]['regex']
            for name, expected in [('r2-llm','llm'), ('pool-hami-mps-tp','hami-mps-tp')]:
                assert re.fullmatch(pattern, name).group(1) == expected
    values = yaml.safe_load((PACKAGE / 'infrastructure/hami-values.yaml').read_text())
    assert values['scheduler']['defaultSchedulerPolicy']['gpuSchedulerPolicy'] == 'binpack'
    guide = (PACKAGE / 'OPERATIONS_FA.md').read_text(encoding='utf-8')
    blocks = re.findall(r'```bash\n(.*?)```', guide, re.S)
    for block in blocks:
        subprocess.run([bash, '-n'], input=block, text=True, check=True, capture_output=True)
        for target in re.findall(r'-f\s+([\w/.-]+\.yaml)', block):
            assert (PACKAGE / target).is_file(), target
    assert '.md' not in guide and '\u0627\u0633\u06a9\u0631\u06cc\u067e\u062a' not in guide
    receipt = dict(ok=True, byte_identical_sources=len(rows), derived_pool_files=len(derived),
                   guide_shell_blocks=len(blocks), objects=counts,
                   checks=['SHA256 and byte equality against originals', 'Python AST', 'bash -n',
                           'YAML parse', 'pinned KServe v1alpha1 schema', 'TP2/PP2 and MPS invariants',
                           'derived runtime equality and node-pool selection', 'guide bash syntax and YAML paths'],
                   unverified=['destination cluster admission/CEL', 'production traffic and startup',
                               'security and multi-node pool adaptations'])
    (PACKAGE / 'validation/receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
