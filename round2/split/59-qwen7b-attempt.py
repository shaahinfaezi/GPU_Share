#!/usr/bin/env python3
"""One bounded vLLM-parameter attempt with the existing Qwen 7B AWQ model.

No model/image download. Separate 60-second TP2 and PP2 memory screens run first;
if it clears screening, fresh solos and the separate HAMi/MPS series follow.
All shared arms still require full-baseline admission. No automatic retries.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('qwen_series_runner', HERE / '58-tiny-series.py')
series = importlib.util.module_from_spec(spec)
spec.loader.exec_module(series)
base = series.tiny.base
replace_once = series.tiny.replace_once
MODEL = 'Qwen/Qwen2.5-7B-Instruct-AWQ'
REVISION = 'existing-local-checkpoint-see-model-provenance'
PVC, SUBPATH = 'r2-models-pvc', 'models/llm'
PROFILE = 'qwen7b-bounded-fork'
GPU_QUOTA_MIB = 5632
KV_BYTES = 256 * 1024 * 1024


def render_tenant(role, arm, percent='100'):
    if role not in ('tp', 'pp') or arm not in ('dedicated', 'hami', 'hami-mps'):
        raise ValueError('Unknown Qwen attempt role or arm')
    body = base.render(role, arm, percent, 'lowmem-fork')
    body = replace_once(body, '  name: r2-' + role + '\n', '  name: r2-' + role + '\n')
    body = replace_once(body, '    r2-role: ' + role + '\n', '    r2-role: ' + role + '\n')
    body = replace_once(body, '    name: ' + base.ROLES[role]['model'], '    name: ' + MODEL)
    body = replace_once(body, '    uri: ' + base.ROLES[role]['uri'],
                        '    uri: pvc://' + PVC + '/' + SUBPATH)
    # Replace the explicit command's args as a parsed list, keeping the original
    # Qwen/Pythia renderer and its default behavior intact.
    lines = body.splitlines()
    index = next(i for i, line in enumerate(lines) if line.startswith('        args: '))
    args = json.loads(lines[index].split('args: ', 1)[1])
    changes = {'--served-model-name': MODEL, '--max-model-len': '1024',
               '--max-num-seqs': '2', '--max-num-batched-tokens': '256'}
    for flag, value in changes.items():
        if args.count(flag) != 1:
            raise ValueError('Missing or repeated argument: ' + flag)
        args[args.index(flag) + 1] = value
    args += ['--kv-cache-memory-bytes', str(KV_BYTES), '--no-enable-prefix-caching',
             '--safetensors-load-strategy', 'lazy', '--load-format', 'safetensors',
             '--quantization', 'awq_marlin', '--dtype', 'half']
    lines[index] = '        args: ' + json.dumps(args)
    # Keep the legacy environment argument copy consistent with the actual CLI.
    env_index = next(i for i, line in enumerate(lines) if 'name: VLLM_ADDITIONAL_ARGS' in line)
    lines[env_index + 1] = '            value: ' + json.dumps(' '.join(args[args.index('--tensor-parallel-size'):]))
    old = '            nvidia.com/gpumem: "4096"'
    if sum(line == old for line in lines) != 2:
        raise ValueError('Expected matching GPU-memory requests and limits')
    lines = ['            nvidia.com/gpumem: "%d"' % GPU_QUOTA_MIB if line == old else line for line in lines]
    return '\n'.join(lines) + '\n'


def validate_config(config):
    expected = dict(model_type='qwen2', hidden_size=3584, num_hidden_layers=28,
                    num_attention_heads=28, num_key_value_heads=4)
    if any(config.get(key) != value for key, value in expected.items()):
        raise RuntimeError('Existing models/llm is not the expected Qwen2.5-7B architecture')
    quant = config.get('quantization_config', {})
    if quant.get('quant_method') != 'awq' or quant.get('bits') != 4:
        raise RuntimeError('Expected the existing 4-bit AWQ checkpoint, not BF16/FP16 weights')


def prepare_model():
    pvc = json.loads(series.P.kube(['get', 'pvc', PVC, '-o', 'json']))
    if pvc.get('status', {}).get('phase') != 'Bound':
        raise RuntimeError('Existing r2-models-pvc must be Bound')
    pv = json.loads(series.P.kube(['get', 'pv', pvc['spec']['volumeName'], '-o', 'json']))
    location = pv['spec'].get('local') or pv['spec'].get('hostPath') or {}
    root = Path(location.get('path', '/missing-model-volume')).resolve()
    path = (root / SUBPATH).resolve()
    if not root.is_relative_to(Path('/data2')) or not path.is_relative_to(root) or not path.is_dir():
        raise RuntimeError('Expected existing models/llm on the model volume under /data2; nothing was downloaded')
    config_path = path / 'config.json'
    config = json.loads(config_path.read_text())
    validate_config(config)
    marker = path / '.source-repo'
    if marker.exists() and marker.read_text().strip() != MODEL:
        raise RuntimeError('Existing model source marker disagrees with Qwen 7B AWQ')
    index = path / 'model.safetensors.index.json'
    if index.exists():
        names = sorted(set(json.loads(index.read_text())['weight_map'].values()))
    else:
        names = ['model.safetensors']
    weights = []
    for name in names:
        weight = path / name
        if not weight.resolve().is_relative_to(path) or not weight.is_file() or weight.stat().st_size <= 0:
            raise RuntimeError('Missing or out-of-directory model shard: ' + name)
        weights.append(dict(file=name, bytes=weight.stat().st_size, mtime_ns=weight.stat().st_mtime_ns))
    if not (path / 'tokenizer_config.json').is_file() or not (path / 'tokenizer.json').is_file():
        raise RuntimeError('Existing Qwen tokenizer files are missing')
    print('Using existing Qwen 7B AWQ at ' + str(path), flush=True)
    return dict(model=MODEL, path=str(path), revision=REVISION,
                config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
                index_sha256=hashlib.sha256(index.read_bytes()).hexdigest() if index.exists() else None,
                weights=weights, profile=PROFILE,
                note='Checkpoint files reused in place; shard sizes/timestamps are provenance, not content hashes. '
                     'KV bytes limit GPU cache, not total host RAM. Host saving remains unmeasured.')


def model_adapter():
    return SimpleNamespace(MODEL=MODEL, REVISION=REVISION, PROFILE=PROFILE,
                           DESCRIPTION=__doc__, ROLES=('tp', 'pp'),
                           PARALLELISM={'tp': (2, 1), 'pp': (1, 2)},
                           DESIGN='TP2+PP2',
                           CONTEXT=1024, TRACE_KIND='chat', DEFAULT_RATE=.5, DEFAULT_OUTPUT=64,
                           SERIES_LABEL='qwen7b-tp2-pp2', RESULT_PREFIX='S-qwen7b-attempt-',
                           LATEST_FILE='qwen7b-attempt-latest.txt', SCREEN_FIRST=True,
                           CANDIDATE_FILE=__file__, render_tenant=render_tenant,
                           replace_once=replace_once, prepare_model=prepare_model)


if __name__ == '__main__':
    series.main(model=model_adapter())
