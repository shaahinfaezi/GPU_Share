#!/usr/bin/env python3
"""Prepare pinned Pythia-70M and measure ONE TP2 instance; never starts sharing.

Run as root on Box1: python3 -u 57-tiny-tp-probe.py
Downloads only config/tokenizer/safetensors into the existing r2-models-pvc
host directory on /data2. Uses the existing local image; no image build/pull.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
MODEL = 'EleutherAI/pythia-70m'
REVISION = 'a39f36b100fe8a5377810d56c3f4789b9c53ac42'
PVC = 'r2-models-pvc'
SUBPATH = 'models/pythia-70m-split-probe'
FILES = ('config.json', 'model.safetensors', 'special_tokens_map.json',
         'tokenizer.json', 'tokenizer_config.json')


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


probe = load('tiny_serving_probe', '56-serving-probe.py')
base = load('tiny_base_renderer', '50-render-split.py')


def replace_once(body, old, new):
    if body.count(old) != 1:
        raise ValueError('Base renderer changed; expected exactly one: ' + old)
    return body.replace(old, new, 1)


class TinyRenderer:
    ROLES = {'tp': {'model': MODEL}}

    @staticmethod
    def render(role, arm, percent, profile):
        if (role, arm, profile) != ('tp', 'dedicated', 'lowmem-fork'):
            raise ValueError('Tiny probe supports only one dedicated TP2 lowmem-fork tenant')
        return render_tenant(role, arm, percent)


def render_tenant(role, arm, percent='100'):
    """Same TP2 serving settings for both named tenants and all three arms."""
    if role not in ('tp', 'tp-a', 'tp-b') or arm not in ('dedicated', 'hami', 'hami-mps'):
        raise ValueError('Unknown tiny-series role or arm')
    body = base.render('tp', arm, percent, 'lowmem-fork')
    body = replace_once(body, '  name: r2-tp\n', '  name: r2-' + role + '\n')
    body = replace_once(body, '    r2-role: tp\n', '    r2-role: ' + role + '\n')
    body = replace_once(body, '    name: ' + base.ROLES['tp']['model'], '    name: ' + MODEL)
    body = replace_once(body, json.dumps(base.ROLES['tp']['model']), json.dumps(MODEL))
    body = replace_once(body, '    uri: ' + base.ROLES['tp']['uri'],
                        '    uri: pvc://' + PVC + '/' + SUBPATH)
    body = replace_once(body, '--max-model-len 8192', '--max-model-len 2048')
    return replace_once(body, '"--max-model-len", "8192"', '"--max-model-len", "2048"')


def make_trace(role):
    trace = probe.make_trace(role)
    for row in trace['requests']:
        row['kind'] = 'completion'
        row['prompt'] = 'A research team measured how two computers process data. They found that'
        # A base model may emit EOS immediately. Measure a fixed bounded decode
        # rather than count immediate EOS as a useful serving-memory exercise.
        row['ignore_eos'] = True
    trace['trace_id'] = hashlib.sha256(json.dumps(
        trace['requests'], sort_keys=True).encode()).hexdigest()[:16]
    trace['params'].update(model=MODEL, model_revision=REVISION,
                           max_model_len=2048, api='completion', ignore_eos=True)
    return trace


def validate_config(config):
    expected = dict(model_type='gpt_neox', hidden_size=512, num_attention_heads=8,
                    num_hidden_layers=6, max_position_embeddings=2048)
    if any(config.get(key) != value for key, value in expected.items()):
        raise RuntimeError('Model config does not match the pinned Pythia-70M TP2 candidate')


def prepare_model():
    pvc = json.loads(probe.kube(['get', 'pvc', PVC, '-o', 'json']))
    if pvc.get('status', {}).get('phase') != 'Bound':
        raise RuntimeError('r2-models-pvc is not Bound; no model was downloaded')
    pv = json.loads(probe.kube(['get', 'pv', pvc['spec']['volumeName'], '-o', 'json']))
    spec = pv['spec']
    source = spec.get('local') or spec.get('hostPath') or {}
    root = Path(source.get('path', '/missing-model-pv')).resolve()
    if not root.is_dir() or not root.is_relative_to(Path('/data2')):
        raise RuntimeError('Expected the existing model PVC to map to a local directory under /data2')
    destination = root / SUBPATH
    if not destination.resolve().is_relative_to(root):
        raise RuntimeError('Model destination escapes the PVC root')
    marker = destination / '.split-probe-source.json'
    identity = dict(model=MODEL, revision=REVISION, files=list(FILES))
    if destination.exists() and not marker.exists():
        raise RuntimeError('Refusing to overwrite an existing unmarked model directory: ' + str(destination))
    if marker.exists() and json.loads(marker.read_text()) != identity:
        raise RuntimeError('Existing model directory has a different source; refusing to overwrite')
    if shutil.disk_usage(root).free < 512 * 1024 * 1024:
        raise RuntimeError('Need at least 512 MiB free on the model volume before preparation')
    # Existing trace generation already depends on huggingface_hub on Box1.
    # Do not import torch or load weights into host Python for preparation.
    from huggingface_hub import snapshot_download
    destination.mkdir(parents=True, exist_ok=True)
    probe.write(marker, identity)
    print('Preparing %s at revision %s in %s' % (MODEL, REVISION, destination), flush=True)
    snapshot_download(repo_id=MODEL, revision=REVISION, local_dir=str(destination),
                      allow_patterns=list(FILES), max_workers=1)
    for filename in FILES:
        path = destination / filename
        if not path.is_file() or path.stat().st_size == 0 or not path.resolve().is_relative_to(destination.resolve()):
            raise RuntimeError('Missing or out-of-directory model file: ' + str(path))
    validate_config(json.loads((destination / 'config.json').read_text()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()  # Deliberately no shared-arm or override switches.
    probe.main(['--role', 'tp', '--profiles', 'lowmem-fork'],
               renderer_override=TinyRenderer, trace_factory=make_trace,
               root_prefix='S-tiny-tp-probe-', prepare=prepare_model)


if __name__ == '__main__':
    main()
