"""Derive explicit pool/site variants from tested manifests; never deploy."""
from pathlib import Path
import copy
import hashlib
import json
import yaml

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / 'deployment'
POOL_KEY = 'gpu-share.io/pool'


def main():
    provenance = []
    def write(path, data, source, changes):
        dest = OUT / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text('# Based on tested configurations; site bindings are recorded in pool-provenance.json.\n' +
                        yaml.safe_dump(data, sort_keys=False), encoding='utf-8')
        provenance.append(dict(file=path, source=source.relative_to(REPO).as_posix(),
                               source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                               sha256=hashlib.sha256(dest.read_bytes()).hexdigest(), changes=changes))
    for pool, arm in (('dedicated','dedicated'), ('hami','hami'), ('hami-mps','hami-mps')):
        for role in ('llm', 'embed'):
            source = OUT / 'configurations/llm-embedding' / arm / ('r2-' + arm + '-' + role + '.yaml')
            original = yaml.safe_load(source.read_text(encoding='utf-8'))
            for scope in ('pools', 'box1'):
                d = copy.deepcopy(original)
                d['metadata']['name'] = 'pool-' + pool + '-' + role
                d['metadata']['labels'][POOL_KEY] = pool
                d['spec']['template']['nodeSelector'] = {POOL_KEY: pool}
                d['spec']['template']['runtimeClassName'] = 'nvidia'
                changes = ['unique service name and pool label', 'pod nodeSelector for intended pool', 'explicit existing nvidia runtime class']
                if scope == 'pools':
                    d['spec']['model']['uri'] = original['spec']['model']['uri'].replace('r2-models-pvc', 'pool-' + pool + '-models')
                    changes.append('separate destination PVC; create matching local storage first')
                else:
                    d['spec']['template']['nodeSelector']['kubernetes.io/hostname'] = 'mr-gholami'
                    changes.append('Box1 hostname and existing model PVC')
                if pool == 'dedicated':
                    for section in ('requests', 'limits'):
                        d['spec']['template']['containers'][0]['resources'][section]['nvidia.com/gpucores'] = '100'
                    changes.append('gpucores=100: pool isolation adaptation, not the historical quota-matched baseline')
                write(scope+'/'+pool+'/'+('10-' if role == 'llm' else '11-')+role+'.yaml', d, source, changes)
        storage_source = REPO/'round2/01-models-storage.yaml'
        storage = list(yaml.safe_load_all(storage_source.read_text(encoding='utf-8')))
        pv,pvc = storage
        name='pool-'+pool+'-models'
        pv['metadata']['name']=name
        pv['metadata']['labels']={'gpu-share.io/volume':name}
        pv['spec']['nodeAffinity']['required']['nodeSelectorTerms'][0]['matchExpressions'][0]['values']=['SET-'+pool+'-NODE']
        pvc['metadata']['name']=name
        pvc['spec']['volumeName']=name
        pvc['spec']['selector']={'matchLabels':{'gpu-share.io/volume':name}}
        write('pools/'+pool+'/00-storage.yaml', {'apiVersion':'v1','kind':'List','items':[pv,pvc]},storage_source,
              ['per-pool volume names', 'explicit destination hostname placeholder', 'matching PV/PVC binding'])
    source=REPO/'qwen-kserve-benchmark/20-mps-daemon.yaml'
    d=yaml.safe_load(source.read_text(encoding='utf-8'))
    d['spec']['template']['spec']['nodeSelector']={POOL_KEY:'hami-mps','gpu-share.io/mps-enabled':'true'}
    d['spec']['template']['spec']['runtimeClassName']='nvidia'
    write('infrastructure/mps-pool.yaml',d,source,['restrict supervisor to enabled MPS pool; same existing supervisor identity', 'explicit existing nvidia runtime class'])
    # Existing successful split manifests retain all measured runtime parameters.
    for pool in ('hami','hami-mps'):
        for role in ('tp','pp'):
            source=OUT/'configurations/split-tp2-pp2'/pool/(role+'.yaml')
            original=yaml.safe_load(source.read_text(encoding='utf-8'))
            for scope in ('pools','box1'):
                d=copy.deepcopy(original)
                d['metadata']['name']='pool-'+pool+'-'+role
                d['metadata']['labels'][POOL_KEY]=pool
                d['spec']['template']['nodeSelector']={POOL_KEY:pool}
                d['spec']['template']['runtimeClassName']='nvidia'
                if scope=='box1': d['spec']['template']['nodeSelector']['kubernetes.io/hostname']='mr-gholami'
                else: d['spec']['model']['uri']=d['spec']['model']['uri'].replace('r2-models-pvc','pool-'+pool+'-models')
                write(scope+'/'+pool+'/split/'+role+'.yaml',d,source,
                      ['unique service name and pool selection','site PVC or existing Box1 PVC; measured runtime flags unchanged', 'explicit existing nvidia runtime class'])
    source=REPO/'round2/monitoring/60-servicemonitors.yaml'
    d=next(x for x in yaml.safe_load_all(source.read_text(encoding='utf-8'))
           if x and x['metadata']['name']=='r2-vllm-kserve')
    d['spec']['podMetricsEndpoints'][0]['relabelings'][0]['regex']='(?:r2-|pool-)(.+)'
    write('monitoring/pool-workloads.yaml',d,source,
          ['extend role mapping to pool service names; retain one existing monitor identity to avoid duplicate scraping'])
    (OUT/'pool-provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    print('Derived',len(provenance),'pool and Box1 YAMLs')


if __name__ == '__main__': main()
