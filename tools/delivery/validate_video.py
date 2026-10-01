"""Second offline check: guide syntax, metrics, dashboard repair and mocked response paths."""
import ast,copy,hashlib,json,re,subprocess
from pathlib import Path
from unittest.mock import patch
import yaml
ROOT=Path(__file__).resolve().parents[2]
PKG=ROOT/'deliverable'
BASH='C:/Program Files/Git/bin/bash.exe'
def main():
    guide=(PKG/'OPERATIONS_FA.md').read_text(encoding='utf-8')
    video_file=ROOT/'tools/delivery/VIDEO_README_FA.md'
    video=video_file.read_text(encoding='utf-8')
    assert '```' not in video
    for target in re.findall(r'\]\(([^)]+)\)',video):
        if not target.startswith('https://'): assert (video_file.parent/target).resolve().exists(),target
    blocks=re.findall(r'```bash\n(.*?)```',guide,re.S)
    for b in blocks:
        subprocess.run([BASH,'-n'],input=b,text=True,check=True,capture_output=True)
        for f in re.findall(r'-f\s+([\w/.-]+\.yaml)',b):assert (PKG/f).is_file(),f
    codes=re.findall(r"python3[^\n]*<<'PY'\n(.*?)\nPY",guide,re.S)
    for c in codes:ast.parse(c)
    repair=next(c for c in codes if "dashboard['panels']" in c)
    cm=yaml.safe_load((PKG/'monitoring/dashboard.yaml').read_text())
    captured=[]
    with patch('subprocess.check_output',return_value=json.dumps(cm)),patch('subprocess.run',side_effect=lambda args,**kw:captured.append(args)):
        exec(compile(repair,'<dashboard repair>','exec'),{})
    d=json.loads(json.loads(captured[0][-1])['data']['r2-gpu-sharing.json'])
    qs=[t['expr'] for p in d['panels'] for t in p.get('targets',[])]
    assert any('vllm:kv_cache_usage_perc' in q for q in qs)
    assert any('vllm:inter_token_latency_seconds_bucket' in q for q in qs)
    assert not any(any(x in q for x in ['prefix_cache_hit_rate','gpu_cache_usage_perc','time_per_output_token_seconds_bucket']) for q in qs)
    raw=json.loads((ROOT.parent/'T15-20260906-182642/hami-mps__llm2a+llm2b/prom.json').read_text())
    actual='\n'.join(v.get('expr') or '' for v in raw['series'].values() if v.get('result'))
    for q in re.findall(r'```promql\n(.*?)```',guide,re.S):
        for metric in re.findall(r'\b(?:vllm:[a-z_]+|DCGM_FI_[A-Z_]+|hami_gpu_[a-z_]+)',q):assert metric in actual,metric
    objects={d['metadata']['name']:d for d in yaml.safe_load_all((PKG/'monitoring/servicemonitors.yaml').read_text())}
    patches=re.findall(r"patch servicemonitor (\S+) --type=json \\\n\s*-p='([^']+)'",guide)
    assert len(patches)==2,len(patches)
    for name,data in patches:
        obj=copy.deepcopy(objects[name])
        for op in json.loads(data):
            parts=op['path'].strip('/').split('/');parent=obj
            for part in parts[:-1]:parent=parent[int(part)] if isinstance(parent,list) else parent[part]
            key=int(parts[-1]) if isinstance(parent,list) else parts[-1]
            assert op['op']=='replace'
            assert isinstance(parent,list) or key in parent
            parent[key]=op['value']
        if name=='r2-hami-scheduler':assert obj['spec']['endpoints'][0]['port']=='monitor'
        if name=='r2-dcgm':assert obj['spec']['namespaceSelector']['matchNames']==['gpu-operator']
    func=next(b for b in blocks if b.startswith('show_model()'))
    mocks=r'''
python3() { /c/Python314/python.exe "$@"; }
sleep() { :; }
kubectl() {
 case "$*" in
 *'get pods'*) [ "$CASE" = no_pod ] || echo demo-pod ;;
 *'wait --for=condition=Ready'*) [ "$CASE" != unready ] ;;
 *'port-forward'*) exec /usr/bin/sleep 30 ;;
 *) return 2 ;;
 esac
}
curl() {
 local out= url= arg
 while [ "$#" -gt 0 ]; do
  arg="$1"; shift
  case "$arg" in -o) out="$1"; shift ;; http*) url="$arg" ;; esac
 done
 if [[ "$url" == */v1/models ]]; then
  printf '{"data":[{"id":"%s"}]}' "$MOCK_MODEL"
 elif [ "$CASE" = network ]; then return 7
 elif [ "$CASE" = http400 ]; then
  printf '{"error":{"message":"bad request"}}' > "$out"; printf 400
 elif [[ "$url" == */embeddings ]]; then
  printf '{"data":[{"embedding":[0.1,0.2]}]}' > "$out"; printf 200
 else
  printf '{"choices":[{"message":{"content":"GPU sharing"}}]}' > "$out"; printf 200
 fi
}
'''
    cases=[('chat','llm chat','Qwen/Qwen2.5-7B-Instruct-AWQ',0),('embed','embed embed','BAAI/bge-m3',0),('wrong_model','embed embed','Qwen/Qwen2.5-7B-Instruct-AWQ',1),('http400','llm chat','Qwen/Qwen2.5-7B-Instruct-AWQ',1),('network','llm chat','Qwen/Qwen2.5-7B-Instruct-AWQ',1),('no_pod','llm chat','',1),('unready','llm chat','',1)]
    for case,args,model,expected in cases:
        prog=mocks+'\n'+func+f'\nCASE={case}\nMOCK_MODEL={model}\nshow_model hami {args}\n'
        r=subprocess.run([BASH],input=prog,text=True,capture_output=True,timeout=20)
        assert r.returncode==expected,(case,r.returncode,r.stdout,r.stderr)
    assert guide==(ROOT/'tools/delivery/OPERATIONS_FA.md').read_text(encoding='utf-8')
    result=dict(ok=True,shell_blocks=len(blocks),python_blocks=len(codes),mock_cases=[c[0] for c in cases],dashboard_repair=True,metric_names_backed_by_T15=True,patch_paths=True,video_links=True,operations_sha256=hashlib.sha256(guide.encode()).hexdigest(),limits='Offline checks and mocked responses only; no live cluster or SSH commands executed.')
    (PKG/'validation/video-receipt.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
