#!/usr/bin/env python3
"""Render the two split tenants for one arm.

WHAT THIS SERIES ASKS
---------------------
Round two measured SHARING one card between whole models. The coworker's work
splits ONE model across two cards. Nobody has run both at once, and the two
mechanisms pull in opposite directions:

    sharing  many tenants -> one device
    splitting  one tenant -> many devices

Put together on a two-card box, two 2-way-split tenants means FOUR vLLM ranks
on TWO physical cards, two ranks per card. That is the configuration under
test. It is not exotic -- it is what happens the first time someone runs a
tensor-parallel model on a HAMi node that already has a neighbour.

WHY THE COWORKER'S YAML IS NOT APPLIED AS-IS
-------------------------------------------
Four changes, each of which would otherwise end the run before it produced a
number. They are changes to the DEPLOYMENT wrapper, not to the split config --
--tensor-parallel-size 2 and --pipeline-parallel-size 2 are his, untouched.

1. BOTH FILES DECLARE metadata.name: qwen2-0b5-instruct-local. Applying both
   to llm-demo does not create two tenants; the second overwrites the first
   and the co-tenancy arm silently becomes a solo arm. Renamed r2-tp / r2-pp,
   which is also the shape harness.pods_for() already knows how to find
   (app.kubernetes.io/name=r2-<role>).

2. THE router BLOCK IS DROPPED. Its image, ghcr.io/llm-d/llm-d-inference-
   scheduler, 403s on this node -- measured in round two, which is why
   render.py has --router off by default. Left in, the pod never goes Ready.
   The client addresses pod IPs directly anyway, so nothing is lost.

3. nvidia.com/gpumem AND nvidia.com/gpucores ARE ADDED. His manifests request
   only nvidia.com/gpu, and round two established what that means under HAMi:
   a tenant that names no gpucores consumes NO scheduler quota (Fit() compares
   Coresreq against the remainder, and Coresreq is zero), and a tenant that
   names no gpumem is capped at nothing at all. Two such tenants do not share
   a card, they race for it. Without these two lines the arm measures a
   collision, not a policy.

4. --gpu-memory-utilization IS LOWERED. His 0.75/0.5 are fractions of the
   WHOLE card. Under a gpumem quota the fraction applies to the quota, and
   vLLM does not count the CUDA context or the graph pool inside it. Round
   two's rule: util <= 1 - outside/quota. At a 4096 MiB quota with ~700 MiB
   outside that ceiling is 0.83, so 0.70 is used. This is the single most
   common way a quota'd vLLM pod crashloops.

THE MEMORY BUDGET, PER CARD
---------------------------
    2 ranks x 4096 MiB gpumem   = 8192 MiB of 12045 usable      -> fits
    2 ranks x 50 gpucores       = 100, exactly the Fit() ceiling -> admits

gpumem is IDENTICAL in all three arms including dedicated, which follows
render.py: if the baseline also got a bigger KV cache its extra throughput
would be a memory-budget effect wearing a sharing costume. gpucores is set
only on the sharing arms, matching --dedicated-cores whole.

    python3 50-render-split.py --arm hami --out rendered/
"""
import argparse
import json
import os

NS = os.environ.get("R2_NS", "llm-demo")

# The two split tenants. `extra` is the coworker's split configuration,
# carried through verbatim -- that is the thing under test.
ROLES = {
    "tp": {
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "uri":   "pvc://qwen-local-pv-claim/",
        "gpus":  2,
        "extra": "--tensor-parallel-size 2 --max-num-seqs 32 --max-model-len 8192",
    },
    "pp": {
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "uri":   "pvc://qwen-local-pv-claim/",
        "gpus":  2,
        # max-num-seqs 4 is HIS value and it is kept. It throttles this tenant
        # hard, so pp will look slower than tp in every arm. That is a property
        # of the configuration, not of sharing, and the collector reports each
        # tenant against its OWN solo baseline so the comparison stays honest.
        "extra": "--pipeline-parallel-size 2 --max-num-seqs 4 --max-model-len 8192",
    },
}

GPUMEM   = 4096   # MiB per device
GPUCORES = 50     # per device; the two tenants sum to exactly 100
UTIL     = 0.70   # fraction of the QUOTA, not of the card

BODY = """---
apiVersion: serving.kserve.io/v1alpha1
kind: LLMInferenceService
metadata:
  name: r2-%(role)s
  namespace: %(ns)s
  labels:
    r2-arm: %(arm)s
    r2-role: %(role)s
    r2-series: split
spec:
  model:
    name: %(model)s
    uri: %(uri)s
  replicas: 1
  template:
    containers:
      - name: main
        image: 127.0.0.1:5000/kserve/huggingfaceserver:v0.17.0-gpu
%(launch)s        resources:
          limits:
            cpu: '2'
            memory: %(host_limit)s
            nvidia.com/gpu: "%(gpus)d"
%(mem_lim)s%(core_lim)s          requests:
            cpu: '100m'
            memory: 4Gi
            nvidia.com/gpu: "%(gpus)d"
%(mem_req)s%(core_req)s        env:
          - name: HF_HOME
            value: "/mnt/models"
          - name: VLLM_ADDITIONAL_ARGS
            value: "%(extra)s --gpu-memory-utilization %(util)s"
%(extra_env)s%(mps)s"""

MPS_ENV = """          - name: CUDA_MPS_PIPE_DIRECTORY
            value: /tmp/nvidia-mps
          - name: CUDA_MPS_LOG_DIRECTORY
            value: /tmp/nvidia-mps
          - name: CUDA_MPS_ACTIVE_THREAD_PERCENTAGE
            value: "%(pct)s"
        volumeMounts:
          - {name: mps-pipe, mountPath: /tmp/nvidia-mps}
    volumes:
      - name: mps-pipe
        hostPath: {path: /tmp/nvidia-mps, type: Directory}
"""


def render(role, arm, mps_percent, profile=None):
    profile = profile or os.environ.get('SPLIT_PROFILE', 'original')
    if profile not in ('original', 'lowmem-eager', 'lowmem-serve'):
        raise ValueError('Unknown SPLIT_PROFILE: %s' % profile)
    r = ROLES[role]
    share = arm in ("hami", "hami-mps")
    mem = '            nvidia.com/gpumem: "%d"\n' % GPUMEM
    core = '            nvidia.com/gpucores: "%d"\n' % GPUCORES if share else ""
    mps = ""
    if arm == "hami-mps":
        mps = MPS_ENV % {"pct": mps_percent}
    launch, extra_env = '', ''
    extra = r['extra']
    if profile != 'original':
        extra += ' --enforce-eager'
    if profile == 'lowmem-serve':
        # Candidate serving envelope, not a proven host-memory saving. Keep
        # model/context/quota and split degree; bound TP batch concurrency and
        # prefill work. Explicit TP=1 for PP prevents auto-detected TP expansion.
        extra = ('--tensor-parallel-size %d --pipeline-parallel-size %d '
                 '--max-num-seqs %d --max-model-len 8192 --enforce-eager '
                 '--max-num-batched-tokens 1024 --enable-chunked-prefill '
                 '--distributed-executor-backend mp') % (
                     2 if role == 'tp' else 1, 2 if role == 'pp' else 1,
                     8 if role == 'tp' else 4)
        args = ['--model', '/mnt/models', '--served-model-name', r['model'],
                '--host', '0.0.0.0', '--port', '8000'] + extra.split() + [
                    '--gpu-memory-utilization', str(UTIL)]
        launch = ('        command: ["python", "-m", "vllm.entrypoints.openai.api_server"]\n'
                  '        args: %s\n'
                  '        startupProbe:\n'
                  '          httpGet: {path: /health, port: 8000}\n'
                  '          periodSeconds: 5\n'
                  '          failureThreshold: 180\n'
                  '        readinessProbe:\n'
                  '          httpGet: {path: /health, port: 8000}\n'
                  '          periodSeconds: 5\n') % json.dumps(args)
        # These are allocator/thread controls, not model-weight offloading or
        # a promise to collapse the required vLLM engine/worker processes.
        extra_env = ('          - {name: MALLOC_ARENA_MAX, value: "2"}\n'
                     '          - {name: TOKENIZERS_PARALLELISM, value: "false"}\n'
                     '          - {name: OMP_NUM_THREADS, value: "1"}\n'
                     '          - {name: CUDA_MODULE_LOADING, value: "LAZY"}\n')
    return BODY % {
        "role": role, "ns": NS, "arm": arm,
        "model": r["model"], "uri": r["uri"], "gpus": r["gpus"],
        "mem_lim": mem, "core_lim": core,
        "mem_req": mem, "core_req": core,
        "extra": extra,
        "launch": launch, "extra_env": extra_env,
        "host_limit": '6Gi' if profile == 'lowmem-serve' else '12Gi',
        "util": UTIL, "mps": mps,
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True,
                    choices=("dedicated", "hami", "hami-mps"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", action="append", default=[],
                    help="render just this role (solo baselines run one at a time)")
    # 100 for every client is what round two measured as correct on a saturated
    # card: T14 split the budget so the two clients summed to 100 and came out
    # 6.9% BEHIND no-MPS at a fairness spread of 15.20. The choice on a
    # contended card is binary -- MPS on at 100, or MPS off.
    ap.add_argument("--mps-percent", default="100")
    a = ap.parse_args()

    roles = a.only or sorted(ROLES)
    bad = [r for r in roles if r not in ROLES]
    if bad:
        raise SystemExit("unknown role(s): %s" % ", ".join(bad))

    os.makedirs(a.out, exist_ok=True)
    for role in roles:
        path = os.path.join(a.out, "r2-split-%s-%s.yaml" % (a.arm, role))
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(render(role, a.arm, a.mps_percent))
        print("wrote %s" % path)


if __name__ == "__main__":
    main()
