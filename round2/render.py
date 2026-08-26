#!/usr/bin/env python3
"""Render the deployment manifests for one scenario in one arm.

Requirement 7 says the three arms must be identical apart from the sharing
mechanism. Hand-maintaining nine YAML files cannot guarantee that -- they drift.
So there is ONE template per model type here, and the arm only controls:

    dedicated   no co-tenant. Same gpumem quota, same vLLM flags, same trace.
                Only the neighbour is removed.
    hami        co-tenants, HAMi gpumem + gpucores, no MPS.
    hami-mps    co-tenants, HAMi gpumem + gpucores, MPS env vars added.

The gpumem quota is IDENTICAL in all three arms, including dedicated. That is
deliberate and follows round one: if the baseline also got a bigger KV cache,
its extra throughput would be a memory-budget effect wearing a sharing costume.
`--dedicated-full-card` overrides it when the practical ceiling is wanted
instead of the controlled baseline.

EVERY model is an LLMInferenceService by default, exactly as Qwen is today --
including the pooling ones. Whether the KServe runtime forwards vLLM's pooling
flags is a question for 00-preflight.sh to answer on the box, not something to
assume here. If it turns out not to, `--pooling-via vllm` re-renders the
embedding and rerank tenants as plain `vllm serve` Deployments with the same
image, PVC, quotas and MPS treatment; only KServe leaves the picture.

ONE PHYSICAL GPU. This node has two idle 3080 Tis and HAMi's default GPU
scheduling policy is spread, so nothing stops two "co-tenants" from being
placed on separate cards -- which would make the sharing arm an uncontended arm
and produce a clean, entirely meaningless result. Three consecutive unpinned
runs landed on GPU 1 with GPU 0 at 39 MiB, so this is measured behaviour.

--pin-gpu emits nvidia.com/use-gpuuuid, and on the KSERVE PATH IT DOES NOT
WORK. Confirmed: the annotation never reaches the pod, because
LLMInferenceService's spec.template is a bare PodSpec with no metadata field --
the same reason labels do not propagate. The pod carries only the hami.io/*
annotations HAMi wrote itself. The flag is still emitted because it DOES work
on the plain-Deployment path (--pooling-via vllm), where we own the template.

For the KServe path, co-location comes from the scheduler instead:
10-hami-binpack.sh sets gpuSchedulerPolicy=binpack cluster-wide. Either way
run.py VERIFIES placement afterwards -- against nvidia-smi, which cannot be
fooled by a renamed annotation -- and aborts rather than publish a sharing arm
whose tenants never met.

    python render.py --scenario llm-embed --arm hami-mps --out rendered/
"""
import argparse
import json
import os
import sys

# --------------------------------------------------------------------------
# Per-model facts. gpumem/util come from the accepted plan's configuration B/E;
# weights are the measured on-disk size, kept here so the budget is auditable.
#
# The three BGE models are distributed in FLOAT32, not bfloat16 -- verify-models
# --deep measured 2.14 GiB for the reranker against a 1090 MiB budget. They
# carry --dtype=half so vLLM casts on load; without it each one needs twice the
# VRAM assumed here and the card no longer fits two tenants.
MODELS = {
    "llm": dict(
        repo="Qwen/Qwen2.5-7B-Instruct-AWQ", kind="chat", weights_mib=5370,
        gpumem=8192, util=0.92, max_model_len=4096, max_num_seqs=64,
        extra=["--quantization=awq_marlin"],
    ),
    "llm2": dict(
        repo="Qwen/Qwen2.5-1.5B-Instruct", kind="chat", weights_mib=2950,
        gpumem=5632, util=0.92, max_model_len=4096, max_num_seqs=64, extra=[],
    ),
    "llm-small": dict(
        repo="Qwen/Qwen2.5-0.5B-Instruct", kind="chat", weights_mib=990,
        gpumem=2816, util=0.90, max_model_len=4096, max_num_seqs=32, extra=[],
    ),
    "llm-mid": dict(
        repo="Qwen/Qwen2.5-3B-Instruct", kind="chat", weights_mib=5890,
        gpumem=11264, util=0.93, max_model_len=8192, max_num_seqs=64, extra=[],
    ),
    "embed": dict(
        repo="BAAI/bge-m3", kind="embed", weights_mib=1090,
        gpumem=3072, util=0.90, max_model_len=8192, max_num_seqs=64,
        extra=["--runner=pooling", "--convert=embed", "--dtype=half"],
    ),
    "rerank": dict(
        repo="BAAI/bge-reranker-v2-m3", kind="rerank", weights_mib=1090,
        gpumem=4608, util=0.88, max_model_len=8192, max_num_seqs=64,
        extra=["--runner=pooling", "--convert=classify", "--dtype=half"],
    ),
    "embed-small": dict(
        repo="BAAI/bge-large-en-v1.5", kind="embed", weights_mib=640,
        gpumem=2816, util=0.90, max_model_len=512, max_num_seqs=64,
        extra=["--runner=pooling", "--convert=embed", "--dtype=half"],
    ),
    "vl": dict(
        repo="Qwen/Qwen2-VL-2B-Instruct", kind="chat", weights_mib=4220,
        gpumem=6656, util=0.90, max_model_len=4096, max_num_seqs=32, extra=[],
    ),
}

# Roles per scenario, mirroring workload/gen_trace.py so a trace and a
# deployment cannot disagree about which services exist.
#
# dense4 was ["llm-small", "llm2", "embed-small", "rerank"], which had two
# faults an audit caught before it cost a run: it summed to 15872 MiB against a
# 12288 MiB card (129%), and llm-small is Qwen2.5-0.5B, which was never
# downloaded -- it belonged to round one's PVC, not this one. It now uses four
# models that exist here.
SCENARIOS = {
    "llm-only":    ["llm"],
    "embed-only":  ["embed"],
    "rerank-only": ["rerank"],
    "llm-embed":   ["llm", "embed"],
    "llm-llm":     ["llm", "llm2"],
    "vl-rerank":   ["vl", "rerank"],
    "dense4":      ["llm2", "embed", "embed-small", "rerank"],
}

# A model's quota is not a property of the model alone -- it depends on who it
# is sharing with. The defaults in MODELS suit two tenants; packing four onto
# one card needs each to give something back, and llm-llm pairs the 7B with a
# 1.5B whose default quota assumes it is the only generative tenant.
#
# Every override below still clears the model's measured footprint with room to
# spare. bge-m3 was measured at 1608 MiB resident inside a 3072 MiB quota, so
# 2816 is not tight; Qwen2.5-1.5B is 2960 MiB of bf16 weights, so 4096 leaves
# roughly 800 MiB of KV, about 29k tokens at 28 KiB/token.
SCENARIO_QUOTAS = {
    # 8192 + 5632 = 13824 (112.5%) at the defaults.
    "llm-llm": {"llm2": 3840},
    # four tenants: 4096 + 2816 + 2304 + 2816 = 12032, 97.9% of the card.
    "dense4": {"llm2": 4096, "embed": 2816, "embed-small": 2304,
               "rerank": 2816},
}

IMAGE = "127.0.0.1:5000/kserve/huggingfaceserver:v0.17.0-gpu"
PVC = "r2-models-pvc"
NS = "llm-demo"
CARD_MIB = 12288

# HAMi reads these from the POD's annotations. use-gpuuuid pins the device;
# the binpack policy is a second, independent lever towards the same outcome in
# case the UUID pin is dropped somewhere along the KServe path.
def pin_annotations(uuid, indent):
    """HAMi reads the device pin from the POD's annotations, and the two call
    sites nest at different depths, so the indent is a parameter rather than
    baked into a constant."""
    if not uuid:
        return ""
    pad = "\n" + " " * indent
    return ('%snvidia.com/use-gpuuuid: "%s"'
            '%shami.io/gpu-scheduler-policy: "binpack"' % (pad, uuid, pad))

MPS_ENV = """
          - name: CUDA_MPS_PIPE_DIRECTORY
            value: /tmp/nvidia-mps
          - name: CUDA_MPS_LOG_DIRECTORY
            value: /tmp/nvidia-mps
          - name: CUDA_MPS_ACTIVE_THREAD_PERCENTAGE
            value: "{pct}\""""

MPS_MOUNT = """
        volumeMounts:
          - {name: mps-pipe, mountPath: /tmp/nvidia-mps}"""

# `volumes` is a sibling of `containers` under spec.template, so it sits at the
# same indent as `containers:` (4), NOT at the indent of the container list
# items (6). At 6 the YAML parser reads it as a scalar key interrupting the
# block sequence and rejects the whole document -- which made every hami-mps
# KServe manifest unparseable until a round-trip yaml.safe_load_all caught it.
MPS_VOL = """
    volumes:
      - name: mps-pipe
        hostPath: {path: /tmp/nvidia-mps, type: Directory}"""


def gpu_resources(m, arm, pct):
    """The resource block. gpucores is requested in both HAMi arms because it is
    what the scheduler packs against -- round one established it is a placement
    token, not a compute bound, and that is exactly how it is used here."""
    lines = ['            nvidia.com/gpu: "1"',
             '            nvidia.com/gpumem: "%d"' % m["gpumem"]]
    if arm in ("hami", "hami-mps"):
        lines.append('            nvidia.com/gpucores: "%d"' % m["gpucores"])
    return "\n".join(lines)


ROUTER_BLOCK = """
  router:
    scheduler: {}
    route: {}
    gateway:
      refs:
        - name: kserve-ingress-gateway
          namespace: kserve
"""


def render_llm(role, m, arm, pct, pin="", router=False):
    res = gpu_resources(m, arm, pct)
    pin = pin_annotations(pin, 4) or " {}"
    # The router is OFF by default. `scheduler: {}` makes KServe create a
    # router-scheduler Deployment pulling ghcr.io/llm-d/llm-d-inference-
    # scheduler, which this node cannot fetch -- ghcr returns 403 Forbidden to
    # anonymous pulls for that repo. It crashlooped through the whole check.
    #
    # Nothing is lost: the client deliberately talks straight to the pod IP so
    # that the KServe router is outside the measurement in every arm. The
    # workload pod and its Service (r2-<role>-kserve-workload-svc:8000) exist
    # either way. Pass --router if external ingress is ever wanted.
    router = ROUTER_BLOCK if router else ""
    env = '          - name: HF_HOME\n            value: /mnt/models'
    if arm == "hami-mps":
        env += MPS_ENV.format(pct=pct)
    args = ["--gpu-memory-utilization=%.2f" % m["util"],
            "--max-model-len=%d" % m["max_model_len"],
            "--max-num-seqs=%d" % m["max_num_seqs"]] + m["extra"]
    argblock = "\n".join("          - %s" % a for a in args)
    vol = MPS_MOUNT if arm == "hami-mps" else ""
    volumes = MPS_VOL if arm == "hami-mps" else ""
    return f"""---
apiVersion: serving.kserve.io/v1alpha1
kind: LLMInferenceService
metadata:
  name: r2-{role}
  namespace: {NS}
  labels:
    r2-arm: {arm}
    r2-role: {role}
  annotations:{pin}
spec:
  model:
    name: {m['repo']}
    uri: pvc://{PVC}/models/{role}
  replicas: 1{router}
  template:
    containers:
      - name: main
        image: {IMAGE}
        args:
{argblock}
        resources:
          limits:
            cpu: "4"
            memory: 12Gi
{res}
          requests:
            cpu: "200m"
            memory: 2Gi
{res}
        env:
{env}{vol}{volumes}
"""


def render_pooling(role, m, arm, pct, pin=""):
    """Plain vllm serve. Same image, same PVC, same HAMi/MPS treatment -- the
    only difference from the LLM path is that KServe is not in front of it."""
    res = gpu_resources(m, arm, pct)
    pin_ann = pin_annotations(pin, 8)
    env = ('          - name: HF_HOME\n            value: /mnt/models\n'
           '          - name: VLLM_LOGGING_LEVEL\n            value: INFO')
    if arm == "hami-mps":
        env += MPS_ENV.format(pct=pct)
    args = ["serve", m["repo"],
            "--served-model-name=%s" % role,
            "--host=0.0.0.0", "--port=8000",
            "--gpu-memory-utilization=%.2f" % m["util"],
            "--max-model-len=%d" % m["max_model_len"],
            "--max-num-seqs=%d" % m["max_num_seqs"],
            "--download-dir=/mnt/models"] + m["extra"]
    argblock = "\n".join("          - %s" % a for a in args)
    mps_mount = ("\n          - {name: mps-pipe, mountPath: /tmp/nvidia-mps}"
                 if arm == "hami-mps" else "")
    mps_vol = ("\n        - name: mps-pipe\n"
               "          hostPath: {path: /tmp/nvidia-mps, type: Directory}"
               if arm == "hami-mps" else "")
    return f"""---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: r2-{role}
  namespace: {NS}
  labels: {{r2-arm: {arm}, r2-role: {role}, app: r2-{role}}}
spec:
  replicas: 1
  selector:
    matchLabels: {{app: r2-{role}}}
  template:
    metadata:
      labels: {{app: r2-{role}, r2-arm: {arm}, r2-role: {role}}}
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8000"
        prometheus.io/path: "/metrics"{pin_ann}
    spec:
      containers:
      - name: main
        image: {IMAGE}
        command: ["vllm"]
        args:
{argblock}
        ports:
          - {{name: http, containerPort: 8000}}
        resources:
          limits:
            cpu: "4"
            memory: 12Gi
{res}
          requests:
            cpu: "200m"
            memory: 2Gi
{res}
        env:
{env}
        volumeMounts:
          - {{name: models, mountPath: /mnt/models}}{mps_mount}
        readinessProbe:
          httpGet: {{path: /health, port: 8000}}
          initialDelaySeconds: 30
          periodSeconds: 10
          failureThreshold: 60
      volumes:
        - name: models
          persistentVolumeClaim: {{claimName: {PVC}}}{mps_vol}
---
apiVersion: v1
kind: Service
metadata:
  name: r2-{role}
  namespace: {NS}
  labels: {{app: r2-{role}, r2-role: {role}}}
spec:
  selector: {{app: r2-{role}}}
  ports:
    - {{name: http, port: 80, targetPort: 8000}}
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", required=True, choices=sorted(SCENARIOS))
    ap.add_argument("--arm", required=True, choices=("dedicated", "hami", "hami-mps"))
    ap.add_argument("--out", required=True, help="directory for rendered YAML")
    ap.add_argument("--mps-percent", default="", help="per-role, e.g. llm=75,embed=25; "
                                                      "default 100 for every role")
    ap.add_argument("--only", action="append", default=[],
                    help="render just this role (dedicated baselines run one at a time)")
    ap.add_argument("--dedicated-full-card", action="store_true",
                    help="give the dedicated arm the whole card instead of the shared quota")
    ap.add_argument("--router", action="store_true",
                    help="also create the KServe router/scheduler. Off by "
                         "default: its image (ghcr.io/llm-d/llm-d-inference-"
                         "scheduler) 403s on this node, and the client bypasses "
                         "the router anyway.")
    ap.add_argument("--pin-gpu", default="",
                    help="HAMi device UUID to pin every tenant to. Without this "
                         "HAMi may spread co-tenants across the two cards, and "
                         "the sharing arms silently stop sharing.")
    ap.add_argument("--pooling-via", choices=("kserve", "vllm"), default="kserve",
                    help="how to serve embedding/rerank tenants. kserve keeps every "
                         "model on one deployment path; switch to vllm only if "
                         "00-preflight.sh shows the runtime drops the pooling flags")
    a = ap.parse_args()

    roles = SCENARIOS[a.scenario]
    if a.only:
        unknown = [r for r in a.only if r not in roles]
        if unknown:
            sys.exit("role(s) %s not in scenario %s (%s)"
                     % (unknown, a.scenario, roles))
        roles = [r for r in roles if r in a.only]

    pcts = {}
    for kv in filter(None, a.mps_percent.split(",")):
        k, v = kv.split("=")
        pcts[k] = int(v)

    # gpucores is the scheduler's packing token: split the card between the
    # co-tenants so HAMi's admission arithmetic reflects the intended density.
    share = max(1, 100 // max(1, len(SCENARIOS[a.scenario])))

    os.makedirs(a.out, exist_ok=True)
    written, budget = [], 0
    for role in roles:
        m = dict(MODELS[role])
        override = SCENARIO_QUOTAS.get(a.scenario, {}).get(role)
        if override:
            m["gpumem"] = override
        m["gpucores"] = share
        if a.arm == "dedicated" and a.dedicated_full_card:
            m["gpumem"] = CARD_MIB - 512
        pct = pcts.get(role, 100)
        as_kserve = m["kind"] == "chat" or a.pooling_via == "kserve"
        body = (render_llm(role, m, a.arm, pct, a.pin_gpu, a.router) if as_kserve
                else render_pooling(role, m, a.arm, pct, a.pin_gpu))
        path = os.path.join(a.out, "r2-%s-%s.yaml" % (a.arm, role))
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(body)
        written.append(path)
        budget += m["gpumem"]
        print("%-10s %-32s gpumem %5d  util %.2f  cores %3d  mps %3d%%  -> %s"
              % (role, m["repo"], m["gpumem"], m["util"], m["gpucores"], pct,
                 os.path.basename(path)))

    manifest = dict(scenario=a.scenario, arm=a.arm, roles=roles,
                    pooling_via=a.pooling_via, pin_gpu=a.pin_gpu,
                    router=a.router,
                    gpumem_total_mib=budget, card_mib=CARD_MIB,
                    fill_percent=round(budget / CARD_MIB * 100, 1),
                    mps_percent={r: pcts.get(r, 100) for r in roles},
                    files=[os.path.basename(p) for p in written],
                    models={r: MODELS[r]["repo"] for r in roles})
    # NOT in a.out itself. `kubectl apply -f DIR` reads every .json/.yaml/.yml
    # directly inside the directory, so a metadata file sitting beside the
    # manifests gets submitted to the API server as if it were an object:
    #     error validating ...manifest.json: [apiVersion not set, kind not set]
    # and apply exits non-zero AFTER partially creating the real resources.
    # Directory reads are non-recursive without -R, so a subdirectory is
    # invisible to apply while staying next to the YAML for auditing.
    meta_dir = os.path.join(a.out, "meta")
    os.makedirs(meta_dir, exist_ok=True)
    with open(os.path.join(meta_dir, "r2-%s-manifest.json" % a.arm), "w",
              encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    print("\narm %s  scenario %s" % (a.arm, a.scenario))
    print("gpumem requested %d of %d MiB = %.1f%% of the card"
          % (budget, CARD_MIB, budget / CARD_MIB * 100))
    if budget > CARD_MIB:
        print("ERROR: over-subscribed by %d MiB. HAMi will refuse to place "
              "these together, so this would fail at admission rather than "
              "produce a result. Adjust SCENARIO_QUOTAS." % (budget - CARD_MIB))
        sys.exit(3)
    print("apply with: kubectl apply -f %s" % a.out)


if __name__ == "__main__":
    main()
