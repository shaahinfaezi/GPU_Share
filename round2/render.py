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
# outside_mib is the CUDA context plus the CUDA-graph pool. It is the number
# that broke llm2, and it is worth stating plainly:
#
#   HAMi's nvidia.com/gpumem caps TOTAL device memory for the process.
#   vLLM's --gpu-memory-utilization is a fraction of the device total, and it
#   does NOT include the CUDA context or the graph pool.
#
# A generative model grows its KV cache to fill vLLM's budget -- but only UP TO
# what `max_num_seqs x max_model_len` can actually use. llm and llm2 ask for
# 64 x 4096 = 262k tokens, far more than their budgets hold, so they do fill it.
# vl asks for 32 x 4096 = 131k, and on the whole card vLLM allocated exactly
# 137,920 tokens (that figure rounded up to block boundaries) and left 1586 MiB
# of its budget UNUSED. So the rule below is the worst case, not the typical
# one, which is the right side to be wrong on: it reserves for a full budget.
#
# In the filling case, resident settles at (util x quota) + outside_mib
# regardless of how big the weights are. The feasibility test is therefore:
#
#       util x quota + outside_mib <= quota      ->    util <= 1 - outside/quota
#
# Measured, not guessed: llm2 at an 11776 MiB quota reported a 10834 MiB budget
# and sat at 11498 MiB resident -> outside = 664 MiB. The 7B AWQ in T1 budgeted
# 7537 MiB and peaked at 7984 -> outside = 448 MiB.
#
# util is a CEILING here, not a setting. effective_util() clamps it to whatever
# the quota can actually support, so a quota change cannot silently produce a
# pod that crashloops in _dummy_sampler_run. Pooling models do not fill their
# budget with KV, so the clamp does not apply to them.
# THE PROFILING PEAK IS A SECOND, LARGER CONSTRAINT, and util does not touch it.
#
# vLLM runs a dummy forward pass at max batch and max sequence -- for a
# multimodal model, at max image/video tokens too -- BEFORE it sizes the KV
# cache, precisely in order to measure the peak. That run is bounded by the
# HAMi cap and by nothing else. Lowering --gpu-memory-utilization cannot help,
# because it only shrinks the KV cache that gets allocated afterwards.
#
# vl proved it. Two probes at 6656 MiB, util 0.872 and 0.812, produced a
# BYTE-IDENTICAL OOM: same 160 MiB request, same 6.44 GiB in use, same 137.87
# MiB free. PyTorch had allocated 6021 MiB, which is ABOVE the 5804 MiB budget
# at the higher util -- so it cannot be the KV cache, which is sized to fit.
#
# So a quota must clear TWO bars: the profiling peak, and then steady state.
# profile_peak_mib records the first where it has been measured.
OUTSIDE_DEFAULT = 700   # unmeasured generative models; probe with 18-vram-probe.sh
OUTSIDE_POOLING = 450   # pooling models: no KV cache, no cudagraph pool of note
OUTSIDE_MARGIN = 150    # slack between predicted resident and the HAMi cap
KV_FLOOR_MIB = 512      # below this a generative tenant thrashes rather than runs

MODELS = {
    "llm": dict(
        repo="Qwen/Qwen2.5-7B-Instruct-AWQ", kind="chat", weights_mib=5370,
        gpumem=8192, util=0.92, max_model_len=4096, max_num_seqs=64,
        outside_mib=448,          # measured, T1
        extra=["--quantization=awq_marlin"],
    ),
    # Was Qwen2.5-1.5B-Instruct in bf16: 2938 MiB of weights, which put the
    # floor to run at all at 4115 MiB and made T2's 3840 MiB quota impossible
    # before a single KV block was allocated. The AWQ build is 1540 MiB, fits
    # 3840 with ~34k tokens of KV, and has the side benefit of making T2 a
    # comparison of MODEL SIZE rather than of size confounded with precision --
    # the 7B it shares the card with is AWQ too.
    "llm2": dict(
        repo="Qwen/Qwen2.5-1.5B-Instruct-AWQ", kind="chat", weights_mib=1540,
        gpumem=5632, util=0.92, max_model_len=4096, max_num_seqs=64,
        # 664 measured on the bf16 build at an 11776 MiB quota. The AWQ build
        # then measured 558 at 3840 (3584 resident against a 3026 budget), so
        # this is deliberately ~106 MiB conservative: the validated T2 config
        # is util 0.788 with 256 MiB of slack, and tightening it to 0.816 for
        # 107 MiB more KV is not worth re-validating.
        outside_mib=664,
        extra=["--quantization=awq_marlin"],
    ),
    # The symmetric pair: two names, one model, one directory on the PVC.
    # 2 x 5632 = 11264 MiB against CARD_MIB 12288, so it fits with ~1 GiB
    # spare. Two 7Bs do not (2 x 8192 = 16384), which is why the pair is
    # built from the 1.5B.
    "llm2a": dict(
        repo="Qwen/Qwen2.5-1.5B-Instruct-AWQ", kind="chat", weights_mib=1540,
        gpumem=5632, util=0.92, max_model_len=4096, max_num_seqs=64,
        outside_mib=664, model_dir="llm2",
        extra=["--quantization=awq_marlin"],
    ),
    "llm2b": dict(
        repo="Qwen/Qwen2.5-1.5B-Instruct-AWQ", kind="chat", weights_mib=1540,
        gpumem=5632, util=0.92, max_model_len=4096, max_num_seqs=64,
        outside_mib=664, model_dir="llm2",
        extra=["--quantization=awq_marlin"],
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
        outside_mib=450,        # bge-m3
        extra=["--runner=pooling", "--convert=embed", "--dtype=half"],
    ),
    "rerank": dict(
        repo="BAAI/bge-reranker-v2-m3", kind="rerank", weights_mib=1090,
        gpumem=4608, util=0.88, max_model_len=8192, max_num_seqs=64,
        outside_mib=450,        # bge-reranker-v2-m3
        extra=["--runner=pooling", "--convert=classify", "--dtype=half"],
    ),
    "embed-small": dict(
        repo="BAAI/bge-large-en-v1.5", kind="embed", weights_mib=640,
        gpumem=2816, util=0.90, max_model_len=512, max_num_seqs=64,
        outside_mib=450,        # bge-large-en-v1.5, measured 246 MiB at startup
        extra=["--runner=pooling", "--convert=embed", "--dtype=half"],
    ),
    # Multimodal init is the most expensive in the matrix: a vision encoder AND
    # a language model compiled, then 11 cudagraph sizes captured. T3 died here
    # with the default outside_mib of 700. The OOM gave the real number: budget
    # 5804 MiB, process at 6595 asking for 160 more, against a cap the process
    # correctly saw as 6.50 GiB -- so outside is at least 951. 1100 leaves ~300
    # MiB of genuine slack rather than the ~0 that killed it.
    "vl": dict(
        repo="Qwen/Qwen2-VL-2B-Instruct", kind="chat", weights_mib=4220,
        gpumem=6656, util=0.90, max_model_len=4096, max_num_seqs=32,
        outside_mib=1100,         # measured from T3's OOM; >=951
        profile_peak_mib=6755,    # measured; util cannot reduce this
        extra=[],
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
    "llm2-only":   ["llm2"],
    "vl-only":     ["vl"],
    "embed-small-only": ["embed-small"],
    "llm-embed":   ["llm", "embed"],
    "llm-llm":     ["llm", "llm2"],
    "llm2-llm2":   ["llm2a", "llm2b"],
    "dense3":      ["llm2", "embed-small", "embed"],
    "dense3b":     ["llm2a", "llm2b", "embed-small"],
    "vl-rerank":   ["vl", "rerank"],
    "dense4":      ["llm2", "embed", "embed-small", "rerank"],
}

# A model's quota is not a property of the model alone -- it depends on who it
# is sharing with. The defaults in MODELS suit two tenants; packing four onto
# one card needs each to give something back, and llm-llm pairs the 7B with a
# 1.5B whose default quota assumes it is the only generative tenant.
#
# The arithmetic here used to reason from weights alone -- "2960 MiB of bf16
# weights, so 4096 leaves roughly 800 MiB of KV" -- and that is how llm2 came
# to crashloop. It ignored both the 0.92 utilisation scaling and the 664 MiB of
# CUDA context that HAMi charges but vLLM does not budget for. The real figure
# was 82 MiB of KV against a cap the process exceeded by 357 MiB.
#
# effective_util() now derives the ceiling from the quota, and the KV_FLOOR_MIB
# check refuses to render a tenant that would thrash, so a bad split fails at
# render time instead of ten minutes into a run.
SCENARIO_QUOTAS = {
    # 8192 + 5632 = 13824 (112.5%) at the defaults.
    "llm-llm": {"llm2": 3840},
    # vl is sized by its PROFILING PEAK, not its weights. Measured twice:
    #   quota 11776, util 0.900 -> budget 10598, KV 3768  => peak 6830
    #   quota  8192, util 0.847 -> budget  6939, KV  133  => peak 6806
    # so KV = util x quota - 6830. 8192 started but gave 1.21x concurrency,
    # which would have thrashed through the tier and produced numbers that
    # looked like a sharing cost. 9216 gives ~46k tokens and 11.3x, against
    # roughly 2 req/s of vl traffic at T3's rate. rerank measured 1568 MiB
    # resident at a 3840 quota, so 2816 leaves it comfortable room.
    "vl-rerank": {"vl": 9216, "rerank": 2816},
    # four tenants: 4096 + 2816 + 2304 + 2816 = 12032, 97.9% of the card.
    # three tenants: 3840 + 2304 + 2816 = 8960 MiB, 72.9% of the card.
    "dense3": {"llm2": 3840, "embed-small": 2304, "embed": 2816},
    # three tenants, two of them identical decoders: 3840 + 3840 + 2304
    # = 9984 MiB, 81.2% of the card. Deliberately roomier than dense3 on
    # the GPU because host RAM, not VRAM, is what refuses this tier.
    "dense3b": {"llm2a": 3840, "llm2b": 3840, "embed-small": 2304},
    "dense4": {"llm2": 4096, "embed": 2816, "embed-small": 2304,
               "rerank": 2816},
}

IMAGE = "127.0.0.1:5000/kserve/huggingfaceserver:v0.17.0-gpu"
PVC = "r2-models-pvc"
NS = "llm-demo"
CARD_MIB = 12288

def effective_util(m):
    """Clamp --gpu-memory-utilization to what the HAMi quota can support.

    Returns (util, safe_cap, clamped).

    THIS APPLIES TO EVERY MODEL KIND. An earlier version exempted pooling
    models, reasoning that they never fill their budget with KV cache so the
    declared value could not bind. That reasoning was wrong, and embed-small
    proved it: BEFORE allocating anything, vLLM checks

        free_memory >= gpu_memory_utilization x total_memory

    and raises ValueError if it fails -- for every model kind, KV cache or not.
    At a 2304 MiB quota with a 246 MiB CUDA context already resident, util 0.90
    asks for 2074 MiB against 2058 MiB free and dies 15 MiB short.

    The check is exactly the inequality this function already enforces, so the
    fix is only to stop skipping it. Left unfixed it would also have taken out
    T4: embed at a 2816 MiB quota had 36 MiB of slack on the same check.
    """
    declared = m["util"]
    default = OUTSIDE_POOLING if m["kind"] != "chat" else OUTSIDE_DEFAULT
    outside = m.get("outside_mib", default) + OUTSIDE_MARGIN
    cap = 1.0 - float(outside) / float(m["gpumem"])
    if declared <= cap:
        return declared, cap, False
    return round(cap, 3), cap, True


# HAMi reads these from the POD's annotations. use-gpuuuid pins the device;
# the binpack policy is a second, independent lever towards the same outcome in
# case the UUID pin is dropped somewhere along the KServe path.
def pin_annotations(uuid, indent):
    """Emit the spec.annotations block that KServe copies onto the pod.

    CORRECTED 2026-09-09. These used to be emitted under the
    LLMInferenceService OWN metadata.annotations, which KServe does NOT
    copy onto the workload pod -- so the pin was written, looked correct in
    the manifest, and never reached HAMi. That is why this project believed
    for weeks that pinning was impossible on the KServe path and reached for
    a cluster-wide binpack instead.

    The field that DOES propagate is spec.annotations: KServe WorkloadSpec
    carries first-class Labels and Annotations as siblings of Template, and
    0.17 ships propagation to downstream workload resources. Our serving
    image is v0.17.0. FINDINGS 9.1.

    Returns the whole block including the annotations: key, so an empty pin
    emits nothing at all rather than a dangling mapping.
    """
    if not uuid:
        return ""
    pad = "\n" + " " * indent
    inner = pad + "  "
    return ("%sannotations:"
            "%snvidia.com/use-gpuuuid: \"%s\""
            "%shami.io/gpu-scheduler-policy: \"binpack\""
            % (pad, inner, uuid, inner))

# HAMi's core limiter is policy-gated and its DEFAULT is "default", whose
# behaviour HAMi does not document. Only "force" is documented to "always limit
# the core utilization below nvidia.com/gpucores". Round one set force on every
# pod; round two set nothing at all, so the two rounds ran different
# configurations and their numbers are not comparable. FINDINGS 5.14b / 5.14c.
CORE_POLICY_ENV = """
          - name: GPU_CORE_UTILIZATION_POLICY
            value: "{policy}\""""

# Set from --core-policy in main(). A list so both render functions can read it
# without adding a parameter to two signatures and every call site.
CORE_POLICY = [""]

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
    # DEDICATED MEANS THE WHOLE CARD: no gpucores annotation at all, which is
    # how HAMi expresses "no core allocation" (hami_gpu_core_allocated_ratio
    # reads 0). This is a deliberate choice of CROSS-TIER COMPARABILITY over
    # removing a confound.
    #
    # The confound is real: dedicated-versus-shared then differs in two ways at
    # once -- co-tenancy AND whether a core token was requested -- so every
    # co-location cost is an upper bound rather than a measurement. T1 through
    # T3 are reported that way and T4 onward must match them, because comparing
    # a tier's dedicated baseline across tiers is worth more than tightening one
    # comparison inside a tier.
    #
    # THE CONFOUND IS STILL OPEN. Two previous versions of this comment claimed
    # it was inert. Both were wrong:
    #
    #   1. "hami_cores reads 100 while pods request 50" -- not evidence. 100 is
    #      the SUM of the two 50s, which is what a device gauge should report.
    #   2. "T5's 7B got 85.9% of solo throughput at gpucores=50, so 50 does not
    #      bind" -- not evidence either. LLM decode is memory-bandwidth-bound,
    #      so a tenant held to 50% of SM time can keep far more than 50% of its
    #      tokens/s. And HAMi documents compute limiting as a CONVERGING RATE
    #      LIMITER (spin-wait before kernel submission), not a hard cap, so the
    #      percentage does not predict a throughput ratio at all.
    #
    # T7 does not settle it either: its hami arm runs gpucores=100, where there
    # is nothing to limit. So no run has exercised the limiter at a binding
    # setting, and every dedicated-vs-shared figure remains an upper bound.
    #
    # To settle it:  run.py --test T7 --gpucores llm=50   (one tenant, no
    # neighbour, so any delta IS the limiter). See FINDINGS 5.14.
    #
    # --dedicated-cores share  gives the tight control instead, at the cost of
    # comparability with T1-T3. Use it for a within-tier study, not the matrix.
    lines = ['            nvidia.com/gpu: "1"',
             '            nvidia.com/gpumem: "%d"' % m["gpumem"]]
    if arm in ("hami", "hami-mps") or m.get("dedicated_cores") == "share":
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
    # 2 spaces: spec.annotations is a sibling of spec.model, not nested in it.
    pin = pin_annotations(pin, 2)
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
    if CORE_POLICY[0] and arm in ("hami", "hami-mps"):
        env += CORE_POLICY_ENV.format(policy=CORE_POLICY[0])
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
spec:{pin}
  model:
    name: {m['repo']}
    uri: pvc://{PVC}/models/{m.get('model_dir', role)}
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
    if CORE_POLICY[0] and arm in ("hami", "hami-mps"):
        env += CORE_POLICY_ENV.format(policy=CORE_POLICY[0])
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
    ap.add_argument("--dedicated-cores", choices=("whole", "share"),
                    default="whole",
                    help="whole (default): the dedicated arm requests no "
                         "gpucores, i.e. the whole card, matching T1-T3. "
                         "share: it requests the same gpucores as the sharing "
                         "arms, which removes the two-variable confound but "
                         "breaks comparability with T1-T3.")
    ap.add_argument("--gpumem", default="",
                    help="override a role's VRAM quota, e.g. llm2=11776. For "
                         "18-vram-probe.sh, which measures what a model needs "
                         "instead of inferring it from weights on disk.")
    ap.add_argument("--gpucores", default="",
                    help="override a role's gpucores, e.g. llm=80,llm2=80. "
                         "Normally 100/len(scenario) so the co-tenants sum to "
                         "exactly 100. This exists to offer HAMi MORE than 100 "
                         "and find out whether it refuses -- see "
                         "26-gpucores-probe.sh. T5 showed the value is not a "
                         "compute bound (gpucores=50 still returned 85.9%% of "
                         "solo throughput), so whether the SUM is admission-"
                         "controlled is the only open question about it.")
    ap.add_argument("--core-policy", default="",
                    choices=("", "default", "force", "disable"),
                    help="GPU_CORE_UTILIZATION_POLICY on the HAMi arms. Empty "
                         "(the default here) sets nothing, which is what every "
                         "round-two tier ran, so HAMi applies its own default "
                         "policy. 'force' is the only value documented to "
                         "always limit core utilisation below gpucores, and is "
                         "what ROUND ONE used -- so the rounds are not "
                         "comparable until this is matched. See FINDINGS 5.14c.")
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

    gpumem_over = {}
    for kv in filter(None, a.gpumem.split(",")):
        k, v = kv.split("=")
        gpumem_over[k] = int(v)

    CORE_POLICY[0] = a.core_policy

    gpucores_over = {}
    for kv in filter(None, a.gpucores.split(",")):
        k, v = kv.split("=")
        gpucores_over[k] = int(v)

    # gpucores is the scheduler's packing token: split the card between the
    # co-tenants so HAMi's admission arithmetic reflects the intended density.
    share = max(1, 100 // max(1, len(SCENARIOS[a.scenario])))

    os.makedirs(a.out, exist_ok=True)
    written, budget, clamps = [], 0, []
    for role in roles:
        m = dict(MODELS[role])
        override = SCENARIO_QUOTAS.get(a.scenario, {}).get(role)
        if override:
            m["gpumem"] = override
        m["gpucores"] = share
        m["dedicated_cores"] = a.dedicated_cores
        if a.arm == "dedicated" and a.dedicated_full_card:
            m["gpumem"] = CARD_MIB - 512
        if role in gpumem_over:
            m["gpumem"] = gpumem_over[role]
        if role in gpucores_over:
            m["gpucores"] = gpucores_over[role]

        # The quota is final at this point, so the utilisation ceiling can be
        # resolved against it. Doing this AFTER every override is the whole
        # point: llm2 crashlooped because a scenario override cut its quota
        # while util stayed at the value that suited a much larger one.
        m["util"], cap, clamped = effective_util(m)
        if clamped:
            clamps.append((role, MODELS[role]["util"], m["util"], m["gpumem"]))
        peak = m.get("profile_peak_mib")
        if peak and peak >= m["gpumem"]:
            sys.exit(
                "%s: vLLM's profiling run peaks at %d MiB and the quota is "
                "%d. That run happens BEFORE the KV cache is sized and is not "
                "bounded by --gpu-memory-utilization, so no utilisation value "
                "can make this fit. Raise the quota above %d in "
                "SCENARIO_QUOTAS, or reduce max_num_seqs / max_model_len / "
                "the multimodal limits." % (role, peak, m["gpumem"], peak))
        if m["kind"] == "chat":
            # vLLM sizes KV as (util x quota) MINUS the peak it measured during
            # profiling -- not minus the weights. For a text model those are
            # close enough that subtracting weights works. For vl they are not:
            # at an 8192 MiB quota this check saw 6939 - 4220 = 2719 MiB and
            # passed, while vLLM actually allocated 133 MiB, giving 1.21x
            # concurrency. The pod came up healthy and would have thrashed
            # through the entire tier, and the resulting numbers would have
            # been reported as a cost of sharing.
            floor_against = max(m["weights_mib"], m.get("profile_peak_mib") or 0)
            kv = m["gpumem"] * m["util"] - floor_against
            if kv < KV_FLOOR_MIB:
                what = ("a %d MiB profiling peak" % floor_against
                        if floor_against != m["weights_mib"]
                        else "%d MiB of weights" % floor_against)
                sys.exit(
                    "%s: a %d MiB quota leaves %d MiB of KV cache after %s, "
                    "below the %d MiB floor. It will start and then thrash, "
                    "which is worse than crashing because the run completes "
                    "and the numbers look real. Raise the quota in "
                    "SCENARIO_QUOTAS." % (role, m["gpumem"], kv, what,
                                          KV_FLOOR_MIB))
        pct = pcts.get(role, 100)
        as_kserve = m["kind"] == "chat" or a.pooling_via == "kserve"
        body = (render_llm(role, m, a.arm, pct, a.pin_gpu, a.router) if as_kserve
                else render_pooling(role, m, a.arm, pct, a.pin_gpu))
        path = os.path.join(a.out, "r2-%s-%s.yaml" % (a.arm, role))
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(body)
        written.append(path)
        budget += m["gpumem"]
        print("%-10s %-34s gpumem %5d  util %.3f  cores %3d  mps %3d%%  -> %s"
              % (role, m["repo"], m["gpumem"], m["util"], m["gpucores"], pct,
                 os.path.basename(path)))

    for role, was, now, q in clamps:
        print("  util clamped for %s: %.2f -> %.3f, because %.2f x %d plus %d "
              "MiB of CUDA context exceeds the %d MiB HAMi cap"
              % (role, was, now, was, q,
                 MODELS[role].get("outside_mib", OUTSIDE_DEFAULT), q))

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
