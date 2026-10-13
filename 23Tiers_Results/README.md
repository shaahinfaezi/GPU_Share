# 23 Tiers: recorded benchmark results

This bundle contains runs with saved inference replay results from the local
research workspace. Successful and adverse outcomes are both retained. Each run
directory is copied in full, including its logs, metrics, request records,
configuration, and any diagnostic context present. Merely planned workloads,
empty attempts and local dry runs are excluded. A run with a completed replay
can still have missing or failed other arms; consult ARMS.csv before comparing it.

## Start here

- **RUNS.csv**: one row per run; the list of recorded experiments.
- **ARMS.csv**: one row per recorded replay, with successful/failed request counts,
  CSV cross-checks, recorded status, telemetry availability and review flags.
- **METRICS.csv**: Prometheus metric keys with actual finite samples, by run/arm.
  Missing metrics are not inferred. Inspect original labels and timestamps.
- **FILES.csv**: every copied artifact, original source path, source SHA-256,
  published SHA-256 and whether the copy is byte-identical.
- **report-source-manifests/**: existing report asset provenance. Canonical
  selection comes from the base report builder; supplemental graphs may use other
  runs listed in these manifests. Selection does not certify a result's validity.

## Scope and interpretation

The report's experiment 23 is the Qwen 7B TP2+PP2 run
`S-qwen7b-attempt-20260925-134248`. No completed T19 replay was found in the supplied
workspace. This folder name does not assert that all 23 planned tiers completed.
The split run includes separate short memory screens; these are not full
benchmark arms. Prometheus was disabled for this split run; its client, GPU and
host samples remain available.

Recorded requests are usable to chart successes, errors, arrivals and observed
latencies, including bad outcomes. This is a source inventory, not a declaration
that every replay is scientifically comparable. Check trace identity, model,
rate, duration, placement, MPS state, subrun status and fault metadata before
comparing arms. T10 includes intentional fault injections; t10.json records what
actually happened. A replay with errors is not an all-success capacity result.
The dedicated model baselines remain separate; they have not been summed.

## Artifact guide

- `replay.requests.csv`: request-level scheduled/send/first/last times, tokens,
  warm-up markers and success/error outcomes. Exclude warm-up rows for live plots.
- `replay.summary.json`: recorded aggregates and per-service latency/throughput.
- `replay.itl.json`: recorded streaming timing samples; inspect its schema before
  interpreting chunk gaps as individual-token latency.
- `prom.json`: saved queries/series; missing, empty and NaN series are not zero.
- `gpu.csv`, `host.csv`: sampled device and host measurements, when collected.
- `trace.json`: workload schedule and prompts; a planned trace alone is not
  evidence that requests executed.
- `events.jsonl`, logs, `subrun.json`, `t10.json`, `arm.json`, `quality.json`,
  placement and pod snapshots: completion, fault, serving and placement evidence.
- `rendered/`: exact saved deployment inputs for that run, when available.

## Integrity and download

Workload traces, streaming timings, request CSVs and Prometheus snapshots use
Git LFS. After cloning, run `git lfs pull` to fetch their contents. An archive download may otherwise contain pointer files.
All measured result files are byte-identical to their originals. Some source
ShareGPT prompts contained credential-like strings/private-key blocks; only
these strings were redacted in publication copies of traces. Original local
files were not changed. PUBLICATION_REDACTIONS.json lists affected JSON paths,
and FILES.csv records both hashes. Redacted traces retain schedules and recorded
token metadata, but are not byte-identical replay inputs. Original report source
hashes refer to the original files, not the sanitized copies.

Verify the published artifacts from this directory:

```python
import csv, hashlib
from pathlib import Path
with open('FILES.csv', newline='', encoding='utf-8') as index:
    for item in csv.DictReader(index):
        with Path(item['path']).open('rb') as artifact:
            actual = hashlib.file_digest(artifact, 'sha256').hexdigest()
        assert actual == item['published_sha256'], item['path']
print('All published artifact hashes match.')
```

## Recorded runs

Failure counts below count replay arms with one or more request errors, not
failed requests. See ARMS.csv for individual request counts and review flags.

| Tier | Run directory | Recorded replays | Replays with request errors |
| --- | --- | ---: | ---: |
| T1 | [T1-20260826-175322](T1-20260826-175322/) | 4 | 3 |
| T1 | [T1-20260902-144618](T1-20260902-144618/) | 4 | 0 |
| T2 | [T2-20260902-152143](T2-20260902-152143/) | 4 | 0 |
| T2 | [T2-20260903-115145](T2-20260903-115145/) | 4 | 3 |
| T2 | [T2-20260903-122626](T2-20260903-122626/) | 4 | 3 |
| T2 | [T2-20260906-102419](T2-20260906-102419/) | 4 | 0 |
| T2 | [T2-20260909-161151](T2-20260909-161151/) | 4 | 0 |
| T2 | [T2-20260909-195331](T2-20260909-195331/) | 1 | 0 |
| T2 | [T2-20260910-172940](T2-20260910-172940/) | 1 | 0 |
| T2 | [T2-20260911-014443](T2-20260911-014443/) | 4 | 0 |
| T3 | [T3-20260902-155701](T3-20260902-155701/) | 4 | 0 |
| T4 | [T4-20260903-130204](T4-20260903-130204/) | 4 | 1 |
| T4 | [T4-20260904-100610](T4-20260904-100610/) | 4 | 1 |
| T5 | [T5-20260904-103901](T5-20260904-103901/) | 4 | 4 |
| T5 | [T5-20260906-075026](T5-20260906-075026/) | 4 | 4 |
| T5 | [T5-20260911-011315](T5-20260911-011315/) | 4 | 4 |
| T6 | [T6-20260911-000539](T6-20260911-000539/) | 4 | 4 |
| T7 | [T7-20260904-111030](T7-20260904-111030/) | 3 | 0 |
| T7 | [T7-20260905-113832](T7-20260905-113832/) | 3 | 0 |
| T7 | [T7-20260905-133623](T7-20260905-133623/) | 3 | 0 |
| T7 | [T7-20260906-083326](T7-20260906-083326/) | 3 | 0 |
| T7 | [T7-20260906-172226](T7-20260906-172226/) | 3 | 0 |
| T7 | [T7-20260906-225515](T7-20260906-225515/) | 3 | 1 |
| T7 | [T7-20260909-191616](T7-20260909-191616/) | 3 | 0 |
| T7 | [T7-20260910-233846](T7-20260910-233846/) | 3 | 0 |
| T8 | [T8-20260906-111040](T8-20260906-111040/) | 4 | 3 |
| T9 | [T9-hami-20260910-225617](T9-hami-20260910-225617/) | 1 | 1 |
| T10 | [T10-global-fault-hami-20260910-135228](T10-global-fault-hami-20260910-135228/) | 1 | 0 |
| T10 | [T10-global-fault-hami-20260910-220052](T10-global-fault-hami-20260910-220052/) | 1 | 0 |
| T10 | [T10-global-fault-hami-20260911-155223](T10-global-fault-hami-20260911-155223/) | 1 | 0 |
| T10 | [T10-mps-kill-hami-20260910-061056](T10-mps-kill-hami-20260910-061056/) | 1 | 0 |
| T10 | [T10-mps-kill-hami-mps-20260905-213058](T10-mps-kill-hami-mps-20260905-213058/) | 1 | 1 |
| T10 | [T10-tenant-fault-hami-20260906-185601](T10-tenant-fault-hami-20260906-185601/) | 1 | 1 |
| T10 | [T10-tenant-fault-hami-20260907-205634](T10-tenant-fault-hami-20260907-205634/) | 1 | 0 |
| T10 | [T10-tenant-fault-hami-20260909-183833](T10-tenant-fault-hami-20260909-183833/) | 1 | 0 |
| T10 | [T10-tenant-fault-hami-mps-20260907-092933](T10-tenant-fault-hami-mps-20260907-092933/) | 1 | 0 |
| T10 | [T10-tenant-fault-hami-mps-20260907-100943](T10-tenant-fault-hami-mps-20260907-100943/) | 1 | 0 |
| T10 | [T10-tenant-fault-hami-mps-20260909-184811](T10-tenant-fault-hami-mps-20260909-184811/) | 1 | 0 |
| T10 | [T10-tenant-kill-hami-20260910-223803](T10-tenant-kill-hami-20260910-223803/) | 1 | 1 |
| T10 | [T10-tenant-kill-hami-mps-20260905-213939](T10-tenant-kill-hami-mps-20260905-213939/) | 1 | 1 |
| T10 | [T10-vram-escape-hami-20260905-211333](T10-vram-escape-hami-20260905-211333/) | 1 | 0 |
| T10 | [T10-vram-escape-hami-20260906-052924](T10-vram-escape-hami-20260906-052924/) | 1 | 0 |
| T10 | [T10-vram-escape-hami-mps-20260905-212222](T10-vram-escape-hami-mps-20260905-212222/) | 1 | 0 |
| T10 | [T10-vram-escape-hami-mps-20260906-053821](T10-vram-escape-hami-mps-20260906-053821/) | 1 | 0 |
| T10 | [T10-vram-hog-hami-20260906-191559](T10-vram-hog-hami-20260906-191559/) | 1 | 0 |
| T10 | [T10-vram-hog-hami-20260907-000230](T10-vram-hog-hami-20260907-000230/) | 1 | 0 |
| T10 | [T10-vram-hog-hami-20260909-185717](T10-vram-hog-hami-20260909-185717/) | 1 | 0 |
| T10 | [T10-vram-hog-hami-20260910-171129](T10-vram-hog-hami-20260910-171129/) | 1 | 0 |
| T10 | [T10-vram-hog-hami-20260910-221956](T10-vram-hog-hami-20260910-221956/) | 1 | 0 |
| T10 | [T10-vram-hog-hami-20260911-161115](T10-vram-hog-hami-20260911-161115/) | 1 | 0 |
| T10 | [T10-vram-hog-hami-20260912-121324](T10-vram-hog-hami-20260912-121324/) | 1 | 0 |
| T10 | [T10-vram-hog-hami-mps-20260912-122229](T10-vram-hog-hami-mps-20260912-122229/) | 1 | 0 |
| T11 | [T11-20260906-092256](T11-20260906-092256/) | 4 | 3 |
| T12 | [T12-20260911-024916](T12-20260911-024916/) | 1 | 1 |
| T12 | [T12-20260911-065446](T12-20260911-065446/) | 1 | 1 |
| T13 | [T13-20260906-114453](T13-20260906-114453/) | 4 | 4 |
| T13 | [T13-20260911-021908](T13-20260911-021908/) | 4 | 4 |
| T14 | [T14-20260906-193411](T14-20260906-193411/) | 4 | 4 |
| T14 | [T14-20260906-233012](T14-20260906-233012/) | 4 | 4 |
| T14 | [T14-20260908-115149](T14-20260908-115149/) | 4 | 4 |
| T14 | [T14-20260908-122621](T14-20260908-122621/) | 4 | 4 |
| T14 | [T14-20260910-050241](T14-20260910-050241/) | 4 | 4 |
| T14 | [T14-20260910-053547](T14-20260910-053547/) | 4 | 4 |
| T14 | [T14-20260911-004001](T14-20260911-004001/) | 4 | 4 |
| T15 | [T15-20260906-182642](T15-20260906-182642/) | 4 | 4 |
| T16 | [T16-20260906-175130](T16-20260906-175130/) | 4 | 0 |
| T16 | [T16-20260910-042857](T16-20260910-042857/) | 4 | 0 |
| T17 | [T17-20260912-123151](T17-20260912-123151/) | 4 | 4 |
| T18 | [T18-20260910-141059](T18-20260910-141059/) | 3 | 1 |
| T18 | [T18-20260912-070817](T18-20260912-070817/) | 3 | 1 |
| T20 | [T20-20260912-073411](T20-20260912-073411/) | 4 | 4 |
| T21 | [T21-20260912-081009](T21-20260912-081009/) | 4 | 4 |
| T22 | [T22-20260912-085910](T22-20260912-085910/) | 3 | 3 |
| T23 | [S-qwen7b-attempt-20260925-134248](S-qwen7b-attempt-20260925-134248/) | 6 | 0 |
