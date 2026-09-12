#!/usr/bin/env bash
# T22: three tenants, run safely rather than marginally.
#
# WHY THIS EXISTS SEPARATELY FROM 33-final-series.sh
# --------------------------------------------------
# 33-final-series.sh had a real bug: it called 28-series-b.sh, which
# BACKGROUNDS ITSELF AND RETURNS. The wrapper's block therefore finished
# immediately, its EXIT trap fired, and the observability stack was restored
# within seconds of launch -- handing back the very memory the tier needed,
# 90 minutes before the series ended. This script does not delegate to a
# self-backgrounding wrapper: it calls run.py directly and blocks, so the
# trap fires when the work is actually done.
#
# THE MEMORY PROBLEM, AND WHY SWAP IS HONEST HERE
# ------------------------------------------------
# Three tenants need ~8,600 MiB raw. The node frees to ~8,156 MiB with the
# observability stack parked, and the harness adds 1,024 MiB of safety
# headroom on top -- headroom that exists because T4's OOM killed two vLLM
# engines and Prometheus mid-run.
#
# Rather than quietly lowering that headroom and hoping, this adds swap as a
# BACKSTOP and then checks whether it was used. Swap converts "the kernel
# kills an engine and the run is lost" into "the run is slower and you can
# measure that it was". If SwapUsed moves meaningfully, the tier is VOID and
# this script says so -- a run that only fit because it swapped has measured
# swapping, not three-tenant co-tenancy.
#
#   ./36-three-body.sh
set -uo pipefail

cd "$(dirname "$0")"
STAMP="$(date -u +%Y%m%d-%H%M%S)"
LOG="logs/three-body-${STAMP}.log"
mkdir -p logs

swap_used() { awk '/SwapTotal/{t=$2}/SwapFree/{f=$2}END{printf "%d", (t-f)/1024}' /proc/meminfo; }
avail()     { awk '/MemAvailable/{printf "%d", $2/1024}' /proc/meminfo; }

SWAP_BEFORE=0

cleanup() {
  echo | tee -a "${LOG}"
  echo "=== teardown" | tee -a "${LOG}"
  local after; after="$(swap_used)"
  echo "  swap used: ${SWAP_BEFORE} MiB before -> ${after} MiB after" | tee -a "${LOG}"
  if [[ "$(( after - SWAP_BEFORE ))" -gt 64 ]]; then
    echo "  *** VOID: the run paged $(( after - SWAP_BEFORE )) MiB." | tee -a "${LOG}"
    echo "  *** Latency figures are contaminated. Report the density ceiling," | tee -a "${LOG}"
    echo "  *** not these numbers." | tee -a "${LOG}"
  else
    echo "  swap essentially untouched -- the tier fit in real RAM." | tee -a "${LOG}"
  fi
  ./35-swapfile.sh off 2>&1 | sed 's/^/  /' | tee -a "${LOG}"
  ./31-freeram.sh up   2>&1 | sed 's/^/  /' | tee -a "${LOG}"
  echo "=== done. log: ${LOG}" | tee -a "${LOG}"
}
trap cleanup EXIT INT TERM

{
  echo "=== T22 three-body ${STAMP}"
  echo "=== dense3b: llm2a + llm2b + embed-small"
  echo "===   two IDENTICAL decoders at equal depth, plus one shallow tenant."
  echo "===   VRAM 9984 of 12288 (81.2%). Host RAM is the constraint."
  echo

  echo "--- step 1: park the observability stack"
  ./31-freeram.sh down
  echo

  echo "--- step 2: swap backstop"
  ./35-swapfile.sh on
  SWAP_BEFORE="$(swap_used)"
  echo "  swap used at start: ${SWAP_BEFORE} MiB"
  echo "  host available    : $(avail) MiB"
  echo

  echo "--- step 3: T22 (blocking; the trap must not fire until this returns)"
  python3 -u run.py --test T22
} 2>&1 | tee -a "${LOG}"
