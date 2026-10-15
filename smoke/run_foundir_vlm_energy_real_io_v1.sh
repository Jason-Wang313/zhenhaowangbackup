#!/usr/bin/env bash
set -euo pipefail

ROOT=/inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/zhenhaowang
PYTHON=/inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/.envs/gpu-train-stable-hdd3/bin/python
RUN_ID="${VLM_ENERGY_REAL_IO_RUN_ID:-$(hostname)}"
RECORDS="${VLM_ENERGY_REAL_IO_RECORDS:-4096}"
WORKERS="${VLM_ENERGY_REAL_IO_WORKERS:-32}"

cd "$ROOT"
export VLM_ENERGY_REAL_IO_RUN_ID="$RUN_ID"
export VLM_ENERGY_REAL_IO_RECORDS="$RECORDS"
export VLM_ENERGY_REAL_IO_WORKERS="$WORKERS"
echo "VLM_ENERGY_REAL_IO_START run_id=$RUN_ID records=$RECORDS workers=$WORKERS utc=$(date -u +%FT%TZ)"
"$PYTHON" smoke/benchmark_foundir_vlm_energy_real_io_v1.py
echo "VLM_ENERGY_REAL_IO_END status=0 utc=$(date -u +%FT%TZ)"
