#!/usr/bin/env bash
set -euo pipefail

ROOT=/inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/zhenhaowang
INIT="${VLM_ENERGY_REAL_INIT_CHECKPOINT:-checkpoints/foundir_vlm_energy_intersection_v1/train/formal_3node_v1/model_epoch_018_resume.pt}"
QUAL_RUN_ID="${VLM_ENERGY_REAL_RUN_ID:-qualification_epoch18_v1}"
IO_RUN_ID="${VLM_ENERGY_REAL_IO_RUN_ID:-$(hostname)_real_v1}"

cd "$ROOT"
export VLM_ENERGY_REAL_IO_RUN_ID="$IO_RUN_ID"
export VLM_ENERGY_REAL_IO_RECORDS="${VLM_ENERGY_REAL_IO_RECORDS:-4096}"
export VLM_ENERGY_REAL_IO_WORKERS=32
bash smoke/run_foundir_vlm_energy_real_io_v1.sh

export VLM_ENERGY_REAL_RUN_ID="$QUAL_RUN_ID"
export VLM_ENERGY_REAL_INIT_CHECKPOINT="$INIT"
export VLM_ENERGY_REAL_MAX_STEPS="${VLM_ENERGY_REAL_MAX_STEPS:-4}"
export VLM_ENERGY_REAL_MAX_VALIDATION_RECORDS="${VLM_ENERGY_REAL_MAX_VALIDATION_RECORDS:-264}"
bash smoke/run_foundir_vlm_energy_real_generalist_v1_qualification_8gpu.sh

echo "VLM_ENERGY_REAL_NODE_QUALIFIED io_result=checkpoints/foundir_vlm_energy_real_generalist_v1/io_benchmark/${IO_RUN_ID}.json gpu_result=checkpoints/foundir_vlm_energy_real_generalist_v1/${QUAL_RUN_ID}/result.json"
