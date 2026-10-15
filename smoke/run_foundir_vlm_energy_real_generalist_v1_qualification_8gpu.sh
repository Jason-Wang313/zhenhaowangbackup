#!/usr/bin/env bash
set -euo pipefail

ROOT=/inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/zhenhaowang
PYTHON=/inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/.envs/gpu-train-stable-hdd3/bin/python
RUN_ID="${VLM_ENERGY_REAL_RUN_ID:-qualification_8gpu_v1}"
INIT="${VLM_ENERGY_REAL_INIT_CHECKPOINT:?set VLM_ENERGY_REAL_INIT_CHECKPOINT}"
MASTER_PORT="${MASTER_PORT:-29651}"
LOG="$ROOT/logs/foundir_vlm_energy_real_${RUN_ID}.log"

[[ "${CUDA_VISIBLE_DEVICES:-}" == "0,1,2,3,4,5,6,7" ]] || {
  echo "ERROR: expose exactly GPUs 0-7" >&2
  exit 2
}

cd "$ROOT"
mkdir -p "$ROOT/logs"
export TORCH_HOME="$ROOT/models/torch"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export VLM_ENERGY_REAL_RUN_ID="$RUN_ID"
export VLM_ENERGY_REAL_INIT_CHECKPOINT="$INIT"
export VLM_ENERGY_REAL_MAX_STEPS="${VLM_ENERGY_REAL_MAX_STEPS:-4}"
export VLM_ENERGY_REAL_MAX_VALIDATION_RECORDS="${VLM_ENERGY_REAL_MAX_VALIDATION_RECORDS:-264}"

exec > >(tee -a "$LOG") 2>&1
echo "VLM_ENERGY_REAL_QUALIFICATION_START run_id=$RUN_ID init=$INIT utc=$(date -u +%FT%TZ)"
"$PYTHON" -m torch.distributed.run \
  --standalone --nproc_per_node=8 --master_port="$MASTER_PORT" \
  smoke/train_foundir_vlm_energy_real_generalist_v1.py
echo "VLM_ENERGY_REAL_QUALIFICATION_END status=0 utc=$(date -u +%FT%TZ)"
