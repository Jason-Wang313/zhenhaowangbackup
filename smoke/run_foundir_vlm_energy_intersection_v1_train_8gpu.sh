#!/usr/bin/env bash
set -euo pipefail

ROOT=/inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/zhenhaowang
PYTHON=/inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/.envs/gpu-train-stable-hdd3/bin/python
RUN_ID=${VLM_ENERGY_TRAIN_RUN_ID:-formal_v1}
LOG="$ROOT/logs/foundir_vlm_energy_intersection_v1_train_${RUN_ID}.log"

cd "$ROOT"
export TORCH_HOME="$ROOT/models/torch"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLM_ENERGY_TRAIN_RUN_ID="$RUN_ID"
if [[ "${CUDA_VISIBLE_DEVICES:-}" != "0,1,2,3,4,5,6,7" ]]; then
  echo "ERROR: set CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7" >&2
  exit 2
fi

echo "FOUNDIR_VLM_ENERGY_TRAIN_START utc=$(date -u +%FT%TZ) run_id=$RUN_ID" | tee -a "$LOG"
echo "PRELAUNCH_ESTIMATE wall_time_hours=50_to_60 gpu_hours=400_to_480 allocated_gpus=8" | tee -a "$LOG"
sha256sum \
  configs/foundir_vlm_energy_intersection_v1.json \
  configs/foundir_vlm_energy_intersection_v1_train.json \
  lowlight_restoration/vlm_energy_restoration.py \
  lowlight_restoration/vlm_energy_objectives.py \
  smoke/train_foundir_vlm_energy_intersection_v1_8gpu.py | tee -a "$LOG"

set +e
"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 \
  smoke/train_foundir_vlm_energy_intersection_v1_8gpu.py 2>&1 | tee -a "$LOG"
status=${PIPESTATUS[0]}
set -e
echo "FOUNDIR_VLM_ENERGY_TRAIN_END status=$status utc=$(date -u +%FT%TZ)" | tee -a "$LOG"
exit "$status"
