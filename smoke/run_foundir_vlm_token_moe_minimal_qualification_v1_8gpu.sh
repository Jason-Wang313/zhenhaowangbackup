#!/usr/bin/env bash
set -euo pipefail

ROOT=/inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/zhenhaowang
PYTHON=/inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/.envs/gpu-train-stable-hdd3/bin/python
CONFIG=configs/foundir_vlm_token_moe_minimal_qualification_v1.json
ARM=${VLM_TOKEN_MOE_ARM:?set VLM_TOKEN_MOE_ARM=image_only or vlm_assisted}

if [[ "$ARM" != image_only && "$ARM" != vlm_assisted ]]; then
  echo "ERROR: VLM_TOKEN_MOE_ARM must be image_only or vlm_assisted" >&2
  exit 2
fi
if [[ "${CUDA_VISIBLE_DEVICES:-}" != "0,1,2,3,4,5,6,7" ]]; then
  echo "ERROR: expose exactly CUDA devices 0-7" >&2
  exit 2
fi

cd "$ROOT"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1

echo "VLM_TOKEN_MOE_QUALIFICATION_START arm=$ARM utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "PRELAUNCH_ESTIMATE wall_time_hours=0.25_to_1.00 allocated_gpu_hours=2.00_to_8.00 gpus=8"
sha256sum "$CONFIG" \
  lowlight_restoration/vlm_guided_moe.py \
  smoke/train_foundir_vlm_token_moe_minimal_qualification_v1_ddp.py

"$PYTHON" -B smoke/train_foundir_vlm_token_moe_minimal_qualification_v1_ddp.py \
  --config "$CONFIG" --arm "$ARM" --preflight-only

"$PYTHON" -m torch.distributed.run \
  --standalone \
  --nproc_per_node=8 \
  smoke/train_foundir_vlm_token_moe_minimal_qualification_v1_ddp.py \
  --config "$CONFIG" \
  --arm "$ARM"

status=$?
echo "VLM_TOKEN_MOE_QUALIFICATION_END arm=$ARM status=$status utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit "$status"
