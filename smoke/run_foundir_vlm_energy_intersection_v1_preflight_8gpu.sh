#!/usr/bin/env bash
set -euo pipefail

ROOT=/inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/zhenhaowang
PYTHON=/inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/.envs/gpu-train-stable-hdd3/bin/python
RUN_ID=${VLM_ENERGY_RUN_ID:-preflight_v1}
LOG="$ROOT/logs/foundir_vlm_energy_intersection_v1_${RUN_ID}.log"

cd "$ROOT"
export TORCH_HOME="$ROOT/models/torch"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
if [[ "${CUDA_VISIBLE_DEVICES:-}" != "0,1,2,3,4,5,6,7" ]]; then
  echo "ERROR: set CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7" >&2
  exit 2
fi

echo "FOUNDIR_VLM_ENERGY_PREFLIGHT_START utc=$(date -u +%FT%TZ)" | tee "$LOG"
echo "PRELAUNCH_ESTIMATE wall_time_hours=0.02_to_0.20 gpu_hours=0.16_to_1.60 allocated_gpus=8" | tee -a "$LOG"
sha256sum \
  configs/foundir_vlm_energy_intersection_v1.json \
  lowlight_restoration/vlm_energy_restoration.py \
  lowlight_restoration/vlm_energy_objectives.py \
  smoke/preflight_foundir_vlm_energy_intersection_v1_8gpu.py | tee -a "$LOG"

set +e
"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 \
  smoke/preflight_foundir_vlm_energy_intersection_v1_8gpu.py 2>&1 | tee -a "$LOG"
status=${PIPESTATUS[0]}
set -e
echo "FOUNDIR_VLM_ENERGY_PREFLIGHT_END status=$status utc=$(date -u +%FT%TZ)" | tee -a "$LOG"
exit "$status"
