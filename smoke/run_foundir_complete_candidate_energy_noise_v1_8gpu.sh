#!/usr/bin/env bash
set -euo pipefail

ROOT=/inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/zhenhaowang
PYTHON=/inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/.envs/gpu-train-stable-hdd3/bin/python
RUN_ID=${COMPLETE_CANDIDATE_ENERGY_RUN_ID:-qualification_v1}
TOTAL_STEPS=${COMPLETE_CANDIDATE_ENERGY_TOTAL_STEPS:-25}
LOG="$ROOT/logs/foundir_complete_candidate_energy_noise_v1_${RUN_ID}.log"

cd "$ROOT"
if [[ "${CUDA_VISIBLE_DEVICES:-}" != "0,1,2,3,4,5,6,7" ]]; then
  echo "ERROR: expose exactly local GPUs 0-7" >&2
  exit 2
fi
case "$TOTAL_STEPS" in
  25|1706|3412|5118|6824|8530) ;;
  *) echo "ERROR: total steps must be a frozen milestone" >&2; exit 2 ;;
esac
export TORCH_HOME="$ROOT/models/torch"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export COMPLETE_CANDIDATE_ENERGY_RUN_ID="$RUN_ID"
export COMPLETE_CANDIDATE_ENERGY_TOTAL_STEPS="$TOTAL_STEPS"
mkdir -p "$ROOT/logs" "$ROOT/checkpoints/foundir_complete_candidate_energy_noise_v1"

echo "FOUNDIR_COMPLETE_CANDIDATE_ENERGY_START utc=$(date -u +%FT%TZ) run_id=$RUN_ID total_steps=$TOTAL_STEPS allocated_gpus=8" | tee -a "$LOG"
echo "SCOPE Noise_only train_GT_risk_targets inference_no_GT frozen_proposal_and_basis no_protected_data" | tee -a "$LOG"
sha256sum \
  configs/foundir_complete_candidate_energy_noise_v1.json \
  lowlight_restoration/complete_candidate_energy.py \
  smoke/train_foundir_complete_candidate_energy_noise_v1_ddp.py | tee -a "$LOG"
set +e
"$PYTHON" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 \
  smoke/train_foundir_complete_candidate_energy_noise_v1_ddp.py 2>&1 | tee -a "$LOG"
status=${PIPESTATUS[0]}
set -e
echo "FOUNDIR_COMPLETE_CANDIDATE_ENERGY_END status=$status utc=$(date -u +%FT%TZ)" | tee -a "$LOG"
exit "$status"
