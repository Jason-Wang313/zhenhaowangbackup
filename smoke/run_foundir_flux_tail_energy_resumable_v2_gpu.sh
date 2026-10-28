#!/usr/bin/env bash
set -euo pipefail
ROOT=/inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/zhenhaowang
PYTHON=/inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/.envs/gpu-train-stable-hdd3/bin/python
RUN_ID=${1:-qualification_8gpu_v2}
GPU_COUNT=${2:-8}
case "$GPU_COUNT" in
  8) VISIBLE=0,1,2,3,4,5,6,7; WALL_CAP=5400 ;;
  4) VISIBLE=0,1,2,3; WALL_CAP=10800 ;;
  *) echo 'ERROR: use one homogeneous 4/8 H100/H200 node'; exit 2 ;;
esac
[[ "$RUN_ID" =~ ^[A-Za-z0-9_-]+$ ]] || exit 2
cd "$ROOT"
exec 7>/tmp/zhenhaowang_flux_tail_switch_v2.lock
flock -n 7 || { echo 'ERROR: another tail-energy switch/run is active on this node'; exit 3; }
export CUDA_VISIBLE_DEVICES="$VISIBLE" OMP_NUM_THREADS=2 NCCL_NVLS_ENABLE=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCH_HOME="$ROOT/models/torch"
export PYTHONPATH="$ROOT/third_party/flux2/src:$ROOT"
OUTPUT="$ROOT/checkpoints/foundir_flux_tail_energy_resumable_v2/$RUN_ID"
LOG="$ROOT/logs/foundir_flux_tail_energy_resumable_v2_${RUN_ID}.log"
mkdir -p logs
exec > >(tee -a "$LOG") 2>&1
STARTED=$SECONDS
trap 'status=$?; elapsed=$((SECONDS-STARTED)); echo "TAIL_ENERGY_V2_LAUNCH_END status=$status elapsed_seconds=$elapsed used_gpu_seconds=$((GPU_COUNT*elapsed)) utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"' EXIT
echo "TAIL_ENERGY_V2_START run_id=$RUN_ID gpus=$GPU_COUNT utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "TAIL_ENERGY_V2_BUDGET wall_cap_seconds=$WALL_CAP per_attempt_used_gpu_hours_cap=12; preserve_all_steps; no_auto_restart; no_auto_formal"
SCRIPT=smoke/qualify_foundir_flux_tail_energy_resumable_v2.py
run_with_budget() {
  local remaining=$((WALL_CAP-(SECONDS-STARTED)))
  ((remaining>0)) || return 124
  timeout --signal=TERM --kill-after=30s "$remaining" "$@"
}
run_with_budget "$PYTHON" -u "$SCRIPT" prepare --run-id "$RUN_ID" --gpus "$GPU_COUNT"
if [[ -f "$OUTPUT/result.json" ]]; then
  run_with_budget "$PYTHON" -u "$SCRIPT" verify --run-id "$RUN_ID" --gpus "$GPU_COUNT"
  exit 0
fi
[[ ! -e "$OUTPUT/STOP_AFTER_STEP" ]] || { echo 'ERROR: explicit stop marker exists; do not silently resume'; exit 75; }
# This user-authorized switch targets only the superseded four-GPU run.
run_with_budget "$PYTHON" -S -u smoke/manage_flux_tail_energy_jobs_v2.py --output "$OUTPUT" --gpus "$GPU_COUNT"
exec 9>/tmp/zhenhaowang_dual_retry_v1.lock
flock -n 9 || { echo 'ERROR: another project job owns this node'; exit 3; }
exec 8>/tmp/zhenhaowang_rain_blur_image_moe_v1.lock
flock -n 8 || { echo 'ERROR: Rain/Blur job owns this node'; exit 3; }
run_with_budget "$PYTHON" -u "$SCRIPT" hardware --gpus "$GPU_COUNT"
if [[ ! -e "$OUTPUT/training_complete.json" ]]; then
  run_with_budget "$PYTHON" -u -m torch.distributed.run --standalone --nproc_per_node="$GPU_COUNT" --max-restarts=0 \
    "$SCRIPT" train --run-id "$RUN_ID" --gpus "$GPU_COUNT"
fi
run_with_budget "$PYTHON" -u -m torch.distributed.run --standalone --nproc_per_node="$GPU_COUNT" --max-restarts=0 \
  "$SCRIPT" evaluate --run-id "$RUN_ID" --gpus "$GPU_COUNT"
run_with_budget "$PYTHON" -u "$SCRIPT" verify --run-id "$RUN_ID" --gpus "$GPU_COUNT"
