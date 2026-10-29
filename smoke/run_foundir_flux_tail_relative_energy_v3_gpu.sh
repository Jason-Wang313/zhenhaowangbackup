#!/usr/bin/env bash
set -euo pipefail
ROOT=/inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/zhenhaowang
PYTHON=/inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/.envs/gpu-train-stable-hdd3/bin/python
MODE=${1:-qualification}
RUN_ID=${2:-qualification_v1}
GPU_COUNT=${3:-8}
QUALIFICATION_RUN_ID=${4:-qualification_v1}
case "$MODE" in
  qualification) CAP8=5400; MILESTONES=(4 8); FINAL_STEP=8 ;;
  trainfit) CAP8=36000; MILESTONES=(50 100 150 200); FINAL_STEP=200 ;;
  *) echo 'ERROR: mode must be qualification or trainfit'; exit 2 ;;
esac
case "$GPU_COUNT" in
  8) VISIBLE=0,1,2,3,4,5,6,7 ;;
  4) VISIBLE=0,1,2,3 ;;
  *) echo 'ERROR: choose one homogeneous 4/8 H100/H200 node'; exit 2 ;;
esac
[[ "$RUN_ID" =~ ^[A-Za-z0-9_-]+$ ]] || exit 2
[[ "$QUALIFICATION_RUN_ID" =~ ^[A-Za-z0-9_-]+$ ]] || exit 2
cd "$ROOT"
OUTPUT="$ROOT/checkpoints/foundir_flux_tail_relative_energy_v3/$RUN_ID"
mkdir -p "$ROOT/checkpoints/foundir_flux_tail_relative_energy_v3" logs
# Shared-filesystem run lock as well as node-local project locks: no duplicate
# cache writers, no job kills, and no cross-node training.
exec 6>"$ROOT/checkpoints/foundir_flux_tail_relative_energy_v3/.active.lock"
flock -n 6 || { echo 'ERROR: another V3 run/cache writer is active'; exit 3; }
exec 7>/tmp/zhenhaowang_flux_tail_switch_v2.lock
flock -n 7 || { echo 'ERROR: another tail-energy job owns the node'; exit 3; }
exec 9>/tmp/zhenhaowang_dual_retry_v1.lock
flock -n 9 || { echo 'ERROR: another project job owns the node'; exit 3; }
exec 8>/tmp/zhenhaowang_rain_blur_image_moe_v1.lock
flock -n 8 || { echo 'ERROR: another project job owns the node'; exit 3; }
export CUDA_VISIBLE_DEVICES="$VISIBLE" OMP_NUM_THREADS=2 NCCL_NVLS_ENABLE=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCH_HOME="$ROOT/models/torch"
export PYTHONPATH="$ROOT/third_party/flux2/src:$ROOT"
LOG="$ROOT/logs/foundir_flux_tail_relative_energy_v3_${RUN_ID}.log"
exec > >(tee -a "$LOG") 2>&1
STARTED=$SECONDS
WALL_CAP=$((CAP8*8/GPU_COUNT))
trap 'status=$?; elapsed=$((SECONDS-STARTED)); echo "TAIL_RELATIVE_V3_END status=$status elapsed_seconds=$elapsed used_gpu_seconds=$((GPU_COUNT*elapsed)) utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"' EXIT
echo "TAIL_RELATIVE_V3_START mode=$MODE run_id=$RUN_ID gpus=$GPU_COUNT utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "TAIL_RELATIVE_V3_BUDGET wall_cap_seconds=$WALL_CAP used_gpu_hours_cap=$((CAP8*8/3600)) no_auto_trainfit no_auto_formal"
SCRIPT=smoke/run_foundir_flux_tail_relative_energy_v3.py
ARGS=(--mode "$MODE" --run-id "$RUN_ID" --gpus "$GPU_COUNT" --qualification-run-id "$QUALIFICATION_RUN_ID")
budget() {
  local remaining=$((WALL_CAP-(SECONDS-STARTED)))
  ((remaining>0)) || return 124
  timeout --signal=TERM --kill-after=30s "$remaining" "$@"
}
phase() { budget "$PYTHON" -u "$SCRIPT" "$@" "${ARGS[@]}"; }
ddp() {
  budget "$PYTHON" -u -m torch.distributed.run --standalone --nproc_per_node="$GPU_COUNT" --max-restarts=0 \
    "$SCRIPT" "$@" "${ARGS[@]}"
}
if [[ -f "$OUTPUT/result.json" ]]; then phase verify; exit 0; fi
[[ ! -e "$OUTPUT/STOP_AFTER_STEP" ]] || { echo 'ERROR: explicit stop marker exists; not resuming'; exit 75; }
phase hardware
phase prepare
ddp cache
ddp mine --generation 0
for target_step in "${MILESTONES[@]}"; do
  ddp train --until "$target_step"
  if ((target_step<FINAL_STEP)); then ddp mine --generation "$target_step"; fi
done
ddp probe
phase verify
