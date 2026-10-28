#!/usr/bin/env bash
set -euo pipefail
ROOT=/inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/zhenhaowang
PYTHON=/inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/.envs/gpu-train-stable-hdd3/bin/python
RUN_ID=${1:-mechanism_v1}
GPU_COUNT=${2:-8}
case "$GPU_COUNT" in
  4) DEFAULT_VISIBLE=0,1,2,3; WALL_CAP=10800 ;;
  8) DEFAULT_VISIBLE=0,1,2,3,4,5,6,7; WALL_CAP=5400 ;;
  *) echo 'ERROR: GPU count must be 4 or 8'; exit 2 ;;
esac
[[ "$RUN_ID" =~ ^[A-Za-z0-9_-]+$ ]] || { echo 'ERROR: invalid fresh run ID'; exit 2; }
cd "$ROOT"
command -v flock >/dev/null
command -v timeout >/dev/null
exec 9>/tmp/zhenhaowang_dual_retry_v1.lock
flock -n 9 || { echo 'ERROR: another project job owns this node'; exit 3; }
exec 8>/tmp/zhenhaowang_rain_blur_image_moe_v1.lock
flock -n 8 || { echo 'ERROR: Rain/Blur job owns this node'; exit 3; }
if pgrep -f '[t]rain_foundir_novel_ppt_|[t]rain_foundir_local_action_energy_|[q]ualify_foundir_flux_energy_v1.py (cache|train)' >/dev/null; then
  echo 'ERROR: old job still exists on this node; no automatic kill'; exit 3
fi
LOG="$ROOT/logs/foundir_flux_tail_oracle_v1_${RUN_ID}.log"
OUTPUT="$ROOT/checkpoints/foundir_flux_tail_oracle_v1/$RUN_ID"
[[ ! -e "$LOG" && ! -e "$OUTPUT" ]] || { echo 'ERROR: preserve previous artifacts; use a fresh ID'; exit 3; }
mkdir -p logs
(set -o noclobber; : > "$LOG") || exit 3
exec > >(tee -a "$LOG") 2>&1
STARTED=$SECONDS
trap 'status=$?; elapsed=$((SECONDS-STARTED)); echo "TAIL_ORACLE_LAUNCH_END status=$status elapsed_seconds=$elapsed allocated_gpu_seconds=$((GPU_COUNT*elapsed)) utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"' EXIT
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$DEFAULT_VISIBLE}"
export OMP_NUM_THREADS=2 NCCL_NVLS_ENABLE=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCH_HOME="$ROOT/models/torch"
export PYTHONPATH="$ROOT/third_party/flux2/src:$ROOT"
SCRIPT=smoke/diagnose_foundir_flux_tail_oracle_v1.py
echo "TAIL_ORACLE_START run_id=$RUN_ID utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "BUDGET_CAP gpus=$GPU_COUNT wall_seconds=$WALL_CAP allocated_gpu_hours=12 plus_at_most_30s_cleanup; not_a_runtime_prediction; GT_train_only; no_training"
timeout --signal=TERM --kill-after=30s "$WALL_CAP" "$PYTHON" -u "$SCRIPT" hardware --gpus "$GPU_COUNT"
REMAINING=$((WALL_CAP-(SECONDS-STARTED)))
((REMAINING>0)) || exit 124
timeout --signal=TERM --kill-after=30s "$REMAINING" "$PYTHON" -u "$SCRIPT" prepare --run-id "$RUN_ID" --gpus "$GPU_COUNT"
REMAINING=$((WALL_CAP-(SECONDS-STARTED)))
((REMAINING>0)) || exit 124
timeout --signal=TERM --kill-after=30s "$REMAINING" "$PYTHON" -u -m torch.distributed.run \
  --standalone --nproc_per_node="$GPU_COUNT" --max-restarts=0 "$SCRIPT" run --run-id "$RUN_ID" --gpus "$GPU_COUNT"
REMAINING=$((WALL_CAP-(SECONDS-STARTED)))
((REMAINING>0)) || exit 124
timeout --signal=TERM --kill-after=30s "$REMAINING" "$PYTHON" -u "$SCRIPT" verify --run-id "$RUN_ID" --gpus "$GPU_COUNT"
