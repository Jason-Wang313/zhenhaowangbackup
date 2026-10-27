#!/usr/bin/env bash
set -euo pipefail
ROOT=/inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/zhenhaowang
PYTHON=/inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/.envs/gpu-train-stable-hdd3/bin/python
RUN_ID=${1:-stage1_v1}
RESUME_RECEIPT=${2:-}
[[ "$RUN_ID" =~ ^[A-Za-z0-9_-]+$ ]] || { echo 'ERROR: invalid fresh run ID'; exit 2; }
if [[ -n "$RESUME_RECEIPT" && "${PPT_PREVIOUS_RUN_STOPPED:-}" != 1 ]]; then
  echo 'ERROR: confirm previous PPT run stopped before resuming'; exit 2
fi
cd "$ROOT"
command -v flock >/dev/null
command -v timeout >/dev/null
exec 9>/tmp/zhenhaowang_dual_retry_v1.lock
flock -n 9 || { echo 'ERROR: another PPT/MoE job uses this node'; exit 3; }
exec 8>/tmp/zhenhaowang_rain_blur_image_moe_v1.lock
flock -n 8 || { echo 'ERROR: another Rain/Blur job uses this node'; exit 3; }
LOG="$ROOT/logs/foundir_novel_ppt_full_data_stage1_v1_${RUN_ID}.log"
OUTPUT="$ROOT/checkpoints/foundir_novel_ppt_full_data_stage1_v1/$RUN_ID"
[[ ! -e "$LOG" && ! -e "$OUTPUT" ]] || { echo 'ERROR: preserve existing run; use a fresh ID'; exit 3; }
mkdir -p logs
(set -o noclobber; : > "$LOG") || exit 3
exec > >(tee -a "$LOG") 2>&1
STARTED=$SECONDS
trap 'status=$?; echo "PPT_FULL_DATA_LAUNCH_END status=$status elapsed_seconds=$((SECONDS-STARTED)) utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"' EXIT
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export OMP_NUM_THREADS=2 NCCL_NVLS_ENABLE=0 NCCL_P2P_DISABLE=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCH_HOME="$ROOT/models/torch"
[[ "$CUDA_VISIBLE_DEVICES" == 0,1,2,3,4,5,6,7 ]] || { echo 'ERROR: expose exactly GPUs 0-7'; exit 2; }
echo "PPT_FULL_DATA_LAUNCH_START run_id=$RUN_ID stop_step=1706 lr_horizon=8530 utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo 'PRELAUNCH_ESTIMATE full_stage_wall_hours=10.5_to_13 allocated_gpu_hours=84_to_104 no_auto_extension=1'
"$PYTHON" -u -c 'import torch; p=[torch.cuda.get_device_properties(i) for i in range(torch.cuda.device_count())]; assert len(p)==8, f"Expected 8 GPUs, got {len(p)}"; assert len({x.name for x in p})==1 and any(x in p[0].name for x in ("H100","H200")); assert all(x.total_memory>=75*1024**3 for x in p); assert all(torch.cuda.mem_get_info(i)[0]>.8*x.total_memory for i,x in enumerate(p)), "GPU occupied"; print("GPU_PASS",p[0].name,"count=8",flush=True)'
SCRIPT=smoke/train_foundir_novel_ppt_full_data_stage1_v1_ddp.py
ARGS=(--run-id "$RUN_ID")
if [[ -n "$RESUME_RECEIPT" ]]; then ARGS+=(--resume-receipt "$RESUME_RECEIPT"); fi
"$PYTHON" -u "$SCRIPT" "${ARGS[@]}" --preflight
timeout --signal=TERM --kill-after=30s 180s "$PYTHON" -u -m torch.distributed.run \
  --standalone --nproc_per_node=8 --max-restarts=0 smoke/cure_nccl_rehearsal_v1.py
echo 'PPT_FULL_DATA_COMM_PASS'
"$PYTHON" -u -m torch.distributed.run --standalone --nproc_per_node=8 --max-restarts=0 \
  "$SCRIPT" "${ARGS[@]}"
