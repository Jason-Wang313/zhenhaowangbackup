#!/usr/bin/env bash
set -euo pipefail
ROOT=/inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/zhenhaowang
PYTHON=/inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/.envs/gpu-train-stable-hdd3/bin/python
RUN_ID=${1:-resume_v1}
[[ "$RUN_ID" =~ ^[A-Za-z0-9_-]+$ ]] || { echo 'ERROR: invalid fresh resume ID'; exit 2; }
[[ "${OLD_MOE_STOPPED:-}" == 1 ]] || { echo 'ERROR: confirm old job is stopped, then set OLD_MOE_STOPPED=1'; exit 2; }
cd "$ROOT"
command -v flock >/dev/null
command -v timeout >/dev/null
exec 9>/tmp/zhenhaowang_dual_retry_v1.lock
flock -n 9 || { echo 'ERROR: another PPT/MoE job uses this node'; exit 3; }
exec 8>/tmp/zhenhaowang_rain_blur_image_moe_v1.lock
flock -n 8 || { echo 'ERROR: another Rain/Blur job uses this node'; exit 3; }
LOG="$ROOT/logs/foundir_moe_blur_lpips_resume_v1_${RUN_ID}.log"
OUTPUT="$ROOT/checkpoints/foundir_moe_blur_lpips_resume_v1/$RUN_ID"
[[ ! -e "$LOG" && ! -e "$OUTPUT" ]] || { echo 'ERROR: preserve existing run; use a fresh ID'; exit 3; }
mkdir -p logs
(set -o noclobber; : > "$LOG") || exit 3
exec > >(tee -a "$LOG") 2>&1
STARTED=$SECONDS
trap 'status=$?; echo "MOE_MIGRATION_LAUNCH_END status=$status elapsed_seconds=$((SECONDS-STARTED)) utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"' EXIT
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export OMP_NUM_THREADS=2 NCCL_NVLS_ENABLE=0 NCCL_P2P_DISABLE=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCH_HOME="$ROOT/models/torch"
[[ "$CUDA_VISIBLE_DEVICES" == 0,1,2,3,4,5,6,7 ]] || { echo 'ERROR: expose exactly GPUs 0-7'; exit 2; }
echo "MOE_MIGRATION_LAUNCH_START run_id=$RUN_ID from=10500 to=11000 old_stop_is_user_ack=1 utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo 'PRELAUNCH_ESTIMATE wall_hours=0.5_to_0.9 allocated_GPU_hours=4_to_7.2 no_auto_PPT_or_extension=1'
"$PYTHON" -u -c 'import torch; p=[torch.cuda.get_device_properties(i) for i in range(torch.cuda.device_count())]; assert len(p)==8, f"Expected 8 GPUs, got {len(p)}"; assert len({x.name for x in p})==1 and any(x in p[0].name for x in ("H100","H200")); assert all(x.total_memory>=75*1024**3 for x in p); assert all(torch.cuda.mem_get_info(i)[0]>.8*x.total_memory for i,x in enumerate(p)), "GPU occupied"; print("GPU_PASS",p[0].name,"count=8",flush=True)'
"$PYTHON" -u smoke/train_foundir_moe_blur_lpips_resume_v1_ddp.py --preflight
timeout --signal=TERM --kill-after=30s 180s "$PYTHON" -u -m torch.distributed.run \
  --standalone --nproc_per_node=8 --max-restarts=0 smoke/cure_nccl_rehearsal_v1.py
echo 'MOE_MIGRATION_COMM_PASS'
"$PYTHON" -u -m torch.distributed.run --standalone --nproc_per_node=8 --max-restarts=0 \
  smoke/train_foundir_moe_blur_lpips_resume_v1_ddp.py --run-id "$RUN_ID"
"$PYTHON" -u -m smoke.moe_blur_lpips_resume_v1 --compare-output "$OUTPUT"
