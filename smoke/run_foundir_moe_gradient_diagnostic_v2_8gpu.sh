#!/usr/bin/env bash
set -euo pipefail
ROOT=/inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/zhenhaowang
PYTHON=/inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/.envs/gpu-train-stable-hdd3/bin/python
BRANCH=moe
RUN_ID=${1:-diagnostic_v2}
[[ "$RUN_ID" =~ ^[A-Za-z0-9_-]+$ ]] || { echo 'ERROR: provide a fresh run ID'; exit 2; }
cd "$ROOT"
command -v flock >/dev/null
command -v timeout >/dev/null
exec 9>/tmp/zhenhaowang_dual_retry_v1.lock
flock -n 9 || { echo 'ERROR: another retry is using this node; use a different node or wait'; exit 3; }
exec 8>/tmp/zhenhaowang_rain_blur_image_moe_v1.lock
flock -n 8 || { echo 'ERROR: previous Rain/Blur job still holds this node'; exit 3; }
LOG="$ROOT/logs/foundir_dual_retry_v1_${BRANCH}_${RUN_ID}.log"
OUTPUT="$ROOT/checkpoints/foundir_dual_retry_v1/$BRANCH/$RUN_ID"
[[ ! -e "$LOG" && ! -e "$OUTPUT" ]] || { echo 'ERROR: existing run; preserve it and use a fresh ID'; exit 3; }
mkdir -p logs
(set -o noclobber; : > "$LOG") || exit 3
exec > >(tee -a "$LOG") 2>&1
STARTED=$SECONDS
trap 'status=$?; echo "DUAL_RETRY_END branch=$BRANCH status=$status elapsed_seconds=$((SECONDS-STARTED)) utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"' EXIT
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export OMP_NUM_THREADS=2
export NCCL_NVLS_ENABLE=0
export NCCL_P2P_DISABLE=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_HOME="$ROOT/models/torch"
[[ "$CUDA_VISIBLE_DEVICES" == 0,1,2,3,4,5,6,7 ]] || { echo 'ERROR: expose exactly GPUs 0-7'; exit 2; }
echo "DUAL_RETRY_START branch=$BRANCH run_id=$RUN_ID utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
"$PYTHON" -u -c 'import torch; p=[torch.cuda.get_device_properties(i) for i in range(torch.cuda.device_count())]; assert len(p)==8, f"Expected 8 GPUs, got {len(p)}"; assert len({x.name for x in p})==1 and any(x in p[0].name for x in ("H100","H200")); assert all(x.total_memory>=75*1024**3 for x in p); assert all(torch.cuda.mem_get_info(i)[0]>.8*x.total_memory for i,x in enumerate(p)), "GPU memory occupied: choose an idle node"; print("GPU_PASS",p[0].name,"count=8",flush=True)'
echo 'PRELAUNCH_ESTIMATE wall_hours=0.25_to_0.75 allocated_gpu_hours=2_to_6 optimizer_steps=0 train_pixels_only=1'
echo 'MOE_GRADIENT_DIAGNOSTIC_VERSION v2_integer_indices'
SCRIPT=smoke/diagnose_foundir_joint_moe_train_gradients_v2.py
sha256sum configs/foundir_dual_retry_execution_v1.json "$SCRIPT" lowlight_restoration/novel_ppt_energy.py
timeout --signal=TERM --kill-after=30s 180s "$PYTHON" -u -m torch.distributed.run \
  --standalone --nproc_per_node=8 --max-restarts=0 smoke/cure_nccl_rehearsal_v1.py
echo 'DUAL_RETRY_COMM_PASS'
"$PYTHON" -u -m torch.distributed.run --standalone --nproc_per_node=8 --max-restarts=0 \
  "$SCRIPT" --run-id "$RUN_ID"
