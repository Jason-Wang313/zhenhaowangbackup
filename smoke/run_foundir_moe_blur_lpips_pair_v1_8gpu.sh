#!/usr/bin/env bash
set -euo pipefail
ROOT=/inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/zhenhaowang
PYTHON=/inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/.envs/gpu-train-stable-hdd3/bin/python
RUN_ID=${1:-pair_v1}
[[ "$RUN_ID" =~ ^[A-Za-z0-9_-]+$ ]] || { echo 'ERROR: invalid fresh pair ID'; exit 2; }
cd "$ROOT"
command -v flock >/dev/null
command -v timeout >/dev/null
exec 9>/tmp/zhenhaowang_dual_retry_v1.lock
flock -n 9 || { echo 'ERROR: another PPT/MoE retry uses this node; wait for it to finish'; exit 3; }
exec 8>/tmp/zhenhaowang_rain_blur_image_moe_v1.lock
flock -n 8 || { echo 'ERROR: another Rain/Blur job uses this node; do not overlap'; exit 3; }
LOG="$ROOT/logs/foundir_moe_blur_lpips_pair_v1_${RUN_ID}.log"
PAIR="$ROOT/checkpoints/foundir_moe_blur_lpips_continuation_v1/$RUN_ID"
[[ ! -e "$LOG" && ! -e "$PAIR" ]] || { echo 'ERROR: existing pair; preserve it, use a fresh ID'; exit 3; }
mkdir -p logs
(set -o noclobber; : > "$LOG") || exit 3
exec > >(tee -a "$LOG") 2>&1
STARTED=$SECONDS
trap 'status=$?; echo "MOE_LPIPS_PAIR_END status=$status elapsed_seconds=$((SECONDS-STARTED)) utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"' EXIT
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export OMP_NUM_THREADS=2
export NCCL_NVLS_ENABLE=0
export NCCL_P2P_DISABLE=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_HOME="$ROOT/models/torch"
[[ "$CUDA_VISIBLE_DEVICES" == 0,1,2,3,4,5,6,7 ]] || { echo 'ERROR: expose GPUs 0-7'; exit 2; }
echo "MOE_LPIPS_PAIR_START run_id=$RUN_ID same_node=1 sequential=1 GPUs_per_variant=8 utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo 'PRELAUNCH_ESTIMATE pair_wall_hours=2_to_3 pair_allocated_GPU_hours=16_to_24 additional_steps_per_variant=1000 no_automatic_extension=1'
gpu_check() {
  "$PYTHON" -u -c 'import torch; p=[torch.cuda.get_device_properties(i) for i in range(torch.cuda.device_count())]; assert len(p)==8, f"Expected 8 GPUs, got {len(p)}"; assert len({x.name for x in p})==1 and any(x in p[0].name for x in ("H100","H200")); assert all(x.total_memory>=75*1024**3 for x in p); assert all(torch.cuda.mem_get_info(i)[0]>.8*x.total_memory for i,x in enumerate(p)), "GPU occupied; do not overlap jobs"; print("GPU_PASS",p[0].name,"count=8",flush=True)'
}
gpu_check
"$PYTHON" -u smoke/train_foundir_moe_blur_lpips_continuation_v1_ddp.py --preflight
timeout --signal=TERM --kill-after=30s 180s "$PYTHON" -u -m torch.distributed.run \
  --standalone --nproc_per_node=8 --max-restarts=0 smoke/cure_nccl_rehearsal_v1.py
echo 'MOE_LPIPS_COMM_PASS'
# Keep both node locks across the whole pair. Each child exits and releases
# its CUDA context before the next starts. Never seed treatment from control.
for VARIANT in control blur_lpips4x; do
  gpu_check
  echo "MOE_LPIPS_VARIANT_START variant=$VARIANT parent_step=10000 utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  "$PYTHON" -u -m torch.distributed.run --standalone --nproc_per_node=8 --max-restarts=0 \
    smoke/train_foundir_moe_blur_lpips_continuation_v1_ddp.py --variant "$VARIANT" --run-id "$RUN_ID"
  "$PYTHON" -S -u -c 'import json,sys; r=json.load(open(sys.argv[1])); assert r["status"]=="complete" and r["engineering_pass"] and r["steps"]==11000; print("VARIANT_COMPLETE",r["variant"],flush=True)' "$PAIR/$VARIANT/result.json"
done
"$PYTHON" -u -m smoke.moe_blur_lpips_continuation_v1 --compare-pair "$PAIR"
