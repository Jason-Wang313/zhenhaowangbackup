#!/usr/bin/env bash
set -euo pipefail
ROOT=/inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/zhenhaowang
PYTHON=/inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/.envs/gpu-train-stable-hdd3/bin/python
RUN_ID=${1:-continue_v1}
[[ "$RUN_ID" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'ERROR: invalid run ID'; exit 2; }
cd "$ROOT"
mkdir -p logs
# Hold a node-local lock for this exact continuation, including all subprocesses.
exec 9>/tmp/zhenhaowang_rain_precision_resume_v1.lock
flock -n 9 || { echo 'ERROR: continuation already running on this node'; exit 3; }
LOG="$ROOT/logs/foundir_rain_flux2_precision_resume_v1_${RUN_ID}.log"
OUTPUT="$ROOT/checkpoints/foundir_rain_flux2_precision_resume_v1/$RUN_ID"
[[ ! -e "$LOG" && ! -e "$OUTPUT" ]] || { echo "ERROR: preserve existing run $RUN_ID"; exit 3; }
# Exclusive creation also prevents the same ID launching from another node.
(set -o noclobber; : > "$LOG") || exit 3
exec > >(tee -a "$LOG") 2>&1
STARTED=$SECONDS
trap 'status=$?; echo "RAIN_RESUME_LAUNCH_END status=$status elapsed_seconds=$((SECONDS-STARTED)) utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"' EXIT
echo "RAIN_RESUME_LAUNCH_START run_id=$RUN_ID utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export OMP_NUM_THREADS=2
[[ "$CUDA_VISIBLE_DEVICES" == 0,1,2,3,4,5,6,7 ]] || { echo 'ERROR: expose exactly GPUs 0-7'; exit 2; }
"$PYTHON" -u -c 'import torch; p=[torch.cuda.get_device_properties(i) for i in range(torch.cuda.device_count())]; assert len(p)==8, f"Expected 8 GPUs, got {len(p)}"; assert len({x.name for x in p})==1 and any(x in p[0].name for x in ("H100","H200")), "Expected homogeneous H100/H200"; assert all(x.total_memory >= 75*1024**3 for x in p); print("GPU_PASS", p[0].name, "count=8", flush=True)'
echo 'PRELAUNCH_ESTIMATE wall_minutes=40_to_50 allocated_gpu_hours=5.3_to_6.7 start_step=3000 end_step=4000 validation=3500,4000'
sha256sum configs/foundir_rain_flux2_precision_resume_v1.json smoke/train_foundir_rain_flux2_precision_resume_v1_ddp.py lowlight_restoration/rain_precision_resume.py
"$PYTHON" -u -m torch.distributed.run --standalone --nproc_per_node=8 \
  smoke/train_foundir_rain_flux2_precision_resume_v1_ddp.py \
  --config configs/foundir_rain_flux2_precision_resume_v1.json --run-id "$RUN_ID"
