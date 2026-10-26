#!/usr/bin/env bash
set -euo pipefail
ROOT=/inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/zhenhaowang
PYTHON=/inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/.envs/gpu-train-stable-hdd3/bin/python
MODE=${1:-qualification}
ARM=${2:-image_only}
RUN_ID=${3:-qualification_v1}
Q_IMAGE=${4:-$ROOT/checkpoints/foundir_rain_blur_joint_vlm_moe_v2/image_only/qualification_v1/result.json}
Q_VLM=${5:-$ROOT/checkpoints/foundir_rain_blur_joint_vlm_moe_v2/vlm_assisted/qualification_v1/result.json}
RESUME=${6:-}
[[ "$MODE" == qualification || "$MODE" == formal ]] || { echo 'ERROR: invalid mode'; exit 2; }
[[ "$ARM" == image_only || "$ARM" == vlm_assisted ]] || { echo 'ERROR: invalid arm'; exit 2; }
[[ "$RUN_ID" =~ ^[A-Za-z0-9_-]+$ ]] || { echo 'ERROR: invalid run ID'; exit 2; }
[[ -z "$RESUME" || "$MODE" == formal ]] || { echo 'ERROR: resume is formal-only'; exit 2; }
cd "$ROOT"
mkdir -p logs
exec 9>/tmp/zhenhaowang_rain_blur_image_moe_v1.lock
flock -n 9 || { echo 'ERROR: another Rain/Blur job holds this node lock'; exit 3; }
LOG="$ROOT/logs/foundir_rain_blur_joint_vlm_moe_v2_${ARM}_${RUN_ID}.log"
OUTPUT="$ROOT/checkpoints/foundir_rain_blur_joint_vlm_moe_v2/$ARM/$RUN_ID"
[[ ! -e "$LOG" && ! -e "$OUTPUT" ]] || { echo 'ERROR: preserve existing run; use a fresh ID'; exit 3; }
(set -o noclobber; : > "$LOG") || exit 3
exec > >(tee -a "$LOG") 2>&1
STARTED=$SECONDS
trap 'status=$?; echo "JOINT_MOE_LAUNCH_END status=$status elapsed_seconds=$((SECONDS-STARTED)) utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"' EXIT
echo "JOINT_MOE_LAUNCH_START mode=$MODE arm=$ARM run_id=$RUN_ID utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export OMP_NUM_THREADS=2
[[ "$CUDA_VISIBLE_DEVICES" == 0,1,2,3,4,5,6,7 ]] || { echo 'ERROR: expose exactly GPUs 0-7'; exit 2; }
"$PYTHON" -u -c 'import torch; p=[torch.cuda.get_device_properties(i) for i in range(torch.cuda.device_count())]; assert len(p)==8, f"Expected 8 GPUs, got {len(p)}"; assert len({x.name for x in p})==1 and any(x in p[0].name for x in ("H100","H200")); assert all(x.total_memory>=75*1024**3 for x in p); print("GPU_PASS",p[0].name,"count=8",flush=True)'
ARGS=(--mode "$MODE" --arm "$ARM" --run-id "$RUN_ID")
if [[ "$MODE" == qualification ]]; then
  echo 'PRELAUNCH_ESTIMATE wall_hours_per_arm=0.20_to_0.75 allocated_gpu_hours=1.6_to_6.0 provisional engineering_only no_auto_formal'
else
  [[ -f "$Q_IMAGE" && -f "$Q_VLM" ]] || { echo 'ERROR: both engineering qualification receipts required'; exit 3; }
  "$PYTHON" -S -u -c 'import json,sys; qs=[json.load(open(p)) for p in sys.argv[1:]]; assert all(q["engineering_pass"] and q["mode"]=="qualification" for q in qs); s=max(q["median_training_step_seconds"] for q in qs); base=10000*s/3600; lo=base+0.75; hi=base*1.5+2; print(f"PRELAUNCH_ESTIMATE per_arm_wall_hours={lo:.2f}_to_{hi:.2f} per_arm_allocated_gpu_hours={8*lo:.2f}_to_{8*hi:.2f} provisional_includes_IO_and_eval",flush=True)' "$Q_IMAGE" "$Q_VLM"
  ARGS+=(--qualifications "$Q_IMAGE" "$Q_VLM")
fi
if [[ -n "$RESUME" ]]; then
  [[ -f "$RESUME" ]] || { echo 'ERROR: complete resume receipt missing'; exit 3; }
  ARGS+=(--resume-receipt "$RESUME")
fi
sha256sum configs/foundir_rain_blur_joint_vlm_moe_v2.json smoke/train_foundir_rain_blur_joint_vlm_moe_v2_ddp.py
"$PYTHON" -u -m torch.distributed.run --standalone --nproc_per_node=8 \
  smoke/train_foundir_rain_blur_joint_vlm_moe_v2_ddp.py "${ARGS[@]}"
