#!/usr/bin/env bash
set -euo pipefail
ROOT=/inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/zhenhaowang
PYTHON=/inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/.envs/gpu-train-stable-hdd3/bin/python
MODE=${1:-qualification}
ARM=${2:-moe}
RUN_ID=${3:-${MODE}_v1}
QUALIFICATION=${4:-$ROOT/checkpoints/foundir_rain_blur_image_moe_v1/$ARM/qualification_v1/result.json}
[[ "$MODE" == qualification || "$MODE" == formal ]] || { echo 'ERROR: invalid mode'; exit 2; }
[[ "$ARM" == moe || "$ARM" == dense ]] || { echo 'ERROR: invalid arm'; exit 2; }
[[ "$RUN_ID" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'ERROR: invalid run ID'; exit 2; }
cd "$ROOT"
mkdir -p logs
exec 9>/tmp/zhenhaowang_rain_blur_image_moe_v1.lock
flock -n 9 || { echo 'ERROR: a Rain/Blur arm is already running on this node'; exit 3; }
LOG="$ROOT/logs/foundir_rain_blur_image_moe_v1_${ARM}_${RUN_ID}.log"
OUTPUT="$ROOT/checkpoints/foundir_rain_blur_image_moe_v1/$ARM/$RUN_ID"
[[ ! -e "$LOG" && ! -e "$OUTPUT" ]] || { echo 'ERROR: preserve existing run'; exit 3; }
(set -o noclobber; : > "$LOG") || exit 3
exec > >(tee -a "$LOG") 2>&1
STARTED=$SECONDS
trap 'status=$?; echo "RAIN_BLUR_MOE_LAUNCH_END status=$status elapsed_seconds=$((SECONDS-STARTED)) utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"' EXIT
echo "RAIN_BLUR_MOE_LAUNCH_START mode=$MODE arm=$ARM run_id=$RUN_ID utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export OMP_NUM_THREADS=2
[[ "$CUDA_VISIBLE_DEVICES" == 0,1,2,3,4,5,6,7 ]] || { echo 'ERROR: expose exactly GPUs 0-7'; exit 2; }
"$PYTHON" -u -c 'import torch; p=[torch.cuda.get_device_properties(i) for i in range(torch.cuda.device_count())]; assert len(p)==8, f"Expected 8 GPUs, got {len(p)}"; assert len({x.name for x in p})==1 and any(x in p[0].name for x in ("H100","H200")); assert all(x.total_memory>=75*1024**3 for x in p); print("GPU_PASS",p[0].name,"count=8",flush=True)'
if [[ "$MODE" == qualification ]]; then
  echo 'PRELAUNCH_ESTIMATE wall_hours=0.25_to_1.0 allocated_gpu_hours=2_to_8 allocated_gpus=8 qualification_only'
else
  [[ -f "$QUALIFICATION" ]] || { echo 'ERROR: qualification result missing'; exit 3; }
  "$PYTHON" -u -c 'import json,sys; q=json.load(open(sys.argv[1])); assert q["engineering_pass"] and q["mode"]=="qualification" and q["arm"]==sys.argv[2]; s=q["median_training_step_seconds"]; base=3000*s/3600; low=base*1.1+0.3; high=base*1.5+1.0; print(f"PRELAUNCH_ESTIMATE measured_seconds_per_step={s} formal_wall_hours={low:.2f}_to_{high:.2f} allocated_gpu_hours={low*8:.2f}_to_{high*8:.2f} includes_provisional_eval_and_checkpoint_IO",flush=True)' "$QUALIFICATION" "$ARM"
fi
sha256sum configs/foundir_rain_blur_image_moe_v1.json smoke/train_foundir_rain_blur_image_moe_v1_ddp.py lowlight_restoration/rain_blur_image_moe.py lowlight_restoration/rain_blur_moe_data.py
ARGS=(--mode "$MODE" --arm "$ARM" --run-id "$RUN_ID")
if [[ "$MODE" == formal ]]; then ARGS+=(--qualification-result "$QUALIFICATION"); fi
"$PYTHON" -u -m torch.distributed.run --standalone --nproc_per_node=8 \
  smoke/train_foundir_rain_blur_image_moe_v1_ddp.py "${ARGS[@]}"
