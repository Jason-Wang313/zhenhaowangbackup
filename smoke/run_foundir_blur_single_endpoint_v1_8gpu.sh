#!/usr/bin/env bash
set -euo pipefail
ROOT=/inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/zhenhaowang
PYTHON=/inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/.envs/gpu-train-stable-hdd3/bin/python
MODE=${1:-qualification}
RUN_ID=${2:-${MODE}_v1}
QUALIFICATION=${3:-$ROOT/checkpoints/foundir_blur_single_endpoint_v1/qualification_v1/result.json}
RESUME=${4:-}
[[ "$MODE" == qualification || "$MODE" == formal ]] || { echo 'ERROR: invalid mode'; exit 2; }
[[ "$RUN_ID" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'ERROR: invalid run ID'; exit 2; }
[[ -z "$RESUME" || "$MODE" == formal ]] || { echo 'ERROR: resume is formal-only'; exit 2; }
cd "$ROOT"
mkdir -p logs
# Same lock as the parent experiment: no accidental same-node concurrent jobs.
exec 9>/tmp/zhenhaowang_rain_blur_image_moe_v1.lock
flock -n 9 || { echo 'ERROR: Rain/Blur job already running on this node'; exit 3; }
LOG="$ROOT/logs/foundir_blur_single_endpoint_v1_${RUN_ID}.log"
OUTPUT="$ROOT/checkpoints/foundir_blur_single_endpoint_v1/$RUN_ID"
[[ ! -e "$LOG" && ! -e "$OUTPUT" ]] || { echo 'ERROR: preserve existing run; use a fresh ID'; exit 3; }
(set -o noclobber; : > "$LOG") || exit 3
exec > >(tee -a "$LOG") 2>&1
STARTED=$SECONDS
trap 'status=$?; echo "BLUR_SINGLE_LAUNCH_END status=$status elapsed_seconds=$((SECONDS-STARTED)) utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"' EXIT
echo "BLUR_SINGLE_LAUNCH_START mode=$MODE run_id=$RUN_ID utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export OMP_NUM_THREADS=2
[[ "$CUDA_VISIBLE_DEVICES" == 0,1,2,3,4,5,6,7 ]] || { echo 'ERROR: expose exactly GPUs 0-7'; exit 2; }
"$PYTHON" -u -c 'import torch; p=[torch.cuda.get_device_properties(i) for i in range(torch.cuda.device_count())]; assert len(p)==8, f"Expected 8 GPUs, got {len(p)}"; assert len({x.name for x in p})==1 and any(x in p[0].name for x in ("H100","H200")); assert all(x.total_memory>=75*1024**3 for x in p); print("GPU_PASS",p[0].name,"count=8",flush=True)'
if [[ "$MODE" == qualification ]]; then
  echo 'PRELAUNCH_ESTIMATE wall_hours=0.15_to_0.50 allocated_gpu_hours=1.2_to_4.0 qualification_only'
else
  [[ -f "$QUALIFICATION" ]] || { echo 'ERROR: qualification receipt missing'; exit 3; }
  "$PYTHON" -S -u -c 'import json,sys; q=json.load(open(sys.argv[1])); assert q["engineering_pass"] and q["mode"]=="qualification"; s=q["median_training_step_seconds"]; base=3000*s/3600; low=base+0.35; high=base*1.5+1.0; print(f"PRELAUNCH_ESTIMATE full_formal_wall_hours={low:.2f}_to_{high:.2f} allocated_gpu_hours={low*8:.2f}_to_{high*8:.2f} checkpoint_and_eval_IO_provisional",flush=True)' "$QUALIFICATION"
fi
sha256sum configs/foundir_blur_single_endpoint_v1.json smoke/train_foundir_blur_single_endpoint_v1_ddp.py lowlight_restoration/blur_single_endpoint.py
ARGS=(--mode "$MODE" --run-id "$RUN_ID")
if [[ "$MODE" == formal ]]; then ARGS+=(--qualification-result "$QUALIFICATION"); fi
if [[ -n "$RESUME" ]]; then
  [[ -f "$RESUME" ]] || { echo 'ERROR: resume completion receipt missing'; exit 3; }
  ARGS+=(--resume-receipt "$RESUME")
fi
"$PYTHON" -u -m torch.distributed.run --standalone --nproc_per_node=8 \
  smoke/train_foundir_blur_single_endpoint_v1_ddp.py "${ARGS[@]}"
