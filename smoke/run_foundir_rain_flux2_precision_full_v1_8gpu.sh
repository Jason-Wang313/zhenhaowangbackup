#!/usr/bin/env bash
set -euo pipefail
ROOT=/inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/zhenhaowang
PYTHON=/inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/.envs/gpu-train-stable-hdd3/bin/python
MODE=${1:-qualification}
RUN_ID=${2:-${MODE}_v1}
QUALIFICATION_RESULT=${3:-$ROOT/checkpoints/foundir_rain_flux2_precision_full_v1/qualification_v1/result.json}
[[ "$MODE" == qualification || "$MODE" == formal ]] || { echo 'ERROR: mode must be qualification or formal'; exit 2; }
[[ "$RUN_ID" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'ERROR: invalid run ID'; exit 2; }
cd "$ROOT"
LOG="$ROOT/logs/foundir_rain_flux2_precision_full_v1_${RUN_ID}.log"
OUTPUT="$ROOT/checkpoints/foundir_rain_flux2_precision_full_v1/$RUN_ID"
[[ ! -e "$LOG" && ! -e "$OUTPUT" ]] || { echo "ERROR: preserve and inspect existing run $RUN_ID"; exit 3; }
mkdir -p logs
exec > >(tee "$LOG") 2>&1
STARTED=$SECONDS
trap 'status=$?; echo "RAIN_FULL_LAUNCH_END status=$status elapsed_seconds=$((SECONDS-STARTED)) utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"' EXIT
echo "RAIN_FULL_LAUNCH_START mode=$MODE run_id=$RUN_ID utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export OMP_NUM_THREADS=2
[[ "$CUDA_VISIBLE_DEVICES" == 0,1,2,3,4,5,6,7 ]] || { echo 'ERROR: expose exactly GPUs 0-7'; exit 2; }
"$PYTHON" -u -c 'import torch; p=[torch.cuda.get_device_properties(i) for i in range(torch.cuda.device_count())]; assert len(p)==8, f"Expected 8 GPUs, got {len(p)}"; assert len({x.name for x in p})==1 and any(x in p[0].name for x in ("H100","H200")), "Expected homogeneous H100/H200"; assert all(x.total_memory >= 75*1024**3 for x in p); print("GPU_PASS", p[0].name, "count=8", flush=True)'
echo 'PRELAUNCH_ESTIMATE qualification_wall_hours=0.10_to_0.50 qualification_gpu_hours=0.8_to_4 formal_3000step_wall_hours=1.5_to_3.0 formal_gpu_hours=12_to_24 provisional_until_full_data_qualification'
sha256sum configs/foundir_rain_flux2_precision_full_v1.json smoke/train_foundir_rain_flux2_precision_full_v1_ddp.py lowlight_restoration/flux2_endpoint_precision.py
ARGS=(--config configs/foundir_rain_flux2_precision_full_v1.json --mode "$MODE" --run-id "$RUN_ID")
if [[ "$MODE" == formal ]]; then
  [[ -f "$QUALIFICATION_RESULT" ]] || { echo "ERROR: qualification result missing: $QUALIFICATION_RESULT"; exit 2; }
  ARGS+=(--qualification-result "$QUALIFICATION_RESULT")
fi
"$PYTHON" -u -m torch.distributed.run --standalone --nproc_per_node=8 \
  smoke/train_foundir_rain_flux2_precision_full_v1_ddp.py "${ARGS[@]}"
