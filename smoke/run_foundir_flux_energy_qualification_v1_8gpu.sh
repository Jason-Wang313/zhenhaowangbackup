#!/usr/bin/env bash
set -euo pipefail
ROOT=/inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/zhenhaowang
PYTHON=/inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/.envs/gpu-train-stable-hdd3/bin/python
RUN_ID=${1:-qualification_v1}
[[ "$RUN_ID" =~ ^[A-Za-z0-9_-]+$ ]] || { echo 'ERROR: invalid fresh run ID'; exit 2; }
cd "$ROOT"
command -v flock >/dev/null
exec 9>/tmp/zhenhaowang_dual_retry_v1.lock
flock -n 9 || { echo 'ERROR: PPT/MoE job already owns this node'; exit 3; }
exec 8>/tmp/zhenhaowang_rain_blur_image_moe_v1.lock
flock -n 8 || { echo 'ERROR: Rain/Blur job already owns this node'; exit 3; }
if pgrep -f '[t]rain_foundir_novel_ppt_|[t]rain_foundir_local_action_energy_' >/dev/null; then
  echo 'ERROR: old PPT job still exists on this node; no automatic kill'; exit 3
fi
LOG="$ROOT/logs/foundir_flux_energy_qualification_v1_${RUN_ID}.log"
OUTPUT="$ROOT/checkpoints/foundir_flux_energy_qualification_v1/$RUN_ID"
[[ ! -e "$LOG" && ! -e "$OUTPUT" ]] || { echo 'ERROR: preserve old run; use a fresh ID'; exit 3; }
mkdir -p logs
(set -o noclobber; : > "$LOG") || exit 3
exec > >(tee -a "$LOG") 2>&1
STARTED=$SECONDS
trap 'status=$?; elapsed=$((SECONDS-STARTED)); echo "FLUX_ENERGY_LAUNCH_END status=$status elapsed_seconds=$elapsed allocated_gpu_seconds=$((8*elapsed)) utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"' EXIT
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=2 NCCL_NVLS_ENABLE=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCH_HOME="$ROOT/models/torch"
export PYTHONPATH="$ROOT/third_party/flux2/src:$ROOT"
SCRIPT=smoke/qualify_foundir_flux_energy_v1.py
echo "FLUX_ENERGY_LAUNCH_START run_id=$RUN_ID utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo 'ESTIMATE wall_hours=0.5_to_1.5 allocated_gpu_hours=4_to_12; engineering_only; no_auto_formal'
"$PYTHON" -u "$SCRIPT" hardware
"$PYTHON" -u "$SCRIPT" prepare --run-id "$RUN_ID"
# The text encoder exits before any FLUX process is loaded.
CUDA_VISIBLE_DEVICES=0 "$PYTHON" -u "$SCRIPT" prompts --run-id "$RUN_ID"
"$PYTHON" -u -m torch.distributed.run --standalone --nproc_per_node=8 --max-restarts=0 \
  "$SCRIPT" cache --run-id "$RUN_ID"
# All FLUX/AE processes have exited: energy trains alone, never alongside FLUX.
"$PYTHON" -u -m torch.distributed.run --standalone --nproc_per_node=8 --max-restarts=0 \
  "$SCRIPT" train --run-id "$RUN_ID"
"$PYTHON" -u -c 'import json,sys; r=json.load(open(sys.argv[1])); assert r["engineering_pass"] and r["steps"]==25 and not r["scientific_gate_evaluated"]; print("FLUX_ENERGY_RESULT_VERIFIED")' "$OUTPUT/result.json"
