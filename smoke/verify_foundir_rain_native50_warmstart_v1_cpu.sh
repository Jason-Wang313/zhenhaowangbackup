#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ "$ROOT" == /inspire/hdd/project/multimodal-discrete-diffusion/liuxiaohong-25080/zhenhaowang ]] || { echo 'ERROR: use migrated CPU workspace'; exit 2; }
[[ $# == 2 ]] || { echo 'Usage: cpu_verify <run_id> <completed_shared_launch_id>'; exit 2; }
RUN_ID=$1 LAUNCH_ID=$2
[[ "$RUN_ID" =~ ^[A-Za-z0-9_-]{1,80}$ && "$LAUNCH_ID" =~ ^[A-Za-z0-9_-]{1,80}$ ]] || exit 2
cd "$ROOT"
PYTHON="${PROJECT_PYTHON:-$ROOT/.conda/gpu-train-stable/bin/python}"
[[ -x "$PYTHON" ]] || { echo 'ERROR: verified Python unavailable'; exit 2; }
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$ROOT:$ROOT/third_party/flux2/src" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
OUTPUT="$ROOT/checkpoints/foundir_rain_native50_warmstart_v1/$RUN_ID"
[[ -f "$OUTPUT/final_lora.json" ]] || { echo 'ERROR: GPU final export missing'; exit 3; }
mkdir -p logs
exec 9>"$OUTPUT/cpu_verify.lock"
flock -n 9 || { echo 'ERROR: CPU verifier already active'; exit 3; }
# CPU-only audit, no new inference/training and no GPU allocation requirement.
# A timeout never authorizes GPU restart or changing the training recipe.
timeout --signal=TERM --kill-after=30s 5400s "$PYTHON" -u smoke/train_foundir_rain_native50_warmstart_v1.py verify \
  --run-id "$RUN_ID" --launch-id "$LAUNCH_ID" --node-rank 0 \
  2>&1 | tee -a "logs/foundir_rain_native50_warmstart_v1_${RUN_ID}_cpu_verify.log"
echo 'RAIN_NATIVE50_WARMSTART_CPU_VERIFY_END status=0 gpu_used=false'
