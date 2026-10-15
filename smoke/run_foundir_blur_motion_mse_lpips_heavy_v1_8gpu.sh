#!/usr/bin/env bash
set -euo pipefail

ROOT=/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
CONFIG="$ROOT/configs/foundir_blur_motion_mse_lpips_heavy_v1.json"
OUTPUT="$ROOT/checkpoints/foundir_blur_motion_mse_lpips_heavy_v1"
PYTHON="$ROOT/.venv/flux2-smoke/bin/python"
LOG="$ROOT/logs/foundir_blur_motion_mse_lpips_heavy_v1.log"
LOCK="$ROOT/logs/foundir_blur_motion_mse_lpips_heavy_v1.lock"

[[ "${CUDA_VISIBLE_DEVICES:-}" == "0,1,2,3,4,5,6,7" ]] || { echo "ERROR: expose exactly GPUs 0-7" >&2; exit 1; }
command -v nvidia-smi >/dev/null || { echo "ERROR: nvidia-smi required" >&2; exit 1; }
[[ ! -e "$OUTPUT/development_selection.json" ]] || { echo "ERROR: Blur terminal selection already exists" >&2; exit 1; }
cd "$ROOT"
mkdir -p "$ROOT/logs"
exec 9>>"$LOCK"
flock -n 9 || { echo "ERROR: Blur training is already running" >&2; exit 1; }
exec > >(tee -a "$LOG") 2>&1
export TORCH_HOME="$ROOT/models/torch" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
"$PYTHON" smoke/train_foundir_task_native_restormer_adaptation_ddp.py --config "$CONFIG" --preflight
"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 \
  smoke/train_foundir_task_native_restormer_adaptation_ddp.py --config "$CONFIG"
echo "FOUNDIR_BLUR_MOTION_MSE_LPIPS_END status=0 utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
