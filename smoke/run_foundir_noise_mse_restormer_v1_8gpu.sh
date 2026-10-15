#!/usr/bin/env bash
set -euo pipefail

ROOT=/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
PYTHON="$ROOT/.venv/flux2-smoke/bin/python"
TRAINER="$ROOT/smoke/train_foundir_task_native_restormer_adaptation_ddp.py"
CONFIG="$ROOT/configs/foundir_noise_mse_restormer_v1.json"
OUTPUT="$ROOT/checkpoints/foundir_noise_mse_restormer_v1"
RESULT="$OUTPUT/development_selection.json"
LOG="$ROOT/logs/foundir_noise_mse_restormer_v1.log"
GPU_LOG="$ROOT/logs/foundir_noise_mse_restormer_v1_gpu.csv"
LOCK="$ROOT/logs/foundir_noise_mse_restormer_v1.lock"

cd "$ROOT"
mkdir -p "$ROOT/logs"
exec > >(tee -a "$LOG") 2>&1
telemetry_pid=
finish_run() {
  status=$?
  if [[ -n "$telemetry_pid" ]]; then
    kill "$telemetry_pid" 2>/dev/null || true
    wait "$telemetry_pid" 2>/dev/null || true
  fi
  echo "FOUNDIR_NOISE_MSE_END status=$status utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  return "$status"
}
trap finish_run EXIT
exec 9>>"$LOCK"
flock -n 9 || { echo "ERROR: Noise MSE pilot already running" >&2; exit 1; }
[[ "${CUDA_VISIBLE_DEVICES:-}" == "0,1,2,3,4,5,6,7" ]] || { echo "ERROR: expose exactly GPUs 0-7" >&2; exit 1; }
[[ ! -f "$RESULT" ]] || { echo "ERROR: terminal selection already exists" >&2; exit 1; }
mapfile -t names < <(nvidia-smi --query-gpu=name --format=csv,noheader)
[[ "${#names[@]}" -eq 8 ]] || { echo "ERROR: exactly eight GPUs required" >&2; exit 1; }
for name in "${names[@]}"; do
  [[ "$name" == *H100* || "$name" == *H200* ]] || { echo "ERROR: unsupported GPU $name" >&2; exit 1; }
done
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TORCH_HOME="$ROOT/models/torch" TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
echo "FOUNDIR_NOISE_MSE_START utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "SCOPE training_only Noise 1000_steps Restormer RGB_MSE no_alpha_sweep no_official_pixels_in_gradient"
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw --format=csv -l 5 >"$GPU_LOG" 2>&1 &
telemetry_pid=$!
"$PYTHON" "$TRAINER" --config "$CONFIG" --preflight
"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 "$TRAINER" --config "$CONFIG"
[[ -f "$RESULT" ]] || { echo "ERROR: terminal selection missing" >&2; exit 1; }
