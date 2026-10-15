#!/usr/bin/env bash
set -euo pipefail
ROOT=/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
PYTHON="$ROOT/.venv/flux2-smoke/bin/python"
CONFIG="$ROOT/configs/foundir_blur_noise_restormer_pilot_v1.json"
TRAINER="$ROOT/smoke/train_foundir_task_native_restormer_adaptation_ddp.py"
OUTPUT="$ROOT/checkpoints/foundir_blur_noise_restormer_pilot_v1"
RESULT="$OUTPUT/development_selection.json"
LOG="$ROOT/logs/foundir_blur_noise_restormer_pilot_v1.log"
GPU_LOG="$ROOT/logs/foundir_blur_noise_restormer_pilot_v1_gpu.csv"
LOCK="$ROOT/logs/foundir_blur_noise_restormer_pilot_v1.lock"
cd "$ROOT"
mkdir -p "$ROOT/logs"
exec > >(tee -a "$LOG") 2>&1
telemetry_pid=
finish_run() {
  status=$?
  if [[ -n "$telemetry_pid" ]]; then kill "$telemetry_pid" 2>/dev/null || true; wait "$telemetry_pid" 2>/dev/null || true; fi
  echo "BLUR_NOISE_RESTORMER_PILOT_END status=$status utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  return "$status"
}
trap finish_run EXIT
exec 9>>"$LOCK"
flock -n 9 || { echo "ERROR: Blur+Noise Restormer pilot already running" >&2; exit 1; }
[[ "${CUDA_VISIBLE_DEVICES:-}" == "0,1,2,3,4,5,6,7" ]] || { echo "ERROR: set eight visible GPUs" >&2; exit 1; }
[[ ! -f "$RESULT" ]] || { echo "ERROR: terminal selection already exists" >&2; exit 1; }
mapfile -t names < <(nvidia-smi --query-gpu=name --format=csv,noheader)
[[ "${#names[@]}" -eq 8 ]] || { echo "ERROR: exactly eight GPUs required" >&2; exit 1; }
for name in "${names[@]}"; do [[ "$name" == *H100* || "$name" == *H200* ]] || { echo "ERROR: unsupported GPU $name" >&2; exit 1; }; done
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TORCH_HOME="$ROOT/models/torch"
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 TORCH_NCCL_TRACE_BUFFER_SIZE=1048576
echo "BLUR_NOISE_RESTORMER_PILOT_START utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "SCOPE training_only_2048_train_128_validation 1000_steps residual_Restormer rgb_L1 identity_mix_25pct no_development no_official_test"
echo "TOTAL_TIME_ESTIMATE H100=40-55min(center47) H200=30-45min(center37) one_8GPU_instance GPU_HOURS_H100=5.3-7.3 GPU_HOURS_H200=4.0-6.0"
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw --format=csv -l 5 >"$GPU_LOG" 2>&1 &
telemetry_pid=$!
"$PYTHON" "$TRAINER" --config "$CONFIG" --preflight
"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 "$TRAINER" --config "$CONFIG"
[[ -f "$RESULT" ]] || { echo "ERROR: terminal selection missing" >&2; exit 1; }
