#!/usr/bin/env bash
set -euo pipefail

ROOT=/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
/usr/bin/python3 "$ROOT/smoke/advisor_governance.py" check
PYTHON="$ROOT/.venv/flux2-smoke/bin/python"
CONFIG="$ROOT/configs/foundir_noise_rain_task_native_capacity_v1.json"
EVALUATOR="$ROOT/smoke/evaluate_foundir_noise_rain_task_native_capacity_ddp.py"
MODEL_CODE="$ROOT/lowlight_restoration/task_native_restormer.py"
BASELINE="$ROOT/checkpoints/foundir_five_type_flux2_dit_v3/noise_rain_full_development_audit_v1/result.json"
CHECKPOINT_DIR="$ROOT/models/restormer-task-native"
LOG="$ROOT/logs/foundir_noise_rain_task_native_capacity_v1.log"
GPU_LOG="$ROOT/logs/foundir_noise_rain_task_native_capacity_v1_gpu.csv"
LOCK="$ROOT/logs/foundir_noise_rain_task_native_capacity_v1.lock"

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
  echo "FOUNDIR_NOISE_RAIN_TASK_NATIVE_CAPACITY_V1_END status=$status utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
trap finish_run EXIT

exec 9>>"$LOCK"
flock -n 9 || { echo "ERROR: another task-native capacity audit holds $LOCK" >&2; exit 1; }
[[ "${CUDA_VISIBLE_DEVICES:-}" == "0,1,2,3,4,5,6,7" ]] || {
  echo "ERROR: expose exactly CUDA devices 0,1,2,3,4,5,6,7" >&2
  exit 1
}
export TORCH_HOME="$ROOT/models/torch"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export FOUNDIR_EVALUATION_COORDINATION_TOKEN=noise-rain-task-native-capacity-v1

echo "FOUNDIR_NOISE_RAIN_TASK_NATIVE_CAPACITY_V1_START utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "FOUNDIR_NOISE_RAIN_TASK_NATIVE_CAPACITY_V1_GIT $(git rev-parse HEAD)"
echo "FOUNDIR_NOISE_RAIN_TASK_NATIVE_CAPACITY_V1_CONFIG_SHA256 $(sha256sum "$CONFIG" | cut -d' ' -f1)"
echo "FOUNDIR_NOISE_RAIN_TASK_NATIVE_CAPACITY_V1_EVALUATOR_SHA256 $(sha256sum "$EVALUATOR" | cut -d' ' -f1)"
echo "FOUNDIR_NOISE_RAIN_TASK_NATIVE_CAPACITY_V1_MODEL_CODE_SHA256 $(sha256sum "$MODEL_CODE" | cut -d' ' -f1)"
echo "FOUNDIR_NOISE_RAIN_TASK_NATIVE_CAPACITY_V1_BASELINE_SHA256 $(sha256sum "$BASELINE" | cut -d' ' -f1)"
for name in deraining real_denoising gaussian_color_denoising_blind; do
  echo "FOUNDIR_NOISE_RAIN_TASK_NATIVE_CAPACITY_V1_${name^^}_SHA256 $(sha256sum "$CHECKPOINT_DIR/$name.pth" | cut -d' ' -f1)"
done
echo "FOUNDIR_NOISE_RAIN_TASK_NATIVE_CAPACITY_V1_SCOPE development_isolated_capacity_and_actual_qwen_hybrid_only"
echo "FOUNDIR_NOISE_RAIN_TASK_NATIVE_CAPACITY_V1_INTERNAL sealed_internal_not_accessed"
echo "FOUNDIR_NOISE_RAIN_TASK_NATIVE_CAPACITY_V1_PROTECTED_OFFICIAL_TEST not_accessed"

{
  echo "RUN_START,$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw \
    --format=csv -l 5
} >>"$GPU_LOG" 2>&1 &
telemetry_pid=$!

"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 \
  "$EVALUATOR" --config "$CONFIG"
