#!/usr/bin/env bash
set -euo pipefail

ROOT=/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
/usr/bin/python3 "$ROOT/smoke/advisor_governance.py" check
PYTHON="$ROOT/.venv/flux2-smoke/bin/python"
CONFIG="$ROOT/configs/foundir_rain_restormer_adaptation_v2_8gpu.json"
TRAINER="$ROOT/smoke/train_foundir_task_native_restormer_adaptation_ddp.py"
LOG="$ROOT/logs/foundir_rain_restormer_adaptation_v2_8gpu.log"
GPU_LOG="$ROOT/logs/foundir_rain_restormer_adaptation_v2_8gpu_gpu.csv"
LOCK="$ROOT/logs/foundir_rain_restormer_adaptation_v2_8gpu.lock"

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
  echo "FOUNDIR_RAIN_RESTORMER_ADAPTATION_V2_8GPU_END status=$status utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
trap finish_run EXIT

exec 9>>"$LOCK"
flock -n 9 || { echo "ERROR: another Rain v2 scale-up holds $LOCK" >&2; exit 1; }
[[ "${CUDA_VISIBLE_DEVICES:-}" == "0,1,2,3,4,5,6,7" ]] || {
  echo "ERROR: expose exactly CUDA devices 0,1,2,3,4,5,6,7" >&2
  exit 1
}
export TORCH_HOME="$ROOT/models/torch"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1 TORCH_NCCL_DESYNC_DEBUG=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

echo "FOUNDIR_RAIN_RESTORMER_ADAPTATION_V2_8GPU_START utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "FOUNDIR_RAIN_RESTORMER_ADAPTATION_V2_8GPU_WORKSPACE $ROOT"
echo "FOUNDIR_RAIN_RESTORMER_ADAPTATION_V2_8GPU_GIT $(git rev-parse HEAD)"
echo "FOUNDIR_RAIN_RESTORMER_ADAPTATION_V2_8GPU_CONFIG_SHA256 $(sha256sum "$CONFIG" | cut -d' ' -f1)"
echo "FOUNDIR_RAIN_RESTORMER_ADAPTATION_V2_8GPU_TRAINER_SHA256 $(sha256sum "$TRAINER" | cut -d' ' -f1)"
echo "FOUNDIR_RAIN_RESTORMER_ADAPTATION_V2_8GPU_DEVICES $CUDA_VISIBLE_DEVICES"
echo "FOUNDIR_RAIN_RESTORMER_ADAPTATION_V2_8GPU_EQUIVALENCE global_batch_32_absolute_steps_lr_seed_loss_optimizer_preserved"
echo "FOUNDIR_RAIN_RESTORMER_ADAPTATION_V2_8GPU_INTERNAL sealed_internal_not_accessed"
echo "FOUNDIR_RAIN_RESTORMER_ADAPTATION_V2_8GPU_PROTECTED_OFFICIAL_TEST not_accessed"

{
  echo "RUN_START,$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw \
    --format=csv -l 5
} >>"$GPU_LOG" 2>&1 &
telemetry_pid=$!

"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 \
  "$TRAINER" --config "$CONFIG"
