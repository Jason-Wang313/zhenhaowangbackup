#!/usr/bin/env bash
set -euo pipefail

ROOT=/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
PYTHON="$ROOT/.venv/flux2-smoke/bin/python"
CONFIG="$ROOT/configs/foundir_blur_flux2_low_frequency_anchor_gate_v1.json"
SCRIPT="$ROOT/smoke/export_foundir_blur_flux2_calibration_cache_ddp.py"
EVALUATOR="$ROOT/smoke/evaluate_foundir_blur_flux2_low_frequency_anchor.py"
OUTPUT="$ROOT/checkpoints/foundir_blur_flux2_low_frequency_anchor_gate_v1/calibration_flux_cache"
CACHE_RESULT="$OUTPUT/result.json"
RESULT="$ROOT/checkpoints/foundir_blur_flux2_low_frequency_anchor_gate_v1/result.json"
LOG="$ROOT/logs/foundir_blur_flux2_calibration_cache_v1.log"
GPU_LOG="$ROOT/logs/foundir_blur_flux2_calibration_cache_v1_gpu.csv"
LOCK="$ROOT/logs/foundir_blur_flux2_calibration_cache_v1.lock"

cd "$ROOT"
/usr/bin/python3 smoke/advisor_governance.py check
mkdir -p "$ROOT/logs" "$ROOT/.cache/triton/foundir_blur_flux2_calibration_cache_v1"
exec > >(tee -a "$LOG") 2>&1

telemetry_pid=
finish_run() {
  status=$?
  if [[ -n "$telemetry_pid" ]]; then
    kill "$telemetry_pid" 2>/dev/null || true
    wait "$telemetry_pid" 2>/dev/null || true
  fi
  echo "BLUR_CALIBRATION_CACHE_END status=$status utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  return "$status"
}
trap finish_run EXIT

exec 9>>"$LOCK"
flock -n 9 || { echo "ERROR: another calibration-cache run holds $LOCK" >&2; exit 1; }
[[ "${CUDA_VISIBLE_DEVICES:-}" == "0,1,2,3,4,5,6,7" ]] || {
  echo "ERROR: set CUDA_VISIBLE_DEVICES exactly to 0,1,2,3,4,5,6,7" >&2; exit 1;
}
[[ ! -f "$RESULT" ]] || { echo "ERROR: terminal gate result already exists" >&2; exit 1; }

export TORCH_HOME="$ROOT/models/torch"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 TORCH_NCCL_TRACE_BUFFER_SIZE=1048576
export TORCH_NCCL_DUMP_ON_TIMEOUT=1 TORCH_NCCL_DESYNC_DEBUG=1
export TRITON_CACHE_DIR="$ROOT/.cache/triton/foundir_blur_flux2_calibration_cache_v1"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

echo "BLUR_CALIBRATION_CACHE_START utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) git=$(git rev-parse HEAD)"
echo "SCOPE immutable_FLUX_step10k 384_training_only_calibration no_weight_update no_development no_sealed no_official_test"
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw \
  --format=csv -l 5 >"$GPU_LOG" 2>&1 &
telemetry_pid=$!

if [[ ! -f "$CACHE_RESULT" ]]; then
  "$PYTHON" "$SCRIPT" --gate-config "$CONFIG" --preflight
  "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 \
    "$SCRIPT" --gate-config "$CONFIG"
else
  echo "BLUR_CALIBRATION_CACHE_REUSE $CACHE_RESULT"
fi

set +e
"$PYTHON" "$EVALUATOR" --config "$CONFIG"
gate_status=$?
set -e
if [[ "$gate_status" -eq 0 ]]; then
  echo "BLUR_LOW_FREQUENCY_ANCHOR_GATE_PASS_STOP"
elif [[ "$gate_status" -eq 2 ]]; then
  echo "BLUR_LOW_FREQUENCY_ANCHOR_SCIENTIFIC_FAILURE_STOP"
else
  echo "ERROR: low-frequency evaluator failed with status $gate_status" >&2
  exit "$gate_status"
fi
exit "$gate_status"
