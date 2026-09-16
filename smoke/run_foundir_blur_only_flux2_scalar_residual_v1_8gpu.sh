#!/usr/bin/env bash
set -euo pipefail

ROOT=/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
PYTHON="$ROOT/.venv/flux2-smoke/bin/python"
CONFIG="$ROOT/configs/foundir_blur_only_flux2_scalar_residual_feasibility_v1.json"
SCRIPT="$ROOT/smoke/evaluate_foundir_blur_only_flux2_scalar_residual_ddp.py"
OUTPUT="$ROOT/checkpoints/foundir_blur_only_flux2_scalar_residual_feasibility_v1"
RESULT="$OUTPUT/result.json"
LOG="$ROOT/logs/foundir_blur_only_flux2_scalar_residual_feasibility_v1.log"
GPU_LOG="$ROOT/logs/foundir_blur_only_flux2_scalar_residual_feasibility_v1_gpu.csv"
LOCK="$ROOT/logs/foundir_blur_only_flux2_scalar_residual_feasibility_v1.lock"

cd "$ROOT"
/usr/bin/python3 smoke/advisor_governance.py check
mkdir -p "$ROOT/logs" "$ROOT/.cache/triton/foundir_blur_flux2_scalar_residual_v1"
exec > >(tee -a "$LOG") 2>&1

telemetry_pid=
finish_run() {
  status=$?
  if [[ -n "$telemetry_pid" ]]; then
    kill "$telemetry_pid" 2>/dev/null || true
    wait "$telemetry_pid" 2>/dev/null || true
  fi
  echo "FLUX2_SCALAR_RESIDUAL_END status=$status utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  return "$status"
}
trap finish_run EXIT

exec 9>>"$LOCK"
flock -n 9 || { echo "ERROR: another scalar residual run holds $LOCK" >&2; exit 1; }
[[ "${CUDA_VISIBLE_DEVICES:-}" == "0,1,2,3,4,5,6,7" ]] || {
  echo "ERROR: set CUDA_VISIBLE_DEVICES exactly to 0,1,2,3,4,5,6,7" >&2; exit 1;
}
[[ -x "$PYTHON" ]] || { echo "ERROR: missing FLUX Python $PYTHON" >&2; exit 1; }
[[ ! -f "$RESULT" ]] || { echo "ERROR: terminal result already exists" >&2; exit 1; }

export TORCH_HOME="$ROOT/models/torch"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 TORCH_NCCL_TRACE_BUFFER_SIZE=1048576
export TORCH_NCCL_DUMP_ON_TIMEOUT=1 TORCH_NCCL_DESYNC_DEBUG=1
export TRITON_CACHE_DIR="$ROOT/.cache/triton/foundir_blur_flux2_scalar_residual_v1"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

echo "FLUX2_SCALAR_RESIDUAL_START utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) git=$(git rev-parse HEAD)"
echo "SCOPE FLUX.2_4B step10k no_weight_update 384_calibration 128_validation fixed_alpha_grid no_development no_sealed no_official_test no_distillation no_decoder_tuning"
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw \
  --format=csv -l 5 >"$GPU_LOG" 2>&1 &
telemetry_pid=$!

"$PYTHON" "$SCRIPT" --config "$CONFIG" --preflight
set +e
"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 \
  "$SCRIPT" --config "$CONFIG"
torchrun_status=$?
set -e

if [[ ! -f "$RESULT" ]]; then
  echo "FLUX2_SCALAR_RESIDUAL_RUNTIME_FAILURE torchrun_status=$torchrun_status" >&2
  /usr/bin/python3 smoke/advisor_governance.py runtime-failure \
    --gate flux2_blur_scalar_residual_runtime \
    --evidence reports/flux2_blur_scalar_residual_feasibility_authorization_v1.json \
    --config "$CONFIG" \
    --model "FLUX.2 Klein Base 4B scalar residual feasibility" \
    --status "$torchrun_status" --log "${LOG#$ROOT/}" --diagnostic-only
  exit 1
fi

/usr/bin/python3 smoke/advisor_governance.py emit \
  --gate flux2_blur_scalar_residual_feasibility_result \
  --evidence "$RESULT" --config "$CONFIG" \
  --model "FLUX.2 Klein Base 4B step-10000 with non-trainable scalar residual correction" \
  --split "training-only frozen 384 calibration plus 128 validation; no development or protected split" \
  --outcome auto \
  --root-cause-or-status "The no-weight-update scalar residual feasibility gate completed. No automatic full-development evaluation or trainable correction follows." \
  --advisor-decision-question "根据冻结 validation gate，是否终止该方向，或另行授权一次 full-development evaluation？" \
  --diagnostic-only --require-decision

if [[ "$torchrun_status" -eq 0 ]]; then
  echo "FLUX2_SCALAR_RESIDUAL_FEASIBILITY_PASS_STOP"
elif [[ "$torchrun_status" -eq 2 ]]; then
  echo "FLUX2_SCALAR_RESIDUAL_SCIENTIFIC_FAILURE_STOP"
else
  echo "ERROR: unexpected torchrun status $torchrun_status" >&2
  exit 1
fi
exit "$torchrun_status"
