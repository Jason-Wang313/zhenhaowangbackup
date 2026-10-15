#!/usr/bin/env bash
set -euo pipefail

ROOT=/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
/usr/bin/python3 "$ROOT/smoke/advisor_governance.py" check
PYTHON="$ROOT/.venv/flux2-smoke/bin/python"
CONFIG="$ROOT/configs/foundir_noise_rain_full_development_audit_v1.json"
EVALUATOR="$ROOT/smoke/evaluate_foundir_noise_rain_full_development_ddp.py"
ROUTER="$ROOT/configs/foundir_five_type_qwen_routed_integrated_v2.json"
CONDITIONS="$ROOT/manifests/foundir_five_type_all_qwen_conditions_v3.csv"
GENERALIST="$ROOT/third_party/foundir/premodel/model-2000.pt"
BLUR="$ROOT/checkpoints/foundir_five_type_flux2_dit_v3/protocol_v4_isolated/blur_generalist_specialization_gate_v2/step_003000.pt"
LOWLIGHT="$ROOT/third_party/foundir/specialist_model/ckpt/lowlight.pth"
WEATHER="$ROOT/third_party/foundir/specialist_model/ckpt/weather.pth"
LOG="$ROOT/logs/foundir_noise_rain_full_development_audit_v1.log"
GPU_LOG="$ROOT/logs/foundir_noise_rain_full_development_audit_v1_gpu.csv"
LOCK="$ROOT/logs/foundir_noise_rain_full_development_audit_v1.lock"

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
  echo "FOUNDIR_NOISE_RAIN_FULL_DEVELOPMENT_AUDIT_V1_END status=$status utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
trap finish_run EXIT

exec 9>>"$LOCK"
flock -n 9 || {
  echo "ERROR: another Noise/Rain full-development audit holds $LOCK" >&2
  exit 1
}
[[ "${CUDA_VISIBLE_DEVICES:-}" == "0,1,2,3,4,5,6,7" ]] || {
  echo "ERROR: expose exactly CUDA devices 0,1,2,3,4,5,6,7" >&2
  exit 1
}

export TORCH_HOME="$ROOT/models/torch"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export FOUNDIR_EVALUATION_COORDINATION_TOKEN=noise-rain-full-development-audit-v1

echo "FOUNDIR_NOISE_RAIN_FULL_DEVELOPMENT_AUDIT_V1_START utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "FOUNDIR_NOISE_RAIN_FULL_DEVELOPMENT_AUDIT_V1_WORKSPACE $ROOT"
echo "FOUNDIR_NOISE_RAIN_FULL_DEVELOPMENT_AUDIT_V1_GIT $(git rev-parse HEAD)"
echo "FOUNDIR_NOISE_RAIN_FULL_DEVELOPMENT_AUDIT_V1_CONFIG_SHA256 $(sha256sum "$CONFIG" | cut -d' ' -f1)"
echo "FOUNDIR_NOISE_RAIN_FULL_DEVELOPMENT_AUDIT_V1_EVALUATOR_SHA256 $(sha256sum "$EVALUATOR" | cut -d' ' -f1)"
echo "FOUNDIR_NOISE_RAIN_FULL_DEVELOPMENT_AUDIT_V1_ROUTER_SHA256 $(sha256sum "$ROUTER" | cut -d' ' -f1)"
echo "FOUNDIR_NOISE_RAIN_FULL_DEVELOPMENT_AUDIT_V1_CONDITIONS_SHA256 $(sha256sum "$CONDITIONS" | cut -d' ' -f1)"
echo "FOUNDIR_NOISE_RAIN_FULL_DEVELOPMENT_AUDIT_V1_GENERALIST_SHA256 $(sha256sum "$GENERALIST" | cut -d' ' -f1)"
echo "FOUNDIR_NOISE_RAIN_FULL_DEVELOPMENT_AUDIT_V1_BLUR_SHA256 $(sha256sum "$BLUR" | cut -d' ' -f1)"
echo "FOUNDIR_NOISE_RAIN_FULL_DEVELOPMENT_AUDIT_V1_LOWLIGHT_SHA256 $(sha256sum "$LOWLIGHT" | cut -d' ' -f1)"
echo "FOUNDIR_NOISE_RAIN_FULL_DEVELOPMENT_AUDIT_V1_WEATHER_SHA256 $(sha256sum "$WEATHER" | cut -d' ' -f1)"
echo "FOUNDIR_NOISE_RAIN_FULL_DEVELOPMENT_AUDIT_V1_DEVICES $CUDA_VISIBLE_DEVICES"
echo "FOUNDIR_NOISE_RAIN_FULL_DEVELOPMENT_AUDIT_V1_SCOPE development_only_7175_records"
echo "FOUNDIR_NOISE_RAIN_FULL_DEVELOPMENT_AUDIT_V1_INTERNAL sealed_internal_not_accessed"
echo "FOUNDIR_NOISE_RAIN_FULL_DEVELOPMENT_AUDIT_V1_PROTECTED_OFFICIAL_TEST not_accessed"

{
  echo "RUN_START,$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw \
    --format=csv -l 5
} >>"$GPU_LOG" 2>&1 &
telemetry_pid=$!

"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 \
  "$EVALUATOR" --config "$CONFIG"
