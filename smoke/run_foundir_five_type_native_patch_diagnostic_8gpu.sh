#!/usr/bin/env bash
set -euo pipefail

ROOT=/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
/usr/bin/python3 "$ROOT/smoke/advisor_governance.py" check
CONFIG="$ROOT/configs/foundir_five_type_native_patch_diagnostic_v1.json"
PYTHON="$ROOT/.conda/gpu-train-stable/bin/python"
LOG="$ROOT/logs/foundir_five_type_native_patch_diagnostic_v1.log"
GPU_LOG="$ROOT/logs/foundir_five_type_native_patch_diagnostic_v1_gpu.csv"
LOCK="$ROOT/logs/foundir_five_type_native_patch_diagnostic_v1.lock"

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
  echo "FOUNDIR_NATIVE_PATCH_END status=$status utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
trap finish_run EXIT

exec 9>>"$LOCK"
if ! flock -n 9; then
  echo "ERROR: another native-patch diagnostic holds $LOCK" >&2
  exit 1
fi
if [[ "${CUDA_VISIBLE_DEVICES:-}" != "0,1,2,3,4,5,6,7" ]]; then
  echo "ERROR: set CUDA_VISIBLE_DEVICES exactly to 0,1,2,3,4,5,6,7" >&2
  exit 1
fi
if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: workspace Qwen Python environment is missing: $PYTHON" >&2
  exit 1
fi
for executable in flock nvidia-smi; do
  command -v "$executable" >/dev/null 2>&1 || {
    echo "ERROR: required runtime executable is missing: $executable" >&2
    exit 1
  }
done

export TORCH_HOME="$ROOT/models/torch"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

echo "FOUNDIR_NATIVE_PATCH_START utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "FOUNDIR_NATIVE_PATCH_GIT $(git rev-parse HEAD)"
echo "FOUNDIR_NATIVE_PATCH_SCOPE training_only_128_records_no_restoration"
echo "FOUNDIR_NATIVE_PATCH_PROTECTED_OFFICIAL_TEST not_accessed"

nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw \
  --format=csv -l 5 >"$GPU_LOG" 2>&1 &
telemetry_pid=$!
sleep 1
if ! kill -0 "$telemetry_pid" 2>/dev/null; then
  echo "ERROR: nvidia-smi telemetry failed to stay active" >&2
  exit 1
fi

for candidate in joint per_type; do
  output="$ROOT/reports/foundir_five_type_native_patch_${candidate}_v1.json"
  if [[ -f "$output" ]]; then
    "$PYTHON" smoke/evaluate_foundir_five_type_qwen_native_patches_ddp.py \
      --config "$CONFIG" --candidate "$candidate" --verify-existing
  else
    "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 \
      smoke/evaluate_foundir_five_type_qwen_native_patches_ddp.py \
      --config "$CONFIG" --candidate "$candidate"
  fi
done

"$PYTHON" smoke/evaluate_foundir_five_type_qwen_native_patches_ddp.py \
  --config "$CONFIG" --finalize
echo "FOUNDIR_NATIVE_PATCH_PASS utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
