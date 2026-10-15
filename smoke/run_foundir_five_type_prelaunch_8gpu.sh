#!/usr/bin/env bash
set -euo pipefail

ROOT=/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
/usr/bin/python3 "$ROOT/smoke/advisor_governance.py" check
CONFIG="$ROOT/configs/foundir_five_type_flux2_dit_v1.json"
PRELAUNCH="$ROOT/configs/foundir_five_type_prelaunch_gate_v1.json"
QWEN_PYTHON="$ROOT/.conda/gpu-train-stable/bin/python"
FLUX_PYTHON="$ROOT/.venv/flux2-smoke/bin/python"
PER_TYPE_JSON="$ROOT/manifests/foundir_five_type_overfit_qwen_per_type_predictions_v1.json"
LOG="$ROOT/logs/foundir_five_type_prelaunch_v1.log"
GPU_LOG="$ROOT/logs/foundir_five_type_prelaunch_v1_gpu.csv"
LOCK="$ROOT/logs/foundir_five_type_prelaunch_v1.lock"

cd "$ROOT"
mkdir -p "$ROOT/logs" "$ROOT/.cache/triton/foundir_five_type" \
  "$ROOT/.cache/torch_extensions/foundir_five_type"
exec > >(tee -a "$LOG") 2>&1

telemetry_pid=
finish_run() {
  status=$?
  if [[ -n "$telemetry_pid" ]]; then
    kill "$telemetry_pid" 2>/dev/null || true
    wait "$telemetry_pid" 2>/dev/null || true
  fi
  echo "FOUNDIR_FIVE_TYPE_PRELAUNCH_END status=$status utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
trap finish_run EXIT

exec 9>>"$LOCK"
if ! flock -n 9; then
  echo "ERROR: another FoundIR five-type prelaunch holds $LOCK" >&2
  exit 1
fi
if [[ "${CUDA_VISIBLE_DEVICES:-}" != "0,1,2,3,4,5,6,7" ]]; then
  echo "ERROR: set CUDA_VISIBLE_DEVICES exactly to 0,1,2,3,4,5,6,7" >&2
  exit 1
fi
if [[ ! -x "$QWEN_PYTHON" || ! -x "$FLUX_PYTHON" ]]; then
  echo "ERROR: workspace Qwen or FLUX Python environment is missing" >&2
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
export TRITON_CACHE_DIR="$ROOT/.cache/triton/foundir_five_type"
export TORCH_EXTENSIONS_DIR="$ROOT/.cache/torch_extensions/foundir_five_type"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

echo "FOUNDIR_FIVE_TYPE_PRELAUNCH_START utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "FOUNDIR_FIVE_TYPE_PRELAUNCH_GIT $(git rev-parse HEAD)"
echo "FOUNDIR_FIVE_TYPE_PRELAUNCH_HARDWARE exactly_8x_homogeneous_H100_or_H200"
echo "FOUNDIR_FIVE_TYPE_PRELAUNCH_PROTECTED_OFFICIAL_TEST not_accessed"

nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw \
  --format=csv -l 5 >"$GPU_LOG" 2>&1 &
telemetry_pid=$!
sleep 1
if ! kill -0 "$telemetry_pid" 2>/dev/null; then
  echo "ERROR: nvidia-smi telemetry failed to stay active" >&2
  exit 1
fi

# Prove the eight-rank transport first so a broken NCCL setup fails before any
# large model is loaded.
"$FLUX_PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 \
  smoke/foundir_five_type_distributed_canary.py

# The complete 685,495-pair manifest was already fully decoded, aligned, and
# hashed.  The frozen Joint-Qwen 128-row pilot is its content-addressed subset.
# Re-reading the 265 MiB CSV several times here adds no new evidence.  The
# per-type exporter instead verifies the exact hashes of the 128 LQ files it
# will actually inspect, and the final gate verifies every frozen artifact.
if [[ -f "$PER_TYPE_JSON" ]]; then
  "$QWEN_PYTHON" smoke/export_foundir_five_type_qwen_per_type_pilot.py \
    --prelaunch-config "$PRELAUNCH" --verify-existing
else
  "$QWEN_PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 \
    smoke/export_foundir_five_type_qwen_per_type_pilot.py \
    --prelaunch-config "$PRELAUNCH"
fi
"$QWEN_PYTHON" smoke/evaluate_foundir_five_type_qwen_training_gate.py \
  --prelaunch-config "$PRELAUNCH"

# Only candidates that pass the scientific gate pay the cost of checking the
# complete future FLUX/released-baseline runtime. This is still before any
# restoration optimizer step.
"$FLUX_PYTHON" smoke/foundir_five_type_runtime_preflight.py \
  --config "$CONFIG" --require-cuda --full-deserialization
"$QWEN_PYTHON" smoke/freeze_foundir_five_type_prelaunch.py \
  --prelaunch-config "$PRELAUNCH"

echo "FOUNDIR_FIVE_TYPE_PRELAUNCH_PASS utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
