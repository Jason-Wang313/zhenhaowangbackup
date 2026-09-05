#!/usr/bin/env bash
set -euo pipefail

ROOT=/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
PYTHON="$ROOT/.conda/gpu-train-stable/bin/python"
CONFIG="$ROOT/configs/qwen_multidegradation_100k_mild_v7_data.json"
LOG="$ROOT/logs/qwen_multidegradation_100k_mild_v7_generation.log"
LOCK="$ROOT/logs/qwen_multidegradation_100k_mild_v7_generation.lock"
WORKERS=${QWEN_V7_GENERATION_WORKERS:-64}
CHUNK_SIZE=${QWEN_V7_GENERATION_CHUNK_SIZE:-128}

cd "$ROOT"
mkdir -p "$ROOT/logs"
exec > >(tee -a "$LOG") 2>&1

finish_run() {
  status=$?
  echo "QWEN_MILD_V7_GENERATION_END status=$status utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
trap finish_run EXIT

exec 9>>"$LOCK"
if ! flock -n 9; then
  echo "ERROR: another Qwen mild-v7 generation holds $LOCK" >&2
  exit 1
fi
if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: workspace Qwen Python is missing: $PYTHON" >&2
  exit 1
fi
if ! [[ "$WORKERS" =~ ^[1-9][0-9]*$ && "$CHUNK_SIZE" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: generation workers and chunk size must be positive integers" >&2
  exit 1
fi

export PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
echo "QWEN_MILD_V7_GENERATION_START utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "QWEN_MILD_V7_GENERATION_GIT $(git rev-parse HEAD)"
echo "QWEN_MILD_V7_GENERATION_POLICY synthetic_only_no_foundir_training_or_test"
echo "QWEN_MILD_V7_GENERATION_RESOURCES workers=$WORKERS chunk_size=$CHUNK_SIZE"

"$PYTHON" smoke/build_qwen_multidegradation_100k.py --config "$CONFIG"
"$PYTHON" smoke/generate_qwen_multidegradation_100k.py \
  --config "$CONFIG" \
  --workers "$WORKERS" \
  --chunk-size "$CHUNK_SIZE" \
  --group-by-source

echo "QWEN_MILD_V7_GENERATION_PASS utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
