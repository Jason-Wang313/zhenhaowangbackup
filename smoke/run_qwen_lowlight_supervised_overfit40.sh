#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${QWEN_SUPERVISED_PYTHON:-$ROOT/.venv/flux2-smoke/bin/python}"
CONFIG="${QWEN_SUPERVISED_CONFIG:-$ROOT/configs/qwen_lowlight_supervised_overfit40.json}"
DEVICE_SET="${CUDA_VISIBLE_DEVICES:-0}"
LOG="$ROOT/logs/qwen_lowlight_supervised_overfit40.log"
LOCK="$ROOT/logs/qwen_lowlight_supervised_overfit40.lock"

if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: validated workspace Python is unavailable: $PYTHON" >&2
  exit 1
fi
if [[ "$DEVICE_SET" == *,* ]]; then
  echo "ERROR: the bounded overfit smoke must expose exactly one GPU: $DEVICE_SET" >&2
  exit 1
fi

mkdir -p "$ROOT/logs"
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "ERROR: supervised Qwen overfit gate is already active" >&2
  exit 1
fi

cd "$ROOT"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

{
  echo "QWEN_SUPERVISED_LAUNCH_BEGIN $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "workspace=$ROOT"
  echo "config=$CONFIG"
  echo "CUDA_VISIBLE_DEVICES=$DEVICE_SET"
  nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
  CUDA_VISIBLE_DEVICES="$DEVICE_SET" "$PYTHON" -c \
    'import torch; print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available(), "count", torch.cuda.device_count())'

  CUDA_VISIBLE_DEVICES="$DEVICE_SET" "$PYTHON" -u \
    smoke/train_qwen_lowlight_supervised.py \
    --config "$CONFIG" --resume auto --stop-after-step 2

  CUDA_VISIBLE_DEVICES="$DEVICE_SET" "$PYTHON" -u \
    smoke/train_qwen_lowlight_supervised.py \
    --config "$CONFIG" --resume auto
  echo "QWEN_SUPERVISED_LAUNCH_END $(date -u +%Y-%m-%dT%H:%M:%SZ)"
} 2>&1 | tee -a "$LOG"
