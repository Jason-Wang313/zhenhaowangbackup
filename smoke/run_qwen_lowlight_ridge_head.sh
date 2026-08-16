#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${QWEN_LOWLIGHT_PYTHON:-$ROOT/.venv/flux2-smoke/bin/python}"
CONFIG="${QWEN_LOWLIGHT_RIDGE_CONFIG:-$ROOT/configs/qwen_lowlight_ridge_head.json}"
LOG="$ROOT/logs/qwen_lowlight_ridge_head.log"
LOCK="$ROOT/logs/qwen_lowlight_ridge_head.lock"

if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: validated workspace Python is unavailable: $PYTHON" >&2
  exit 1
fi

mkdir -p "$ROOT/logs"
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "ERROR: Qwen ridge-head run is already active" >&2
  exit 1
fi

cd "$ROOT"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

"$PYTHON" smoke/qwen_lowlight_ridge_preflight.py --config "$CONFIG"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  "$PYTHON" -u train_qwen_lowlight_ridge_head.py --config "$CONFIG" \
  2>&1 | tee -a "$LOG"
