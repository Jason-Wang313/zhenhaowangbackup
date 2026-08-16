#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${QWEN_LOWLIGHT_PYTHON:-$ROOT/.venv/flux2-smoke/bin/python}"
CONFIG="${QWEN_LOWLIGHT_CONFIG:-$ROOT/configs/qwen_lowlight_smoke25.json}"
LOG="$ROOT/logs/qwen_lowlight_smoke25.log"

if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: validated workspace Python is unavailable: $PYTHON" >&2
  exit 1
fi

cd "$ROOT"
"$PYTHON" smoke/qwen_lowlight_preflight.py --config "$CONFIG"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  "$PYTHON" -u predict_qwen_lowlight.py --config "$CONFIG" 2>&1 | tee -a "$LOG"
