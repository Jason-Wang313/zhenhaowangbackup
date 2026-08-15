#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$ROOT/.venv/flux2-smoke/bin/python"
CONFIG="$ROOT/configs/lowlight_flux2_overfit16.json"
LOG="$ROOT/logs/lowlight_baseline_overfit16.log"
LOCK="$ROOT/logs/lowlight_baseline_overfit16.lock"

if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: validated workspace Python is unavailable: $PYTHON" >&2
  exit 1
fi
mkdir -p "$ROOT/logs"
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "ERROR: another low-light baseline smoke is running: $LOCK" >&2
  exit 1
fi

cd "$ROOT"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

"$PYTHON" -u "$ROOT/smoke/lowlight_baseline_preflight.py" --config "$CONFIG" 2>&1 | tee -a "$LOG"
"$PYTHON" -u "$ROOT/train_flux2_lowlight.py" --config "$CONFIG" --resume 2>&1 | tee -a "$LOG"
"$PYTHON" -u "$ROOT/smoke/evaluate_lowlight_overfit.py" --config "$CONFIG" 2>&1 | tee -a "$LOG"

echo "LOWLIGHT BASELINE OVERFIT AUTOMATIC GATE PASS; inspect contact sheets for identity" | tee -a "$LOG"
