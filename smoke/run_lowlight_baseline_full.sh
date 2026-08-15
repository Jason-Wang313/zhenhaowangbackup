#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$ROOT/.venv/flux2-smoke/bin/python"
CONFIG="$ROOT/configs/lowlight_flux2_full.json"
LOG="$ROOT/logs/lowlight_baseline_full.log"
LOCK="$ROOT/logs/lowlight_baseline_full.lock"

if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: validated workspace Python is unavailable: $PYTHON" >&2
  exit 1
fi
mkdir -p "$ROOT/logs"
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "ERROR: another full low-light baseline run is active: $LOCK" >&2
  exit 1
fi

cd "$ROOT"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_HOME="$ROOT/models/torch"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

"$PYTHON" -u "$ROOT/smoke/lowlight_baseline_preflight.py" --config "$CONFIG" 2>&1 | tee -a "$LOG"
"$PYTHON" -u "$ROOT/train_flux2_lowlight.py" --config "$CONFIG" --resume 2>&1 | tee -a "$LOG"
"$PYTHON" -u "$ROOT/evaluate_flux2_lowlight_split.py" --config "$CONFIG" --split validation --evaluation-name validation_full 2>&1 | tee -a "$LOG"
"$PYTHON" -u "$ROOT/evaluate_flux2_lowlight_split.py" --config "$CONFIG" --split test --evaluation-name test_full 2>&1 | tee -a "$LOG"

echo "LOWLIGHT FULL BASELINE INTERNAL EVALUATION COMPLETE; inspect validation/test contact sheets" | tee -a "$LOG"
