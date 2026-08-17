#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$ROOT/.venv/flux2-smoke/bin/python"
CONFIG="$ROOT/configs/lowlight_flux2_qwen_conditioned_full.json"
BASELINE_CONFIG="$ROOT/configs/lowlight_flux2_full.json"
LOG="$ROOT/logs/lowlight_qwen_conditioned_full.log"
LOCK="$ROOT/logs/lowlight_qwen_conditioned_full.lock"

if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: validated workspace Python is unavailable: $PYTHON" >&2
  exit 1
fi
mkdir -p "$ROOT/logs"
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "ERROR: another Qwen-conditioned Lowlight run is active: $LOCK" >&2
  exit 1
fi

cd "$ROOT"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_HOME="$ROOT/models/torch"
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

"$PYTHON" -u smoke/lowlight_baseline_preflight.py --config "$BASELINE_CONFIG" 2>&1 | tee -a "$LOG"
"$PYTHON" -u smoke/lowlight_qwen_conditioned_preflight.py --config "$CONFIG" 2>&1 | tee -a "$LOG"
"$PYTHON" -u smoke/export_qwen_lowlight_condition_predictions.py --config "$CONFIG" 2>&1 | tee -a "$LOG"
"$PYTHON" -u smoke/lowlight_qwen_conditioned_preflight.py --config "$CONFIG" --require-predictions 2>&1 | tee -a "$LOG"
"$PYTHON" -u smoke/train_flux2_lowlight_qwen_conditioned.py --config "$CONFIG" --resume 2>&1 | tee -a "$LOG"
"$PYTHON" -u smoke/evaluate_flux2_lowlight_qwen_conditioned.py --config "$CONFIG" --split validation --evaluation-name validation_full 2>&1 | tee -a "$LOG"
"$PYTHON" -u smoke/evaluate_flux2_lowlight_qwen_conditioned.py --config "$CONFIG" --split test --evaluation-name test_full 2>&1 | tee -a "$LOG"
"$PYTHON" -u smoke/compare_lowlight_baseline_qwen.py --config "$CONFIG" 2>&1 | tee -a "$LOG"

echo "LOWLIGHT QWEN-CONDITIONED INTERNAL COMPARISON COMPLETE; inspect paired contact sheets" | tee -a "$LOG"
