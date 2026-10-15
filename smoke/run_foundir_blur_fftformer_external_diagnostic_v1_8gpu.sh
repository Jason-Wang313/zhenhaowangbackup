#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="$ROOT/configs/foundir_blur_fftformer_external_diagnostic_v1.json"
SCRIPT="$ROOT/smoke/evaluate_foundir_blur_fftformer_external_diagnostic_ddp.py"
RESTORE_PYTHON="$ROOT/.venv/flux2-smoke/bin/python"
METRIC_PYTHON="$ROOT/.venv/foundir-metrics/bin/python"
OUTPUT="$ROOT/checkpoints/foundir_blur_fftformer_external_diagnostic_v1"

cd "$ROOT"
export TORCH_HOME="$ROOT/models/torch"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
[[ -x "$RESTORE_PYTHON" && -x "$METRIC_PYTHON" ]] || {
  echo "Frozen restoration/metric Python environments are missing" >&2
  exit 1
}
"$RESTORE_PYTHON" "$SCRIPT" --config "$CONFIG" --stage check
"$METRIC_PYTHON" -c 'import pyiqa; print("FFT_DIAGNOSTIC_PYIQA_PREFLIGHT_PASS")'
if [[ ! -f "$OUTPUT/restoration_complete.json" ]]; then
  "$RESTORE_PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 \
    "$SCRIPT" --config "$CONFIG" --stage restore
else
  echo "FFT_DIAGNOSTIC_RESTORE_REUSED records=150"
fi
"$METRIC_PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 \
  "$SCRIPT" --config "$CONFIG" --stage metrics
