#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$ROOT/.venv/flux2-smoke/bin/python"
LOG="$ROOT/logs/flux2_gpu_smoke.log"
LOCK="$ROOT/logs/flux2_gpu_smoke.lock"

cd "$ROOT"
mkdir -p "$ROOT/logs"

if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: dedicated FLUX.2 Python environment is missing: $PYTHON" >&2
  exit 1
fi

exec 9>"$LOCK"
if ! flock -n 9; then
  echo "ERROR: another FLUX.2 smoke process is already running" >&2
  exit 1
fi

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "[$(date -u +%FT%TZ)] Starting FLUX.2 end-to-end GPU smoke" | tee "$LOG"
"$PYTHON" -u "$ROOT/smoke/run_flux2_gpu_smoke.py" 2>&1 | tee -a "$LOG"
