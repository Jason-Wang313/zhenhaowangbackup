#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
RUN_ID=${1:-memory_isolated_vjp_v5}
[[ "$RUN_ID" =~ ^[A-Za-z0-9_-]+$ ]] || exit 2
PYTHON=${FLUX_NATIVE_PYTHON:-"$ROOT/.conda/gpu-train-stable/bin/python"}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCH_HOME="$ROOT/models/torch"
export PYTHONPATH="$ROOT:$ROOT/third_party/flux2/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p logs
exec 8>/tmp/zhenhaowang_flux_native_energy.lock
flock -n 8 || { echo 'ERROR: another native-energy job owns this node'; exit 3; }
START_EPOCH=$(date -u +%s)
LOG="logs/flux_native_energy_memory_${RUN_ID}_$(date -u +%Y%m%dT%H%M%SZ).log"
(
  trap 'rc=$?; echo "NATIVE_ENERGY_MEMORY_END status=$rc wall_seconds=$(( $(date -u +%s)-START_EPOCH )) gpus=1"' EXIT
  echo 'Single-GPU cached-sample memory diagnostic; full three updates/backward; no optimizer step; cap1200s.'
  timeout --signal=TERM --kill-after=30s 1200s "$PYTHON" -u \
    smoke/profile_foundir_flux_native_energy_memory_v1.py --run-id "$RUN_ID"
) 2>&1 | tee "$LOG"
