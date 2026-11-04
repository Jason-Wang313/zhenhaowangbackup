#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ "$ROOT" == /inspire/hdd/project/multimodal-discrete-diffusion/liuxiaohong-25080/zhenhaowang && $# == 0 ]] || { echo 'ERROR: destination CPU audit only'; exit 2; }
cd "$ROOT"
PYTHON="$ROOT/.conda/gpu-train-stable/bin/python"
[[ -x "$PYTHON" ]] || { echo 'ERROR: qualified Python missing'; exit 2; }
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH="$ROOT:$ROOT/third_party/flux2/src"
mkdir -p logs
exec 9>/tmp/zhenhaowang_router_generalization_audit.lock
flock -n 9 || { echo 'ERROR: audit already running'; exit 3; }
exec > >(tee -a logs/foundir_router_generalization_audit_v1.log) 2>&1
START=$SECONDS
finish() {
  status=$?
  printf 'ROUTER_GENERALIZATION_AUDIT_END status=%s seconds=%s GPU_used=false\n' "$status" "$((SECONDS-START))"
}
trap finish EXIT
echo 'ROUTER_GENERALIZATION_AUDIT_START CPU_threads=8 training=false GPU_used=false cap_seconds=600'
timeout --signal=TERM --kill-after=30s 600s "$PYTHON" -u smoke/analyze_foundir_router_generalization_v1.py coverage
# No automatic retry, training, GPU work, or follow-up.
