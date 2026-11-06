#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ $# == 2 && "$1" == 3 && "$2" =~ ^(cache|train|official)$ ]] || exit 2
cd "$ROOT"
export CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCH_HOME="$ROOT/models/torch"
export CUDNN_CONV_WSCAP_DBG=256 TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT=64
export PYTHONPATH="$ROOT:$ROOT/third_party/flux2/src"
PY="$ROOT/.conda/gpu-train-stable/bin/python"
"$PY" -S smoke/foundir_fullstack_node3_extension_v1.py inventory 3
exec 9>/tmp/zhenhaowang_existing_flux_replay.lock
flock -n 9 || exit 3
ACTIVE_PROCESSES=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
[[ -z "$ACTIVE_PROCESSES" ]] || { echo 'ERROR: existing GPU process; preserve it'; exit 3; }
PHASE="$2"
SCRIPT=smoke/run_foundir_fullstack_blur_coverage_v2.py
if [[ "$PHASE" == official ]]; then
  SCRIPT=smoke/evaluate_foundir_blur_coverage_official_v2.py
  PHASE=run
fi
set +e
"$PY" -u -m torch.distributed.run --standalone --nproc_per_node=2 --max_restarts=0 "$SCRIPT" "$PHASE"
status=$?
set -e
printf 'BLUR_COVERAGE_LAUNCH_END node=3 status=%s\n' "$status"
exit "$status"
