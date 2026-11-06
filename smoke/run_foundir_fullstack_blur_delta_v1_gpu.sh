#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ $# == 2 && "$1" == 3 && "$2" =~ ^(qualify|cache|train|evaluate)$ ]] || exit 2
cd "$ROOT"
export CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCH_HOME="$ROOT/models/torch"
export CUDNN_CONV_WSCAP_DBG=256 TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT=64
export PYTHONPATH="$ROOT:$ROOT/third_party/flux2/src"
PY="$ROOT/.conda/gpu-train-stable/bin/python"
"$PY" -S smoke/foundir_fullstack_node3_extension_v1.py inventory 3
exec 9>/tmp/zhenhaowang_existing_flux_replay.lock
flock -n 9 || { echo 'ERROR: local GPU job active'; exit 3; }
ACTIVE_PROCESSES=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
[[ -z "$ACTIVE_PROCESSES" ]] || { echo 'ERROR: GPU compute process present; preserve it'; exit 3; }
set +e
"$PY" -u -m torch.distributed.run --standalone --nproc_per_node=2 --max_restarts=0 \
 smoke/run_foundir_fullstack_blur_delta_v1.py "$2"
status=$?
set -e
printf 'BLUR_DELTA_LAUNCH_END phase=%s node=3 status=%s\n' "$2" "$status"
exit "$status"
