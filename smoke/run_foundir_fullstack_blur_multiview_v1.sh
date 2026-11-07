#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ $# == 1 && "$1" =~ ^(1|3)$ ]] || exit 2
cd "$ROOT"
export CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCH_HOME="$ROOT/models/torch"
export CUDNN_CONV_WSCAP_DBG=256 TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT=64
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NCCL_IB_DISABLE=1
export PYTHONPATH="$ROOT:$ROOT/third_party/flux2/src"
PY="$ROOT/.conda/gpu-train-stable/bin/python"
if [[ "$1" == 3 ]]; then
  "$PY" -S smoke/foundir_fullstack_node3_extension_v1.py inventory 3
else
  "$PY" -S smoke/foundir_fullstack_executor_3x2_v2.py inventory 1
fi
exec 9>/tmp/zhenhaowang_existing_flux_replay.lock
flock -n 9 || exit 3
ACTIVE_PROCESSES=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
[[ -z "$ACTIVE_PROCESSES" ]] || { echo 'ERROR: GPU process present; preserve it'; exit 3; }
set +e
"$PY" -u -m torch.distributed.run --nnodes=2 --nproc_per_node=2 --node_rank="$(( ($1-1)/2 ))" \
  --master_addr=10.244.47.55 --master_port=29883 --max_restarts=0 smoke/train_foundir_blur_multiview_v1.py run
status=$?
set -e
printf 'BLUR_MULTIVIEW_LAUNCH_END node=%s status=%s\n' "$1" "$status"
exit "$status"
