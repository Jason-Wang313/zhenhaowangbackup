#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ $# == 1 && "$1" =~ ^[0-2]$ ]] || exit 2
cd "$ROOT"
export CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$ROOT:$ROOT/third_party/flux2/src"
PY="$ROOT/.conda/gpu-train-stable/bin/python"
"$PY" -S smoke/foundir_fullstack_executor_3x2_v2.py inventory "$1"
exec 9>/tmp/zhenhaowang_existing_flux_replay.lock
flock -n 9 || { echo 'ERROR: local GPU job active'; exit 3; }
ACTIVE_PROCESSES=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
[[ -z "$ACTIVE_PROCESSES" ]] || { echo 'ERROR: GPU busy'; exit 3; }
set +e
timeout --signal=TERM --kill-after=30 1800 "$PY" -u -m torch.distributed.run --standalone --nproc_per_node=2 --max_restarts=0 \
 smoke/train_foundir_fullstack_noise_texture_router_v3.py run --node "$1"
status=$?
set -e
printf 'TEXTURE_LAUNCH_END node=%s status=%s\n' "$1" "$status"
exit "$status"
