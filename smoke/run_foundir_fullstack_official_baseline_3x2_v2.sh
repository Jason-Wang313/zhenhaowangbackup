#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ "$ROOT" == /inspire/hdd/project/multimodal-discrete-diffusion/liuxiaohong-25080/zhenhaowang ]] || exit 2
[[ $# == 1 && "$1" =~ ^[0-2]$ ]] || exit 2
cd "$ROOT"
# This variable selects the frozen 4-node parent only; new runner partitions into 6 workers.
export FOUNDIR_OFFICIAL_NNODES=4 CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCH_HOME="$ROOT/models/torch"
export PYTHONPATH="$ROOT:$ROOT/third_party/flux2/src"
"$ROOT/.conda/gpu-train-stable/bin/python" -S smoke/foundir_fullstack_executor_3x2_v2.py inventory "$1"
exec 9>/tmp/zhenhaowang_existing_flux_replay.lock
flock -n 9 || { echo 'ERROR: local GPU job active'; exit 3; }
ACTIVE_PROCESSES=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
[[ -z "$ACTIVE_PROCESSES" ]] || { echo 'ERROR: GPU compute process present; preserve it'; exit 3; }
"$ROOT/.conda/gpu-train-stable/bin/python" -u -m torch.distributed.run --standalone --nproc_per_node=2 --max_restarts=0 \
  smoke/audit_foundir_fullstack_official_baseline_3x2_v2.py run --node "$1"
