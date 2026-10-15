#!/usr/bin/env bash
set -euo pipefail
ROOT=/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
PYTHON="$ROOT/.venv/flux2-smoke/bin/python"
LOG="$ROOT/logs/foundir_noise_route_cross_compatibility_v1.log"
cd "$ROOT"
[[ "${CUDA_VISIBLE_DEVICES:-}" == "0,1,2,3,4,5,6,7" ]] || exit 1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCH_HOME="$ROOT/models/torch" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 smoke/evaluate_foundir_noise_route_cross_compatibility_ddp.py 2>&1 | tee "$LOG"
