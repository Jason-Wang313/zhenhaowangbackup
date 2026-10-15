#!/usr/bin/env bash
set -euo pipefail
ROOT=/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
PYTHON="$ROOT/.conda/gpu-train-stable/bin/python"
CONFIG="$ROOT/configs/foundir_qwen_noise_lowlight_physical_holdout_v3.json"
LOG="$ROOT/logs/foundir_qwen_noise_lowlight_physical_holdout_v3.log"
cd "$ROOT"
[[ "${CUDA_VISIBLE_DEVICES:-}" == "0,1,2,3,4,5,6,7" ]] || exit 1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 smoke/evaluate_foundir_qwen_noise_lowlight_physical_holdout_v3_ddp.py --config "$CONFIG" 2>&1 | tee "$LOG"
