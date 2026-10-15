#!/usr/bin/env bash
set -euo pipefail

ROOT=/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
CONFIG="$ROOT/configs/foundir_official_sow600_qwen_v6_blur_motion_audit.json"
OUTPUT="$ROOT/checkpoints/foundir_official_sow600_qwen_v6_blur_motion_audit"
RESTORE_PYTHON="$ROOT/.venv/flux2-smoke/bin/python"
METRIC_PYTHON="$ROOT/.venv/foundir-metrics/bin/python"
LOG="$ROOT/logs/foundir_official_sow600_blur_v6.log"
LOCK="$ROOT/logs/foundir_official_sow600_blur_v6.lock"

[[ "${CUDA_VISIBLE_DEVICES:-}" == "0,1,2,3,4,5,6,7" ]] || { echo "ERROR: expose exactly GPUs 0-7" >&2; exit 1; }
cd "$ROOT"
/usr/bin/python3 smoke/advisor_governance.py check
mkdir -p "$ROOT/logs"
exec 9>>"$LOCK"
flock -n 9 || { echo "ERROR: v6 Blur audit is already running" >&2; exit 1; }
exec > >(tee -a "$LOG") 2>&1
export TORCH_HOME="$ROOT/models/torch" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
"$RESTORE_PYTHON" smoke/restore_foundir_official_sow600_tiled_ddp.py --config "$CONFIG" --stage identity --check-only
"$RESTORE_PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 \
  smoke/restore_foundir_official_sow600_tiled_ddp.py --config "$CONFIG" --stage blur
[[ "$(find "$OUTPUT/restored" -maxdepth 1 -type f -name '*.png' | wc -l)" -eq 600 ]] || { echo "ERROR: v6 outputs incomplete" >&2; exit 1; }
[[ -f "$OUTPUT/stages/blur.json" ]] || { echo "ERROR: v6 Blur stage is incomplete" >&2; exit 1; }
"$METRIC_PYTHON" smoke/evaluate_foundir_official_sow600_pyiqa_ddp.py --config "$CONFIG" --check-only
"$METRIC_PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 \
  smoke/evaluate_foundir_official_sow600_pyiqa_ddp.py --config "$CONFIG"
echo "FOUNDIR_OFFICIAL_BLUR_V6_COMPLETE result=$OUTPUT/official_result.json"
