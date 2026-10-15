#!/usr/bin/env bash
set -euo pipefail

ROOT=/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
CONFIG="$ROOT/configs/foundir_official_sow600_qwen_v2_rain_haze.json"
OUTPUT="$ROOT/checkpoints/foundir_official_sow600_qwen_v2_rain_haze"
PYTHON="$ROOT/.venv/flux2-smoke/bin/python"
LOG="$ROOT/logs/foundir_official_sow600_rain_haze_v2.log"
LOCK="$ROOT/logs/foundir_official_sow600_rain_haze_v2.lock"

[[ "${CUDA_VISIBLE_DEVICES:-}" == "0,1,2,3,4,5,6,7" ]] || { echo "ERROR: expose exactly GPUs 0-7" >&2; exit 1; }
cd "$ROOT"
/usr/bin/python3 smoke/advisor_governance.py check
mkdir -p "$ROOT/logs"
exec 9>>"$LOCK"
flock -n 9 || { echo "ERROR: v2 Rain/Haze job is already running" >&2; exit 1; }
exec > >(tee -a "$LOG") 2>&1
export TORCH_HOME="$ROOT/models/torch" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TORCH_NCCL_ASYNC_ERROR_HANDLING=1

"$PYTHON" smoke/restore_foundir_official_sow600_tiled_ddp.py --config "$CONFIG" --stage identity --check-only
"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 \
  smoke/restore_foundir_official_sow600_tiled_ddp.py --config "$CONFIG" --stage rain

[[ "$(find "$OUTPUT/restored" -maxdepth 1 -type f -name '*.png' | wc -l)" -eq 600 ]] || {
  echo "ERROR: v2 outputs are incomplete" >&2
  exit 1
}
echo "FOUNDIR_OFFICIAL_RAIN_HAZE_V2_COMPLETE utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
