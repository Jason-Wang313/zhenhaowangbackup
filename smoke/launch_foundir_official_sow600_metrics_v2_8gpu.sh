#!/usr/bin/env bash
set -euo pipefail

ROOT=/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
CONFIG="$ROOT/configs/foundir_official_sow600_qwen_v2_rain_haze.json"
OUTPUT="$ROOT/checkpoints/foundir_official_sow600_qwen_v2_rain_haze"
PYTHON="$ROOT/.venv/foundir-metrics/bin/python"
LOG="$ROOT/logs/foundir_official_sow600_metrics_v2.log"
LOCK="$ROOT/logs/foundir_official_sow600_metrics_v2.lock"

[[ "${CUDA_VISIBLE_DEVICES:-}" == "0,1,2,3,4,5,6,7" ]] || { echo "ERROR: expose exactly GPUs 0-7" >&2; exit 1; }
[[ -f "$OUTPUT/stages/haze_feedback.json" ]] || { echo "ERROR: Haze v2 is not prepared" >&2; exit 1; }
[[ -f "$OUTPUT/stages/rain.json" ]] || { echo "ERROR: Rain v2 is not complete" >&2; exit 1; }
[[ ! -f "$OUTPUT/official_result.json" ]] || { echo "ERROR: v2 terminal result already exists" >&2; exit 1; }
[[ "$(find "$OUTPUT/restored" -maxdepth 1 -type f -name '*.png' | wc -l)" -eq 600 ]] || { echo "ERROR: expected 600 outputs" >&2; exit 1; }
cd "$ROOT"
mkdir -p "$ROOT/logs"
exec 9>>"$LOCK"
flock -n 9 || { echo "ERROR: v2 metrics are already running" >&2; exit 1; }
exec > >(tee -a "$LOG") 2>&1
export TORCH_HOME="$ROOT/models/torch" HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
"$PYTHON" smoke/evaluate_foundir_official_sow600_pyiqa_ddp.py --config "$CONFIG" --check-only
"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 \
  smoke/evaluate_foundir_official_sow600_pyiqa_ddp.py --config "$CONFIG"
echo "FOUNDIR_OFFICIAL_METRICS_V2_COMPLETE result=$OUTPUT/official_result.json"
