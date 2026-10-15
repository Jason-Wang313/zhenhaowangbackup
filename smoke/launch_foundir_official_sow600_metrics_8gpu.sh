#!/usr/bin/env bash
set -euo pipefail

ROOT=/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
PYTHON="$ROOT/.venv/foundir-metrics/bin/python"
CONFIG="$ROOT/configs/foundir_official_sow600_qwen_v1.json"
OUTPUT="$ROOT/checkpoints/foundir_official_sow600_qwen_v1"
LOG="$ROOT/logs/foundir_official_sow600_metrics_v1.log"
LOCK="$ROOT/logs/foundir_official_sow600_metrics_v1.lock"

cd "$ROOT"
[[ "${CUDA_VISIBLE_DEVICES:-}" == "0,1,2,3,4,5,6,7" ]] || { echo "ERROR: expose exactly GPUs 0-7" >&2; exit 1; }
[[ ! -f "$OUTPUT/official_result.json" ]] || { echo "ERROR: official terminal result already exists" >&2; exit 1; }
for stage in rain noise low_light blur haze; do
  for shard in 0 1 2 3 4; do
    [[ -f "$OUTPUT/stages/${stage}.shard_${shard}_of_5.json" ]] || {
      echo "ERROR: missing completed shard marker: stage=$stage shard=$shard" >&2
      exit 1
    }
  done
done
mkdir -p "$ROOT/logs"
exec 9>>"$LOCK"
flock -n 9 || { echo "ERROR: official metrics are already running" >&2; exit 1; }
exec > >(tee -a "$LOG") 2>&1
export TORCH_HOME="$ROOT/models/torch" HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
echo "FOUNDIR_OFFICIAL_METRICS_START utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
"$PYTHON" smoke/evaluate_foundir_official_sow600_pyiqa_ddp.py --config "$CONFIG" --check-only
"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 \
  smoke/evaluate_foundir_official_sow600_pyiqa_ddp.py --config "$CONFIG"
[[ -f "$OUTPUT/official_result.json" ]] || { echo "ERROR: official result missing" >&2; exit 1; }
echo "FOUNDIR_OFFICIAL_METRICS_COMPLETE result=$OUTPUT/official_result.json"
