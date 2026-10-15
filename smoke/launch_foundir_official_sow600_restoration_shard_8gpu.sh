#!/usr/bin/env bash
set -euo pipefail

ROOT=/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
PYTHON="$ROOT/.venv/flux2-smoke/bin/python"
CONFIG="$ROOT/configs/foundir_official_sow600_qwen_v1.json"
OUTPUT="$ROOT/checkpoints/foundir_official_sow600_qwen_v1"
SHARD_COUNT="${SHARD_COUNT:-5}"
SHARD_INDEX="${SHARD_INDEX:?set SHARD_INDEX to 0, 1, 2, 3, or 4}"
LOG="$ROOT/logs/foundir_official_sow600_restoration_shard${SHARD_INDEX}_of_${SHARD_COUNT}.log"
GPU_LOG="$ROOT/logs/foundir_official_sow600_restoration_shard${SHARD_INDEX}_of_${SHARD_COUNT}_gpu.csv"
LOCK="$ROOT/logs/foundir_official_sow600_restoration_shard${SHARD_INDEX}_of_${SHARD_COUNT}.lock"

[[ "$SHARD_COUNT" == "5" ]] || { echo "ERROR: this frozen launch expects SHARD_COUNT=5" >&2; exit 1; }
[[ "$SHARD_INDEX" == "0" || "$SHARD_INDEX" == "1" || "$SHARD_INDEX" == "2" || "$SHARD_INDEX" == "3" || "$SHARD_INDEX" == "4" ]] || {
  echo "ERROR: SHARD_INDEX must be 0, 1, 2, 3, or 4" >&2
  exit 1
}
[[ "${CUDA_VISIBLE_DEVICES:-}" == "0,1,2,3,4,5,6,7" ]] || {
  echo "ERROR: expose exactly GPUs 0-7" >&2
  exit 1
}
[[ -f "$OUTPUT/router_decisions.csv" ]] || { echo "ERROR: official Qwen decisions are missing" >&2; exit 1; }
[[ ! -f "$OUTPUT/official_result.json" ]] || { echo "ERROR: official terminal result already exists" >&2; exit 1; }

cd "$ROOT"
/usr/bin/python3 smoke/advisor_governance.py check
mkdir -p "$ROOT/logs" "$ROOT/.cache/triton/foundir_official_sow600_shard${SHARD_INDEX}"
exec 9>>"$LOCK"
flock -n 9 || { echo "ERROR: shard $SHARD_INDEX is already running" >&2; exit 1; }
mapfile -t gpu_names < <(nvidia-smi --query-gpu=name --format=csv,noheader)
[[ "${#gpu_names[@]}" -eq 8 ]] || { echo "ERROR: exactly eight visible GPUs are required" >&2; exit 1; }
first_name=${gpu_names[0]}
for name in "${gpu_names[@]}"; do
  [[ "$name" == "$first_name" ]] || { echo "ERROR: mixed GPU models on one node are forbidden" >&2; exit 1; }
  [[ "$name" == *H100* || "$name" == *H200* ]] || { echo "ERROR: unsupported GPU: $name" >&2; exit 1; }
done

exec > >(tee -a "$LOG") 2>&1
telemetry_pid=
finish_run() {
  status=$?
  if [[ -n "$telemetry_pid" ]]; then
    kill "$telemetry_pid" 2>/dev/null || true
    wait "$telemetry_pid" 2>/dev/null || true
  fi
  echo "FOUNDIR_OFFICIAL_SHARD_END shard=$SHARD_INDEX status=$status utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  return "$status"
}
trap finish_run EXIT

export TORCH_HOME="$ROOT/models/torch" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 TORCH_NCCL_TRACE_BUFFER_SIZE=1048576
export TRITON_CACHE_DIR="$ROOT/.cache/triton/foundir_official_sow600_shard${SHARD_INDEX}"

echo "FOUNDIR_OFFICIAL_SHARD_START shard=$SHARD_INDEX/$SHARD_COUNT gpu=$first_name utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "SCOPE one_fifth_of_each_frozen_expert no_cross_node_communication shared_GPFS deterministic_record_sharding"
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw --format=csv -l 10 >"$GPU_LOG" 2>&1 &
telemetry_pid=$!

"$PYTHON" smoke/restore_foundir_official_sow600_tiled_ddp.py --config "$CONFIG" --stage identity --check-only
for stage in rain noise low_light blur haze; do
  marker="$OUTPUT/stages/${stage}.shard_${SHARD_INDEX}_of_${SHARD_COUNT}.json"
  if [[ -f "$marker" ]]; then
    echo "FOUNDIR_OFFICIAL_SHARD_STAGE_REUSED stage=$stage shard=$SHARD_INDEX"
    continue
  fi
  echo "FOUNDIR_OFFICIAL_SHARD_STAGE_START stage=$stage shard=$SHARD_INDEX utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 \
    smoke/restore_foundir_official_sow600_tiled_ddp.py \
    --config "$CONFIG" --stage "$stage" \
    --shard-count "$SHARD_COUNT" --shard-index "$SHARD_INDEX"
done
echo "FOUNDIR_OFFICIAL_SHARD_COMPLETE shard=$SHARD_INDEX/$SHARD_COUNT utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
