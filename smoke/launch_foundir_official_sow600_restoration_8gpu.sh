#!/usr/bin/env bash
set -euo pipefail

ROOT=/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
PYTHON="$ROOT/.venv/flux2-smoke/bin/python"
METRICS_PYTHON="$ROOT/.venv/foundir-metrics/bin/python"
CONFIG="$ROOT/configs/foundir_official_sow600_qwen_v1.json"
OUTPUT="$ROOT/checkpoints/foundir_official_sow600_qwen_v1"
LOG="$ROOT/logs/foundir_official_sow600_restoration_v1.log"
GPU_LOG="$ROOT/logs/foundir_official_sow600_restoration_v1_gpu.csv"
LOCK="$ROOT/logs/foundir_official_sow600_restoration_v1.lock"
TERMINAL="$OUTPUT/official_result.json"

cd "$ROOT"
/usr/bin/python3 smoke/advisor_governance.py check
[[ "${CUDA_VISIBLE_DEVICES:-}" == "0,1,2,3,4,5,6,7" ]] || {
  echo "ERROR: set CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7" >&2
  exit 1
}
[[ -x "$PYTHON" && -x "$METRICS_PYTHON" ]] || { echo "ERROR: frozen Python environments are missing" >&2; exit 1; }
[[ -f "$OUTPUT/router_decisions.csv" && -f "$OUTPUT/router_decisions.json" ]] || {
  echo "ERROR: completed official Qwen routing is required" >&2
  exit 1
}
[[ ! -f "$TERMINAL" ]] || { echo "ERROR: official terminal result already exists" >&2; exit 1; }

mkdir -p "$ROOT/logs" "$OUTPUT" "$ROOT/.cache/triton/foundir_official_sow600_v1"
exec 9>>"$LOCK"
flock -n 9 || { echo "ERROR: official SOW600 restoration is already running" >&2; exit 1; }
mapfile -t gpu_names < <(nvidia-smi --query-gpu=name --format=csv,noheader)
[[ "${#gpu_names[@]}" -eq 8 ]] || { echo "ERROR: exactly eight visible GPUs are required" >&2; exit 1; }
first_name=${gpu_names[0]}
for name in "${gpu_names[@]}"; do
  [[ "$name" == "$first_name" ]] || { echo "ERROR: mixed GPU models are forbidden" >&2; exit 1; }
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
  echo "FOUNDIR_OFFICIAL_SOW600_END status=$status utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  return "$status"
}
trap finish_run EXIT

export TORCH_HOME="$ROOT/models/torch" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 TORCH_NCCL_TRACE_BUFFER_SIZE=1048576
export TRITON_CACHE_DIR="$ROOT/.cache/triton/foundir_official_sow600_v1"

echo "FOUNDIR_OFFICIAL_SOW600_START utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) gpu=$first_name"
echo "SCOPE official_600 full_resolution real_Qwen 0_wrong_routes 12830_FLUX_tiles 4388_Restormer_tiles official_pyiqa"
echo "CONSERVATIVE_TIME_ESTIMATE H200=5-8h H100=7-12h; update_after_measured_progress"
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw --format=csv -l 10 >"$GPU_LOG" 2>&1 &
telemetry_pid=$!

"$PYTHON" smoke/restore_foundir_official_sow600_tiled_ddp.py --config "$CONFIG" --stage identity --check-only
TORCH_HOME="$ROOT/models/torch" "$METRICS_PYTHON" smoke/evaluate_foundir_official_sow600_pyiqa_ddp.py --config "$CONFIG" --check-only
"$PYTHON" smoke/restore_foundir_official_sow600_tiled_ddp.py --config "$CONFIG" --stage identity

for stage in rain noise low_light blur haze; do
  marker="$OUTPUT/stages/$stage.json"
  if [[ -f "$marker" ]]; then
    echo "FOUNDIR_OFFICIAL_STAGE_REUSED stage=$stage marker=$marker"
    continue
  fi
  echo "FOUNDIR_OFFICIAL_STAGE_START stage=$stage utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 \
    smoke/restore_foundir_official_sow600_tiled_ddp.py --config "$CONFIG" --stage "$stage"
done

echo "FOUNDIR_OFFICIAL_METRICS_START utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
"$METRICS_PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 \
  smoke/evaluate_foundir_official_sow600_pyiqa_ddp.py --config "$CONFIG"
[[ -f "$TERMINAL" ]] || { echo "ERROR: official terminal result is missing" >&2; exit 1; }
echo "FOUNDIR_OFFICIAL_SOW600_COMPLETE result=$TERMINAL"
