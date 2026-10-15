#!/usr/bin/env bash
set -euo pipefail
ROOT=/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
PYTHON="$ROOT/.venv/flux2-smoke/bin/python"
CONFIG="$ROOT/configs/foundir_noise_family_integrated_development_v1.json"
SCRIPT="$ROOT/smoke/evaluate_foundir_mixed_restoration_pilot_ddp.py"
OUTPUT="$ROOT/checkpoints/foundir_noise_family_integrated_development_v1"
RESULT="$OUTPUT/result.json"
LOG="$ROOT/logs/foundir_noise_family_integrated_development_v1.log"
GPU_LOG="$ROOT/logs/foundir_noise_family_integrated_development_v1_gpu.csv"
LOCK="$ROOT/logs/foundir_noise_family_integrated_development_v1.lock"
cd "$ROOT"
/usr/bin/python3 smoke/advisor_governance.py check
mkdir -p "$ROOT/logs" "$ROOT/.cache/triton/foundir_noise_family_integrated_development_v1"
exec > >(tee -a "$LOG") 2>&1
telemetry_pid=
finish_run() {
  status=$?
  if [[ -n "$telemetry_pid" ]]; then kill "$telemetry_pid" 2>/dev/null || true; wait "$telemetry_pid" 2>/dev/null || true; fi
  echo "NOISE_FAMILY_INTEGRATED_DEVELOPMENT_END status=$status utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  return "$status"
}
trap finish_run EXIT
exec 9>>"$LOCK"
flock -n 9 || { echo "ERROR: integration evaluation already running" >&2; exit 1; }
[[ "${CUDA_VISIBLE_DEVICES:-}" == "0,1,2,3,4,5,6,7" ]] || { echo "ERROR: set eight visible GPUs" >&2; exit 1; }
[[ ! -f "$RESULT" ]] || { echo "ERROR: terminal result already exists" >&2; exit 1; }
mapfile -t gpu_names < <(nvidia-smi --query-gpu=name --format=csv,noheader | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
[[ "${#gpu_names[@]}" -eq 8 ]] || { echo "ERROR: exactly eight GPUs required" >&2; exit 1; }
first_name=${gpu_names[0]}
for name in "${gpu_names[@]}"; do
  [[ "$name" == "$first_name" ]] || { echo "ERROR: mixed GPU models forbidden" >&2; exit 1; }
  [[ "$name" == *H100* || "$name" == *H200* ]] || { echo "ERROR: unsupported GPU $name" >&2; exit 1; }
done
export TORCH_HOME="$ROOT/models/torch"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 TORCH_NCCL_TRACE_BUFFER_SIZE=1048576
export TRITON_CACHE_DIR="$ROOT/.cache/triton/foundir_noise_family_integrated_development_v1"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
echo "NOISE_FAMILY_INTEGRATED_DEVELOPMENT_START utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) gpu=$first_name"
echo "SCOPE 1842_accepted_development_records 3010_frozen_expert_applications no_training no_official_test"
echo "TOTAL_TIME_ESTIMATE H200=25-40min H100=45-60min GPU_HOURS_H200=3.3-5.3 GPU_HOURS_H100=6-8"
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw --format=csv -l 5 >"$GPU_LOG" 2>&1 &
telemetry_pid=$!
"$PYTHON" "$SCRIPT" --config "$CONFIG" --preflight
for stage in noise low_light metrics; do
  echo "NOISE_FAMILY_INTEGRATED_STAGE_START stage=$stage utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 "$SCRIPT" --config "$CONFIG" --stage "$stage"
done
[[ -f "$RESULT" ]] || { echo "ERROR: terminal result missing" >&2; exit 1; }
"$PYTHON" - "$RESULT" <<'PY'
import json, pathlib, sys
p=json.loads(pathlib.Path(sys.argv[1]).read_text())
assert p["records"] == 1842
assert p["model_weight_updates"] is False
assert p["development_pixels_accessed"] is True
assert p["official_foundir_test_accessed"] is False
print("NOISE_FAMILY_INTEGRATED_AUTOMATIC_GATE", "PASS" if p["automatic_metric_gate_pass"] else "FAIL")
PY
