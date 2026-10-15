#!/usr/bin/env bash
set -euo pipefail

ROOT=/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
PYTHON="$ROOT/.venv/flux2-smoke/bin/python"
CONFIG="$ROOT/configs/foundir_haze_flux2_step3000_full_development_v1.json"
SCRIPT="$ROOT/smoke/evaluate_foundir_haze_flux2_step3000_full_development_ddp.py"
OUTPUT="$ROOT/checkpoints/foundir_haze_flux2_step3000_full_development_v1"
RESULT="$OUTPUT/result.json"
LOG="$ROOT/logs/foundir_haze_flux2_step3000_full_development_v1.log"
GPU_LOG="$ROOT/logs/foundir_haze_flux2_step3000_full_development_v1_gpu.csv"
LOCK="$ROOT/logs/foundir_haze_flux2_step3000_full_development_v1.lock"

cd "$ROOT"
/usr/bin/python3 smoke/advisor_governance.py check
mkdir -p "$ROOT/logs" "$ROOT/.cache/triton/foundir_haze_step3000_full_dev_v1"
exec > >(tee -a "$LOG") 2>&1

telemetry_pid=
finish_run() {
  status=$?
  if [[ -n "$telemetry_pid" ]]; then
    kill "$telemetry_pid" 2>/dev/null || true
    wait "$telemetry_pid" 2>/dev/null || true
  fi
  echo "HAZE_STEP3000_FULL_DEVELOPMENT_END status=$status utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  return "$status"
}
trap finish_run EXIT

exec 9>>"$LOCK"
flock -n 9 || { echo "ERROR: another Haze full-development run holds $LOCK" >&2; exit 1; }
[[ "${CUDA_VISIBLE_DEVICES:-}" == "0,1,2,3,4,5,6,7" ]] || {
  echo "ERROR: set CUDA_VISIBLE_DEVICES exactly to 0,1,2,3,4,5,6,7" >&2; exit 1;
}
[[ -x "$PYTHON" ]] || { echo "ERROR: missing FLUX Python $PYTHON" >&2; exit 1; }
[[ ! -f "$RESULT" ]] || { echo "ERROR: terminal result already exists" >&2; exit 1; }

mapfile -t gpu_names < <(nvidia-smi --query-gpu=name --format=csv,noheader | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
[[ "${#gpu_names[@]}" -eq 8 ]] || { echo "ERROR: exactly eight GPUs are required" >&2; exit 1; }
first_name=${gpu_names[0]}
for name in "${gpu_names[@]}"; do
  [[ "$name" == "$first_name" ]] || { echo "ERROR: mixed GPU models are forbidden" >&2; exit 1; }
  [[ "$name" == *H100* || "$name" == *H200* ]] || { echo "ERROR: unsupported GPU $name" >&2; exit 1; }
done

export TORCH_HOME="$ROOT/models/torch"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 TORCH_NCCL_TRACE_BUFFER_SIZE=1048576
export TORCH_NCCL_DUMP_ON_TIMEOUT=1 TORCH_NCCL_DESYNC_DEBUG=1
export TRITON_CACHE_DIR="$ROOT/.cache/triton/foundir_haze_step3000_full_dev_v1"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

echo "HAZE_STEP3000_FULL_DEVELOPMENT_START utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) git=$(git rev-parse HEAD) gpu=$first_name"
echo "SCOPE immutable_FLUX_Haze_step3000 all_1580_source_disjoint_development 50_step_inference SOW_mean_gate source_group_bootstrap no_training no_tuning no_released_FoundIR no_sealed no_official_test"
echo "TOTAL_TIME_ESTIMATE H100=23-31min(center27) H200=17-24min(center20) one_8GPU_instance GPU_HOURS_H100=3.1-4.1 GPU_HOURS_H200=2.3-3.2 includes_preflight_loading_1580_inferences_metrics_bootstrap_visuals_reporting"
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw --format=csv -l 5 >"$GPU_LOG" 2>&1 &
telemetry_pid=$!

"$PYTHON" "$SCRIPT" --config "$CONFIG" --preflight
set +e
"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 "$SCRIPT" --config "$CONFIG"
torchrun_status=$?
set -e

if [[ ! -f "$RESULT" ]]; then
  /usr/bin/python3 smoke/advisor_governance.py runtime-failure \
    --gate flux2_haze_step3000_full_development_runtime \
    --evidence "$CONFIG" --config "$CONFIG" \
    --model "FLUX.2 Klein Base 4B isolated-Haze LoRA step-3000" \
    --status "$torchrun_status" --log "${LOG#$ROOT/}" || true
  exit 1
fi

/usr/bin/python3 smoke/advisor_governance.py emit \
  --gate flux2_haze_step3000_full_development_result \
  --evidence "$RESULT" --config "$CONFIG" \
  --model "FLUX.2 Klein Base 4B isolated-Haze LoRA step-3000" \
  --split "all 1580 source-group-disjoint isolated-Haze development records; no protected split" \
  --outcome auto --require-decision \
  --root-cause-or-status "User-authorized immutable Haze step-3000 full-development confirmation completed. SOW mean PSNR/LPIPS are formal; source-group bootstrap and tails are mandatory descriptive evidence." \
  --advisor-decision-question "根据完整 Haze development 的 SOW metrics、source-group bootstrap 与固定视觉，是否接受当前 Haze checkpoint，或授权一次 bounded improvement？"

set +e
"$PYTHON" - "$RESULT" <<'PY'
import json, pathlib, sys
p=json.loads(pathlib.Path(sys.argv[1]).read_text())
assert p["records"] == 1580 and p["source_groups"] == 158
assert p["model_weight_updates"] is False and p["automatic_follow_on"] is False
assert p["official_foundir_test_accessed"] is False and p["sealed_internal_accessed"] is False
raise SystemExit(0 if p["automatic_metric_gate_pass"] else 2)
PY
scientific_status=$?
set -e
if [[ "$scientific_status" -eq 0 ]]; then
  echo "HAZE_STEP3000_FULL_DEVELOPMENT_SOW_PASS_STOP_FOR_VISUAL_REVIEW"
else
  echo "HAZE_STEP3000_FULL_DEVELOPMENT_SOW_FAILURE_STOP"
fi
exit "$scientific_status"
