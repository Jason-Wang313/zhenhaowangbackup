#!/usr/bin/env bash
set -euo pipefail

ROOT=/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
PYTHON="$ROOT/.venv/flux2-smoke/bin/python"
CONFIG="$ROOT/configs/foundir_lowlight_flux2_scalar_residual_full_development_v1.json"
SCRIPT="$ROOT/smoke/evaluate_foundir_lowlight_flux2_scalar_residual_full_development_ddp.py"
OUTPUT="$ROOT/checkpoints/foundir_lowlight_flux2_scalar_residual_full_development_v1"
RESULT="$OUTPUT/result.json"
LOG="$ROOT/logs/foundir_lowlight_flux2_scalar_residual_full_development_v1.log"
GPU_LOG="$ROOT/logs/foundir_lowlight_flux2_scalar_residual_full_development_v1_gpu.csv"
LOCK="$ROOT/logs/foundir_lowlight_flux2_scalar_residual_full_development_v1.lock"

cd "$ROOT"
/usr/bin/python3 smoke/advisor_governance.py check
mkdir -p "$ROOT/logs" "$ROOT/.cache/triton/foundir_lowlight_scalar_full_dev_v1"
exec > >(tee -a "$LOG") 2>&1

telemetry_pid=
finish_run() {
  status=$?
  if [[ -n "$telemetry_pid" ]]; then
    kill "$telemetry_pid" 2>/dev/null || true
    wait "$telemetry_pid" 2>/dev/null || true
  fi
  echo "LOWLIGHT_SCALAR_FULL_DEVELOPMENT_END status=$status utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  return "$status"
}
trap finish_run EXIT

exec 9>>"$LOCK"
flock -n 9 || { echo "ERROR: another Low-light scalar full-development run holds $LOCK" >&2; exit 1; }
[[ "${CUDA_VISIBLE_DEVICES:-}" == "0,1,2,3,4,5,6,7" ]] || {
  echo "ERROR: set CUDA_VISIBLE_DEVICES exactly to 0,1,2,3,4,5,6,7" >&2; exit 1;
}
[[ -x "$PYTHON" ]] || { echo "ERROR: missing FLUX Python $PYTHON" >&2; exit 1; }
[[ ! -f "$RESULT" ]] || { echo "ERROR: terminal result already exists" >&2; exit 1; }

export TORCH_HOME="$ROOT/models/torch"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 TORCH_NCCL_TRACE_BUFFER_SIZE=1048576
export TORCH_NCCL_DUMP_ON_TIMEOUT=1 TORCH_NCCL_DESYNC_DEBUG=1
export TRITON_CACHE_DIR="$ROOT/.cache/triton/foundir_lowlight_scalar_full_dev_v1"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

echo "LOWLIGHT_SCALAR_FULL_DEVELOPMENT_START utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) git=$(git rev-parse HEAD)"
echo "SCOPE 1400_source_disjoint_Lowlight_development frozen_alpha0.925 SOW_mean_gate tail_descriptive no_tuning no_weight_update no_sealed no_official_test"
echo "TOTAL_TIME_ESTIMATE H100=20-27min(center23) H200=15-21min(center17) includes_preflight_loading_inference_metrics_visuals_reporting; one_8GPU_instance; GPU_HOURS H100=2.7-3.6 H200=2.0-2.8"
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw --format=csv -l 5 >"$GPU_LOG" 2>&1 &
telemetry_pid=$!

"$PYTHON" "$SCRIPT" --config "$CONFIG" --preflight
set +e
"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 "$SCRIPT" --config "$CONFIG"
torchrun_status=$?
set -e

if [[ ! -f "$RESULT" ]]; then
  /usr/bin/python3 smoke/advisor_governance.py runtime-failure \
    --gate flux2_lowlight_scalar_full_development_runtime \
    --evidence reports/foundir_lowlight_flux2_scalar_residual_full_development_authorization_v1.json \
    --config "$CONFIG" --model "FLUX.2 Klein Base 4B Low-light step-400 plus frozen alpha=.925" \
    --status "$torchrun_status" --log "${LOG#$ROOT/}" || true
  exit 1
fi

/usr/bin/python3 smoke/advisor_governance.py emit \
  --gate flux2_lowlight_scalar_full_development_result \
  --evidence "$RESULT" --config "$CONFIG" \
  --model "FLUX.2 Klein Base 4B isolated-Low-light step-400 plus frozen alpha=.925 scalar residual" \
  --split "all 1400 source-group-disjoint isolated-Low-light development records; no protected split" \
  --outcome auto --require-decision \
  --root-cause-or-status "Frozen-alpha Low-light full-development confirmation completed. SOW mean PSNR/LPIPS are formal; tail statistics are descriptive per explicit user acceptance scope. Human review remains required." \
  --advisor-decision-question "根据完整 Low-light development SOW metrics 与固定视觉审查，是否接受该 Low-light correction 并继续 Haze？"

set +e
"$PYTHON" - "$RESULT" <<'PY'
import json, pathlib, sys
p=json.loads(pathlib.Path(sys.argv[1]).read_text())
assert p["records"] == 1400 and p["source_groups"] == 21 and p["frozen_alpha"] == 0.925
assert p["tail_statistics_descriptive_only"] is True
assert p["official_foundir_test_accessed"] is False and p["sealed_internal_accessed"] is False
raise SystemExit(0 if p["automatic_metric_gate_pass"] else 2)
PY
scientific_status=$?
set -e
if [[ "$scientific_status" -eq 0 ]]; then
  echo "LOWLIGHT_SCALAR_FULL_DEVELOPMENT_SOW_PASS_STOP_FOR_VISUAL_REVIEW"
else
  echo "LOWLIGHT_SCALAR_FULL_DEVELOPMENT_SOW_FAILURE_STOP"
fi
exit "$scientific_status"
