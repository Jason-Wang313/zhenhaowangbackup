#!/usr/bin/env bash
set -euo pipefail
ROOT=/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
PYTHON="$ROOT/.venv/flux2-smoke/bin/python"
CONFIG="$ROOT/configs/foundir_blur_only_flux2_scalar_residual_full_development_v1.json"
SCRIPT="$ROOT/smoke/evaluate_foundir_blur_only_flux2_scalar_residual_full_development_ddp.py"
OUTPUT="$ROOT/checkpoints/foundir_blur_only_flux2_scalar_residual_full_development_v1"
RESULT="$OUTPUT/result.json"
LOG="$ROOT/logs/foundir_blur_only_flux2_scalar_residual_full_development_v1.log"
GPU_LOG="$ROOT/logs/foundir_blur_only_flux2_scalar_residual_full_development_v1_gpu.csv"
LOCK="$ROOT/logs/foundir_blur_only_flux2_scalar_residual_full_development_v1.lock"
cd "$ROOT"
/usr/bin/python3 smoke/advisor_governance.py check
mkdir -p "$ROOT/logs" "$ROOT/.cache/triton/foundir_blur_flux2_scalar_full_dev_v1"
exec > >(tee -a "$LOG") 2>&1
telemetry_pid=
finish_run() { status=$?; if [[ -n "$telemetry_pid" ]]; then kill "$telemetry_pid" 2>/dev/null || true; wait "$telemetry_pid" 2>/dev/null || true; fi; echo "FLUX2_SCALAR_FULL_DEVELOPMENT_END status=$status utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"; return "$status"; }
trap finish_run EXIT
exec 9>>"$LOCK"
flock -n 9 || { echo "ERROR: another scalar full-development run holds $LOCK" >&2; exit 1; }
[[ "${CUDA_VISIBLE_DEVICES:-}" == "0,1,2,3,4,5,6,7" ]] || { echo "ERROR: set CUDA_VISIBLE_DEVICES exactly to 0,1,2,3,4,5,6,7" >&2; exit 1; }
[[ ! -f "$RESULT" ]] || { echo "ERROR: terminal result already exists" >&2; exit 1; }
export TORCH_HOME="$ROOT/models/torch" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 TORCH_NCCL_TRACE_BUFFER_SIZE=1048576 TORCH_NCCL_DUMP_ON_TIMEOUT=1 TORCH_NCCL_DESYNC_DEBUG=1
export TRITON_CACHE_DIR="$ROOT/.cache/triton/foundir_blur_flux2_scalar_full_dev_v1" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
echo "FLUX2_SCALAR_FULL_DEVELOPMENT_START utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) git=$(git rev-parse HEAD)"
echo "SCOPE FLUX.2_4B step10k alpha0.75_frozen 2595_source_disjoint_development original_crop_seed original_inference_seed paired_alpha1_replay no_tuning no_weight_update no_sealed no_official_test"
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw --format=csv -l 5 >"$GPU_LOG" 2>&1 &
telemetry_pid=$!
"$PYTHON" "$SCRIPT" --config "$CONFIG" --preflight
set +e
"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 "$SCRIPT" --config "$CONFIG"
torchrun_status=$?
set -e
if [[ ! -f "$RESULT" ]]; then
  /usr/bin/python3 smoke/advisor_governance.py runtime-failure --gate flux2_blur_scalar_full_development_runtime --evidence reports/flux2_blur_scalar_residual_full_development_authorization_v1.json --config "$CONFIG" --model "FLUX.2 step10k frozen alpha=.75 full development" --status "$torchrun_status" --log "${LOG#$ROOT/}"
  exit 1
fi
/usr/bin/python3 smoke/advisor_governance.py emit --gate flux2_blur_scalar_residual_full_development_result --evidence "$RESULT" --config "$CONFIG" --model "FLUX.2 Klein Base 4B step-10000 plus frozen alpha=.75 scalar residual" --split "all 2595 source-group-disjoint Blur development records; no protected split" --outcome auto --root-cause-or-status "One-shot frozen-alpha full-development confirmation completed; inspect paired replay, SOW gate, tail risk, and all 16 fixed visuals. No automatic follow-on." --advisor-decision-question "根据完整 paired metrics 与视觉审计，是否接受该 deterministic correction 或终止该方向？" --require-decision
set +e
"$PYTHON" - "$RESULT" <<'PY'
import json, pathlib, sys
p=json.loads(pathlib.Path(sys.argv[1]).read_text())
assert p['records']==2595 and p['frozen_alpha']==0.75
assert p['development_gt_optimization'] is False
assert p['sealed_internal_accessed'] is False and p['official_foundir_test_accessed'] is False
raise SystemExit(0 if p['automatic_metric_gate_pass'] else 2)
PY
normalized=$?
set -e
if [[ "$normalized" -eq 0 ]]; then echo "FLUX2_SCALAR_FULL_DEVELOPMENT_METRIC_PASS_STOP_FOR_VISUAL_REVIEW"; else echo "FLUX2_SCALAR_FULL_DEVELOPMENT_SCIENTIFIC_FAILURE_STOP"; fi
exit "$normalized"
