#!/usr/bin/env bash
set -euo pipefail

ROOT=/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
PYTHON="$ROOT/.venv/flux2-smoke/bin/python"
CONFIG="$ROOT/configs/foundir_blur_only_flux2_full_development_v1.json"
EVALUATOR="$ROOT/smoke/evaluate_foundir_blur_only_flux2_full_development_ddp.py"
OUTPUT="$ROOT/checkpoints/foundir_blur_only_flux2_full_v1/full_development_evaluation_v1"
LOG="$ROOT/logs/foundir_blur_only_flux2_full_development_v1.log"
GPU_LOG="$ROOT/logs/foundir_blur_only_flux2_full_development_v1_gpu.csv"
LOCK="$ROOT/logs/foundir_blur_only_flux2_full_development_v1.lock"

cd "$ROOT"
/usr/bin/python3 "$ROOT/smoke/advisor_governance.py" check
mkdir -p "$ROOT/logs"
exec > >(tee -a "$LOG") 2>&1

telemetry_pid=
finish_run() {
  status=$?
  if [[ -n "$telemetry_pid" ]]; then
    kill "$telemetry_pid" 2>/dev/null || true
    wait "$telemetry_pid" 2>/dev/null || true
  fi
  echo "FOUNDIR_BLUR_ONLY_FULL_DEVELOPMENT_V1_END status=$status utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  if [[ "$status" -ne 0 ]]; then
    /usr/bin/python3 "$ROOT/smoke/advisor_governance.py" runtime-failure \
      --gate blur_only_full_development_runtime_failure \
      --evidence "$OUTPUT/result.json" \
      --config "$CONFIG" \
      --model "FLUX.2 Klein Base 4B isolated-Blur LoRA" \
      --status "$status" \
      --log "logs/foundir_blur_only_flux2_full_development_v1.log" \
      --diagnostic-only || true
  fi
}
trap finish_run EXIT

exec 9>>"$LOCK"
flock -n 9 || {
  echo "ERROR: another Blur-only full-development evaluation holds $LOCK" >&2
  exit 1
}
[[ "${CUDA_VISIBLE_DEVICES:-}" == "0,1,2,3,4,5,6,7" ]] || {
  echo "ERROR: expose exactly CUDA devices 0,1,2,3,4,5,6,7" >&2
  exit 1
}

export TORCH_HOME="$ROOT/models/torch"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

echo "FOUNDIR_BLUR_ONLY_FULL_DEVELOPMENT_V1_START utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "FOUNDIR_BLUR_ONLY_FULL_DEVELOPMENT_V1_GIT $(git rev-parse HEAD)"
echo "FOUNDIR_BLUR_ONLY_FULL_DEVELOPMENT_V1_CONFIG_SHA256 $(sha256sum "$CONFIG" | cut -d' ' -f1)"
echo "FOUNDIR_BLUR_ONLY_FULL_DEVELOPMENT_V1_EVALUATOR_SHA256 $(sha256sum "$EVALUATOR" | cut -d' ' -f1)"
echo "FOUNDIR_BLUR_ONLY_FULL_DEVELOPMENT_V1_SCOPE development_only_2595_records_five_checkpoints"
echo "FOUNDIR_BLUR_ONLY_FULL_DEVELOPMENT_V1_INTERNAL sealed_internal_not_accessed"
echo "FOUNDIR_BLUR_ONLY_FULL_DEVELOPMENT_V1_OFFICIAL_TEST not_accessed"

{
  echo "RUN_START,$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw \
    --format=csv -l 5
} >>"$GPU_LOG" 2>&1 &
telemetry_pid=$!

"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 \
  "$EVALUATOR" --config "$CONFIG"

/usr/bin/python3 "$ROOT/smoke/advisor_governance.py" emit \
  --gate blur_only_full_development_evaluation \
  --evidence "$OUTPUT/result.json" \
  --config "$CONFIG" \
  --model "FLUX.2 Klein Base 4B isolated-Blur DiT LoRA rank/alpha 32/32" \
  --split source_group_disjoint_blur_development_2595 \
  --outcome information \
  --root-cause-or-status "Five-checkpoint full Blur development curve completed; use the curve and fixed visual outputs to judge overfitting." \
  --advisor-decision-question "请根据全量 curve 与视觉结果判断：继续调整 Blur-only，还是先审计 hallucination/training-inference code？" \
  --diagnostic-only \
  --require-decision

echo "FOUNDIR_BLUR_ONLY_FULL_DEVELOPMENT_V1_ADVISOR_REPORT $OUTPUT/advisor_report.md"
