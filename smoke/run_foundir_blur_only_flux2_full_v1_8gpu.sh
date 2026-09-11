#!/usr/bin/env bash
set -euo pipefail

ROOT=/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
CONFIG="$ROOT/configs/foundir_blur_only_flux2_full_v1.json"
PYTHON="$ROOT/.venv/flux2-smoke/bin/python"
OUTPUT="$ROOT/checkpoints/foundir_blur_only_flux2_full_v1"
SUMMARY="$OUTPUT/training_summary.json"
LOG="$ROOT/logs/foundir_blur_only_flux2_full_v1.log"
GPU_LOG="$ROOT/logs/foundir_blur_only_flux2_full_v1_gpu.csv"
LOCK="$ROOT/logs/foundir_blur_only_flux2_full_v1.lock"
MODEL="FLUX.2 Klein Base 4B DiT LoRA rank32 isolated-Blur-only"

cd "$ROOT"
/usr/bin/python3 smoke/advisor_governance.py check
mkdir -p "$ROOT/logs" "$OUTPUT" "$ROOT/.cache/triton/foundir_blur_only_full_v1"
exec > >(tee -a "$LOG") 2>&1

telemetry_pid=
finish_run() {
  status=$?
  if [[ -n "$telemetry_pid" ]]; then
    kill "$telemetry_pid" 2>/dev/null || true
    wait "$telemetry_pid" 2>/dev/null || true
  fi
  if [[ "$status" -ne 0 && ! -f "$SUMMARY" ]]; then
    /usr/bin/python3 smoke/advisor_governance.py runtime-failure \
      --gate blur_only_full_training_runtime_failure --evidence "$CONFIG" \
      --config "$CONFIG" --model "$MODEL" --status "$status" --log "$LOG" || true
  fi
  echo "FOUNDIR_BLUR_ONLY_FULL_END status=$status utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  return "$status"
}
trap finish_run EXIT

exec 9>>"$LOCK"
flock -n 9 || { echo "ERROR: another Blur-only full run holds $LOCK" >&2; exit 1; }
[[ "${CUDA_VISIBLE_DEVICES:-}" == "0,1,2,3,4,5,6,7" ]] || {
  echo "ERROR: set CUDA_VISIBLE_DEVICES exactly to 0,1,2,3,4,5,6,7" >&2; exit 1;
}
[[ -x "$PYTHON" ]] || { echo "ERROR: missing workspace FLUX Python: $PYTHON" >&2; exit 1; }

export TORCH_HOME="$ROOT/models/torch"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 TORCH_NCCL_TRACE_BUFFER_SIZE=1048576
export TORCH_NCCL_DUMP_ON_TIMEOUT=1 TORCH_NCCL_DESYNC_DEBUG=1
export TRITON_CACHE_DIR="$ROOT/.cache/triton/foundir_blur_only_full_v1"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

echo "FOUNDIR_BLUR_ONLY_FULL_START utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "SCOPE FLUX.2_4B isolated_Blur_only DiT_LoRA_only no_Qwen_degradation no_mixed no_pixel_refiner no_sealed no_official_test"

nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw \
  --format=csv -l 5 >"$GPU_LOG" 2>&1 &
telemetry_pid=$!

"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 \
  smoke/train_foundir_blur_only_flux2_full_ddp.py --config "$CONFIG"

"$PYTHON" - "$CONFIG" "$SUMMARY" <<'PY'
import hashlib,json,pathlib,sys
sha=lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
config,summary=map(pathlib.Path,sys.argv[1:])
p=json.loads(summary.read_text())
assert p["status"] == "training_complete_awaiting_human_visual_review"
assert p["automatic_checkpoint_selection"] is False
assert p["human_visual_review"] == "pending"
assert p["config_sha256"] == sha(config)
assert p["qwen_degradation_conditioning"] is False
assert p["mixed_degradations"] is False
assert p["pixel_refiner"] is False
assert p["sealed_internal_accessed"] is False
assert p["protected_foundir_test_accessed"] is False
checkpoint=pathlib.Path.cwd()/p["final_checkpoint"]
assert sha(checkpoint) == p["final_checkpoint_sha256"]
print("FOUNDIR_BLUR_ONLY_FULL_RESULT_VERIFY_PASS")
PY

/usr/bin/python3 smoke/advisor_governance.py emit \
  --gate blur_only_full_training_completion --evidence "$SUMMARY" --config "$CONFIG" \
  --model "$MODEL" --split source_group_disjoint_development_fixed_visual_subset \
  --outcome information --require-decision \
  --root-cause-or-status "Blur-only full training completed; every candidate has fixed development triptychs; checkpoint selection is intentionally paused before metrics for human visual screening." \
  --advisor-decision-question "请先肉眼检查每个 candidate 的固定 Input|Output|GT；确认 visually valid candidates 后，是否授权仅在这些 candidates 上比较完整 source-group-disjoint Blur development PSNR/LPIPS？"

echo "FOUNDIR_BLUR_ONLY_FULL_TRAINING_COMPLETE_STOP_FOR_ADVISOR_VISUAL_REVIEW"
