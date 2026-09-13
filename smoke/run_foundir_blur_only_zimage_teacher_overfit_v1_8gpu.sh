#!/usr/bin/env bash
set -euo pipefail

ROOT=/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
CONFIG="$ROOT/configs/foundir_blur_only_zimage_teacher_overfit_v1.json"
PYTHON="$ROOT/.conda/gpu-train-stable/bin/python"
OUTPUT="$ROOT/checkpoints/foundir_blur_only_zimage_teacher_overfit_v1"
RESULT="$OUTPUT/evaluation/result.json"
LOG="$ROOT/logs/foundir_blur_only_zimage_teacher_overfit_v1.log"
GPU_LOG="$ROOT/logs/foundir_blur_only_zimage_teacher_overfit_v1_gpu.csv"
LOCK="$ROOT/logs/foundir_blur_only_zimage_teacher_overfit_v1.lock"
MODEL="Z-Image 6B undistilled 50-step Blur-only teacher with frozen LQ ControlNet and DiT LoRA rank32"

cd "$ROOT"
/usr/bin/python3 smoke/advisor_governance.py check
mkdir -p "$ROOT/logs" "$OUTPUT" "$ROOT/.cache/triton/foundir_blur_only_zimage_teacher_v1"
exec > >(tee -a "$LOG") 2>&1

telemetry_pid=
finish_run() {
  status=$?
  if [[ -n "$telemetry_pid" ]]; then
    kill "$telemetry_pid" 2>/dev/null || true
    wait "$telemetry_pid" 2>/dev/null || true
  fi
  if [[ "$status" -ne 0 && ! -f "$RESULT" ]]; then
    /usr/bin/python3 smoke/advisor_governance.py runtime-failure \
      --gate zimage_blur_teacher_tiny_overfit_runtime_failure --evidence "$CONFIG" \
      --config "$CONFIG" --model "$MODEL" --status "$status" --log "$LOG" || true
  fi
  echo "FOUNDIR_ZIMAGE_TEACHER_OVERFIT_V1_END status=$status utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  return "$status"
}
trap finish_run EXIT

exec 9>>"$LOCK"
flock -n 9 || { echo "ERROR: another Z-Image teacher run holds $LOCK" >&2; exit 1; }
[[ "${CUDA_VISIBLE_DEVICES:-}" == "0,1,2,3,4,5,6,7" ]] || {
  echo "ERROR: set CUDA_VISIBLE_DEVICES exactly to 0,1,2,3,4,5,6,7" >&2; exit 1;
}
[[ -x "$PYTHON" ]] || { echo "ERROR: missing workspace Python: $PYTHON" >&2; exit 1; }

export PYTHONPATH="$ROOT/.deps/zimage:$ROOT/third_party/diffsynth-studio:$ROOT"
export DIFFSYNTH_SKIP_DOWNLOAD=True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export TORCH_HOME="$ROOT/models/torch"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 TORCH_NCCL_TRACE_BUFFER_SIZE=1048576
export TORCH_NCCL_DUMP_ON_TIMEOUT=1 TORCH_NCCL_DESYNC_DEBUG=1
export TRITON_CACHE_DIR="$ROOT/.cache/triton/foundir_blur_only_zimage_teacher_v1"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

echo "FOUNDIR_ZIMAGE_TEACHER_OVERFIT_V1_START utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "SCOPE Z-Image_6B_Base 50_step_teacher isolated_Blur_only frozen_LQ_ControlNet DiT_LoRA no_distill no_decoder_tuning no_Qwen_degradation no_mixed no_sealed no_official_test"

nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw \
  --format=csv -l 5 >"$GPU_LOG" 2>&1 &
telemetry_pid=$!

"$PYTHON" smoke/preflight_foundir_blur_only_zimage_teacher.py \
  --config "$CONFIG" --require-cuda --write-artifact

"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 \
  smoke/train_foundir_blur_only_zimage_teacher_overfit_ddp.py --config "$CONFIG"

"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 \
  smoke/evaluate_foundir_blur_only_zimage_teacher_overfit_ddp.py --config "$CONFIG"

"$PYTHON" - "$RESULT" <<'PY'
import json,pathlib,sys
p=json.loads(pathlib.Path(sys.argv[1]).read_text())
assert p["status"] == "zimage_blur_teacher_tiny_overfit_evaluation_complete"
assert p["sealed_internal_accessed"] is False
assert p["official_foundir_test_accessed"] is False
assert p["human_visual_review_required"] is True
assert p["human_visual_review_complete"] is False
print("FOUNDIR_ZIMAGE_TEACHER_RESULT_VERIFY_PASS")
PY

/usr/bin/python3 smoke/advisor_governance.py emit \
  --gate zimage_blur_teacher_tiny_overfit --evidence "$RESULT" --config "$CONFIG" \
  --model "$MODEL" --split fixed_16_training_pairs_plus_16_source_disjoint_visual_development_pairs \
  --outcome information \
  --root-cause-or-status "Z-Image teacher tiny-overfit metrics and fixed visual triptychs are complete; full teacher training remains stopped until the visual result is inspected." \
  --advisor-decision-question "请肉眼确认 Z-Image teacher 是否真正学会 Blur restoration；若视觉有效，继续完整 Blur-only teacher training。"

echo "FOUNDIR_ZIMAGE_TEACHER_OVERFIT_COMPLETE_STOP_FOR_VISUAL_REVIEW"
