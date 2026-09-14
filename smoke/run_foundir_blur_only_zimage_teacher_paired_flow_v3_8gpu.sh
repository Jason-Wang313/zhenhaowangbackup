#!/usr/bin/env bash
set -euo pipefail

ROOT=/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
CONFIG="$ROOT/configs/foundir_blur_only_zimage_teacher_paired_flow_continuation_v3.json"
PYTHON="$ROOT/.conda/gpu-train-stable/bin/python"
OUTPUT="$ROOT/checkpoints/foundir_blur_only_zimage_teacher_paired_flow_continuation_v3"
RESULT="$OUTPUT/evaluation/result.json"
LOG="$ROOT/logs/foundir_blur_only_zimage_teacher_paired_flow_continuation_v3.log"
GPU_LOG="$ROOT/logs/foundir_blur_only_zimage_teacher_paired_flow_continuation_v3_gpu.csv"
LOCK="$ROOT/logs/foundir_blur_only_zimage_teacher_paired_flow_continuation_v3.lock"
MODEL="Z-Image 6B undistilled 50-step Blur-only solver-aligned paired-flow teacher v3 with DiT LoRA rank32"

cd "$ROOT"
/usr/bin/python3 "$ROOT/smoke/advisor_governance.py" check
mkdir -p "$ROOT/logs" "$OUTPUT" "$ROOT/.cache/triton/foundir_blur_only_zimage_paired_flow_v3"
exec > >(tee -a "$LOG") 2>&1

telemetry_pid=
finish_run() {
  status=$?
  if [[ -n "$telemetry_pid" ]]; then
    kill "$telemetry_pid" 2>/dev/null || true
    wait "$telemetry_pid" 2>/dev/null || true
  fi
  if [[ "$status" -ne 0 && ! -f "$RESULT" ]]; then
    /usr/bin/python3 "$ROOT/smoke/advisor_governance.py" runtime-failure \
      --gate zimage_blur_teacher_paired_flow_v3_runtime_failure --evidence "$CONFIG" \
      --config "$CONFIG" --model "$MODEL" --status "$status" --log "$LOG" || true
  fi
  echo "FOUNDIR_ZIMAGE_PAIRED_FLOW_V3_END status=$status utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  return "$status"
}
trap finish_run EXIT

exec 9>>"$LOCK"
flock -n 9 || { echo "ERROR: another Z-Image paired-flow v3 run holds $LOCK" >&2; exit 1; }
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
export TRITON_CACHE_DIR="$ROOT/.cache/triton/foundir_blur_only_zimage_paired_flow_v3"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

echo "FOUNDIR_ZIMAGE_PAIRED_FLOW_V3_START utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "SCOPE Z-Image_6B_Base paired_LQ_to_GT_latent_flow parent_v2_step_400 optimizer_reset exact_50_step_shift_6_grid plain_MSE CFG_1 exact_LQ_initialization isolated_Blur_only DiT_LoRA no_ControlNet no_distill no_decoder_tuning no_Qwen_degradation no_mixed no_sealed no_official_test"

nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw \
  --format=csv -l 5 >"$GPU_LOG" 2>&1 &
telemetry_pid=$!

"$PYTHON" smoke/preflight_foundir_blur_only_zimage_teacher_paired_flow_v3.py \
  --config "$CONFIG" --require-cuda --write-artifact

"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 \
  smoke/train_foundir_blur_only_zimage_teacher_paired_flow_v3_ddp.py --config "$CONFIG"

"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 \
  smoke/evaluate_foundir_blur_only_zimage_teacher_paired_flow_v3_ddp.py --config "$CONFIG"

"$PYTHON" - "$RESULT" <<'PY'
import json,pathlib,sys
p=json.loads(pathlib.Path(sys.argv[1]).read_text())
assert p["status"] == "zimage_blur_teacher_paired_flow_v3_evaluation_complete"
assert p["sealed_internal_accessed"] is False
assert p["official_foundir_test_accessed"] is False
assert p["controlnet_loaded"] is False
assert p["optimizer_resume"] is False
assert p["initialization"] == "exact_lq_vae_latent_at_sigma_1"
assert p["scheduler_shift"] == 6.0 and p["cfg_scale"] == 1.0
assert p["timestep_sampling"] == "uniform_exact_50_step_shift_6_inference_grid"
assert p["loss_weighting"] == "plain_unweighted_velocity_mse"
assert p["human_visual_review_required"] is True
assert p["human_visual_review_complete"] is False
print("FOUNDIR_ZIMAGE_PAIRED_FLOW_V3_RESULT_VERIFY_PASS")
PY

/usr/bin/python3 "$ROOT/smoke/advisor_governance.py" emit \
  --gate zimage_blur_teacher_paired_flow_tiny_overfit_v3 --evidence "$RESULT" --config "$CONFIG" \
  --model "$MODEL" --split fixed_16_training_pairs_plus_16_source_disjoint_visual_development_pairs \
  --outcome auto --diagnostic-only --hold-on-failure \
  --root-cause-or-status "Solver-aligned v3 continuation from immutable v2 step 400 completed; automatic PSNR/LPIPS and fixed visuals determine whether full teacher training may begin." \
  --advisor-decision-question "The user already authorized best-effort same-mainline continuation without another advisor round. Continue only according to the frozen teacher/distillation/decoder ordering."

echo "FOUNDIR_ZIMAGE_PAIRED_FLOW_V3_COMPLETE_STOP_FOR_VISUAL_REVIEW"
