#!/usr/bin/env bash
set -uo pipefail

ROOT=/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
PYTHON="$ROOT/.conda/gpu-train-stable/bin/python"
CONFIG="$ROOT/configs/qwen_multidegradation_visibility_v2b_hard_triple.json"
LOG="$ROOT/logs/qwen_multidegradation_visibility_v2b_hard_triple.log"
GPU_LOG="$ROOT/logs/qwen_multidegradation_visibility_v2b_hard_triple_gpu.csv"
LOCK="$ROOT/logs/qwen_multidegradation_visibility_v2b_hard_triple.lock"
OUTPUT="$ROOT/checkpoints/qwen_multidegradation_visibility_v2b_hard_triple"
VARIANT=merger_trainable
FINAL_CHECKPOINT="$OUTPUT/$VARIANT/checkpoints/V2B_C3_hard_triple_epoch8/adapter.safetensors"
VALIDATION_RESULT="$OUTPUT/$VARIANT/validation_result.json"
FIXED_TEST_RESULT="$OUTPUT/$VARIANT/fixed_test_result.json"

cd "$ROOT" || exit 1
mkdir -p "$ROOT/logs"
exec > >(tee -a "$LOG") 2>&1
echo "QWEN_VISIBILITY_V2B_RUN_START_UTC $(date -u +%Y-%m-%dT%H:%M:%SZ)"

telemetry_pid=
finish_run() {
  status=$?
  if [[ -n "$telemetry_pid" ]]; then
    kill "$telemetry_pid" 2>/dev/null || true
    wait "$telemetry_pid" 2>/dev/null || true
  fi
  echo "QWEN_VISIBILITY_V2B_RUN_END_UTC $(date -u +%Y-%m-%dT%H:%M:%SZ) status=$status"
}
trap finish_run EXIT

exec 9>>"$LOCK"
if ! flock -n 9; then
  echo "ERROR: another v2b Qwen pipeline holds $LOCK" >&2
  exit 1
fi
case "${CUDA_VISIBLE_DEVICES:-}" in
  0,1) NPROC=2; ACCUMULATION=4 ;;
  0,1,2,3,4,5,6,7) NPROC=8; ACCUMULATION=1 ;;
  *) echo "ERROR: set CUDA_VISIBLE_DEVICES exactly to 0,1 or 0,1,2,3,4,5,6,7" >&2; exit 1 ;;
esac
if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: missing workspace interpreter: $PYTHON" >&2
  exit 1
fi

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

echo "QWEN_VISIBILITY_V2B_WORKSPACE $ROOT"
echo "QWEN_VISIBILITY_V2B_CONFIG $CONFIG"
echo "QWEN_VISIBILITY_V2B_GIT_REVISION $(git rev-parse HEAD)"
echo "QWEN_VISIBILITY_V2B_VISIBLE_DEVICES $CUDA_VISIBLE_DEVICES"
echo "QWEN_VISIBILITY_V2B_WORLD_SIZE $NPROC"
echo "QWEN_VISIBILITY_V2B_BATCH micro_per_rank=25 accumulation=$ACCUMULATION global=200"
echo "QWEN_VISIBILITY_V2B_SCHEDULE continuation_epochs=3 dialogues=126000 steps=630"
echo "QWEN_VISIBILITY_V2B_GATE fresh_validation_triple_exact_ge_0.83 protected_test_ge_0.80"

"$PYTHON" smoke/qwen_multidegradation_lora_preflight.py --config "$CONFIG" --full-hashes
preflight_status=$?
if [[ $preflight_status -ne 0 ]]; then
  echo "QWEN_VISIBILITY_V2B_PREFLIGHT_FAILED status=$preflight_status"
  exit "$preflight_status"
fi

nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw --format=csv -l 5 > "$GPU_LOG" 2>&1 &
telemetry_pid=$!

training_mode=()
if [[ -f "$FINAL_CHECKPOINT" && ! -f "$VALIDATION_RESULT" ]]; then
  training_mode=(--validation-only)
  echo "QWEN_VISIBILITY_V2B_VALIDATION_ONLY_RECOVERY checkpoint=$FINAL_CHECKPOINT"
fi
"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$NPROC" \
  smoke/train_qwen_multidegradation_lora_ddp.py --config "$CONFIG" --variant "$VARIANT" "${training_mode[@]}"
validation_status=$?
if [[ ! -f "$VALIDATION_RESULT" ]]; then
  echo "QWEN_VISIBILITY_V2B_TRAIN_OR_VALIDATION_CRASH outer_status=$validation_status" >&2
  exit 1
fi
if [[ $validation_status -ne 0 ]]; then
  echo "QWEN_VISIBILITY_V2B_VALIDATION_SCIENTIFIC_FAILURE outer_status=$validation_status"
  echo "QWEN_VISIBILITY_V2B_PROTECTED_TEST_NOT_ACCESSED"
  exit 2
fi
echo "QWEN_VISIBILITY_V2B_VALIDATION_GATE_PASS"

"$PYTHON" smoke/select_qwen_multidegradation_lora.py --config "$CONFIG"
selection_status=$?
if [[ $selection_status -ne 0 ]]; then
  echo "QWEN_VISIBILITY_V2B_SELECTION_FAILED"
  exit 2
fi

"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$NPROC" \
  smoke/train_qwen_multidegradation_lora_ddp.py --config "$CONFIG" --variant "$VARIANT" --test-only
test_status=$?
if [[ $test_status -ne 0 ]]; then
  if [[ -f "$FIXED_TEST_RESULT" ]]; then
    echo "QWEN_VISIBILITY_V2B_PROTECTED_TEST_SCIENTIFIC_FAILURE outer_status=$test_status"
    exit 2
  fi
  echo "QWEN_VISIBILITY_V2B_PROTECTED_TEST_CRASH outer_status=$test_status" >&2
  exit 1
fi

echo "QWEN_VISIBILITY_V2B_RUN_PASS"
exit 0
