#!/usr/bin/env bash
set -uo pipefail

ROOT=/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
PYTHON="$ROOT/.conda/gpu-train-stable/bin/python"
CONFIG="$ROOT/configs/qwen_multidegradation_visibility_v2_rank32.json"
LOG="$ROOT/logs/qwen_multidegradation_visibility_v2_rank32.log"
GPU_LOG="$ROOT/logs/qwen_multidegradation_visibility_v2_rank32_gpu.csv"
LOCK="$ROOT/logs/qwen_multidegradation_visibility_v2_rank32.lock"
OUTPUT="$ROOT/checkpoints/qwen_multidegradation_visibility_v2_rank32"
OVERFIT="$ROOT/checkpoints/qwen_multidegradation_visibility_v2_rank32_overfit/result.json"
VARIANT=merger_trainable
FINAL_CHECKPOINT="$OUTPUT/$VARIANT/checkpoints/V2_H5_per_type_dialogue_epoch5/adapter.safetensors"
VALIDATION_RESULT="$OUTPUT/$VARIANT/validation_result.json"
FIXED_TEST_RESULT="$OUTPUT/$VARIANT/fixed_test_result.json"

cd "$ROOT" || exit 1
mkdir -p "$ROOT/logs"
exec > >(tee -a "$LOG") 2>&1
echo "QWEN_VISIBILITY_V2_RANK32_RUN_START_UTC $(date -u +%Y-%m-%dT%H:%M:%SZ)"

telemetry_pid=
finish_run() {
  status=$?
  if [[ -n "$telemetry_pid" ]]; then
    kill "$telemetry_pid" 2>/dev/null || true
    wait "$telemetry_pid" 2>/dev/null || true
  fi
  echo "QWEN_VISIBILITY_V2_RANK32_RUN_END_UTC $(date -u +%Y-%m-%dT%H:%M:%SZ) status=$status"
}
trap finish_run EXIT

exec 9>>"$LOCK"
if ! flock -n 9; then
  echo "ERROR: another visibility-v2 Qwen pipeline holds $LOCK" >&2
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

echo "QWEN_VISIBILITY_V2_RANK32_WORKSPACE $ROOT"
echo "QWEN_VISIBILITY_V2_RANK32_CONFIG $CONFIG"
echo "QWEN_VISIBILITY_V2_RANK32_GIT_REVISION $(git rev-parse HEAD)"
echo "QWEN_VISIBILITY_V2_RANK32_VISIBLE_DEVICES $CUDA_VISIBLE_DEVICES"
echo "QWEN_VISIBILITY_V2_RANK32_WORLD_SIZE $NPROC"
echo "QWEN_VISIBILITY_V2_RANK32_BATCH micro_per_rank=25 accumulation=$ACCUMULATION global=200"
echo "QWEN_VISIBILITY_V2_RANK32_SCHEDULE epochs=5 dialogues=210000 steps=1050"
echo "QWEN_VISIBILITY_V2_RANK32_PROTECTED_TEST_POLICY validation_gate_before_one_test_access"

"$PYTHON" smoke/qwen_multidegradation_lora_preflight.py --config "$CONFIG" --full-hashes
preflight_status=$?
if [[ $preflight_status -ne 0 ]]; then
  echo "QWEN_VISIBILITY_V2_RANK32_PREFLIGHT_FAILED status=$preflight_status"
  exit "$preflight_status"
fi

nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw --format=csv -l 5 > "$GPU_LOG" 2>&1 &
telemetry_pid=$!

if [[ ! -f "$FINAL_CHECKPOINT" ]]; then
  "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$NPROC" \
    smoke/train_qwen_multidegradation_overfit_ddp.py --config "$CONFIG"
  overfit_status=$?
  if [[ ! -f "$OVERFIT" ]]; then
    echo "QWEN_VISIBILITY_V2_RANK32_OVERFIT_CRASH outer_status=$overfit_status" >&2
    exit 1
  fi
  if [[ $overfit_status -ne 0 ]]; then
    echo "QWEN_VISIBILITY_V2_RANK32_OVERFIT_SCIENTIFIC_FAILURE outer_status=$overfit_status"
    exit 2
  fi
  echo "QWEN_VISIBILITY_V2_RANK32_OVERFIT_GATE_PASS"
fi

training_mode=()
if [[ -f "$FINAL_CHECKPOINT" && ! -f "$VALIDATION_RESULT" ]]; then
  training_mode=(--validation-only)
  echo "QWEN_VISIBILITY_V2_RANK32_VALIDATION_ONLY_RECOVERY checkpoint=$FINAL_CHECKPOINT"
fi
"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$NPROC" \
  smoke/train_qwen_multidegradation_lora_ddp.py --config "$CONFIG" --variant "$VARIANT" "${training_mode[@]}"
validation_status=$?
if [[ ! -f "$VALIDATION_RESULT" ]]; then
  echo "QWEN_VISIBILITY_V2_RANK32_TRAIN_OR_VALIDATION_CRASH outer_status=$validation_status" >&2
  exit 1
fi
if [[ $validation_status -ne 0 ]]; then
  echo "QWEN_VISIBILITY_V2_RANK32_VALIDATION_SCIENTIFIC_FAILURE outer_status=$validation_status"
  exit 2
fi
echo "QWEN_VISIBILITY_V2_RANK32_VALIDATION_GATE_PASS"

"$PYTHON" smoke/select_qwen_multidegradation_lora.py --config "$CONFIG"
selection_status=$?
if [[ $selection_status -ne 0 ]]; then
  echo "QWEN_VISIBILITY_V2_RANK32_SELECTION_FAILED"
  exit 2
fi

"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$NPROC" \
  smoke/train_qwen_multidegradation_lora_ddp.py --config "$CONFIG" --variant "$VARIANT" --test-only
test_status=$?
if [[ $test_status -ne 0 ]]; then
  if [[ -f "$FIXED_TEST_RESULT" ]]; then
    echo "QWEN_VISIBILITY_V2_RANK32_FIXED_TEST_SCIENTIFIC_FAILURE outer_status=$test_status"
    exit 2
  fi
  echo "QWEN_VISIBILITY_V2_RANK32_FIXED_TEST_CRASH outer_status=$test_status" >&2
  exit 1
fi

echo "QWEN_VISIBILITY_V2_RANK32_RUN_PASS"
exit 0
