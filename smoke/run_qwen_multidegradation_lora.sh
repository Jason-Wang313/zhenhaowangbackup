#!/usr/bin/env bash
set -uo pipefail

ROOT=/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
PYTHON="$ROOT/.conda/gpu-train-stable/bin/python"
CONFIG="$ROOT/configs/qwen_multidegradation_lora_v1.json"
LOG="$ROOT/logs/qwen_multidegradation_lora_v1.log"
GPU_LOG="$ROOT/logs/qwen_multidegradation_lora_v1_gpu.csv"
LOCK="$ROOT/logs/qwen_multidegradation_lora_v1.lock"

cd "$ROOT" || exit 1
mkdir -p "$ROOT/logs"
exec > >(tee -a "$LOG") 2>&1
echo "QWEN_MULTIDEGRADATION_RUN_START_UTC $(date -u +%Y-%m-%dT%H:%M:%SZ)"

telemetry_pid=
finish_run() {
  status=$?
  if [[ -n "$telemetry_pid" ]]; then
    kill "$telemetry_pid" 2>/dev/null || true
    wait "$telemetry_pid" 2>/dev/null || true
  fi
  echo "QWEN_MULTIDEGRADATION_RUN_END_UTC $(date -u +%Y-%m-%dT%H:%M:%SZ) status=$status"
}
trap finish_run EXIT

exec 9>>"$LOCK"
if ! flock -n 9; then
  echo "ERROR: another Qwen multi-degradation pipeline holds $LOCK" >&2
  exit 1
fi
case "${CUDA_VISIBLE_DEVICES:-}" in
  0,1) NPROC=2 ;;
  0,1,2,3) NPROC=4 ;;
  *)
    echo "ERROR: set CUDA_VISIBLE_DEVICES exactly to 0,1 or 0,1,2,3" >&2
    exit 1
    ;;
esac
if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: missing workspace interpreter: $PYTHON" >&2
  exit 1
fi

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

echo "QWEN_MULTIDEGRADATION_WORKSPACE $ROOT"
echo "QWEN_MULTIDEGRADATION_GIT_REVISION $(git rev-parse HEAD)"
echo "QWEN_MULTIDEGRADATION_VISIBLE_DEVICES $CUDA_VISIBLE_DEVICES"
echo "QWEN_MULTIDEGRADATION_WORLD_SIZE $NPROC"

"$PYTHON" smoke/qwen_multidegradation_lora_preflight.py --config "$CONFIG" --full-hashes
preflight_status=$?
if [[ $preflight_status -ne 0 ]]; then
  echo "QWEN_MULTIDEGRADATION_PREFLIGHT_FAILED status=$preflight_status"
  exit "$preflight_status"
fi

nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw --format=csv -l 5 > "$GPU_LOG" 2>&1 &
telemetry_pid=$!

"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$NPROC" \
  smoke/train_qwen_multidegradation_overfit_ddp.py --config "$CONFIG"
overfit_status=$?
overfit_result="$ROOT/checkpoints/qwen_multidegradation_lora_overfit_v1/result.json"
if [[ $overfit_status -ne 0 ]]; then
  if [[ -f "$overfit_result" ]]; then
    echo "QWEN_MULTIDEGRADATION_OVERFIT_SCIENTIFIC_FAILURE outer_status=$overfit_status"
    exit 2
  fi
  echo "QWEN_MULTIDEGRADATION_OVERFIT_CRASH outer_status=$overfit_status" >&2
  exit 1
fi

run_variant() {
  local variant=$1
  "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$NPROC" \
    smoke/train_qwen_multidegradation_lora_ddp.py \
    --config "$CONFIG" --variant "$variant"
  local outer_status=$?
  local result="$ROOT/checkpoints/qwen_multidegradation_lora_v1/$variant/validation_result.json"
  if [[ -f "$result" ]]; then
    "$PYTHON" - "$result" <<'PY'
import json,sys
p=json.load(open(sys.argv[1]))
print("QWEN_MULTIDEGRADATION_VARIANT_COMPLETE", p["variant"], p["status"], "outer_status_recorded_separately")
PY
    return 0
  fi
  echo "QWEN_MULTIDEGRADATION_VARIANT_CRASH variant=$variant outer_status=$outer_status" >&2
  return 1
}

run_variant merger_frozen || exit 1
run_variant merger_trainable || exit 1

"$PYTHON" smoke/select_qwen_multidegradation_lora.py
selection_status=$?
if [[ $selection_status -ne 0 ]]; then
  echo "QWEN_MULTIDEGRADATION_NO_VARIANT_PASSED"
  exit 2
fi
selected_variant=$("$PYTHON" - <<'PY'
import json
print(json.load(open('checkpoints/qwen_multidegradation_lora_v1/checkpoint_selection.json'))['selected_variant'])
PY
)
"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$NPROC" \
  smoke/train_qwen_multidegradation_lora_ddp.py \
  --config "$CONFIG" --variant "$selected_variant" --test-only
test_status=$?
if [[ $test_status -ne 0 ]]; then
  test_result="$ROOT/checkpoints/qwen_multidegradation_lora_v1/$selected_variant/fixed_test_result.json"
  if [[ -f "$test_result" ]]; then
    echo "QWEN_MULTIDEGRADATION_FIXED_TEST_SCIENTIFIC_FAILURE outer_status=$test_status"
    exit 2
  fi
  echo "QWEN_MULTIDEGRADATION_FIXED_TEST_CRASH outer_status=$test_status" >&2
  exit 1
fi

"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$NPROC" \
  smoke/evaluate_qwen_multidegradation_real_sanity_ddp.py --config "$CONFIG"
real_status=$?
if [[ $real_status -ne 0 ]]; then
  echo "QWEN_MULTIDEGRADATION_REAL_SANITY_INFERENCE_CRASH status=$real_status" >&2
  exit 1
fi
"$PYTHON" smoke/render_qwen_multidegradation_real_sanity.py
render_status=$?
if [[ $render_status -ne 0 ]]; then
  echo "QWEN_MULTIDEGRADATION_REAL_SANITY_RENDER_CRASH status=$render_status" >&2
  exit 1
fi

echo "QWEN_MULTIDEGRADATION_RUN_PASS selected_variant=$selected_variant"
exit 0
