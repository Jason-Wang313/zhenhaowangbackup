#!/usr/bin/env bash
set -uo pipefail

ROOT=/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
PYTHON="$ROOT/.conda/gpu-train-stable/bin/python"
METHOD=${QWEN_100K_METHOD:-}
case "$METHOD" in
  joint)
    CONFIG="$ROOT/configs/qwen_multidegradation_100k_rank64_joint_v1.json"
    LOG="$ROOT/logs/qwen_multidegradation_100k_rank64_joint_v1.log"
    GPU_LOG="$ROOT/logs/qwen_multidegradation_100k_rank64_joint_v1_gpu.csv"
    LOCK="$ROOT/logs/qwen_multidegradation_100k_rank64_joint_v1.lock"
    ;;
  per_type)
    CONFIG="$ROOT/configs/qwen_multidegradation_100k_rank64_per_type_v1.json"
    LOG="$ROOT/logs/qwen_multidegradation_100k_rank64_per_type_v1.log"
    GPU_LOG="$ROOT/logs/qwen_multidegradation_100k_rank64_per_type_v1_gpu.csv"
    LOCK="$ROOT/logs/qwen_multidegradation_100k_rank64_per_type_v1.lock"
    ;;
  *) echo "ERROR: set QWEN_100K_METHOD to joint or per_type" >&2; exit 1 ;;
esac

cd "$ROOT" || exit 1
mkdir -p "$ROOT/logs"
exec > >(tee -a "$LOG") 2>&1
echo "QWEN_100K_RANK64_METHOD_START method=$METHOD utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

telemetry_pid=
finish_run() {
  status=$?
  if [[ -n "$telemetry_pid" ]]; then
    kill "$telemetry_pid" 2>/dev/null || true
    wait "$telemetry_pid" 2>/dev/null || true
  fi
  echo "QWEN_100K_RANK64_METHOD_END method=$METHOD status=$status utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
trap finish_run EXIT

exec 9>>"$LOCK"
if ! flock -n 9; then
  echo "ERROR: another $METHOD rank-64 run holds $LOCK" >&2
  exit 1
fi

visible=${CUDA_VISIBLE_DEVICES:-}
if [[ -z "$visible" ]]; then
  echo "ERROR: CUDA_VISIBLE_DEVICES is empty" >&2
  exit 1
fi
IFS=, read -r -a devices <<<"$visible"
NPROC=${#devices[@]}
case "$NPROC" in
  1|2|4|8) ;;
  *) echo "ERROR: each method requires 1, 2, 4, or 8 visible GPUs" >&2; exit 1 ;;
esac
if [[ $(printf '%s\n' "${devices[@]}" | sort -u | wc -l) -ne $NPROC ]]; then
  echo "ERROR: CUDA_VISIBLE_DEVICES contains duplicates" >&2
  exit 1
fi
for device in "${devices[@]}"; do
  if [[ ! "$device" =~ ^[0-9]+$ ]]; then
    echo "ERROR: visible GPU IDs must be nonnegative integers" >&2
    exit 1
  fi
done
if [[ ! -x "$PYTHON" || ! -f "$CONFIG" ]]; then
  echo "ERROR: workspace interpreter or frozen config is missing" >&2
  exit 1
fi

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

echo "QWEN_100K_RANK64_METHOD_CONFIG $CONFIG"
echo "QWEN_100K_RANK64_METHOD_VISIBLE $visible"
echo "QWEN_100K_RANK64_METHOD_WORLD $NPROC"
echo "QWEN_100K_RANK64_METHOD_HARDWARE H100_80GB_OR_H200"
echo "QWEN_100K_RANK64_METHOD_LORA visual_rank=64 visual_alpha=64 language_rank=64 language_alpha=64"
echo "QWEN_100K_RANK64_METHOD_TEST_POLICY validation_only_protected_test_not_accessed"

"$PYTHON" smoke/qwen_multidegradation_lora_preflight.py --config "$CONFIG"
preflight_status=$?
if [[ $preflight_status -ne 0 ]]; then
  echo "QWEN_100K_RANK64_PREFLIGHT_FAILED method=$METHOD status=$preflight_status"
  exit "$preflight_status"
fi

nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw --format=csv -l 5 > "$GPU_LOG" 2>&1 &
telemetry_pid=$!

"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$NPROC" \
  smoke/train_qwen_multidegradation_lora_ddp.py \
  --config "$CONFIG" --variant merger_trainable
status=$?
echo "QWEN_100K_RANK64_PROTECTED_TEST_NOT_ACCESSED method=$METHOD"
exit "$status"
