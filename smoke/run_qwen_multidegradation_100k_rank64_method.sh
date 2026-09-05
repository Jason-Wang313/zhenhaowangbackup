#!/usr/bin/env bash
set -uo pipefail

ROOT=/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
PYTHON="$ROOT/.conda/gpu-train-stable/bin/python"
METHOD=${QWEN_100K_METHOD:-}
VERSION=${QWEN_100K_VERSION:-v1}
case "$VERSION:$METHOD" in
  v1:joint)
    CONFIG="$ROOT/configs/qwen_multidegradation_100k_rank64_joint_v1.json"
    LOG="$ROOT/logs/qwen_multidegradation_100k_rank64_joint_v1.log"
    GPU_LOG="$ROOT/logs/qwen_multidegradation_100k_rank64_joint_v1_gpu.csv"
    LOCK="$ROOT/logs/qwen_multidegradation_100k_rank64_joint_v1.lock"
    OUTPUT_DIR="$ROOT/checkpoints/qwen_multidegradation_100k_rank64_joint_v1/merger_trainable"
    ;;
  v1:per_type)
    CONFIG="$ROOT/configs/qwen_multidegradation_100k_rank64_per_type_v1.json"
    LOG="$ROOT/logs/qwen_multidegradation_100k_rank64_per_type_v1.log"
    GPU_LOG="$ROOT/logs/qwen_multidegradation_100k_rank64_per_type_v1_gpu.csv"
    LOCK="$ROOT/logs/qwen_multidegradation_100k_rank64_per_type_v1.lock"
    OUTPUT_DIR="$ROOT/checkpoints/qwen_multidegradation_100k_rank64_per_type_v1/merger_trainable"
    ;;
  v2:joint)
    CONFIG="$ROOT/configs/qwen_multidegradation_100k_rank64_joint_v2.json"
    LOG="$ROOT/logs/qwen_multidegradation_100k_rank64_joint_v2.log"
    GPU_LOG="$ROOT/logs/qwen_multidegradation_100k_rank64_joint_v2_gpu.csv"
    LOCK="$ROOT/logs/qwen_multidegradation_100k_rank64_joint_v2.lock"
    OUTPUT_DIR="$ROOT/checkpoints/qwen_multidegradation_100k_rank64_joint_v2/merger_trainable"
    ;;
  v2:per_type)
    CONFIG="$ROOT/configs/qwen_multidegradation_100k_rank64_per_type_v2.json"
    LOG="$ROOT/logs/qwen_multidegradation_100k_rank64_per_type_v2.log"
    GPU_LOG="$ROOT/logs/qwen_multidegradation_100k_rank64_per_type_v2_gpu.csv"
    LOCK="$ROOT/logs/qwen_multidegradation_100k_rank64_per_type_v2.lock"
    OUTPUT_DIR="$ROOT/checkpoints/qwen_multidegradation_100k_rank64_per_type_v2/merger_trainable"
    ;;
  *) echo "ERROR: set QWEN_100K_VERSION to v1 or v2 and QWEN_100K_METHOD to joint or per_type" >&2; exit 1 ;;
esac

cd "$ROOT" || exit 1
mkdir -p "$ROOT/logs"
exec > >(tee -a "$LOG") 2>&1
echo "QWEN_100K_RANK64_METHOD_START version=$VERSION method=$METHOD utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

telemetry_pid=
finish_run() {
  status=$?
  if [[ -n "$telemetry_pid" ]]; then
    kill "$telemetry_pid" 2>/dev/null || true
    wait "$telemetry_pid" 2>/dev/null || true
  fi
  echo "QWEN_100K_RANK64_METHOD_END version=$VERSION method=$METHOD status=$status utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
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

nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw --format=csv -l 5 > "$GPU_LOG" 2>&1 &
telemetry_pid=$!

# The trainer runs the same fail-closed preflight once on rank zero before it
# loads the model.  Do not repeat the 120k-file scan in this shell wrapper.
"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$NPROC" \
  smoke/train_qwen_multidegradation_lora_ddp.py \
  --config "$CONFIG" --variant merger_trainable
torchrun_status=$?
config_sha256=$(sha256sum "$CONFIG" | awk '{print $1}')
"$PYTHON" smoke/qwen_multidegradation_result_status.py \
  --workspace-root "$ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --config-sha256 "$config_sha256" \
  --variant merger_trainable
status=$?
case "$status" in
  0) echo "QWEN_100K_RANK64_VERIFIED_PASS method=$METHOD" ;;
  2) echo "QWEN_100K_RANK64_VERIFIED_SCIENTIFIC_FAILURE method=$METHOD" ;;
  *) echo "QWEN_100K_RANK64_TRUE_ERROR method=$METHOD torchrun_status=$torchrun_status" >&2 ;;
esac
echo "QWEN_100K_RANK64_PROTECTED_TEST_NOT_ACCESSED method=$METHOD"
exit "$status"
