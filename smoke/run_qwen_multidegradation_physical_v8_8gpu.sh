#!/usr/bin/env bash
set -uo pipefail

ROOT=/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
PYTHON="$ROOT/.conda/gpu-train-stable/bin/python"
CONFIG="$ROOT/configs/qwen_multidegradation_100k_rank64_per_type_physical_v8.json"
OUTPUT="$ROOT/checkpoints/qwen_multidegradation_100k_rank64_per_type_physical_v8/merger_trainable"
LOG="$ROOT/logs/qwen_multidegradation_100k_rank64_per_type_physical_v8.log"
GPU_LOG="$ROOT/logs/qwen_multidegradation_100k_rank64_per_type_physical_v8_gpu.csv"
LOCK="$ROOT/logs/qwen_multidegradation_100k_rank64_per_type_physical_v8.lock"

cd "$ROOT" || exit 1
mkdir -p "$ROOT/logs"
exec > >(tee -a "$LOG") 2>&1
echo "QWEN_PHYSICAL_V8_START utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "QWEN_PHYSICAL_V8_WORKSPACE $ROOT"
echo "QWEN_PHYSICAL_V8_GIT $(git rev-parse HEAD)"
echo "QWEN_PHYSICAL_V8_CONFIG $CONFIG"
echo "QWEN_PHYSICAL_V8_CONFIG_SHA256 $(sha256sum "$CONFIG" | awk '{print $1}')"

telemetry_pid=
finish_run() {
  status=$?
  if [[ -n "$telemetry_pid" ]]; then
    kill "$telemetry_pid" 2>/dev/null || true
    wait "$telemetry_pid" 2>/dev/null || true
  fi
  echo "QWEN_PHYSICAL_V8_END status=$status utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
trap finish_run EXIT

exec 9>>"$LOCK"
if ! flock -n 9; then
  echo "ERROR: another physical-v8 Qwen run holds $LOCK" >&2
  exit 1
fi
if [[ "${CUDA_VISIBLE_DEVICES:-}" != "0,1,2,3,4,5,6,7" ]]; then
  echo "ERROR: set CUDA_VISIBLE_DEVICES exactly to 0,1,2,3,4,5,6,7" >&2
  exit 1
fi
if [[ ! -x "$PYTHON" || ! -f "$CONFIG" ]]; then
  echo "ERROR: workspace interpreter or physical-v8 config is missing" >&2
  exit 1
fi

mapfile -t gpu_rows < <(
  nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader,nounits
)
if [[ ${#gpu_rows[@]} -lt 8 ]]; then
  echo "ERROR: eight visible GPUs are required" >&2
  exit 1
fi
for row in "${gpu_rows[@]:0:8}"; do
  if [[ ! "$row" =~ NVIDIA[[:space:]]+H(100|200) ]]; then
    echo "ERROR: physical-v8 supports only NVIDIA H100/H200; found $row" >&2
    exit 1
  fi
done
echo "QWEN_PHYSICAL_V8_GPU_INVENTORY ${gpu_rows[*]:0:8}"

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

nvidia-smi \
  --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw \
  --format=csv -l 5 >"$GPU_LOG" 2>&1 &
telemetry_pid=$!

"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 \
  smoke/train_qwen_multidegradation_lora_ddp.py \
  --config "$CONFIG" --variant merger_trainable
torchrun_status=$?

config_sha256=$(sha256sum "$CONFIG" | awk '{print $1}')
"$PYTHON" smoke/qwen_multidegradation_result_status.py \
  --workspace-root "$ROOT" \
  --output-dir "$OUTPUT" \
  --config-sha256 "$config_sha256" \
  --variant merger_trainable
status=$?
case "$status" in
  0) echo "QWEN_PHYSICAL_V8_VERIFIED_PASS" ;;
  2) echo "QWEN_PHYSICAL_V8_VERIFIED_SCIENTIFIC_FAILURE" ;;
  *) echo "QWEN_PHYSICAL_V8_TRUE_ERROR torchrun_status=$torchrun_status" >&2 ;;
esac
echo "QWEN_PHYSICAL_V8_FOUNDIR_QWEN_TRAINING_IMAGES 0"
echo "QWEN_PHYSICAL_V8_PROTECTED_TEST_NOT_ACCESSED"
exit "$status"
