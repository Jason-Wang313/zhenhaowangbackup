#!/usr/bin/env bash
set -uo pipefail

ROOT=/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
PYTHON="$ROOT/.conda/gpu-train-stable/bin/python"
CONFIG="$ROOT/configs/qwen_multidegradation_100k_rank64_per_type_v2.json"
OUTPUT="$ROOT/checkpoints/qwen_multidegradation_100k_rank64_per_type_v2/merger_trainable"
LOG="$ROOT/logs/qwen_multidegradation_100k_rank64_per_type_v2_fixed_test.log"
GPU_LOG="$ROOT/logs/qwen_multidegradation_100k_rank64_per_type_v2_fixed_test_gpu.csv"
LOCK="$ROOT/logs/qwen_multidegradation_100k_rank64_per_type_v2_fixed_test.lock"

cd "$ROOT" || exit 1
mkdir -p "$ROOT/logs"
exec > >(tee -a "$LOG") 2>&1
echo "QWEN_100K_V2_FIXED_TEST_START utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

telemetry_pid=
finish_run() {
  status=$?
  if [[ -n "$telemetry_pid" ]]; then
    kill "$telemetry_pid" 2>/dev/null || true
    wait "$telemetry_pid" 2>/dev/null || true
  fi
  echo "QWEN_100K_V2_FIXED_TEST_END status=$status utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
trap finish_run EXIT

exec 9>>"$LOCK"
if ! flock -n 9; then
  echo "ERROR: another v2 fixed-test process holds $LOCK" >&2
  exit 1
fi
case "${CUDA_VISIBLE_DEVICES:-}" in
  0,1) NPROC=2 ;;
  0,1,2,3,4,5,6,7) NPROC=8 ;;
  *) echo "ERROR: set CUDA_VISIBLE_DEVICES exactly to 0,1 or 0,1,2,3,4,5,6,7" >&2; exit 1 ;;
esac
if [[ ! -x "$PYTHON" || ! -f "$CONFIG" ]]; then
  echo "ERROR: workspace interpreter or frozen config is missing" >&2
  exit 1
fi

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

echo "QWEN_100K_V2_FIXED_TEST_WORKSPACE $ROOT"
echo "QWEN_100K_V2_FIXED_TEST_GIT_REVISION $(git rev-parse HEAD)"
echo "QWEN_100K_V2_FIXED_TEST_VISIBLE $CUDA_VISIBLE_DEVICES"
echo "QWEN_100K_V2_FIXED_TEST_WORLD $NPROC"
echo "QWEN_100K_V2_FIXED_TEST_POLICY one_time_synthetic_test_no_foundir"

"$PYTHON" smoke/verify_qwen_multidegradation_100k_v2_fixed_test.py || exit 1

nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw --format=csv -l 5 > "$GPU_LOG" 2>&1 &
telemetry_pid=$!

"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$NPROC" \
  smoke/train_qwen_multidegradation_lora_ddp.py \
  --config "$CONFIG" --variant merger_trainable --test-only
torchrun_status=$?
config_sha256=$(sha256sum "$CONFIG" | awk '{print $1}')
"$PYTHON" smoke/qwen_multidegradation_result_status.py \
  --workspace-root "$ROOT" \
  --output-dir "$OUTPUT" \
  --config-sha256 "$config_sha256" \
  --variant merger_trainable \
  --split test
status=$?
case "$status" in
  0) echo "QWEN_100K_V2_FIXED_TEST_VERIFIED_PASS" ;;
  2) echo "QWEN_100K_V2_FIXED_TEST_VERIFIED_SCIENTIFIC_FAILURE" ;;
  *) echo "QWEN_100K_V2_FIXED_TEST_TRUE_ERROR torchrun_status=$torchrun_status" >&2 ;;
esac
echo "QWEN_100K_V2_FIXED_TEST_FOUNDIR_NOT_ACCESSED"
exit "$status"
