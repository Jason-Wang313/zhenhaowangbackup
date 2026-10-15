#!/usr/bin/env bash
set -uo pipefail

ROOT=/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
/usr/bin/python3 "$ROOT/smoke/advisor_governance.py" check
PYTHON="$ROOT/.conda/gpu-train-stable/bin/python"
CONFIG="$ROOT/configs/foundir_five_type_qwen_real_gate_v2.json"
EVALUATOR="$ROOT/smoke/evaluate_foundir_five_type_qwen_real_gate_v2_ddp.py"
LOG="$ROOT/logs/foundir_five_type_qwen_real_gate_v2.log"
GPU_LOG="$ROOT/logs/foundir_five_type_qwen_real_gate_v2_gpu.csv"
LOCK="$ROOT/logs/foundir_five_type_qwen_real_gate_v2.lock"

cd "$ROOT" || exit 1
mkdir -p "$ROOT/logs"
exec > >(tee -a "$LOG") 2>&1
echo "FOUNDIR_QWEN_REAL_GATE_V2_START utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

telemetry_pid=
finish_run() {
  status=$?
  if [[ -n "$telemetry_pid" ]]; then
    kill "$telemetry_pid" 2>/dev/null || true
    wait "$telemetry_pid" 2>/dev/null || true
  fi
  echo "FOUNDIR_QWEN_REAL_GATE_V2_END status=$status utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
trap finish_run EXIT

exec 9>>"$LOCK"
if ! flock -n 9; then
  echo "ERROR: another FoundIR-training Qwen v2 gate holds $LOCK" >&2
  exit 1
fi
if [[ "${CUDA_VISIBLE_DEVICES:-}" != "0,1,2,3,4,5,6,7" ]]; then
  echo "ERROR: set CUDA_VISIBLE_DEVICES exactly to 0,1,2,3,4,5,6,7" >&2
  exit 1
fi
if [[ ! -x "$PYTHON" || ! -f "$CONFIG" || ! -f "$EVALUATOR" ]]; then
  echo "ERROR: workspace interpreter, config, or evaluator is missing" >&2
  exit 1
fi

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

echo "FOUNDIR_QWEN_REAL_GATE_V2_GIT $(git rev-parse HEAD)"
echo "FOUNDIR_QWEN_REAL_GATE_V2_SCOPE training_images_inference_only"
echo "FOUNDIR_QWEN_REAL_GATE_V2_QWEN_TRAINING false"
echo "FOUNDIR_QWEN_REAL_GATE_V2_OFFICIAL_TEST not_accessed"

"$PYTHON" "$EVALUATOR" --config "$CONFIG" --check-only || exit 1

nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw \
  --format=csv -l 5 > "$GPU_LOG" 2>&1 &
telemetry_pid=$!

"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 \
  "$EVALUATOR" --config "$CONFIG"
torchrun_status=$?
"$PYTHON" "$EVALUATOR" --config "$CONFIG" --verify-existing
status=$?
case "$status" in
  0) echo "FOUNDIR_QWEN_REAL_GATE_V2_VERIFIED_PASS" ;;
  2) echo "FOUNDIR_QWEN_REAL_GATE_V2_VERIFIED_SCIENTIFIC_FAILURE" ;;
  *) echo "FOUNDIR_QWEN_REAL_GATE_V2_TRUE_ERROR torchrun_status=$torchrun_status" >&2 ;;
esac
echo "FOUNDIR_QWEN_REAL_GATE_V2_OFFICIAL_TEST_NOT_ACCESSED"
exit "$status"
