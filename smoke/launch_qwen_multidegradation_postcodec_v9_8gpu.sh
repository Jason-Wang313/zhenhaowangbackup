#!/usr/bin/env bash
set -euo pipefail

ROOT=/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
RUNNER="$ROOT/smoke/run_qwen_multidegradation_postcodec_v9_8gpu.sh"
LOG="$ROOT/logs/qwen_multidegradation_postcodec_v9.log"
PID_FILE="$ROOT/logs/qwen_multidegradation_postcodec_v9.pid"
PYTHON="$ROOT/.conda/gpu-train-stable/bin/python"

cd "$ROOT"
if [[ "${CUDA_VISIBLE_DEVICES:-}" != "0,1,2,3,4,5,6,7" ]]; then
  echo "ERROR: set CUDA_VISIBLE_DEVICES exactly to 0,1,2,3,4,5,6,7" >&2
  exit 1
fi
required=(
  configs/qwen_multidegradation_postcodec_v9_data.json
  configs/qwen_multidegradation_postcodec_v9.json
  lowlight_restoration/qwen_input_transform.py
  smoke/audit_foundir_qwen_noise_codec_order.py
  smoke/build_qwen_multidegradation_postcodec_v9.py
  smoke/qwen_multidegradation_lora_preflight.py
  smoke/train_qwen_multidegradation_lora_ddp.py
  smoke/run_qwen_multidegradation_postcodec_v9_8gpu.sh
  smoke/launch_qwen_multidegradation_postcodec_v9_8gpu.sh
)
"$PYTHON" -m py_compile \
  lowlight_restoration/qwen_input_transform.py \
  smoke/audit_foundir_qwen_noise_codec_order.py \
  smoke/build_qwen_multidegradation_postcodec_v9.py \
  smoke/qwen_multidegradation_lora_preflight.py \
  smoke/train_qwen_multidegradation_lora_ddp.py
bash -n "$RUNNER" "$0"
for relative in "${required[@]}"; do
  git ls-files --error-unmatch "$relative" >/dev/null 2>&1 || {
    echo "ERROR: required v9 file is not committed: $relative" >&2
    exit 1
  }
  git diff --quiet HEAD -- "$relative" || {
    echo "ERROR: required v9 file differs from committed revision: $relative" >&2
    exit 1
  }
done
if [[ "$(git rev-parse HEAD)" != "$(git rev-parse origin/main)" ]]; then
  echo "ERROR: local HEAD differs from origin/main" >&2
  exit 1
fi

mkdir -p "$ROOT/logs"
if [[ -f "$PID_FILE" ]]; then
  old_pid=$(tr -dc '0-9' <"$PID_FILE")
  if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "ERROR: post-codec v9 is already active as PID $old_pid" >&2
    exit 1
  fi
fi
nohup env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" bash "$RUNNER" \
  >/dev/null 2>&1 </dev/null &
pid=$!
printf '%s\n' "$pid" >"$PID_FILE"
sleep 3
if ! kill -0 "$pid" 2>/dev/null; then
  echo "ERROR: post-codec v9 launch exited; inspect $LOG" >&2
  tail -100 "$LOG" >&2 || true
  exit 1
fi
echo "QWEN_POSTCODEC_V9_DETACHED pid=$pid"
echo "Persistent log: $LOG"
echo "Monitor: tail -f $LOG"
