#!/usr/bin/env bash
set -euo pipefail

ROOT=/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
PYTHON="$ROOT/.conda/gpu-train-stable/bin/python"
RUNNER="$ROOT/smoke/run_foundir_qwen_eleven_class_fresh_holdout_v2_8gpu.sh"
CONFIG="$ROOT/configs/foundir_qwen_eleven_class_fresh_holdout_v2.json"
LOG="$ROOT/logs/foundir_qwen_eleven_class_fresh_holdout_v2_launcher.log"
PID_FILE="$ROOT/logs/foundir_qwen_eleven_class_fresh_holdout_v2.pid"

cd "$ROOT"
[[ "${CUDA_VISIBLE_DEVICES:-}" == "0,1,2,3,4,5,6,7" ]] || {
  echo "ERROR: set CUDA_VISIBLE_DEVICES exactly to 0,1,2,3,4,5,6,7" >&2
  exit 1
}
bash -n "$RUNNER" "$0"
"$PYTHON" -m py_compile smoke/run_foundir_qwen_eleven_class_gate_v1_ddp.py
"$PYTHON" smoke/run_foundir_qwen_eleven_class_gate_v1_ddp.py --config "$CONFIG" --check-only
mkdir -p "$ROOT/logs"
if [[ -f "$PID_FILE" ]]; then
  old_pid=$(tr -dc '0-9' < "$PID_FILE")
  if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "ERROR: Qwen eleven-class fresh holdout is already active as PID $old_pid" >&2
    exit 1
  fi
fi
nohup env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash "$RUNNER" >>"$LOG" 2>&1 </dev/null &
pid=$!
printf '%s\n' "$pid" > "$PID_FILE"
sleep 3
kill -0 "$pid" 2>/dev/null || {
  echo "ERROR: launcher exited during startup; inspect $LOG" >&2
  exit 1
}
echo "FOUNDIR_QWEN_ELEVEN_CLASS_FRESH_HOLDOUT_STARTED pid=$pid"
echo "Monitor: tail -f $ROOT/logs/foundir_qwen_eleven_class_fresh_holdout_v2.log"
