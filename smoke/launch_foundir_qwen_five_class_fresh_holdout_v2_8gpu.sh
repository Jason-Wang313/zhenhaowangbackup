#!/usr/bin/env bash
set -euo pipefail

ROOT=/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
PYTHON="$ROOT/.conda/gpu-train-stable/bin/python"
RUNNER="$ROOT/smoke/run_foundir_qwen_five_class_fresh_holdout_v2_8gpu.sh"
CONFIG="$ROOT/configs/foundir_qwen_five_class_fresh_holdout_v2.json"
PID_FILE="$ROOT/logs/foundir_qwen_five_class_fresh_holdout_v2.pid"
LAUNCHER_LOG="$ROOT/logs/foundir_qwen_five_class_fresh_holdout_v2_launcher.log"

cd "$ROOT"
[[ "${CUDA_VISIBLE_DEVICES:-}" == "0,1,2,3,4,5,6,7" ]] || {
  echo "ERROR: set CUDA_VISIBLE_DEVICES exactly to 0,1,2,3,4,5,6,7" >&2
  exit 1
}
bash -n "$RUNNER" "$0"
"$PYTHON" -m py_compile smoke/evaluate_foundir_qwen_five_class_fresh_holdout_v2_ddp.py
"$PYTHON" smoke/evaluate_foundir_qwen_five_class_fresh_holdout_v2_ddp.py --config "$CONFIG" --check-only
mkdir -p "$ROOT/logs"
nohup env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash "$RUNNER" >>"$LAUNCHER_LOG" 2>&1 </dev/null &
pid=$!
printf '%s\n' "$pid" > "$PID_FILE"
sleep 3
kill -0 "$pid" 2>/dev/null || {
  echo "ERROR: v2 launcher exited during startup; inspect $LAUNCHER_LOG" >&2
  exit 1
}
echo "FOUNDIR_QWEN_FIVE_CLASS_FRESH_HOLDOUT_V2_STARTED pid=$pid"
echo "Monitor: tail -f $ROOT/logs/foundir_qwen_five_class_fresh_holdout_v2.log"
