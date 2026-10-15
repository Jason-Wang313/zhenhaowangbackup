#!/usr/bin/env bash
set -euo pipefail

ROOT=/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
PYTHON="$ROOT/.conda/gpu-train-stable/bin/python"
CONFIG="$ROOT/configs/foundir_qwen_compositional_router_v1.json"
LOG="$ROOT/logs/foundir_qwen_compositional_router_v1.log"
PID_FILE="$ROOT/logs/foundir_qwen_compositional_router_v1.pid"

cd "$ROOT"
bash -n "$0"
"$PYTHON" -m py_compile smoke/run_foundir_qwen_compositional_router_v1.py
"$PYTHON" smoke/run_foundir_qwen_compositional_router_v1.py --config "$CONFIG" --check-only
mkdir -p "$ROOT/logs"
if [[ -f "$PID_FILE" ]]; then
  old_pid=$(tr -dc '0-9' < "$PID_FILE")
  if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "ERROR: compositional router is already active as PID $old_pid" >&2
    exit 1
  fi
fi
nohup "$PYTHON" smoke/run_foundir_qwen_compositional_router_v1.py --config "$CONFIG" \
  >"$LOG" 2>&1 </dev/null &
pid=$!
printf '%s\n' "$pid" > "$PID_FILE"
sleep 2
kill -0 "$pid" 2>/dev/null || {
  echo "ERROR: compositional router exited during startup; inspect $LOG" >&2
  exit 1
}
echo "FOUNDIR_QWEN_COMPOSITIONAL_STARTED pid=$pid"
echo "Monitor: tail -f $LOG"
