#!/usr/bin/env bash
set -euo pipefail

ROOT=/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
/usr/bin/python3 "$ROOT/smoke/advisor_governance.py" check
RUNNER="$ROOT/smoke/run_foundir_five_type_qwen_real_gate_physical_v8_8gpu.sh"
LOG="$ROOT/logs/foundir_five_type_qwen_real_gate_physical_v8.log"
LAUNCHER_LOG="$ROOT/logs/foundir_five_type_qwen_real_gate_physical_v8_launcher.log"
PID_FILE="$ROOT/logs/foundir_five_type_qwen_real_gate_physical_v8.pid"
PYTHON="$ROOT/.conda/gpu-train-stable/bin/python"
CONFIG="$ROOT/configs/foundir_five_type_qwen_real_gate_physical_v8.json"
EVALUATOR="$ROOT/smoke/evaluate_foundir_five_type_qwen_real_gate_v2_ddp.py"

if [[ "${CUDA_VISIBLE_DEVICES:-}" != "0,1,2,3,4,5,6,7" ]]; then
  echo "ERROR: set CUDA_VISIBLE_DEVICES exactly to 0,1,2,3,4,5,6,7" >&2
  exit 1
fi
cd "$ROOT"
"$PYTHON" -m py_compile "$EVALUATOR"
bash -n "$RUNNER" "$0"
"$PYTHON" "$EVALUATOR" --config "$CONFIG" --check-only
mkdir -p "$ROOT/logs"
echo "Policy: FoundIR training images only; official and synthetic tests are not accessed"

system_tmux=$(command -v tmux 2>/dev/null || true)
if [[ -z "$system_tmux" || ! -x "$system_tmux" ]]; then
  workspace_tmux="$ROOT/.tools/tmux-env/bin/tmux"
  if [[ -x "$workspace_tmux" ]]; then
    system_tmux=$workspace_tmux
  fi
fi
session=foundir_qwen_real_gate_physical_v8
if [[ -n "$system_tmux" && -x "$system_tmux" ]] && "$system_tmux" -V >/dev/null 2>&1; then
  if "$system_tmux" has-session -t "$session" 2>/dev/null; then
    echo "ERROR: tmux session already exists: $session" >&2
    exit 1
  fi
  "$system_tmux" new-session -d -s "$session" "bash"
  "$system_tmux" set-option -t "$session" remain-on-exit on
  "$system_tmux" send-keys -t "$session" "cd '$ROOT' && CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' bash '$RUNNER'" Enter
  echo "FOUNDIR_QWEN_REAL_GATE_PHYSICAL_V8_TMUX_STARTED session=$session"
  echo "Attach: $system_tmux attach -t $session"
else
  if [[ -f "$PID_FILE" ]]; then
    old_pid=$(tr -dc '0-9' < "$PID_FILE")
    if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
      echo "ERROR: physical-v8 FoundIR-training Qwen gate is already active as PID $old_pid" >&2
      exit 1
    fi
  fi
  nohup env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash "$RUNNER" \
    >> "$LAUNCHER_LOG" 2>&1 </dev/null &
  pid=$!
  printf '%s\n' "$pid" > "$PID_FILE"
  sleep 3
  kill -0 "$pid" 2>/dev/null || {
    echo "ERROR: physical-v8 real-gate launcher exited during startup; inspect $LAUNCHER_LOG" >&2
    exit 1
  }
  echo "FOUNDIR_QWEN_REAL_GATE_PHYSICAL_V8_NOHUP_STARTED pid=$pid"
fi
echo "Monitor: tail -f $LOG"
