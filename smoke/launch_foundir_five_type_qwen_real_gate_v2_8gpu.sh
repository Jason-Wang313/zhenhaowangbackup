#!/usr/bin/env bash
set -euo pipefail

ROOT=/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
RUNNER="$ROOT/smoke/run_foundir_five_type_qwen_real_gate_v2_8gpu.sh"
LAUNCHER_LOG="$ROOT/logs/foundir_five_type_qwen_real_gate_v2_launcher.log"

if [[ "${CUDA_VISIBLE_DEVICES:-}" != "0,1,2,3,4,5,6,7" ]]; then
  echo "ERROR: set CUDA_VISIBLE_DEVICES exactly to 0,1,2,3,4,5,6,7" >&2
  exit 1
fi
cd "$ROOT"
bash -n "$RUNNER"
mkdir -p "$ROOT/logs"
echo "Policy: FoundIR training images only; official test is not accessed"

system_tmux=$(command -v tmux 2>/dev/null || true)
if [[ -z "$system_tmux" || ! -x "$system_tmux" ]]; then
  workspace_tmux="$ROOT/.tools/tmux-env/bin/tmux"
  if [[ -x "$workspace_tmux" ]]; then
    system_tmux=$workspace_tmux
  fi
fi
session=foundir_qwen_real_gate_v2
if [[ -n "$system_tmux" && -x "$system_tmux" ]] && "$system_tmux" -V >/dev/null 2>&1; then
  if "$system_tmux" has-session -t "$session" 2>/dev/null; then
    echo "ERROR: tmux session already exists: $session" >&2
    exit 1
  fi
  "$system_tmux" new-session -d -s "$session" "bash"
  "$system_tmux" set-option -t "$session" remain-on-exit on
  "$system_tmux" send-keys -t "$session" "cd '$ROOT' && CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' bash '$RUNNER'" Enter
  echo "FOUNDIR_QWEN_REAL_GATE_V2_TMUX_STARTED session=$session"
  echo "Attach: $system_tmux attach -t $session"
else
  nohup env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash "$RUNNER" >> "$LAUNCHER_LOG" 2>&1 </dev/null &
  pid=$!
  sleep 2
  kill -0 "$pid" 2>/dev/null || {
    echo "ERROR: Qwen real-gate launcher exited during startup; inspect $LAUNCHER_LOG" >&2
    exit 1
  }
  echo "$pid" > "$ROOT/logs/foundir_five_type_qwen_real_gate_v2.pid"
  echo "FOUNDIR_QWEN_REAL_GATE_V2_NOHUP_STARTED pid=$pid"
fi
echo "Monitor: tail -f $ROOT/logs/foundir_five_type_qwen_real_gate_v2.log"
