#!/usr/bin/env bash
set -euo pipefail

ROOT=/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
LOG="$ROOT/logs/qwen_multidegradation_lora_v1_launcher.log"
RUNNER="$ROOT/smoke/run_qwen_multidegradation_lora.sh"

cd "$ROOT"
if [[ "${CUDA_VISIBLE_DEVICES:-}" != "0,1,2,3" ]]; then
  echo "ERROR: set CUDA_VISIBLE_DEVICES exactly to 0,1,2,3" >&2
  exit 1
fi
bash -n "$RUNNER"
mkdir -p "$ROOT/logs"

system_tmux=$(command -v tmux 2>/dev/null || true)
if [[ -n "$system_tmux" && -x "$system_tmux" ]]; then
  session=qwen_multidegradation_lora_v1
  if "$system_tmux" has-session -t "$session" 2>/dev/null; then
    echo "QWEN_MULTIDEGRADATION_TMUX_ALREADY_EXISTS session=$session"
  else
    "$system_tmux" new-session -d -s "$session" "bash"
    "$system_tmux" set-option -t "$session" remain-on-exit on
    "$system_tmux" send-keys -t "$session" "cd '$ROOT' && CUDA_VISIBLE_DEVICES=0,1,2,3 bash '$RUNNER'" Enter
  fi
  echo "QWEN_MULTIDEGRADATION_TMUX_STARTED session=$session"
  echo "Attach: $system_tmux attach -t $session"
else
  nohup env CUDA_VISIBLE_DEVICES=0,1,2,3 bash "$RUNNER" >> "$LOG" 2>&1 </dev/null &
  pid=$!
  echo "$pid" > "$ROOT/logs/qwen_multidegradation_lora_v1.pid"
  echo "QWEN_MULTIDEGRADATION_DETACHED_NOHUP_STARTED pid=$pid"
fi
echo "Monitor: tail -f $ROOT/logs/qwen_multidegradation_lora_v1.log"
