#!/usr/bin/env bash
set -euo pipefail

ROOT=/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
RUNNER="$ROOT/smoke/run_qwen_multidegradation_100k_rank64_per_type_v2_fixed_test.sh"
LAUNCHER_LOG="$ROOT/logs/qwen_multidegradation_100k_rank64_per_type_v2_fixed_test_launcher.log"
VISIBLE=${CUDA_VISIBLE_DEVICES:-}

case "$VISIBLE" in
  0,1|0,1,2,3,4,5,6,7) ;;
  *) echo "ERROR: set CUDA_VISIBLE_DEVICES exactly to 0,1 or 0,1,2,3,4,5,6,7" >&2; exit 1 ;;
esac
cd "$ROOT"
bash -n "$RUNNER"
mkdir -p "$ROOT/logs"

system_tmux=$(command -v tmux 2>/dev/null || true)
if [[ -z "$system_tmux" || ! -x "$system_tmux" ]]; then
  workspace_tmux="$ROOT/.tools/tmux-env/bin/tmux"
  if [[ -x "$workspace_tmux" ]]; then
    system_tmux=$workspace_tmux
  fi
fi
session=qwen_100k_v2_fixed_test
if [[ -n "$system_tmux" && -x "$system_tmux" ]] && "$system_tmux" -V >/dev/null 2>&1; then
  if "$system_tmux" has-session -t "$session" 2>/dev/null; then
    echo "ERROR: tmux session already exists: $session" >&2
    exit 1
  fi
  "$system_tmux" new-session -d -s "$session" "bash"
  "$system_tmux" set-option -t "$session" remain-on-exit on
  "$system_tmux" send-keys -t "$session" "cd '$ROOT' && CUDA_VISIBLE_DEVICES='$VISIBLE' bash '$RUNNER'" Enter
  echo "QWEN_100K_V2_FIXED_TEST_TMUX_STARTED session=$session"
  echo "Attach: $system_tmux attach -t $session"
else
  nohup env CUDA_VISIBLE_DEVICES="$VISIBLE" bash "$RUNNER" >> "$LAUNCHER_LOG" 2>&1 </dev/null &
  pid=$!
  sleep 2
  kill -0 "$pid" 2>/dev/null || {
    echo "ERROR: fixed-test launcher exited during startup; inspect $LAUNCHER_LOG" >&2
    exit 1
  }
  echo "$pid" > "$ROOT/logs/qwen_multidegradation_100k_rank64_per_type_v2_fixed_test.pid"
  echo "QWEN_100K_V2_FIXED_TEST_NOHUP_STARTED pid=$pid"
fi
echo "Monitor: tail -f $ROOT/logs/qwen_multidegradation_100k_rank64_per_type_v2_fixed_test.log"
