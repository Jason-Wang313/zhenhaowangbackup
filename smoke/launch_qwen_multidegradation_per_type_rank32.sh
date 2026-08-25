#!/usr/bin/env bash
set -euo pipefail

ROOT=/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
LOG="$ROOT/logs/qwen_multidegradation_per_type_rank32_v1_launcher.log"
RUNNER="$ROOT/smoke/run_qwen_multidegradation_per_type_rank32.sh"

cd "$ROOT"
visible=${CUDA_VISIBLE_DEVICES:-}
case "$visible" in
  0,1) nproc=2 ;;
  0,1,2,3) nproc=4 ;;
  *) echo "ERROR: set CUDA_VISIBLE_DEVICES exactly to 0,1 or 0,1,2,3" >&2; exit 1 ;;
esac
bash -n "$RUNNER"
mkdir -p "$ROOT/logs"

system_tmux=$(command -v tmux 2>/dev/null || true)
if [[ -n "$system_tmux" && -x "$system_tmux" ]] && "$system_tmux" -V >/dev/null 2>&1; then
  session=qwen_multidegradation_per_type_rank32_v1
  if "$system_tmux" has-session -t "$session" 2>/dev/null; then
    echo "QWEN_PER_TYPE_RANK32_TMUX_ALREADY_EXISTS session=$session"
    echo "Attach: $system_tmux attach -t $session"
  else
    "$system_tmux" new-session -d -s "$session" "bash"
    "$system_tmux" set-option -t "$session" remain-on-exit on
    "$system_tmux" send-keys -t "$session" "cd '$ROOT' && CUDA_VISIBLE_DEVICES='$visible' bash '$RUNNER'" Enter
    echo "QWEN_PER_TYPE_RANK32_TMUX_STARTED session=$session"
    echo "Attach: $system_tmux attach -t $session"
  fi
else
  nohup env CUDA_VISIBLE_DEVICES="$visible" bash "$RUNNER" >> "$LOG" 2>&1 </dev/null &
  pid=$!
  echo "$pid" > "$ROOT/logs/qwen_multidegradation_per_type_rank32_v1.pid"
  echo "QWEN_PER_TYPE_RANK32_DETACHED_NOHUP_STARTED pid=$pid"
fi
echo "QWEN_PER_TYPE_RANK32_LAUNCH_WORLD_SIZE $nproc"
echo "Monitor: tail -f $ROOT/logs/qwen_multidegradation_per_type_rank32_v1.log"
