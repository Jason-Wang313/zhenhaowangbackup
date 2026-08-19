#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMUX="$ROOT/.tools/tmux-env/bin/tmux"
RUNNER="$ROOT/smoke/run_qwen_lowlight_supervised_overfit40_ddp8.sh"
SESSION="qwen_vlm_overfit40_ddp8"
LOG="$ROOT/logs/qwen_lowlight_supervised_overfit40_ddp8.log"

if [[ ! -x "$TMUX" ]]; then
  echo "ERROR: workspace-local tmux is unavailable: $TMUX" >&2
  exit 1
fi
if [[ ! -f "$RUNNER" ]]; then
  echo "ERROR: checked-in GPU runner is unavailable: $RUNNER" >&2
  exit 1
fi
"$TMUX" -V
bash -n "$RUNNER"

if [[ "${1:-}" == "--check" ]]; then
  echo "QWEN_SUPERVISED_DDP8_TMUX_PREFLIGHT_PASS session=$SESSION"
  echo "Runner: $RUNNER"
  echo "Persistent log: $LOG"
  exit 0
fi
if [[ "$#" -ne 0 ]]; then
  echo "ERROR: unexpected launcher argument: $1" >&2
  exit 1
fi

if "$TMUX" has-session -t "$SESSION" 2>/dev/null; then
  echo "ERROR: tmux session already exists: $SESSION" >&2
  echo "Attach with: $TMUX attach -t $SESSION" >&2
  exit 1
fi

"$TMUX" new-session -d -s "$SESSION" -c "$ROOT"
"$TMUX" set-window-option -t "$SESSION:0" remain-on-exit on
launch_command="CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash smoke/run_qwen_lowlight_supervised_overfit40_ddp8.sh"
"$TMUX" send-keys -t "$SESSION:0.0" -l "$launch_command"
"$TMUX" send-keys -t "$SESSION:0.0" Enter

echo "QWEN_SUPERVISED_DDP8_TMUX_LAUNCHED session=$SESSION"
echo "Attach: $TMUX attach -t $SESSION"
echo "Persistent log: tail -n 80 -f $LOG"
