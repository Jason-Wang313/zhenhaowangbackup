#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMUX="$ROOT/.tools/tmux-env/bin/tmux"
RUNNER="$ROOT/smoke/run_qwen_lowlight_real_validation_constrained.sh"
SESSION="qwen_vlm_real_constrained"
LOG="$ROOT/logs/qwen_lowlight_real_validation_constrained_step600.log"
GPU_LOG="$ROOT/logs/qwen_lowlight_real_validation_constrained_step600_gpu.csv"
OUTPUT="$ROOT/checkpoints/qwen_lowlight_real_validation_constrained_step600"

if [[ ! -x "$TMUX" ]]; then
  echo "ERROR: workspace-local tmux is unavailable: $TMUX" >&2
  exit 1
fi
if [[ ! -f "$RUNNER" ]]; then
  echo "ERROR: checked-in constrained-diagnostic runner is unavailable: $RUNNER" >&2
  exit 1
fi
"$TMUX" -V
bash "$RUNNER" --check

if [[ "${1:-}" == "--check" ]]; then
  if [[ "$#" -ne 1 ]]; then
    echo "ERROR: --check takes no additional arguments" >&2
    exit 1
  fi
  echo "QWEN_REAL_CONSTRAINED_TMUX_PREFLIGHT_PASS session=$SESSION"
  echo "Persistent log: $LOG"
  echo "GPU telemetry: $GPU_LOG"
  echo "Output: $OUTPUT"
  exit 0
fi
if [[ "$#" -ne 0 ]]; then
  echo "ERROR: unexpected constrained-diagnostic launcher argument: $1" >&2
  exit 1
fi

if "$TMUX" has-session -t "$SESSION" 2>/dev/null; then
  echo "QWEN_REAL_CONSTRAINED_TMUX_SESSION_ALREADY_EXISTS session=$SESSION"
  echo "Attaching without launching a duplicate process."
  exec "$TMUX" attach -t "$SESSION"
fi

"$TMUX" new-session -d -s "$SESSION" -c "$ROOT"
"$TMUX" set-window-option -t "$SESSION:0" remain-on-exit on
launch_command="CUDA_VISIBLE_DEVICES=0 bash smoke/run_qwen_lowlight_real_validation_constrained.sh"
"$TMUX" send-keys -t "$SESSION:0.0" -l "$launch_command"
"$TMUX" send-keys -t "$SESSION:0.0" Enter

echo "QWEN_REAL_CONSTRAINED_TMUX_LAUNCHED session=$SESSION"
echo "The launcher is attaching now; detach safely with Ctrl-b then d."
echo "Reattach: $TMUX attach -t $SESSION"
echo "Persistent log: tail -n 100 -f $LOG"
echo "GPU telemetry: tail -n 20 -f $GPU_LOG"
exec "$TMUX" attach -t "$SESSION"
