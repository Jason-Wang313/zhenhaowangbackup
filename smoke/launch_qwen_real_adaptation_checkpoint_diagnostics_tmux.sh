#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMUX="$ROOT/.tools/tmux-env/bin/tmux"
RUNNER="$ROOT/smoke/run_qwen_real_adaptation_checkpoint_diagnostics.sh"
SESSION="qwen_real_adapt_checkpoint_diagnostics"
LOG="$ROOT/logs/qwen_lowlight_real_adaptation_checkpoint_diagnostics_20260821.log"
OUTPUT_BASE="$ROOT/checkpoints/qwen_lowlight_real_adaptation_checkpoint_diagnostics_20260821"
DEVICE_SET="${CUDA_VISIBLE_DEVICES:-}"

if [[ "$DEVICE_SET" != "0,1" && "$DEVICE_SET" != "0,1,2,3,4,5,6,7" ]]; then
  echo "ERROR: set CUDA_VISIBLE_DEVICES exactly to 0,1 or 0,1,2,3,4,5,6,7" >&2
  exit 1
fi
if [[ ! -x "$TMUX" || ! -f "$RUNNER" ]]; then
  echo "ERROR: workspace-local tmux or diagnostic runner is unavailable" >&2
  exit 1
fi
"$TMUX" -V
bash "$RUNNER" --check

if [[ "${1:-}" == "--check" ]]; then
  if [[ "$#" -ne 1 ]]; then
    echo "ERROR: --check takes no additional arguments" >&2
    exit 1
  fi
  echo "QWEN_REAL_ADAPTATION_DIAGNOSTIC_TMUX_CHECK_PASS session=$SESSION"
  exit 0
fi
if [[ "$#" -ne 0 ]]; then
  echo "ERROR: unexpected diagnostic launcher argument: $1" >&2
  exit 1
fi
if "$TMUX" has-session -t "$SESSION" 2>/dev/null; then
  echo "QWEN_REAL_ADAPTATION_DIAGNOSTIC_SESSION_EXISTS session=$SESSION"
  exec "$TMUX" attach -t "$SESSION"
fi

"$TMUX" new-session -d -s "$SESSION" -c "$ROOT"
"$TMUX" set-window-option -t "$SESSION:0" remain-on-exit on
launch_command="CUDA_VISIBLE_DEVICES=$DEVICE_SET bash smoke/run_qwen_real_adaptation_checkpoint_diagnostics.sh"
"$TMUX" send-keys -t "$SESSION:0.0" -l "$launch_command"
"$TMUX" send-keys -t "$SESSION:0.0" Enter

echo "QWEN_REAL_ADAPTATION_DIAGNOSTIC_TMUX_LAUNCHED session=$SESSION"
echo "Detach safely with Ctrl-b then d."
echo "Reattach: $TMUX attach -t $SESSION"
echo "Persistent log: tail -n 100 -f $LOG"
echo "Immutable attempt outputs: $OUTPUT_BASE"
exec "$TMUX" attach -t "$SESSION"
