#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMUX="$ROOT/.tools/tmux-env/bin/tmux"
RUNNER="$ROOT/smoke/run_lowlight_vae_ceiling.sh"
SESSION="lowlight_vae_ceiling_v1"
LOG="$ROOT/logs/lowlight_vae_ceiling_v1.log"
OUTPUT="$ROOT/checkpoints/lowlight_vae_ceiling_v1"
DEVICE_SET="${CUDA_VISIBLE_DEVICES:-}"

if [[ "$DEVICE_SET" != "0,1" ]]; then
  echo "ERROR: set CUDA_VISIBLE_DEVICES exactly to 0,1" >&2
  exit 1
fi
if [[ ! -x "$TMUX" || ! -f "$RUNNER" ]]; then
  echo "ERROR: workspace-local tmux or VAE-ceiling runner is unavailable" >&2
  exit 1
fi
"$TMUX" -V
bash "$RUNNER" --check

if [[ "${1:-}" == "--check" ]]; then
  if [[ "$#" -ne 1 ]]; then
    echo "ERROR: --check takes no additional arguments" >&2
    exit 1
  fi
  echo "LOWLIGHT_VAE_CEILING_TMUX_CHECK_PASS session=$SESSION"
  exit 0
fi
if [[ "$#" -ne 0 ]]; then
  echo "ERROR: unexpected VAE-ceiling launcher argument: $1" >&2
  exit 1
fi
if "$TMUX" has-session -t "$SESSION" 2>/dev/null; then
  echo "LOWLIGHT_VAE_CEILING_SESSION_EXISTS session=$SESSION"
  exec "$TMUX" attach -t "$SESSION"
fi

"$TMUX" new-session -d -s "$SESSION" -c "$ROOT"
"$TMUX" set-window-option -t "$SESSION:0" remain-on-exit on
launch_command="CUDA_VISIBLE_DEVICES=$DEVICE_SET bash smoke/run_lowlight_vae_ceiling.sh"
"$TMUX" send-keys -t "$SESSION:0.0" -l "$launch_command"
"$TMUX" send-keys -t "$SESSION:0.0" Enter

echo "LOWLIGHT_VAE_CEILING_TMUX_LAUNCHED session=$SESSION"
echo "Detach safely with Ctrl-b then d."
echo "Reattach: $TMUX attach -t $SESSION"
echo "Persistent log: tail -n 100 -f $LOG"
echo "Immutable output: $OUTPUT"
exec "$TMUX" attach -t "$SESSION"
