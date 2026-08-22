#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMUX_HELPERS="$ROOT/smoke/tmux_launcher_lib.sh"
RUNNER="$ROOT/smoke/run_qwen_lowlight_real_exposure_ordering.sh"
SESSION="qwen_real_exposure_ordering"
TMUX_SERVER_LABEL="zhenhaowang_jobs"
LOG="$ROOT/logs/qwen_lowlight_real_exposure_ordering_step600.log"
OUTPUT="$ROOT/checkpoints/qwen_lowlight_real_exposure_ordering_step600"
DEVICE_SET="${CUDA_VISIBLE_DEVICES:-}"

if [[ "$DEVICE_SET" != "0,1" ]]; then
  echo "ERROR: set CUDA_VISIBLE_DEVICES exactly to 0,1" >&2
  exit 1
fi
if [[ ! -f "$TMUX_HELPERS" || ! -f "$RUNNER" ]]; then
  echo "ERROR: tmux helpers or exposure-ordering runner are unavailable" >&2
  exit 1
fi
source "$TMUX_HELPERS"
TMUX_BIN_PATH="$(select_tmux_binary)"
tmux_job_server -V
bash "$RUNNER" --check

if [[ "${1:-}" == "--check" ]]; then
  if [[ "$#" -ne 1 ]]; then
    echo "ERROR: --check takes no additional arguments" >&2
    exit 1
  fi
  echo "QWEN_REAL_EXPOSURE_ORDERING_TMUX_CHECK_PASS session=$SESSION"
  exit 0
fi
if [[ "$#" -ne 0 ]]; then
  echo "ERROR: unexpected exposure-ordering launcher argument: $1" >&2
  exit 1
fi
if tmux_job_server has-session -t "$SESSION" 2>/dev/null; then
  echo "QWEN_REAL_EXPOSURE_ORDERING_SESSION_EXISTS session=$SESSION"
  tmux_attach_or_report "$SESSION"
  exit 0
fi

tmux_job_server new-session -d -s "$SESSION" -c "$ROOT"
tmux_job_server set-window-option -t "$SESSION:0" remain-on-exit on
launch_command="CUDA_VISIBLE_DEVICES=$DEVICE_SET bash smoke/run_qwen_lowlight_real_exposure_ordering.sh"
tmux_job_server send-keys -t "$SESSION:0.0" -l "$launch_command"
tmux_job_server send-keys -t "$SESSION:0.0" Enter

echo "QWEN_REAL_EXPOSURE_ORDERING_TMUX_LAUNCHED session=$SESSION"
printf 'Reattach from a separate terminal: '
tmux_attach_hint "$SESSION"
printf '\n'
echo "Persistent log: tail -n 100 -f $LOG"
echo "Immutable output: $OUTPUT"
tmux_attach_or_report "$SESSION"
