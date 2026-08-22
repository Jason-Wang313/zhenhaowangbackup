#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMUX_HELPERS="$ROOT/smoke/tmux_launcher_lib.sh"
RUNNER="$ROOT/smoke/run_qwen_lowlight_supervised_overfit40_ddp8.sh"
SESSION="qwen_vlm_overfit40_ddp8"
TMUX_SERVER_LABEL="zhenhaowang_jobs"
LOG="$ROOT/logs/qwen_lowlight_supervised_overfit40_ddp8.log"

if [[ ! -f "$TMUX_HELPERS" ]]; then
  echo "ERROR: tmux launcher helpers are unavailable: $TMUX_HELPERS" >&2
  exit 1
fi
if [[ ! -f "$RUNNER" ]]; then
  echo "ERROR: checked-in GPU runner is unavailable: $RUNNER" >&2
  exit 1
fi
source "$TMUX_HELPERS"
TMUX_BIN_PATH="$(select_tmux_binary)"
tmux_job_server -V
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

if tmux_job_server has-session -t "$SESSION" 2>/dev/null; then
  echo "ERROR: tmux session already exists: $SESSION" >&2
  printf 'Attach from a separate terminal: ' >&2
  tmux_attach_hint "$SESSION" >&2
  printf '\n' >&2
  exit 1
fi

tmux_job_server new-session -d -s "$SESSION" -c "$ROOT"
tmux_job_server set-window-option -t "$SESSION:0" remain-on-exit on
launch_command="CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash smoke/run_qwen_lowlight_supervised_overfit40_ddp8.sh"
tmux_job_server send-keys -t "$SESSION:0.0" -l "$launch_command"
tmux_job_server send-keys -t "$SESSION:0.0" Enter

echo "QWEN_SUPERVISED_DDP8_TMUX_LAUNCHED session=$SESSION"
printf 'Attach from a separate terminal: '
tmux_attach_hint "$SESSION"
printf '\n'
echo "Persistent log: tail -n 80 -f $LOG"
