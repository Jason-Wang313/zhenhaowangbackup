#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMUX_HELPERS="$ROOT/smoke/tmux_launcher_lib.sh"
RUNNER="$ROOT/smoke/run_qwen_lowlight_supervised_scaling_1_2_4_8_json_retry.sh"
SESSION="qwen_vlm_scaling_json_retry"
TMUX_SERVER_LABEL="zhenhaowang_jobs"
LOG="$ROOT/logs/qwen_lowlight_supervised_scaling_1_2_4_8.log"
GPU_LOG="$ROOT/logs/qwen_lowlight_supervised_scaling_1_2_4_8_gpu.csv"
OUTPUT="$ROOT/checkpoints/qwen_lowlight_supervised_scaling_1_2_4_8"

if [[ ! -f "$TMUX_HELPERS" ]]; then
  echo "ERROR: tmux launcher helpers are unavailable: $TMUX_HELPERS" >&2
  exit 1
fi
if [[ ! -x "$RUNNER" ]]; then
  echo "ERROR: checked-in GPU retry runner is unavailable: $RUNNER" >&2
  exit 1
fi
source "$TMUX_HELPERS"
TMUX_BIN_PATH="$(select_tmux_binary)"
tmux_job_server -V
"$RUNNER" --check

if [[ "${1:-}" == "--check" ]]; then
  if [[ "$#" -ne 1 ]]; then
    echo "ERROR: --check takes no additional arguments" >&2
    exit 1
  fi
  echo "QWEN_SCALING_JSON_RETRY_TMUX_PREFLIGHT_PASS session=$SESSION"
  echo "Persistent log: $LOG"
  echo "GPU telemetry: $GPU_LOG"
  echo "Output: $OUTPUT"
  exit 0
fi
if [[ "$#" -ne 0 ]]; then
  echo "ERROR: unexpected retry-launcher argument: $1" >&2
  exit 1
fi

if tmux_job_server has-session -t "$SESSION" 2>/dev/null; then
  echo "ERROR: tmux retry session already exists: $SESSION" >&2
  printf 'Attach from a separate terminal: ' >&2
  tmux_attach_hint "$SESSION" >&2
  printf '\n' >&2
  exit 1
fi

tmux_job_server new-session -d -s "$SESSION" -c "$ROOT"
tmux_job_server set-window-option -t "$SESSION:0" remain-on-exit on
launch_command="CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash smoke/run_qwen_lowlight_supervised_scaling_1_2_4_8_json_retry.sh"
tmux_job_server send-keys -t "$SESSION:0.0" -l "$launch_command"
tmux_job_server send-keys -t "$SESSION:0.0" Enter

echo "QWEN_SCALING_JSON_RETRY_TMUX_LAUNCHED session=$SESSION"
printf 'Attach from a separate terminal: '
tmux_attach_hint "$SESSION"
printf '\n'
echo "Persistent log: tail -n 100 -f $LOG"
echo "GPU telemetry: tail -n 20 -f $GPU_LOG"
