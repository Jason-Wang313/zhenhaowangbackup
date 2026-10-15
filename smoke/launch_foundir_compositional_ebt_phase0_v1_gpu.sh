#!/usr/bin/env bash
set -euo pipefail

ROOT=/inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/zhenhaowang
RUN_ID=${EBT_PHASE0_RUN_ID:-gpu_smoke_v1}
SESSION="foundir_ebt_phase0_${RUN_ID//./_}"
source "$ROOT/smoke/tmux_launcher_lib.sh"
TMUX_BIN_PATH=$(select_tmux_binary)
TMUX_SERVER_LABEL="foundir-ebt-phase0-${RUN_ID//./-}"

"$TMUX_BIN_PATH" -V
if tmux_job_server has-session -t "$SESSION" 2>/dev/null; then
  echo "refusing duplicate tmux session: $SESSION" >&2
  exit 2
fi

tmux_job_server new-session -d -s "$SESSION" -c "$ROOT"
tmux_job_server set-option -t "$SESSION" remain-on-exit on
command="export EBT_PHASE0_RUN_ID='$RUN_ID'; export CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES:-0}'; bash smoke/run_foundir_compositional_ebt_phase0_v1_gpu.sh"
tmux_job_server send-keys -t "$SESSION" "$command" C-m
echo "launched tmux session: $SESSION"
printf 'attach: '
tmux_attach_hint "$SESSION"
printf '\n'
echo "log: tail -f $ROOT/logs/foundir_compositional_ebt_phase0_v1_${RUN_ID}.log"
