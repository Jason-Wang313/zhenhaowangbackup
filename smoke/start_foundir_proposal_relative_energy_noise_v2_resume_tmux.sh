#!/usr/bin/env bash
set -euo pipefail

ROOT=/inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/zhenhaowang
HELPERS="$ROOT/smoke/tmux_launcher_lib.sh"
RUNNER="$ROOT/smoke/run_foundir_proposal_relative_energy_noise_v2_8gpu.sh"
CHECKPOINT="$ROOT/checkpoints/foundir_proposal_relative_energy_noise_v2/noise_energy_v2_formal1/step_001706.pt"
TERMINAL="$ROOT/checkpoints/foundir_proposal_relative_energy_noise_v2/noise_energy_v2_formal1/terminal_result.json"
SESSION=foundir_proposal_energy_v2_formal1
TMUX_SERVER_LABEL=zhenhaowang_jobs

cd "$ROOT"
[[ -f "$HELPERS" ]] || { echo "ERROR: missing tmux helpers: $HELPERS" >&2; exit 2; }
[[ -f "$RUNNER" ]] || { echo "ERROR: missing V2 runner: $RUNNER" >&2; exit 2; }
[[ -f "$CHECKPOINT" ]] || { echo "ERROR: missing step-1706 resume checkpoint: $CHECKPOINT" >&2; exit 2; }
[[ ! -f "$TERMINAL" ]] || { echo "ERROR: formal1 already has a terminal result; refusing resume" >&2; exit 2; }

source "$HELPERS"
TMUX_BIN_PATH="$(select_tmux_binary)"
tmux_job_server -V

if pgrep -af '[t]rain_foundir_proposal_relative_energy_noise_v2_ddp.py' >/dev/null; then
  echo "ERROR: an existing V2 training process is visible; refusing duplicate launch" >&2
  pgrep -af '[t]rain_foundir_proposal_relative_energy_noise_v2_ddp.py' >&2 || true
  exit 2
fi

if tmux_job_server has-session -t "$SESSION" 2>/dev/null; then
  echo "V2_TMUX_SESSION_ALREADY_EXISTS session=$SESSION"
  echo "Attaching without launching a duplicate process."
  tmux_attach_or_report "$SESSION"
  exit 0
fi

tmux_job_server new-session -d -s "$SESSION" -c "$ROOT"
tmux_job_server set-window-option -t "$SESSION:0" remain-on-exit on
launch_command="PROPOSAL_ENERGY_V2_RUN_ID=noise_energy_v2_formal1 PROPOSAL_ENERGY_V2_TOTAL_STEPS=8530 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash smoke/run_foundir_proposal_relative_energy_noise_v2_8gpu.sh"
tmux_job_server send-keys -t "$SESSION:0.0" -l "$launch_command"
tmux_job_server send-keys -t "$SESSION:0.0" Enter

echo "V2_TMUX_RESUME_LAUNCHED session=$SESSION checkpoint=step_001706"
printf 'Reattach from another terminal: '
tmux_attach_hint "$SESSION"
printf '\n'
echo "Detach safely with: Ctrl-b then d"
echo "Persistent log: $ROOT/logs/foundir_proposal_relative_energy_noise_v2_noise_energy_v2_formal1.log"
tmux_attach_or_report "$SESSION"
