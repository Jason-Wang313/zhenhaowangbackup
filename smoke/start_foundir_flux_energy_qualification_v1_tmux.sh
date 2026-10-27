#!/usr/bin/env bash
set -euo pipefail
ROOT=/inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/zhenhaowang
RUN_ID=${1:-qualification_v1}
[[ "$RUN_ID" =~ ^[A-Za-z0-9_-]+$ ]] || { echo 'ERROR: invalid fresh run ID'; exit 2; }
cd "$ROOT"
[[ ! -e "logs/foundir_flux_energy_qualification_v1_${RUN_ID}.log" && ! -e "checkpoints/foundir_flux_energy_qualification_v1/$RUN_ID" ]] || { echo 'ERROR: use a fresh run ID'; exit 3; }
source smoke/tmux_launcher_lib.sh
TMUX_BIN_PATH=$(select_tmux_binary)
TMUX_SERVER_LABEL=zhenhaowang_jobs
SESSION="flux_energy_${RUN_ID}"
if tmux_job_server has-session -t "$SESSION" 2>/dev/null; then echo 'ERROR: session already exists'; exit 3; fi
printf -v LAUNCH_COMMAND '%q ' bash "$ROOT/smoke/run_foundir_flux_energy_qualification_v1_8gpu.sh" "$RUN_ID"
tmux_job_server new-session -d -s "$SESSION" "$LAUNCH_COMMAND"
echo "FLUX_ENERGY_TMUX_CREATED session=$SESSION (creation is not proof of training)"
echo "LOG=$ROOT/logs/foundir_flux_energy_qualification_v1_${RUN_ID}.log"
