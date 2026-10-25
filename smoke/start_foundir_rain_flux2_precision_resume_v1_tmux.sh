#!/usr/bin/env bash
set -euo pipefail
ROOT=/inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/zhenhaowang
RUN_ID=${1:-continue_v1}
[[ "$RUN_ID" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'ERROR: invalid run ID'; exit 2; }
cd "$ROOT"
command -v flock >/dev/null || { echo 'ERROR: flock is required'; exit 2; }
source smoke/tmux_launcher_lib.sh
TMUX_BIN_PATH=$(select_tmux_binary)
TMUX_SERVER_LABEL=zhenhaowang_jobs
SESSION="rain_resume_${RUN_ID}"
LOG="$ROOT/logs/foundir_rain_flux2_precision_resume_v1_${RUN_ID}.log"
[[ ! -e "$LOG" && ! -e "$ROOT/checkpoints/foundir_rain_flux2_precision_resume_v1/$RUN_ID" ]] || { echo "ERROR: existing run $RUN_ID; inspect log"; exit 3; }
if tmux_job_server has-session -t "$SESSION" 2>/dev/null; then
  echo "ERROR: session already exists: $SESSION"; exit 3
fi
printf -v LAUNCH_COMMAND '%q ' env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash \
  "$ROOT/smoke/run_foundir_rain_flux2_precision_resume_v1_8gpu.sh" "$RUN_ID"
tmux_job_server new-session -d -s "$SESSION" "$LAUNCH_COMMAND"
echo "RAIN_RESUME_TMUX_CREATED session=$SESSION"
echo "LOG=$LOG"
echo 'Session creation does not prove training started; check for RAIN_RESUME_STATE_RESTORED and step 3001.'
