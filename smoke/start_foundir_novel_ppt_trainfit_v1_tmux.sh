#!/usr/bin/env bash
set -euo pipefail
ROOT=/inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/zhenhaowang
RUN_ID=${1:-trainfit_v1}
[[ "$RUN_ID" =~ ^trainfit_[A-Za-z0-9_-]+$ ]] || { echo 'ERROR: provide fresh trainfit_ run ID'; exit 2; }
cd "$ROOT"
LOG="$ROOT/logs/foundir_dual_retry_v1_ppt_${RUN_ID}.log"
[[ ! -e "$LOG" && ! -e "$ROOT/checkpoints/foundir_dual_retry_v1/ppt/$RUN_ID" ]] || { echo 'ERROR: existing run; preserve it'; exit 3; }
source smoke/tmux_launcher_lib.sh
TMUX_BIN_PATH=$(select_tmux_binary)
TMUX_SERVER_LABEL=zhenhaowang_jobs
SESSION="novel_ppt_${RUN_ID}"
if tmux_job_server has-session -t "$SESSION" 2>/dev/null; then echo 'ERROR: existing session'; exit 3; fi
printf -v LAUNCH_COMMAND '%q ' env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  bash "$ROOT/smoke/run_foundir_novel_ppt_trainfit_v1_8gpu.sh" "$RUN_ID"
tmux_job_server new-session -d -s "$SESSION" "$LAUNCH_COMMAND"
echo "PPT_TRAINFIT_TMUX_CREATED session=$SESSION"
echo "LOG=$LOG"
echo 'Train-only diagnostic, not formal training. Confirm progress and final status=0.'
