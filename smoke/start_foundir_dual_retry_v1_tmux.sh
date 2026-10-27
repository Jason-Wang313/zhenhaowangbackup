#!/usr/bin/env bash
set -euo pipefail
ROOT=/inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/zhenhaowang
BRANCH=${1:-}
RUN_ID=${2:-}
[[ "$BRANCH" == moe || "$BRANCH" == ppt ]] || { echo 'ERROR: branch must be moe or ppt'; exit 2; }
[[ "$RUN_ID" =~ ^[A-Za-z0-9_-]+$ ]] || { echo 'ERROR: provide a fresh run ID'; exit 2; }
cd "$ROOT"
LOG="$ROOT/logs/foundir_dual_retry_v1_${BRANCH}_${RUN_ID}.log"
[[ ! -e "$LOG" && ! -e "$ROOT/checkpoints/foundir_dual_retry_v1/$BRANCH/$RUN_ID" ]] || { echo 'ERROR: existing run; inspect it'; exit 3; }
source smoke/tmux_launcher_lib.sh
TMUX_BIN_PATH=$(select_tmux_binary)
TMUX_SERVER_LABEL=zhenhaowang_jobs
SESSION="dual_retry_${BRANCH}_${RUN_ID}"
if tmux_job_server has-session -t "$SESSION" 2>/dev/null; then echo 'ERROR: existing session'; exit 3; fi
printf -v LAUNCH_COMMAND '%q ' env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  bash "$ROOT/smoke/run_foundir_dual_retry_v1_8gpu.sh" "$BRANCH" "$RUN_ID"
tmux_job_server new-session -d -s "$SESSION" "$LAUNCH_COMMAND"
echo "DUAL_RETRY_TMUX_CREATED session=$SESSION"
echo "LOG=$LOG"
echo 'Tmux creation is not GPU success. Check COMM_PASS, progress markers and final status=0.'
echo 'Run the other branch on another idle 8-GPU node, or after this job finishes. Never overlap on one node.'
