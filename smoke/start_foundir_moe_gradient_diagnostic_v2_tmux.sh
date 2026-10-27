#!/usr/bin/env bash
set -euo pipefail
ROOT=/inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/zhenhaowang
RUN_ID=${1:-diagnostic_v2}
[[ "$RUN_ID" =~ ^[A-Za-z0-9_-]+$ ]] || { echo 'ERROR: provide a fresh run ID'; exit 2; }
cd "$ROOT"
LOG="$ROOT/logs/foundir_dual_retry_v1_moe_${RUN_ID}.log"
[[ ! -e "$LOG" && ! -e "$ROOT/checkpoints/foundir_dual_retry_v1/moe/$RUN_ID" ]] || { echo 'ERROR: existing run; preserve it and use a fresh ID'; exit 3; }
source smoke/tmux_launcher_lib.sh
TMUX_BIN_PATH=$(select_tmux_binary)
TMUX_SERVER_LABEL=zhenhaowang_jobs
SESSION="moe_gradient_v2_${RUN_ID}"
if tmux_job_server has-session -t "$SESSION" 2>/dev/null; then echo 'ERROR: existing session'; exit 3; fi
printf -v LAUNCH_COMMAND '%q ' env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  bash "$ROOT/smoke/run_foundir_moe_gradient_diagnostic_v2_8gpu.sh" "$RUN_ID"
tmux_job_server new-session -d -s "$SESSION" "$LAUNCH_COMMAND"
echo "MOE_GRADIENT_V2_TMUX_CREATED session=$SESSION"
echo "LOG=$LOG"
echo 'Tmux creation is not GPU success. Confirm progress and final status=0.'
echo 'Use the idle MoE node; do not overlap with PPT on one node.'
