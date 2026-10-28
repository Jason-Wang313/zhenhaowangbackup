#!/usr/bin/env bash
set -euo pipefail
ROOT=/inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/zhenhaowang
RUN_ID=${1:-mechanism_v1}
GPU_COUNT=${2:-8}
case "$GPU_COUNT" in
  4) DEFAULT_VISIBLE=0,1,2,3 ;;
  8) DEFAULT_VISIBLE=0,1,2,3,4,5,6,7 ;;
  *) echo 'ERROR: GPU count must be 4 or 8'; exit 2 ;;
esac
[[ "$RUN_ID" =~ ^[A-Za-z0-9_-]+$ ]] || { echo 'ERROR: invalid fresh run ID'; exit 2; }
cd "$ROOT"
[[ ! -e "logs/foundir_flux_tail_oracle_v1_${RUN_ID}.log" && ! -e "checkpoints/foundir_flux_tail_oracle_v1/$RUN_ID" ]] || { echo 'ERROR: use a fresh run ID'; exit 3; }
source smoke/tmux_launcher_lib.sh
TMUX_BIN_PATH=$(select_tmux_binary)
TMUX_SERVER_LABEL=zhenhaowang_jobs
SESSION="flux_tail_${RUN_ID}"
if tmux_job_server has-session -t "$SESSION" 2>/dev/null; then echo 'ERROR: session already exists'; exit 3; fi
printf -v LAUNCH_COMMAND '%q ' env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$DEFAULT_VISIBLE}" bash "$ROOT/smoke/run_foundir_flux_tail_oracle_v1_gpu.sh" "$RUN_ID" "$GPU_COUNT"
tmux_job_server new-session -d -s "$SESSION" "$LAUNCH_COMMAND"
echo "TAIL_ORACLE_TMUX_CREATED session=$SESSION gpus=$GPU_COUNT (not proof of GPU progress)"
echo "LOG=$ROOT/logs/foundir_flux_tail_oracle_v1_${RUN_ID}.log"
