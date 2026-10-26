#!/usr/bin/env bash
set -euo pipefail
ROOT=/inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/zhenhaowang
MODE=${1:-qualification}
RUN_ID=${2:-${MODE}_v1}
[[ "$MODE" == qualification || "$MODE" == formal ]] || { echo 'ERROR: invalid mode'; exit 2; }
[[ "$RUN_ID" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'ERROR: invalid run ID'; exit 2; }
cd "$ROOT"
command -v flock >/dev/null || { echo 'ERROR: flock required'; exit 2; }
source smoke/tmux_launcher_lib.sh
TMUX_BIN_PATH=$(select_tmux_binary)
TMUX_SERVER_LABEL=zhenhaowang_jobs
SESSION="blur_single_${RUN_ID}"
LOG="$ROOT/logs/foundir_blur_single_endpoint_v1_${RUN_ID}.log"
[[ ! -e "$LOG" && ! -e "$ROOT/checkpoints/foundir_blur_single_endpoint_v1/$RUN_ID" ]] || { echo 'ERROR: existing run; inspect it'; exit 3; }
if tmux_job_server has-session -t "$SESSION" 2>/dev/null; then echo 'ERROR: existing session'; exit 3; fi
printf -v LAUNCH_COMMAND '%q ' env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash \
  "$ROOT/smoke/run_foundir_blur_single_endpoint_v1_8gpu.sh" "$MODE" "$RUN_ID" \
  "${3:-$ROOT/checkpoints/foundir_blur_single_endpoint_v1/qualification_v1/result.json}" "${4:-}"
tmux_job_server new-session -d -s "$SESSION" "$LAUNCH_COMMAND"
echo "BLUR_SINGLE_TMUX_CREATED session=$SESSION"
echo "LOG=$LOG"
echo 'Check initial Rain/Blur parity and finite steps; tmux creation alone is not training success.'
