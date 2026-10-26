#!/usr/bin/env bash
set -euo pipefail
ROOT=/inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/zhenhaowang
MODE=${1:-qualification}
ARM=${2:-moe}
RUN_ID=${3:-${MODE}_v1}
[[ "$MODE" == qualification || "$MODE" == formal ]] || { echo 'ERROR: invalid mode'; exit 2; }
[[ "$ARM" == moe || "$ARM" == dense ]] || { echo 'ERROR: invalid arm'; exit 2; }
[[ "$RUN_ID" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'ERROR: invalid run ID'; exit 2; }
cd "$ROOT"
command -v flock >/dev/null || { echo 'ERROR: flock required'; exit 2; }
source smoke/tmux_launcher_lib.sh
TMUX_BIN_PATH=$(select_tmux_binary)
TMUX_SERVER_LABEL=zhenhaowang_jobs
SESSION="rain_blur_${ARM}_${RUN_ID}"
LOG="$ROOT/logs/foundir_rain_blur_image_moe_v1_${ARM}_${RUN_ID}.log"
[[ ! -e "$LOG" && ! -e "$ROOT/checkpoints/foundir_rain_blur_image_moe_v1/$ARM/$RUN_ID" ]] || { echo 'ERROR: existing run; inspect it'; exit 3; }
if tmux_job_server has-session -t "$SESSION" 2>/dev/null; then echo 'ERROR: existing session'; exit 3; fi
printf -v LAUNCH_COMMAND '%q ' env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash \
  "$ROOT/smoke/run_foundir_rain_blur_image_moe_v1_8gpu.sh" "$MODE" "$ARM" "$RUN_ID" \
  "${4:-$ROOT/checkpoints/foundir_rain_blur_image_moe_v1/$ARM/qualification_v1/result.json}"
tmux_job_server new-session -d -s "$SESSION" "$LAUNCH_COMMAND"
echo "RAIN_BLUR_MOE_TMUX_CREATED session=$SESSION"
echo "LOG=$LOG"
echo 'Check initial Rain parity and finite training steps; tmux creation alone is not training success.'
