#!/usr/bin/env bash
set -euo pipefail
ROOT=/inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/zhenhaowang
MODE=${1:-qualification}
ARM=${2:-image_only}
RUN_ID=${3:-qualification_v1}
[[ "$MODE" == qualification || "$MODE" == formal ]] || { echo 'ERROR: invalid mode'; exit 2; }
[[ "$ARM" == image_only || "$ARM" == vlm_assisted ]] || { echo 'ERROR: invalid arm'; exit 2; }
[[ "$RUN_ID" =~ ^[A-Za-z0-9_-]+$ ]] || { echo 'ERROR: invalid run ID'; exit 2; }
cd "$ROOT"
command -v flock >/dev/null || { echo 'ERROR: flock required'; exit 2; }
source smoke/tmux_launcher_lib.sh
TMUX_BIN_PATH=$(select_tmux_binary)
TMUX_SERVER_LABEL=zhenhaowang_jobs
SESSION="joint_moe_${ARM}_${RUN_ID}"
LOG="$ROOT/logs/foundir_rain_blur_joint_vlm_moe_v2_${ARM}_${RUN_ID}.log"
[[ ! -e "$LOG" && ! -e "$ROOT/checkpoints/foundir_rain_blur_joint_vlm_moe_v2/$ARM/$RUN_ID" ]] || { echo 'ERROR: existing run; inspect it'; exit 3; }
if tmux_job_server has-session -t "$SESSION" 2>/dev/null; then echo 'ERROR: existing session'; exit 3; fi
printf -v LAUNCH_COMMAND '%q ' env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash \
  "$ROOT/smoke/run_foundir_rain_blur_joint_vlm_moe_v2_8gpu.sh" "$MODE" "$ARM" "$RUN_ID" \
  "${4:-$ROOT/checkpoints/foundir_rain_blur_joint_vlm_moe_v2/image_only/qualification_v1/result.json}" \
  "${5:-$ROOT/checkpoints/foundir_rain_blur_joint_vlm_moe_v2/vlm_assisted/qualification_v1/result.json}" "${6:-}"
tmux_job_server new-session -d -s "$SESSION" "$LAUNCH_COMMAND"
echo "JOINT_MOE_TMUX_CREATED session=$SESSION"
echo "LOG=$LOG"
echo 'Session creation is not training success. Check JOINT_MOE_TRAIN and final engineering_pass.'
