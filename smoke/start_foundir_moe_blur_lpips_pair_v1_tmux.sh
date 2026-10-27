#!/usr/bin/env bash
set -euo pipefail
ROOT=/inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/zhenhaowang
RUN_ID=${1:-pair_v1}
[[ "$RUN_ID" =~ ^[A-Za-z0-9_-]+$ ]] || { echo 'ERROR: invalid fresh pair ID'; exit 2; }
cd "$ROOT"
command -v flock >/dev/null
# Report a busy node in the user's terminal before creating a short-lived
# tmux session. The actual job rechecks and holds both locks for the pair.
for LOCK_FILE in /tmp/zhenhaowang_dual_retry_v1.lock /tmp/zhenhaowang_rain_blur_image_moe_v1.lock; do
  flock -n "$LOCK_FILE" true || { echo 'ERROR: this node is running a PPT/MoE/Rain job; wait for it to finish'; exit 3; }
done
LOG="$ROOT/logs/foundir_moe_blur_lpips_pair_v1_${RUN_ID}.log"
[[ ! -e "$LOG" && ! -e "$ROOT/checkpoints/foundir_moe_blur_lpips_continuation_v1/$RUN_ID" ]] || { echo 'ERROR: existing pair; preserve it'; exit 3; }
source smoke/tmux_launcher_lib.sh
TMUX_BIN_PATH=$(select_tmux_binary)
TMUX_SERVER_LABEL=zhenhaowang_jobs
SESSION="moe_lpips_${RUN_ID}"
if tmux_job_server has-session -t "$SESSION" 2>/dev/null; then echo 'ERROR: existing session'; exit 3; fi
printf -v LAUNCH_COMMAND '%q ' env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash \
  "$ROOT/smoke/run_foundir_moe_blur_lpips_pair_v1_8gpu.sh" "$RUN_ID"
tmux_job_server new-session -d -s "$SESSION" "$LAUNCH_COMMAND"
echo "MOE_LPIPS_TMUX_CREATED session=$SESSION"
echo "LOG=$LOG"
echo 'One node, sequential control then blur_lpips4x, 8 GPUs each. Session creation is not training success.'
