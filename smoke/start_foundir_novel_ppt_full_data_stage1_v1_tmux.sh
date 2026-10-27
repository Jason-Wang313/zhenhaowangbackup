#!/usr/bin/env bash
set -euo pipefail
ROOT=/inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/zhenhaowang
RUN_ID=${1:-stage1_v1}
RESUME_RECEIPT=${2:-}
[[ "$RUN_ID" =~ ^[A-Za-z0-9_-]+$ ]] || { echo 'ERROR: invalid fresh run ID'; exit 2; }
if [[ -n "$RESUME_RECEIPT" && "${PPT_PREVIOUS_RUN_STOPPED:-}" != 1 ]]; then
  echo 'ERROR: confirm previous PPT job stopped before resuming'; exit 2
fi
cd "$ROOT"
command -v flock >/dev/null
for LOCK_FILE in /tmp/zhenhaowang_dual_retry_v1.lock /tmp/zhenhaowang_rain_blur_image_moe_v1.lock; do
  flock -n "$LOCK_FILE" true || { echo 'ERROR: this node is busy; use node 2'; exit 3; }
done
LOG="$ROOT/logs/foundir_novel_ppt_full_data_stage1_v1_${RUN_ID}.log"
[[ ! -e "$LOG" && ! -e "$ROOT/checkpoints/foundir_novel_ppt_full_data_stage1_v1/$RUN_ID" ]] || { echo 'ERROR: preserve existing run ID'; exit 3; }
source smoke/tmux_launcher_lib.sh
TMUX_BIN_PATH=$(select_tmux_binary)
TMUX_SERVER_LABEL=zhenhaowang_jobs
SESSION="novel_ppt_full_${RUN_ID}"
if tmux_job_server has-session -t "$SESSION" 2>/dev/null; then echo 'ERROR: existing session'; exit 3; fi
printf -v LAUNCH_COMMAND '%q ' env PPT_PREVIOUS_RUN_STOPPED="${PPT_PREVIOUS_RUN_STOPPED:-0}" \
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash \
  "$ROOT/smoke/run_foundir_novel_ppt_full_data_stage1_v1_8gpu.sh" "$RUN_ID" "$RESUME_RECEIPT"
tmux_job_server new-session -d -s "$SESSION" "$LAUNCH_COMMAND"
echo "PPT_FULL_DATA_TMUX_CREATED session=$SESSION"
echo "LOG=$LOG"
echo 'Full train pool, stop at 1706; fresh energy unless an explicit stage1 receipt is supplied. Confirm TRAIN progress in log.'
