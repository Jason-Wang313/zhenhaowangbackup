#!/usr/bin/env bash
set -euo pipefail
ROOT=/inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/zhenhaowang
RUN_ID=${1:-qualification_8gpu_v2}
GPU_COUNT=${2:-8}
[[ "$RUN_ID" =~ ^[A-Za-z0-9_-]+$ ]] || exit 2
[[ "$GPU_COUNT" == 4 || "$GPU_COUNT" == 8 ]] || exit 2
cd "$ROOT"
source smoke/tmux_launcher_lib.sh
TMUX_BIN_PATH=$(select_tmux_binary)
TMUX_SERVER_LABEL=zhenhaowang_jobs
SESSION="flux_tail_energy_v2_${RUN_ID}"
if tmux_job_server has-session -t "$SESSION" 2>/dev/null; then
  echo "TAIL_ENERGY_V2_ALREADY_RUNNING session=$SESSION; not launching twice"
  exit 0
fi
printf -v LAUNCH_COMMAND '%q ' bash "$ROOT/smoke/run_foundir_flux_tail_energy_resumable_v2_gpu.sh" "$RUN_ID" "$GPU_COUNT"
tmux_job_server new-session -d -s "$SESSION" "$LAUNCH_COMMAND"
echo "TAIL_ENERGY_V2_TMUX_CREATED session=$SESSION gpus=$GPU_COUNT (verify log for actual progress)"
echo "LOG=$ROOT/logs/foundir_flux_tail_energy_resumable_v2_${RUN_ID}.log"
echo 'Same command resumes the newest committed step; cached images are never regenerated.'
