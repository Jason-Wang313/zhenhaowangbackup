#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/flux_tail_v3_launcher_lib.sh"
configure_tail_v3 "$@"
printf -v LAUNCH_COMMAND '%q ' env -u PYTHONHOME -u NCCL_SOCKET_IFNAME -u GLOO_SOCKET_IFNAME \
  PYTHONNOUSERSITE=1 FLUX_TAIL_V3_DRY_RUN=0 \
  "FLUX_TAIL_V3_PYTHON=$PYTHON" "CUDA_VISIBLE_DEVICES=$VISIBLE" \
  "NNODES=$NNODES" "NODE_RANK=$NODE_RANK" "MASTER_ADDR=$MASTER_ADDR" "MASTER_PORT=$MASTER_PORT" \
  "${TRANSPORT[@]}" \
  bash "$ROOT/smoke/run_foundir_flux_tail_relative_energy_v3_gpu.sh" "$MODE" "$RUN_ID" "$GPU_COUNT" "$QUALIFICATION_RUN_ID"
if [[ "${FLUX_TAIL_V3_DRY_RUN:-0}" == 1 ]]; then
  printf '%s\n' "$LAUNCH_COMMAND"; exit 0
fi
cd "$ROOT"
source smoke/tmux_launcher_lib.sh
TMUX_BIN_PATH=$(select_tmux_binary)
TMUX_SERVER_LABEL=zhenhaowang_jobs
SESSION="flux_tail_relative_v3_${RUN_ID}_node${NODE_RANK}"
if tmux_job_server has-session -t "$SESSION" 2>/dev/null; then
  echo "TAIL_RELATIVE_V3_ALREADY_RUNNING session=$SESSION"; exit 0
fi
tmux_job_server new-session -d -s "$SESSION" "$LAUNCH_COMMAND"
echo "TAIL_RELATIVE_V3_TMUX_CREATED session=$SESSION node_rank=$NODE_RANK world=$GPU_COUNT; run the matching command on all $NNODES nodes"
echo "LOG=$LOG"
echo 'Same topology/device/runtime mapping required on resume; no auto trainfit/formal.'
