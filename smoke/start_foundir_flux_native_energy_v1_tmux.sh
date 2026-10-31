#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/flux_native_energy_launcher_lib.sh"
configure_native_energy "$@"
cd "$ROOT"
# Explicitly forward topology and transport through any pre-existing tmux server.
printf -v COMMAND '%q ' env "CUDA_VISIBLE_DEVICES=$VISIBLE" "FLUX_NATIVE_PYTHON=$PYTHON" \
  "NNODES=$NNODES" "NODE_RANK=$NODE_RANK" "MASTER_ADDR=$MASTER_ADDR" "MASTER_PORT=$MASTER_PORT" \
  "NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-}" "GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-}" \
  "${TRANSPORT[@]}" bash "$ROOT/smoke/run_foundir_flux_native_energy_v1_gpu.sh" "$RUN_ID" "$GPUS" "$RESUME"
if [[ "${FLUX_NATIVE_DRY_RUN:-0}" == 1 ]]; then printf '%s\n' "$COMMAND"; exit 0; fi
source smoke/tmux_launcher_lib.sh
TMUX_BIN_PATH=$(select_tmux_binary)
TMUX_SERVER_LABEL=zhenhaowang_jobs
SESSION="flux_native_energy_${RUN_ID}_node${NODE_RANK}"
if tmux_job_server has-session -t "$SESSION" 2>/dev/null; then
  echo "NATIVE_ENERGY_SESSION_EXISTS session=$SESSION; not a new launch"; exit 0
fi
tmux_job_server new-session -d -s "$SESSION" "$COMMAND"
echo "NATIVE_ENERGY_TMUX_CREATED session=$SESSION nodes=$NNODES; launch all nodes and inspect logs for actual training"
