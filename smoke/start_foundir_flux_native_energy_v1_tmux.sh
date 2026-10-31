#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
RUN_ID=${1:-qualification_v1}; GPUS=${2:-8}; RESUME=${3:-fresh}
[[ "$RUN_ID" =~ ^[A-Za-z0-9_-]+$ && ( "$GPUS" == 4 || "$GPUS" == 8 ) ]] || exit 2
[[ "$RESUME" == fresh || "$RESUME" == resume ]] || exit 2
if [[ "$GPUS" == 8 ]]; then DEFAULT_VISIBLE=0,1,2,3,4,5,6,7
else DEFAULT_VISIBLE=0,1,2,3; fi
VISIBLE=${CUDA_VISIBLE_DEVICES:-$DEFAULT_VISIBLE}
PYTHON=${FLUX_NATIVE_PYTHON:-"$ROOT/.conda/gpu-train-stable/bin/python"}
printf -v COMMAND '%q ' env "CUDA_VISIBLE_DEVICES=$VISIBLE" "FLUX_NATIVE_PYTHON=$PYTHON" \
  bash "$ROOT/smoke/run_foundir_flux_native_energy_v1_gpu.sh" "$RUN_ID" "$GPUS" "$RESUME"
if [[ "${FLUX_NATIVE_DRY_RUN:-0}" == 1 ]]; then printf '%s\n' "$COMMAND"; exit 0; fi
source smoke/tmux_launcher_lib.sh
TMUX_BIN_PATH=$(select_tmux_binary)
TMUX_SERVER_LABEL=zhenhaowang_jobs
SESSION="flux_native_energy_${RUN_ID}"
if tmux_job_server has-session -t "$SESSION" 2>/dev/null; then
  echo "NATIVE_ENERGY_SESSION_EXISTS session=$SESSION; not a new launch"; exit 0
fi
tmux_job_server new-session -d -s "$SESSION" "$COMMAND"
echo "NATIVE_ENERGY_TMUX_CREATED session=$SESSION; inspect logs for actual GPU/training status"
