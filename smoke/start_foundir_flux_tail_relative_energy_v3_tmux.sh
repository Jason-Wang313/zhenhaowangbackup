#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
# An explicit, qualified interpreter is required after relocation. Never fall
# back to system Python or the environment retained by an older tmux server.
PYTHON=${FLUX_TAIL_V3_PYTHON:-}
if [[ "$PYTHON" != /* || ! -f "$PYTHON" || ! -x "$PYTHON" ]]; then
  echo 'ERROR: set FLUX_TAIL_V3_PYTHON to the absolute executable path of the qualified V3 Python' >&2
  exit 2
fi
MODE=${1:-qualification}
RUN_ID=${2:-qualification_v1}
GPU_COUNT=${3:-8}
QUALIFICATION_RUN_ID=${4:-qualification_v1}
[[ "$MODE" == qualification || "$MODE" == trainfit ]] || exit 2
[[ "$RUN_ID" =~ ^[A-Za-z0-9_-]+$ ]] || exit 2
[[ "$QUALIFICATION_RUN_ID" =~ ^[A-Za-z0-9_-]+$ ]] || exit 2
[[ "$GPU_COUNT" == 4 || "$GPU_COUNT" == 8 ]] || exit 2
case "$GPU_COUNT" in
  8) VISIBLE=0,1,2,3,4,5,6,7 ;;
  4) VISIBLE=0,1,2,3 ;;
esac
cd "$ROOT"
source smoke/tmux_launcher_lib.sh
TMUX_BIN_PATH=$(select_tmux_binary)
TMUX_SERVER_LABEL=zhenhaowang_jobs
SESSION="flux_tail_relative_v3_${RUN_ID}"
if tmux_job_server has-session -t "$SESSION" 2>/dev/null; then
  echo "TAIL_RELATIVE_V3_ALREADY_RUNNING session=$SESSION"; exit 0
fi
printf -v LAUNCH_COMMAND '%q ' env -u PYTHONHOME PYTHONNOUSERSITE=1 \
  "FLUX_TAIL_V3_PYTHON=$PYTHON" "CUDA_VISIBLE_DEVICES=$VISIBLE" \
  bash "$ROOT/smoke/run_foundir_flux_tail_relative_energy_v3_gpu.sh" "$MODE" "$RUN_ID" "$GPU_COUNT" "$QUALIFICATION_RUN_ID"
tmux_job_server new-session -d -s "$SESSION" "$LAUNCH_COMMAND"
echo "TAIL_RELATIVE_V3_TMUX_CREATED session=$SESSION gpus=$GPU_COUNT; verify actual progress in log"
echo "LOG=$ROOT/logs/foundir_flux_tail_relative_energy_v3_${RUN_ID}.log"
echo 'Same command resumes this run; no auto-trainfit/formal and no qualification-weight reuse.'
