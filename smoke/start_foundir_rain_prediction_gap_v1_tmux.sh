#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ "$ROOT" == /inspire/hdd/project/multimodal-discrete-diffusion/liuxiaohong-25080/zhenhaowang ]] || { echo 'ERROR: use migrated GPU workspace'; exit 2; }
[[ $# == 4 ]] || { echo 'Usage: tmux_launcher <node_rank 0..3> <current_node0_address> <run_id> <fresh_shared_launch_id>'; exit 2; }
NODE_RANK=$1 MASTER_ADDR=$2 RUN_ID=$3 LAUNCH_ID=$4
[[ "$NODE_RANK" =~ ^[0-3]$ && "$MASTER_ADDR" =~ ^[A-Za-z0-9][A-Za-z0-9.-]*$ ]] || exit 2
[[ "$RUN_ID" =~ ^[A-Za-z0-9_-]{1,80}$ && "$LAUNCH_ID" =~ ^[A-Za-z0-9_-]{1,80}$ ]] || exit 2
MASTER_PORT=${REPLAY_MASTER_PORT:-29692}
[[ "$MASTER_PORT" =~ ^[0-9]{4,5}$ ]] && ((10#$MASTER_PORT <= 65535)) || exit 2
PYTHON="${PROJECT_PYTHON:-$ROOT/.conda/gpu-train-stable/bin/python}"
TMUX_BIN="$ROOT/.tools/tmux-env/bin/tmux"
[[ -x "$PYTHON" && -x "$TMUX_BIN" ]] || { echo 'ERROR: verified Python/tmux unavailable'; exit 2; }
SESSION="foundir_rain_prediction_gap_${RUN_ID}_node${NODE_RANK}"
if env -u TMUX -u TMUX_PANE TMUX_TMPDIR=/tmp "$TMUX_BIN" -L zhenhaowang_jobs has-session -t "$SESSION" 2>/dev/null; then
  echo "ERROR: existing session $SESSION; inspect instead of duplicate launch"; exit 3
fi
printf -v COMMAND 'cd %q && env PROJECT_PYTHON=%q CUDA_VISIBLE_DEVICES=0,1 REPLAY_MASTER_PORT=%q bash %q %q %q %q %q' \
  "$ROOT" "$PYTHON" "$MASTER_PORT" "$ROOT/smoke/run_foundir_rain_prediction_gap_v1_gpu.sh" "$NODE_RANK" "$MASTER_ADDR" "$RUN_ID" "$LAUNCH_ID"
env -u TMUX -u TMUX_PANE TMUX_TMPDIR=/tmp "$TMUX_BIN" -L zhenhaowang_jobs new-session -d -s "$SESSION" "$COMMAND"
echo "RAIN_PREDICTION_GAP_TMUX_CREATED session=$SESSION node=$NODE_RANK nodes=4 local_gpus=2; not_yet_GPU_success"
echo "log=$ROOT/logs/foundir_rain_prediction_gap_v1_${RUN_ID}_node${NODE_RANK}.log"
