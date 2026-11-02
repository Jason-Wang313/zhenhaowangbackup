#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ "$ROOT" == /inspire/hdd/project/multimodal-discrete-diffusion/liuxiaohong-25080/zhenhaowang ]] || { echo 'ERROR: use migrated GPU workspace'; exit 2; }
RUN_ID=${1:-replay8_v1}
GPU_COUNT=${2:-8}
[[ "$RUN_ID" =~ ^[A-Za-z0-9_-]{1,80}$ ]] || { echo 'ERROR: invalid run ID'; exit 2; }
case "$GPU_COUNT" in
  4) VISIBLE=0,1,2,3 ;;
  8) VISIBLE=0,1,2,3,4,5,6,7 ;;
  *) echo 'ERROR: use 4 or 8 GPUs'; exit 2 ;;
esac
PYTHON="${PROJECT_PYTHON:-$ROOT/.conda/gpu-train-stable/bin/python}"
TMUX_BIN="$ROOT/.tools/tmux-env/bin/tmux"
[[ -x "$PYTHON" && -x "$TMUX_BIN" ]] || { echo 'ERROR: verified Python/tmux unavailable'; exit 2; }
SESSION="foundir_existing_flux_replay_${RUN_ID}"
if env -u TMUX -u TMUX_PANE TMUX_TMPDIR=/tmp "$TMUX_BIN" -L zhenhaowang_jobs has-session -t "$SESSION" 2>/dev/null; then
  echo "ERROR: existing session $SESSION; inspect it instead of duplicate launch"; exit 3
fi
printf -v COMMAND 'cd %q && env PROJECT_PYTHON=%q CUDA_VISIBLE_DEVICES=%q bash %q %q %q' \
  "$ROOT" "$PYTHON" "$VISIBLE" "$ROOT/smoke/run_foundir_existing_flux_replay_v1_gpu.sh" "$RUN_ID" "$GPU_COUNT"
env -u TMUX -u TMUX_PANE TMUX_TMPDIR=/tmp "$TMUX_BIN" -L zhenhaowang_jobs new-session -d -s "$SESSION" "$COMMAND"
echo "EXPERT_REPLAY_TMUX_CREATED session=$SESSION; not_yet_GPU_success"
echo "log=$ROOT/logs/foundir_existing_flux_replay_v1_${RUN_ID}.log"
