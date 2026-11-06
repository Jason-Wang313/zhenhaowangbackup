#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ "$ROOT" == /inspire/hdd/project/multimodal-discrete-diffusion/liuxiaohong-25080/zhenhaowang ]] || exit 2
[[ $# == 1 && "$1" =~ ^[0-2]$ ]] || { echo 'Usage: <new-node0-2>'; exit 2; }
TMUX_BIN="$ROOT/.tools/tmux-env/bin/tmux"
"$ROOT/.conda/gpu-train-stable/bin/python" -S "$ROOT/smoke/foundir_fullstack_executor_3x2_v2.py" inventory "$1"
SESSION="foundir_fullstack_worker_3x2_v2_node$1"
if env -u TMUX -u TMUX_PANE TMUX_TMPDIR=/tmp "$TMUX_BIN" -L zhenhaowang_jobs has-session -t "$SESSION" 2>/dev/null; then
  echo "FULLSTACK_WORKER_SESSION_EXISTS node=$1 inspect heartbeat"; exit 0
fi
mkdir -p "$ROOT/logs"
printf -v COMMAND 'cd %q && CUDA_VISIBLE_DEVICES=0,1 %q -S %q worker %q >> %q 2>&1' \
  "$ROOT" "$ROOT/.conda/gpu-train-stable/bin/python" "$ROOT/smoke/foundir_fullstack_executor_3x2_v2.py" "$1" "$ROOT/logs/foundir_fullstack_worker_3x2_v2_node$1.log"
env -u TMUX -u TMUX_PANE TMUX_TMPDIR=/tmp "$TMUX_BIN" -L zhenhaowang_jobs new-session -d -s "$SESSION" "$COMMAND"
echo "FULLSTACK_WORKER_TMUX_CREATED node=$1 verify shared heartbeat"
