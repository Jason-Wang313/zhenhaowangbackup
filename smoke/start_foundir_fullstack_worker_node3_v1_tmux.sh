#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ "$ROOT" == /inspire/hdd/project/multimodal-discrete-diffusion/liuxiaohong-25080/zhenhaowang && $# == 0 ]] || exit 2
PY="$ROOT/.conda/gpu-train-stable/bin/python"
WORKER="$ROOT/smoke/foundir_fullstack_node3_extension_v1.py"
TMUX_BIN="$ROOT/.tools/tmux-env/bin/tmux"
[[ -x "$TMUX_BIN" ]] || { echo 'ERROR: real bundled tmux missing'; exit 3; }
"$PY" -S "$WORKER" inventory 3
SESSION=foundir_fullstack_worker_node3_v1
if env -u TMUX -u TMUX_PANE TMUX_TMPDIR=/tmp "$TMUX_BIN" -L zhenhaowang_jobs has-session -t "$SESSION" 2>/dev/null; then
 echo 'FULLSTACK_WORKER_SESSION_EXISTS node=3 inspect shared heartbeat'; exit 0
fi
mkdir -p "$ROOT/logs"
printf -v COMMAND 'cd %q && CUDA_VISIBLE_DEVICES=0,1 %q -S %q worker 3 >> %q 2>&1' \
 "$ROOT" "$PY" "$WORKER" "$ROOT/logs/foundir_fullstack_worker_node3_v1.log"
env -u TMUX -u TMUX_PANE TMUX_TMPDIR=/tmp "$TMUX_BIN" -L zhenhaowang_jobs new-session -d -s "$SESSION" "$COMMAND"
echo 'FULLSTACK_WORKER_TMUX_CREATED node=3 verify FULLSTACK_WORKER_READY and shared heartbeat; no GPU job submitted'
