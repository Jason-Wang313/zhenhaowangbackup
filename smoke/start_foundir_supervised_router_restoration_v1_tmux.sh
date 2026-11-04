#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ "$ROOT" == /inspire/hdd/project/multimodal-discrete-diffusion/liuxiaohong-25080/zhenhaowang ]] || { echo 'ERROR: destination workspace only'; exit 2; }
[[ $# == 0 ]] || exit 2
PYTHON="${PROJECT_PYTHON:-$ROOT/.conda/gpu-train-stable/bin/python}"
TMUX_BIN="$ROOT/.tools/tmux-env/bin/tmux"
[[ -x "$PYTHON" && -x "$TMUX_BIN" ]] || { echo 'ERROR: qualified Python/tmux missing'; exit 2; }
SESSION=foundir_supervised_router_restoration_v1
if env -u TMUX -u TMUX_PANE TMUX_TMPDIR=/tmp "$TMUX_BIN" -L zhenhaowang_jobs has-session -t "$SESSION" 2>/dev/null; then
  echo 'ERROR: existing evaluation session; inspect instead of duplicate launch'; exit 3
fi
printf -v COMMAND 'cd %q && env PROJECT_PYTHON=%q CUDA_VISIBLE_DEVICES=0,1 bash %q' \
  "$ROOT" "$PYTHON" "$ROOT/smoke/run_foundir_supervised_router_restoration_v1_gpu.sh"
env -u TMUX -u TMUX_PANE TMUX_TMPDIR=/tmp "$TMUX_BIN" -L zhenhaowang_jobs new-session -d -s "$SESSION" "$COMMAND"
echo "MATCHED_RESTORATION_TMUX_CREATED session=$SESSION; not_yet_GPU_success"
echo "log=$ROOT/logs/foundir_supervised_router_restoration_v1.log"
