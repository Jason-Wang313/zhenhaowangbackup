#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ "$ROOT" == /inspire/hdd/project/multimodal-discrete-diffusion/liuxiaohong-25080/zhenhaowang && $# == 0 ]] || { echo 'ERROR: destination CPU workspace only'; exit 2; }
PYTHON="${PROJECT_PYTHON:-$ROOT/.conda/gpu-train-stable/bin/python}"
TMUX_BIN="$ROOT/.tools/tmux-env/bin/tmux"
[[ -x "$PYTHON" && -x "$TMUX_BIN" ]] || { echo 'ERROR: qualified Python/tmux missing'; exit 2; }
SESSION=foundir_router_supervised_repair_v1
if env -u TMUX -u TMUX_PANE TMUX_TMPDIR=/tmp "$TMUX_BIN" -L zhenhaowang_jobs has-session -t "$SESSION" 2>/dev/null; then
  echo 'ERROR: session exists; inspect instead of duplicate launch'; exit 3
fi
printf -v COMMAND 'cd %q && env PROJECT_PYTHON=%q CUDA_VISIBLE_DEVICES= bash %q' \
  "$ROOT" "$PYTHON" "$ROOT/smoke/run_foundir_router_supervised_repair_v1_cpu.sh"
env -u TMUX -u TMUX_PANE TMUX_TMPDIR=/tmp "$TMUX_BIN" -L zhenhaowang_jobs new-session -d -s "$SESSION" "$COMMAND"
echo "SUPERVISED_ROUTER_CPU_TMUX_CREATED session=$SESSION; not_yet_training_success"
echo "log=$ROOT/logs/foundir_router_supervised_repair_v1.log"
