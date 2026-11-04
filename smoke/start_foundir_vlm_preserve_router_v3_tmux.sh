#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ "$ROOT" == /inspire/hdd/project/multimodal-discrete-diffusion/liuxiaohong-25080/zhenhaowang && $# == 0 ]] || exit 2
TMUX_BIN="$ROOT/.tools/tmux-env/bin/tmux"
[[ -x "$TMUX_BIN" ]] || { echo 'ERROR: project tmux missing'; exit 2; }
SESSION=foundir_vlm_preserve_router_v3_cpu
if env -u TMUX -u TMUX_PANE TMUX_TMPDIR=/tmp "$TMUX_BIN" -L zhenhaowang_jobs has-session -t "$SESSION" 2>/dev/null; then
  echo 'ERROR: session exists; inspect rather than duplicate'; exit 3
fi
printf -v COMMAND 'cd %q && bash %q' "$ROOT" "$ROOT/smoke/run_foundir_vlm_preserve_router_v3_cpu.sh"
env -u TMUX -u TMUX_PANE TMUX_TMPDIR=/tmp "$TMUX_BIN" -L zhenhaowang_jobs new-session -d -s "$SESSION" "$COMMAND"
echo "VLM_PRESERVE_V3_TMUX_CREATED session=$SESSION not_yet_training_success"
