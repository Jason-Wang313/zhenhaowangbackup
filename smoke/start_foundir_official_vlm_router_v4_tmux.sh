#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ "$ROOT" == /inspire/hdd/project/multimodal-discrete-diffusion/liuxiaohong-25080/zhenhaowang ]] || exit 2
[[ $# == 2 && "$1" =~ ^(route|render)$ && "$2" =~ ^[0-1]$ ]] || { echo 'Usage: <route|render> <node0-1>'; exit 2; }
PHASE=$1 NODE=$2
TMUX_BIN="$ROOT/.tools/tmux-env/bin/tmux"
[[ -x "$TMUX_BIN" ]] || { echo 'ERROR: tmux runtime missing'; exit 2; }
SESSION="foundir_official_vlm_v4_${PHASE}_node$NODE"
if env -u TMUX -u TMUX_PANE TMUX_TMPDIR=/tmp "$TMUX_BIN" -L zhenhaowang_jobs has-session -t "$SESSION" 2>/dev/null; then
  echo 'ERROR: session exists; inspect instead of duplicate'; exit 3
fi
printf -v COMMAND 'cd %q && bash %q %q %q' "$ROOT" "$ROOT/smoke/run_foundir_official_vlm_router_v4.sh" "$PHASE" "$NODE"
env -u TMUX -u TMUX_PANE TMUX_TMPDIR=/tmp "$TMUX_BIN" -L zhenhaowang_jobs new-session -d -s "$SESSION" "$COMMAND"
echo "OFFICIAL_VLM_V4_TMUX_CREATED session=$SESSION not_compute_success"
