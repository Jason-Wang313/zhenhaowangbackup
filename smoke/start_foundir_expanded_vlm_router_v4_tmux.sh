#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ "$ROOT" == /inspire/hdd/project/multimodal-discrete-diffusion/liuxiaohong-25080/zhenhaowang ]] || exit 2
[[ $# == 1 || $# == 2 ]] || { echo 'Usage: <train|features node0-1|render node0-1>'; exit 2; }
PHASE=$1 NODE=${2:-0}
[[ "$PHASE" =~ ^(features|render)$ && $# == 2 && "$NODE" =~ ^[0-1]$ || "$PHASE" == train && $# == 1 ]] || exit 2
TMUX_BIN="$ROOT/.tools/tmux-env/bin/tmux"
[[ -x "$TMUX_BIN" ]] || exit 2
SESSION="foundir_expanded_vlm_v4_${PHASE}_${NODE}"
if env -u TMUX -u TMUX_PANE TMUX_TMPDIR=/tmp "$TMUX_BIN" -L zhenhaowang_jobs has-session -t "$SESSION" 2>/dev/null; then
  echo 'ERROR: session exists; inspect instead of duplicate'; exit 3
fi
printf -v COMMAND 'cd %q && bash %q %q' "$ROOT" "$ROOT/smoke/run_foundir_expanded_vlm_router_v4.sh" "$PHASE"
if [[ "$PHASE" != train ]]; then printf -v COMMAND '%s %q' "$COMMAND" "$NODE"; fi
env -u TMUX -u TMUX_PANE TMUX_TMPDIR=/tmp "$TMUX_BIN" -L zhenhaowang_jobs new-session -d -s "$SESSION" "$COMMAND"
echo "EXPANDED_V4_TMUX_CREATED session=$SESSION not_compute_success"
