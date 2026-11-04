#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ "$ROOT" == /inspire/hdd/project/multimodal-discrete-diffusion/liuxiaohong-25080/zhenhaowang ]] || exit 2
[[ $# == 1 || $# == 2 ]] || { echo 'Usage: tmux_launcher <features node_rank_0_to_3 | train>'; exit 2; }
PHASE=$1 NODE=${2:-0}
[[ "$PHASE" == features && $# == 2 && "$NODE" =~ ^[0-3]$ || "$PHASE" == train && $# == 1 ]] || exit 2
TMUX_BIN="$ROOT/.tools/tmux-env/bin/tmux"
[[ -x "$TMUX_BIN" ]] || exit 2
SESSION="foundir_frozen_image_v2_${PHASE}_${NODE}"
if env -u TMUX -u TMUX_PANE TMUX_TMPDIR=/tmp "$TMUX_BIN" -L zhenhaowang_jobs has-session -t "$SESSION" 2>/dev/null; then
  echo 'ERROR: session exists; inspect rather than duplicate'; exit 3
fi
printf -v COMMAND 'cd %q && bash %q %q' "$ROOT" "$ROOT/smoke/run_foundir_frozen_image_router_v2.sh" "$PHASE"
if [[ "$PHASE" == features ]]; then printf -v COMMAND '%s %q' "$COMMAND" "$NODE"; fi
env -u TMUX -u TMUX_PANE TMUX_TMPDIR=/tmp "$TMUX_BIN" -L zhenhaowang_jobs new-session -d -s "$SESSION" "$COMMAND"
echo "FROZEN_IMAGE_V2_TMUX_CREATED session=$SESSION not_yet_compute_success"
