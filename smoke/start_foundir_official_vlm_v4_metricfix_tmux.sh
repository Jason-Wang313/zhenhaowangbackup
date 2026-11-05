#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ "$ROOT" == /inspire/hdd/project/multimodal-discrete-diffusion/liuxiaohong-25080/zhenhaowang ]] || exit 2
[[ $# == 1 && "$1" =~ ^[0-3]$ ]] || { echo 'Usage: <node0-3>'; exit 2; }
NODE=$1
TMUX_BIN="$ROOT/.tools/tmux-env/bin/tmux"
[[ -x "$TMUX_BIN" ]] || { echo 'ERROR: tmux missing'; exit 2; }
[[ -f "$ROOT/checkpoints/foundir_official_vlm_router_v4/official_vlm_v4_frozen4800_v1_4x2/metric_identity_fix.json" ]] || { echo 'ERROR: CPU metricfix prepare required'; exit 2; }
SESSION="foundir_official_vlm_v4_4x2_render_node$NODE"
if env -u TMUX -u TMUX_PANE TMUX_TMPDIR=/tmp "$TMUX_BIN" -L zhenhaowang_jobs has-session -t "$SESSION" 2>/dev/null; then
  echo 'ERROR: session exists; inspect instead of duplicate'; exit 3
fi
printf -v COMMAND 'cd %q && bash %q render %q' "$ROOT" "$ROOT/smoke/run_foundir_official_vlm_v4_metricfix.sh" "$NODE"
env -u TMUX -u TMUX_PANE TMUX_TMPDIR=/tmp "$TMUX_BIN" -L zhenhaowang_jobs new-session -d -s "$SESSION" "$COMMAND"
echo "OFFICIAL_VLM_V4_METRICFIX_TMUX_CREATED session=$SESSION not_compute_success"
