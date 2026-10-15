#!/usr/bin/env bash
set -euo pipefail
ROOT=/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
SESSION=foundir_lowlight_haze_restormer_v1
RECOVERY="$ROOT/smoke/recover_foundir_lowlight_haze_restormer_v1_8gpu.sh"
TMUX_BIN=$(command -v tmux || true)
[[ -n "$TMUX_BIN" && -x "$TMUX_BIN" ]] || { echo "ERROR: executable tmux not found in this GPU shell" >&2; exit 1; }
"$TMUX_BIN" -V
command -v nvidia-smi >/dev/null || { echo "ERROR: run this only in the GPU-enabled interface" >&2; exit 1; }
if "$TMUX_BIN" has-session -t "$SESSION" 2>/dev/null; then
  echo "ERROR: tmux session $SESSION already exists; refusing duplicate launch" >&2
  echo "Attach with: $TMUX_BIN attach -t $SESSION" >&2
  exit 1
fi
echo "TOTAL END-TO-END TIME: H200 100-160 min (center 125), 13.3-21.3 GPU-hours (center 16.7); H100 130-200 min (center 160), 17.3-26.7 GPU-hours (center 21.3)."
echo "CONCURRENCY: one homogeneous node, 8 GPUs. Main uncertainty: full-resolution tiled LPIPS and shared-storage hashing throughput."
"$TMUX_BIN" new-session -d -s "$SESSION" -c "$ROOT"
"$TMUX_BIN" set-option -t "$SESSION" remain-on-exit on
"$TMUX_BIN" send-keys -t "$SESSION" "env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash $RECOVERY" C-m
exec "$TMUX_BIN" attach -t "$SESSION"
