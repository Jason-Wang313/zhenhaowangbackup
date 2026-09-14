#!/usr/bin/env bash
set -euo pipefail

ROOT=/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
RUNNER="$ROOT/smoke/run_foundir_blur_only_zimage_teacher_paired_flow_v2_8gpu.sh"
CONFIG="$ROOT/configs/foundir_blur_only_zimage_teacher_paired_flow_overfit_v2.json"
PYTHON="$ROOT/.conda/gpu-train-stable/bin/python"
LOG="$ROOT/logs/foundir_blur_only_zimage_teacher_paired_flow_overfit_v2.log"
TMUX=/usr/bin/tmux
SESSION=foundir-zimage-paired-flow-v2

cd "$ROOT"
/usr/bin/python3 "$ROOT/smoke/advisor_governance.py" check
[[ $# -le 1 && ( $# -eq 0 || "$1" == "--check" ) ]] || {
  echo "ERROR: only optional --check is supported" >&2; exit 1;
}
[[ "${CUDA_VISIBLE_DEVICES:-}" == "0,1,2,3,4,5,6,7" ]] || {
  echo "ERROR: set CUDA_VISIBLE_DEVICES exactly to 0,1,2,3,4,5,6,7" >&2; exit 1;
}
[[ -x "$PYTHON" ]] || { echo "ERROR: missing $PYTHON" >&2; exit 1; }
[[ -x "$TMUX" ]] || { echo "ERROR: missing executable $TMUX" >&2; exit 1; }
"$TMUX" -V

export PYTHONPATH="$ROOT/.deps/zimage:$ROOT/third_party/diffsynth-studio:$ROOT"
"$PYTHON" -m py_compile \
  lowlight_restoration/foundir_zimage_paired_flow.py \
  smoke/preflight_foundir_blur_only_zimage_teacher_paired_flow.py \
  smoke/train_foundir_blur_only_zimage_teacher_paired_flow_ddp.py \
  smoke/evaluate_foundir_blur_only_zimage_teacher_paired_flow_ddp.py
"$PYTHON" -c \
  'import smoke.train_foundir_blur_only_zimage_teacher_paired_flow_ddp; import smoke.evaluate_foundir_blur_only_zimage_teacher_paired_flow_ddp; print("FOUNDIR_ZIMAGE_PAIRED_FLOW_RUNTIME_IMPORT_PASS")'
bash -n "$RUNNER" "$0"

if [[ "${1:-}" == "--check" ]]; then
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$PYTHON" -m pytest -q tests/test_foundir_zimage_paired_flow_teacher.py
  "$PYTHON" smoke/preflight_foundir_blur_only_zimage_teacher_paired_flow.py \
    --config "$CONFIG" --skip-model-artifacts
  echo "FOUNDIR_ZIMAGE_PAIRED_FLOW_LAUNCHER_CHECK_PASS"
  exit 0
fi

mkdir -p "$ROOT/logs"
if "$TMUX" has-session -t "$SESSION" 2>/dev/null; then
  echo "ERROR: tmux session $SESSION already exists; attach and inspect it" >&2
  exit 1
fi
"$TMUX" new-session -d -s "$SESSION" -c "$ROOT"
"$TMUX" set-option -t "$SESSION" remain-on-exit on
"$TMUX" send-keys -t "$SESSION":0.0 \
  "export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7; bash '$RUNNER'" C-m
sleep 3
pane_state=$("$TMUX" display-message -p -t "$SESSION":0.0 '#{pane_dead}:#{pane_dead_status}')
if [[ "$pane_state" != "0:" ]]; then
  echo "ERROR: launcher exited during startup; inspect $LOG" >&2
  tail -100 "$LOG" >&2 || true
  exit 1
fi
echo "FOUNDIR_ZIMAGE_PAIRED_FLOW_TMUX_STARTED session=$SESSION"
echo "Attach: $TMUX attach -t $SESSION"
echo "Persistent log: $LOG"
echo "Monitor: tail -f $LOG"
