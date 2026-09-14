#!/usr/bin/env bash
set -euo pipefail
ROOT=/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
cd "$ROOT"
PYTHON="$ROOT/.conda/gpu-train-stable/bin/python"
RUNNER="$ROOT/smoke/run_foundir_blur_only_zimage_endpoint_distill_v1_8gpu.sh"
LOG="$ROOT/logs/foundir_blur_only_zimage_endpoint_distill_v1.log"
SESSION=foundir-zimage-endpoint-v1
[[ $# -eq 0 || ( $# -eq 1 && "$1" == '--check' ) ]]
/usr/bin/python3 smoke/advisor_governance.py check
export PYTHONPATH="$ROOT/.deps/zimage:$ROOT/third_party/diffsynth-studio:$ROOT"
"$PYTHON" -m py_compile smoke/train_foundir_blur_only_zimage_endpoint_distill_ddp.py
bash -n "$RUNNER" "$0"
if [[ "${1:-}" == '--check' ]]; then
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$PYTHON" -m pytest -q tests/test_zimage_endpoint_distillation.py
  "$PYTHON" smoke/train_foundir_blur_only_zimage_endpoint_distill_ddp.py --config configs/foundir_blur_only_zimage_endpoint_distill_v1.json --preflight
  exit 0
fi
[[ "${CUDA_VISIBLE_DEVICES:-}" == '0,1,2,3,4,5,6,7' ]]
if [[ -e checkpoints/foundir_blur_only_zimage_endpoint_distill_v1/result.json || -e "$LOG" ]]; then
  echo "ERROR: previous job evidence exists; inspect $LOG before retrying"; exit 1
fi
mkdir -p logs
TMUX=$(command -v tmux || true)
if [[ -n "$TMUX" && -x "$TMUX" ]]; then
  "$TMUX" -V
  if "$TMUX" has-session -t "$SESSION" 2>/dev/null; then echo 'ERROR: session already exists'; exit 1; fi
  "$TMUX" new-session -d -s "$SESSION" -c "$ROOT"
  "$TMUX" set-option -t "$SESSION" remain-on-exit on
  "$TMUX" send-keys -t "$SESSION" "export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7; bash '$RUNNER'" C-m
  echo "Attach: $TMUX attach -t $SESSION"
else
  nohup bash "$RUNNER" >logs/foundir_blur_only_zimage_endpoint_distill_v1_startup.log 2>&1 </dev/null &
  pid=$!
  printf '%s\n' "$pid" > logs/foundir_blur_only_zimage_endpoint_distill_v1.pid
  sleep 3
  kill -0 "$pid" || { echo "ERROR: startup failed; inspect $LOG and startup.log"; exit 1; }
fi
echo "Monitor: tail -f $LOG"
