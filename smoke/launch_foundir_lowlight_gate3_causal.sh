#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
RUNNER="$ROOT/smoke/run_foundir_lowlight_gate3_causal.sh"
LOG="$ROOT/logs/foundir_lowlight_gate3_causal_v1.log"
PID_FILE="$ROOT/logs/foundir_lowlight_gate3_causal_v1.pid"
CONFIG="$ROOT/configs/foundir_lowlight_gate3_causal_v1.json"

if [[ "${CUDA_VISIBLE_DEVICES:-}" != "0,1,2,3" ]]; then
  echo "ERROR: set CUDA_VISIBLE_DEVICES exactly to 0,1,2,3" >&2
  exit 1
fi

"$ROOT/.conda/gpu-train-stable/bin/python" -m py_compile \
  "$ROOT/lowlight_restoration/causal_condition.py" \
  "$ROOT/smoke/export_foundir_lowlight_gate3_causal_qwen.py" \
  "$ROOT/smoke/run_foundir_lowlight_gate3_causal_adapter.py" \
  "$ROOT/smoke/evaluate_foundir_lowlight_gate3_causal.py"
"$ROOT/.conda/gpu-train-stable/bin/python" -m unittest \
  tests.test_foundir_lowlight_gate3_causal tests.test_foundir_lowlight_gate3_compact
"$ROOT/.conda/gpu-train-stable/bin/python" \
  "$ROOT/smoke/export_foundir_lowlight_gate3_causal_qwen.py" --config "$CONFIG" --contract-only
"$ROOT/.conda/gpu-train-stable/bin/python" \
  "$ROOT/smoke/run_foundir_lowlight_gate3_causal_adapter.py" --config "$CONFIG" --contract-only
"$ROOT/.venv/foundir-metrics/bin/python" -c \
  'import importlib.metadata as m, lpips, pyiqa; assert m.version("lpips") == "0.1.4"; assert m.version("pyiqa") == "0.1.16"'
bash -n "$RUNNER" "$0"

required=(
  configs/foundir_lowlight_gate3_causal_v1.json
  lowlight_restoration/causal_condition.py
  smoke/export_foundir_lowlight_gate3_causal_qwen.py
  smoke/run_foundir_lowlight_gate3_causal_adapter.py
  smoke/evaluate_foundir_lowlight_gate3_causal.py
  smoke/run_foundir_lowlight_gate3_causal.sh
  smoke/launch_foundir_lowlight_gate3_causal.sh
  tests/test_foundir_lowlight_gate3_causal.py
)
for relative in "${required[@]}"; do
  if ! git -C "$ROOT" ls-files --error-unmatch "$relative" >/dev/null 2>&1; then
    echo "ERROR: required causal file is not committed: $relative" >&2
    exit 1
  fi
  if ! git -C "$ROOT" diff --quiet HEAD -- "$relative"; then
    echo "ERROR: required causal file differs from committed revision: $relative" >&2
    exit 1
  fi
done

mkdir -p "$ROOT/logs"
if [[ -f "$PID_FILE" ]]; then
  old_pid="$(tr -dc '0-9' <"$PID_FILE")"
  if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "ERROR: Gate 3 causal bundle is already active as PID $old_pid" >&2
    exit 1
  fi
fi
nohup env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" bash "$RUNNER" >>"$LOG" 2>&1 </dev/null &
pid="$!"
printf '%s\n' "$pid" >"$PID_FILE"
sleep 2
if ! kill -0 "$pid" 2>/dev/null; then
  echo "ERROR: Gate 3 causal bundle exited during launch; inspect $LOG" >&2
  tail -80 "$LOG" >&2 || true
  exit 1
fi
echo "FOUNDIR_GATE3_CAUSAL_DETACHED pid=$pid"
echo "Persistent log: $LOG"
echo "Expected wall clock: approximately 20-35 minutes; completed adapter and image outputs resume without recomputation."
