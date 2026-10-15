#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
/usr/bin/python3 "$ROOT/smoke/advisor_governance.py" check
cd "$ROOT"
RUNNER="$ROOT/smoke/run_foundir_lowlight_gate3_correction1.sh"
LOG="$ROOT/logs/foundir_lowlight_gate3_correction1_v1.log"
PID_FILE="$ROOT/logs/foundir_lowlight_gate3_correction1_v1.pid"
CONFIG="$ROOT/configs/foundir_lowlight_gate3_correction1_v1.json"

if [[ "${CUDA_VISIBLE_DEVICES:-}" != "0,1,2,3" ]]; then
  echo "ERROR: set CUDA_VISIBLE_DEVICES exactly to 0,1,2,3" >&2
  exit 1
fi

"$ROOT/.conda/gpu-train-stable/bin/python" -m py_compile \
  "$ROOT/lowlight_restoration/structured_qwen_residual.py" \
  "$ROOT/smoke/run_foundir_lowlight_gate3_correction1.py" \
  "$ROOT/smoke/evaluate_foundir_lowlight_gate3_correction1.py"
"$ROOT/.conda/gpu-train-stable/bin/python" -m unittest \
  tests.test_foundir_lowlight_gate3_correction1 tests.test_foundir_lowlight_gate3_causal \
  tests.test_foundir_lowlight_gate3_compact
"$ROOT/.conda/gpu-train-stable/bin/python" \
  "$ROOT/smoke/run_foundir_lowlight_gate3_correction1.py" --config "$CONFIG" --contract-only
"$ROOT/.venv/foundir-metrics/bin/python" -c \
  'import importlib.metadata as m, lpips, pyiqa; assert m.version("lpips") == "0.1.4"; assert m.version("pyiqa") == "0.1.16"'
bash -n "$RUNNER" "$0"

required=(
  configs/foundir_lowlight_gate3_correction1_v1.json
  lowlight_restoration/structured_qwen_residual.py
  smoke/run_foundir_lowlight_gate3_correction1.py
  smoke/evaluate_foundir_lowlight_gate3_correction1.py
  smoke/run_foundir_lowlight_gate3_correction1.sh
  smoke/launch_foundir_lowlight_gate3_correction1.sh
  tests/test_foundir_lowlight_gate3_correction1.py
)
for relative in "${required[@]}"; do
  if ! git -C "$ROOT" ls-files --error-unmatch "$relative" >/dev/null 2>&1; then
    echo "ERROR: required correction-1 file is not committed: $relative" >&2
    exit 1
  fi
  if ! git -C "$ROOT" diff --quiet HEAD -- "$relative"; then
    echo "ERROR: required correction-1 file differs from committed revision: $relative" >&2
    exit 1
  fi
done

mkdir -p "$ROOT/logs"
if [[ -f "$PID_FILE" ]]; then
  old_pid="$(tr -dc '0-9' <"$PID_FILE")"
  if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "ERROR: Gate 3 correction 1 is already active as PID $old_pid" >&2
    exit 1
  fi
fi
nohup env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" bash "$RUNNER" >>"$LOG" 2>&1 </dev/null &
pid="$!"
printf '%s\n' "$pid" >"$PID_FILE"
sleep 2
if ! kill -0 "$pid" 2>/dev/null; then
  echo "ERROR: Gate 3 correction 1 exited during launch; inspect $LOG" >&2
  tail -80 "$LOG" >&2 || true
  exit 1
fi
echo "FOUNDIR_GATE3_CORRECTION1_DETACHED pid=$pid"
echo "Persistent log: $LOG"
echo "Expected wall clock: approximately 10-20 minutes; Qwen features and frozen measured/oracle outputs are reused."
