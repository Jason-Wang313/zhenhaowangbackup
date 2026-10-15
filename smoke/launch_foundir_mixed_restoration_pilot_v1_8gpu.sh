#!/usr/bin/env bash
set -euo pipefail

ROOT=/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
PYTHON="$ROOT/.venv/flux2-smoke/bin/python"
CONFIG="$ROOT/configs/foundir_mixed_restoration_pilot_v1.json"
SCRIPT="$ROOT/smoke/evaluate_foundir_mixed_restoration_pilot_ddp.py"
RUNNER="$ROOT/smoke/run_foundir_mixed_restoration_pilot_v1_8gpu.sh"
LOG="$ROOT/logs/foundir_mixed_restoration_pilot_v1.log"
PID_FILE="$ROOT/logs/foundir_mixed_restoration_pilot_v1.pid"
RESULT="$ROOT/checkpoints/foundir_mixed_restoration_pilot_v1/result.json"

cd "$ROOT"
/usr/bin/python3 smoke/advisor_governance.py check
[[ $# -le 1 && ( $# -eq 0 || "$1" == "--check" ) ]] || {
  echo "ERROR: only optional --check is supported" >&2; exit 1;
}
[[ "${CUDA_VISIBLE_DEVICES:-}" == "0,1,2,3,4,5,6,7" ]] || {
  echo "ERROR: export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7" >&2; exit 1;
}
for path in "$PYTHON" "$CONFIG" "$SCRIPT" "$RUNNER"; do
  [[ -f "$path" ]] || { echo "ERROR: missing dependency $path" >&2; exit 1; }
done
"$PYTHON" -m py_compile "$SCRIPT" smoke/build_foundir_mixed_restoration_pilot_v1.py
bash -n "$RUNNER" "$0"
"$PYTHON" -m pytest -q tests/test_foundir_mixed_restoration_pilot.py
"$PYTHON" "$SCRIPT" --config "$CONFIG" --preflight
echo "TOTAL END-TO-END TIME: H100 15-25 min (center 20); H200 11-19 min (center 15). One homogeneous 8-GPU instance; 2.0-3.3 H100 GPU-hours or 1.5-2.5 H200 GPU-hours."

if [[ "${1:-}" == "--check" ]]; then
  echo "MIXED_RESTORATION_PILOT_LAUNCHER_CHECK_PASS"
  exit 0
fi
command -v nvidia-smi >/dev/null || { echo "ERROR: nvidia-smi is required" >&2; exit 1; }
[[ ! -f "$RESULT" ]] || { echo "ERROR: terminal result already exists" >&2; exit 1; }
mkdir -p "$ROOT/logs"
if [[ -f "$PID_FILE" ]]; then
  old_pid=$(tr -dc '0-9' <"$PID_FILE")
  if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "ERROR: launcher already active as PID $old_pid" >&2; exit 1
  fi
fi
nohup setsid env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" bash "$RUNNER" >/dev/null 2>&1 </dev/null &
pid=$!
printf '%s\n' "$pid" >"$PID_FILE"
sleep 3
kill -0 "$pid" 2>/dev/null || {
  echo "ERROR: runner exited during startup; inspect $LOG" >&2
  tail -100 "$LOG" >&2 || true
  exit 1
}
echo "MIXED_RESTORATION_PILOT_DETACHED process_group=$pid"
echo "Persistent log: $LOG"
echo "Monitor: tail -f $LOG"
echo "Stop safely: kill -- -$pid"
