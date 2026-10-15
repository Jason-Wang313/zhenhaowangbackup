#!/usr/bin/env bash
set -euo pipefail

ROOT=/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
PYTHON="$ROOT/.venv/flux2-smoke/bin/python"
CONFIG="$ROOT/configs/foundir_haze_flux2_step3000_full_development_v1.json"
SCRIPT="$ROOT/smoke/evaluate_foundir_haze_flux2_step3000_full_development_ddp.py"
RUNNER="$ROOT/smoke/run_foundir_haze_flux2_step3000_full_development_v1_8gpu.sh"
RESULT="$ROOT/checkpoints/foundir_haze_flux2_step3000_full_development_v1/result.json"
LOG="$ROOT/logs/foundir_haze_flux2_step3000_full_development_v1.log"
PID_FILE="$ROOT/logs/foundir_haze_flux2_step3000_full_development_v1.pid"

cd "$ROOT"
/usr/bin/python3 smoke/advisor_governance.py check
[[ $# -le 1 && ( $# -eq 0 || "$1" == "--check" ) ]] || {
  echo "ERROR: only optional --check is supported" >&2; exit 1;
}
[[ "${CUDA_VISIBLE_DEVICES:-}" == "0,1,2,3,4,5,6,7" ]] || {
  echo "ERROR: export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7" >&2; exit 1;
}
for command in setsid flock sha256sum; do
  command -v "$command" >/dev/null || { echo "ERROR: $command is required" >&2; exit 1; }
done
for path in "$PYTHON" "$CONFIG" "$SCRIPT" "$RUNNER"; do
  [[ -f "$path" ]] || { echo "ERROR: missing dependency $path" >&2; exit 1; }
done

"$PYTHON" -m py_compile "$SCRIPT" tests/test_foundir_haze_flux2_step3000_full_development.py
bash -n "$RUNNER" "$0"
"$PYTHON" -m pytest -q tests/test_foundir_haze_flux2_step3000_full_development.py
"$PYTHON" "$SCRIPT" --config "$CONFIG" --preflight
echo "TOTAL END-TO-END TIME: H100 23-31 min (center 27); H200 17-24 min (center 20). One homogeneous 8-GPU instance; 3.1-4.1 H100 GPU-hours or 2.3-3.2 H200 GPU-hours. Includes preflight, loading, 1580 50-step inferences, metrics, bootstrap, visuals, aggregation, and report."

if [[ "${1:-}" == "--check" ]]; then
  echo "HAZE_STEP3000_FULL_DEVELOPMENT_LAUNCHER_CHECK_PASS"
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
echo "HAZE_STEP3000_FULL_DEVELOPMENT_DETACHED process_group=$pid"
echo "Persistent log: $LOG"
echo "Monitor: tail -f $LOG"
echo "Stop safely: kill -- -$pid"
