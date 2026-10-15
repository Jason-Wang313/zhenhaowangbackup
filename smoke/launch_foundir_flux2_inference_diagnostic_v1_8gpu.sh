#!/usr/bin/env bash
set -euo pipefail

ROOT=/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
/usr/bin/python3 "$ROOT/smoke/advisor_governance.py" check
RUNNER="$ROOT/smoke/run_foundir_flux2_inference_diagnostic_v1_8gpu.sh"
LOG="$ROOT/logs/foundir_flux2_inference_diagnostic_v1.log"
PID_FILE="$ROOT/logs/foundir_flux2_inference_diagnostic_v1.pid"
PYTHON="$ROOT/.venv/flux2-smoke/bin/python"

cd "$ROOT"
[[ $# -le 1 && ( $# -eq 0 || "$1" == "--check" ) ]] || {
  echo "ERROR: only the optional --check argument is supported" >&2; exit 1;
}
[[ "${CUDA_VISIBLE_DEVICES:-}" == "0,1,2,3,4,5,6,7" ]] || {
  echo "ERROR: set CUDA_VISIBLE_DEVICES exactly to 0,1,2,3,4,5,6,7" >&2; exit 1;
}
"$PYTHON" -m py_compile smoke/diagnose_foundir_flux2_inference_ddp.py
bash -n "$RUNNER" "$0"
[[ -f checkpoints/foundir_five_type_flux2_dit_v3/full_curriculum/step_033750.safetensors ]] || {
  echo "ERROR: step 33750 checkpoint is missing" >&2; exit 1;
}
if [[ "${1:-}" == "--check" ]]; then
  echo "FOUNDIR_INFERENCE_DIAGNOSTIC_LAUNCHER_CHECK_PASS"
  exit 0
fi
mkdir -p "$ROOT/logs"
if [[ -f "$PID_FILE" ]]; then
  old_pid=$(tr -dc '0-9' <"$PID_FILE")
  if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "ERROR: inference diagnostic is already active as PID $old_pid" >&2
    exit 1
  fi
fi
nohup env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" bash "$RUNNER" >/dev/null 2>&1 </dev/null &
pid=$!
printf '%s\n' "$pid" >"$PID_FILE"
sleep 3
if ! kill -0 "$pid" 2>/dev/null; then
  echo "ERROR: inference diagnostic exited during launch; inspect $LOG" >&2
  tail -100 "$LOG" >&2 || true
  exit 1
fi
echo "FOUNDIR_INFERENCE_DIAGNOSTIC_DETACHED pid=$pid"
echo "Persistent log: $LOG"
echo "Monitor: tail -f $LOG"
