#!/usr/bin/env bash
set -euo pipefail

ROOT=/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
PYTHON="$ROOT/.venv/flux2-smoke/bin/python"
CONFIG="$ROOT/configs/foundir_rain_flux2_strength_identity_directional_gate_v1.json"
FREEZER="$ROOT/smoke/freeze_foundir_rain_flux2_strength_identity_directional_gate.py"
RUNNER="$ROOT/smoke/run_foundir_rain_flux2_strength_identity_directional_gate_v1_8gpu.sh"
TEST="$ROOT/tests/test_foundir_rain_flux2_strength_identity_directional_gate.py"
LOG="$ROOT/logs/foundir_rain_flux2_strength_identity_directional_gate_v1.log"
PID_FILE="$ROOT/logs/foundir_rain_flux2_strength_identity_directional_gate_v1.pid"
RESULT="$ROOT/checkpoints/foundir_rain_flux2_strength_identity_directional_gate_v1/result.json"

cd "$ROOT"
/usr/bin/python3 smoke/advisor_governance.py check
[[ $# -le 1 && ( $# -eq 0 || "$1" == "--check" ) ]] || {
  echo "ERROR: only optional --check is supported" >&2; exit 1;
}
[[ "${CUDA_VISIBLE_DEVICES:-}" == "0,1,2,3,4,5,6,7" ]] || {
  echo "ERROR: export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7" >&2; exit 1;
}
for command in setsid flock sha256sum jq; do command -v "$command" >/dev/null || { echo "ERROR: $command is required" >&2; exit 1; }; done
for path in "$PYTHON" "$CONFIG" "$FREEZER" "$RUNNER" "$TEST"; do [[ -f "$path" ]] || { echo "ERROR: missing $path" >&2; exit 1; }; done

"$PYTHON" -m py_compile "$FREEZER" smoke/train_foundir_blur_only_flux2_overfit_ddp.py "$TEST"
bash -n "$RUNNER" "$0"
"$PYTHON" -m pytest -q "$TEST"
"$PYTHON" "$FREEZER" --config "$CONFIG"
"$PYTHON" smoke/train_foundir_blur_only_flux2_overfit_ddp.py --config "$CONFIG" --preflight
if [[ "${1:-}" == "--check" ]]; then
  echo "FOUNDIR_RAIN_STRENGTH_IDENTITY_LAUNCHER_CHECK_PASS"
  exit 0
fi

command -v nvidia-smi >/dev/null || { echo "ERROR: nvidia-smi is required" >&2; exit 1; }
[[ ! -f "$RESULT" ]] || { echo "ERROR: terminal Rain directional result already exists" >&2; exit 1; }
mkdir -p "$ROOT/logs"
nohup setsid env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" bash "$RUNNER" >/dev/null 2>&1 </dev/null &
pid=$!
printf '%s\n' "$pid" >"$PID_FILE"
sleep 3
kill -0 "$pid" 2>/dev/null || {
  echo "ERROR: runner exited during startup; inspect $LOG" >&2
  tail -100 "$LOG" >&2 || true
  exit 1
}
echo "FOUNDIR_RAIN_STRENGTH_IDENTITY_DETACHED process_group=$pid"
echo "Persistent log: $LOG"
echo "Monitor: tail -f $LOG"
echo "Stop safely: kill -- -$pid"
