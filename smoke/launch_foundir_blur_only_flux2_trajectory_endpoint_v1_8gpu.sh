#!/usr/bin/env bash
set -euo pipefail

ROOT=/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
PYTHON="$ROOT/.venv/flux2-smoke/bin/python"
CONFIG="$ROOT/configs/foundir_blur_only_flux2_trajectory_endpoint_gate_v1.json"
SCRIPT="$ROOT/smoke/train_foundir_blur_only_flux2_pixel_fidelity_ddp.py"
RUNNER="$ROOT/smoke/run_foundir_blur_only_flux2_trajectory_endpoint_v1_8gpu.sh"
OUTPUT="$ROOT/checkpoints/foundir_blur_only_flux2_trajectory_endpoint_gate_v1"
RESULT="$OUTPUT/result.json"
LOG="$ROOT/logs/foundir_blur_only_flux2_trajectory_endpoint_gate_v1.log"
PID_FILE="$ROOT/logs/foundir_blur_only_flux2_trajectory_endpoint_gate_v1.pid"

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

"$PYTHON" -m py_compile "$SCRIPT" lowlight_restoration/flux2_trajectory_endpoint.py \
  tests/test_foundir_blur_only_flux2_pixel_fidelity.py tests/test_flux2_trajectory_endpoint.py
bash -n "$RUNNER" "$0"
"$PYTHON" -m pytest -q tests/test_foundir_blur_only_flux2_pixel_fidelity.py \
  tests/test_flux2_trajectory_endpoint.py
"$PYTHON" "$SCRIPT" --config "$CONFIG" --preflight

if [[ "${1:-}" == "--check" ]]; then
  echo "FOUNDIR_BLUR_TRAJECTORY_ENDPOINT_LAUNCHER_CHECK_PASS"
  exit 0
fi
command -v nvidia-smi >/dev/null || { echo "ERROR: nvidia-smi is required" >&2; exit 1; }
[[ ! -f "$RESULT" ]] || { echo "ERROR: terminal result already exists; inspect it" >&2; exit 1; }
mkdir -p "$ROOT/logs"
if [[ -f "$PID_FILE" ]]; then
  old_pid=$(tr -dc '0-9' <"$PID_FILE")
  if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "ERROR: trajectory-endpoint launcher process already active as PID $old_pid" >&2
    exit 1
  fi
fi

nohup setsid env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" bash "$RUNNER" \
  >/dev/null 2>&1 </dev/null &
pid=$!
printf '%s\n' "$pid" >"$PID_FILE"
sleep 3
kill -0 "$pid" 2>/dev/null || {
  echo "ERROR: trajectory-endpoint runner exited during startup; inspect $LOG" >&2
  tail -100 "$LOG" >&2 || true
  exit 1
}
echo "FOUNDIR_BLUR_TRAJECTORY_ENDPOINT_DETACHED process_group=$pid"
echo "Persistent log: $LOG"
echo "Monitor: tail -f $LOG"
echo "Stop safely: kill -- -$pid"
