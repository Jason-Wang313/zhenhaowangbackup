#!/usr/bin/env bash
set -euo pipefail

ROOT=/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
/usr/bin/python3 "$ROOT/smoke/advisor_governance.py" check
RUNNER="$ROOT/smoke/run_foundir_blur_restormer_refinement_v1_8gpu.sh"
TRAINER="$ROOT/smoke/train_foundir_blur_restormer_refinement_ddp.py"
CONFIG="$ROOT/configs/foundir_blur_restormer_refinement_gate_v1.json"
LOG="$ROOT/logs/foundir_blur_restormer_refinement_v1.log"
PID_FILE="$ROOT/logs/foundir_blur_restormer_refinement_v1.pid"
PYTHON="$ROOT/.venv/flux2-smoke/bin/python"

cd "$ROOT"
[[ $# -le 1 && ( $# -eq 0 || "$1" == "--check" ) ]] || {
  echo "ERROR: only optional --check is supported" >&2
  exit 1
}
[[ "${CUDA_VISIBLE_DEVICES:-}" == "0,1,2,3,4,5,6,7" ]] || {
  echo "ERROR: expose exactly CUDA devices 0,1,2,3,4,5,6,7" >&2
  exit 1
}
[[ -x "$PYTHON" && -f "$RUNNER" && -f "$TRAINER" && -f "$CONFIG" ]] || {
  echo "ERROR: Blur Restormer launcher dependency is missing" >&2
  exit 1
}

"$PYTHON" -m py_compile "$TRAINER"
bash -n "$RUNNER" "$0"
"$PYTHON" "$TRAINER" --config "$CONFIG" --preflight

if [[ "${1:-}" == "--check" ]]; then
  echo "FOUNDIR_BLUR_RESTORMER_V1_LAUNCHER_CHECK_PASS"
  exit 0
fi

mkdir -p "$ROOT/logs"
nohup setsid env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" bash "$RUNNER" \
  >/dev/null 2>&1 </dev/null &
pid=$!
printf '%s\n' "$pid" >"$PID_FILE"
sleep 3
kill -0 "$pid" 2>/dev/null || {
  echo "ERROR: Blur Restormer v1 exited during launch; inspect $LOG" >&2
  tail -100 "$LOG" >&2 || true
  exit 1
}
echo "FOUNDIR_BLUR_RESTORMER_V1_DETACHED process_group=$pid"
echo "Persistent log: $LOG"
echo "GPU telemetry: $ROOT/logs/foundir_blur_restormer_refinement_v1_gpu.csv"
echo "Stop safely: kill -- -$pid"
