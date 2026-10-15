#!/usr/bin/env bash
set -euo pipefail

ROOT=/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
PYTHON="$ROOT/.venv/flux2-smoke/bin/python"
CONFIG="$ROOT/configs/foundir_haze_flux2_source_diverse_continuation_v1.json"
RUNNER="$ROOT/smoke/run_foundir_haze_flux2_source_diverse_continuation_v1_8gpu.sh"
LOG="$ROOT/logs/foundir_haze_flux2_source_diverse_continuation_v1.log"
PID_FILE="$ROOT/logs/foundir_haze_flux2_source_diverse_continuation_v1.pid"
RESULT="$ROOT/checkpoints/foundir_haze_flux2_source_diverse_continuation_v1/result.json"

cd "$ROOT"
/usr/bin/python3 smoke/advisor_governance.py check
[[ $# -le 1 && ( $# -eq 0 || "$1" == "--check" ) ]] || {
  echo "ERROR: only optional --check is supported" >&2; exit 1;
}
[[ "${CUDA_VISIBLE_DEVICES:-}" == "0,1,2,3,4,5,6,7" ]] || {
  echo "ERROR: export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7" >&2; exit 1;
}
for command in setsid flock sha256sum jq; do command -v "$command" >/dev/null || { echo "ERROR: $command is required" >&2; exit 1; }; done
for path in "$PYTHON" "$CONFIG" "$RUNNER"; do [[ -f "$path" ]] || { echo "ERROR: missing $path" >&2; exit 1; }; done

"$PYTHON" -m py_compile smoke/train_foundir_blur_only_flux2_overfit_ddp.py tests/test_foundir_haze_flux2_source_diverse_continuation.py
bash -n "$RUNNER" "$0"
"$PYTHON" -m pytest -q tests/test_foundir_haze_flux2_source_diverse_continuation.py
"$PYTHON" smoke/train_foundir_blur_only_flux2_overfit_ddp.py --config "$CONFIG" --preflight
if [[ "${1:-}" == "--check" ]]; then
  echo "FOUNDIR_HAZE_CONTINUATION_LAUNCHER_CHECK_PASS"
  exit 0
fi

command -v nvidia-smi >/dev/null || { echo "ERROR: nvidia-smi is required" >&2; exit 1; }
[[ ! -f "$RESULT" ]] || { echo "ERROR: terminal continuation result already exists" >&2; exit 1; }
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
echo "FOUNDIR_HAZE_CONTINUATION_DETACHED process_group=$pid"
echo "Persistent log: $LOG"
echo "Monitor: tail -f $LOG"
echo "Stop safely: kill -- -$pid"
