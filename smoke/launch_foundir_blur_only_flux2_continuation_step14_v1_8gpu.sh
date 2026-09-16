#!/usr/bin/env bash
set -euo pipefail

ROOT=/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
PYTHON="$ROOT/.venv/flux2-smoke/bin/python"
RUNNER="$ROOT/smoke/run_foundir_blur_only_flux2_continuation_step14_v1_8gpu.sh"
LOG="$ROOT/logs/foundir_blur_only_flux2_continuation_step14_v1.log"
PID_FILE="$ROOT/logs/foundir_blur_only_flux2_continuation_step14_v1.pid"

cd "$ROOT"
/usr/bin/python3 smoke/advisor_governance.py check
[[ $# -le 1 && ( $# -eq 0 || "$1" == "--check" ) ]] || {
  echo "ERROR: only optional --check is supported" >&2; exit 1;
}
[[ "${CUDA_VISIBLE_DEVICES:-}" == "0,1,2,3,4,5,6,7" ]] || {
  echo "ERROR: set CUDA_VISIBLE_DEVICES exactly to 0,1,2,3,4,5,6,7" >&2; exit 1;
}
for command in setsid flock sha256sum; do
  command -v "$command" >/dev/null || { echo "ERROR: $command is required" >&2; exit 1; }
done
"$PYTHON" -m py_compile \
  smoke/check_foundir_blur_only_flux2_continuation_step14_gate.py \
  smoke/train_foundir_blur_only_flux2_continuation_ddp.py \
  smoke/evaluate_foundir_blur_only_flux2_continuation_ddp.py
bash -n "$RUNNER" "$0"

if [[ "${1:-}" == "--check" ]]; then
  "$PYTHON" -m pytest -q tests/test_foundir_blur_only_flux2_continuation.py
  echo "FOUNDIR_BLUR_FLUX2_CONTINUATION_STEP14_LAUNCHER_CHECK_PASS"
  exit 0
fi
"$PYTHON" smoke/check_foundir_blur_only_flux2_continuation_step14_gate.py
command -v nvidia-smi >/dev/null || { echo "ERROR: nvidia-smi is required" >&2; exit 1; }

mkdir -p "$ROOT/logs"
nohup setsid env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" bash "$RUNNER" \
  >/dev/null 2>&1 </dev/null &
pid=$!
printf '%s\n' "$pid" >"$PID_FILE"
sleep 3
kill -0 "$pid" 2>/dev/null || {
  echo "ERROR: step-14 continuation exited during startup; inspect $LOG" >&2
  tail -100 "$LOG" >&2 || true
  exit 1
}
echo "FOUNDIR_BLUR_FLUX2_CONTINUATION_STEP14_DETACHED process_group=$pid"
echo "Persistent log: $LOG"
