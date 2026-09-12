#!/usr/bin/env bash
set -euo pipefail

ROOT=/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
RUNNER="$ROOT/smoke/run_foundir_blur_only_flux2_full_v1_8gpu.sh"
CONFIG="$ROOT/configs/foundir_blur_only_flux2_full_v1.json"
PYTHON="$ROOT/.venv/flux2-smoke/bin/python"
LOG="$ROOT/logs/foundir_blur_only_flux2_full_v1.log"
PID_FILE="$ROOT/logs/foundir_blur_only_flux2_full_v1.pid"

cd "$ROOT"
/usr/bin/python3 "$ROOT/smoke/advisor_governance.py" check
[[ $# -le 1 && ( $# -eq 0 || "$1" == "--check" ) ]] || {
  echo "ERROR: only optional --check is supported" >&2; exit 1;
}
[[ "${CUDA_VISIBLE_DEVICES:-}" == "0,1,2,3,4,5,6,7" ]] || {
  echo "ERROR: set CUDA_VISIBLE_DEVICES exactly to 0,1,2,3,4,5,6,7" >&2; exit 1;
}
[[ -x "$PYTHON" ]] || { echo "ERROR: missing $PYTHON" >&2; exit 1; }

"$PYTHON" -m py_compile smoke/train_foundir_blur_only_flux2_full_ddp.py \
  tests/test_foundir_blur_only_flux2_full.py
bash -n "$RUNNER" "$0"

if [[ "${1:-}" == "--check" ]]; then
  "$PYTHON" -m pytest -q tests/test_foundir_blur_only_flux2_full.py
  "$PYTHON" smoke/train_foundir_blur_only_flux2_full_ddp.py --config "$CONFIG" --preflight
  echo "FOUNDIR_BLUR_ONLY_FULL_LAUNCHER_CHECK_PASS"
  exit 0
fi

mkdir -p "$ROOT/logs"
# PID namespaces differ across GPU instances, so a shared stale PID must never
# be used as a cross-node duplicate detector. The runner's shared flock is the
# authoritative single-run guard.
nohup env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" bash "$RUNNER" >/dev/null 2>&1 </dev/null &
pid=$!
printf '%s\n' "$pid" >"$PID_FILE"
sleep 3
if ! kill -0 "$pid" 2>/dev/null; then
  echo "ERROR: launcher exited during startup; inspect $LOG" >&2
  tail -100 "$LOG" >&2 || true
  exit 1
fi
echo "FOUNDIR_BLUR_ONLY_FULL_DETACHED pid=$pid"
echo "Persistent log: $LOG"
echo "Monitor: tail -f $LOG"
