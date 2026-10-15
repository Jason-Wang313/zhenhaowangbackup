#!/usr/bin/env bash
set -euo pipefail
ROOT=/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
/usr/bin/python3 "$ROOT/smoke/advisor_governance.py" check
RUNNER="$ROOT/smoke/run_foundir_blur_generalist_specialization_v2_8gpu.sh"
LOG="$ROOT/logs/foundir_blur_generalist_specialization_v2.log"
PID_FILE="$ROOT/logs/foundir_blur_generalist_specialization_v2.pid"
PYTHON="$ROOT/.venv/flux2-smoke/bin/python"
cd "$ROOT"
[[ $# -le 1 && ( $# -eq 0 || "$1" == "--check" ) ]] || { echo "ERROR: only optional --check is supported" >&2; exit 1; }
[[ "${CUDA_VISIBLE_DEVICES:-}" == "0,1,2,3,4,5,6,7" ]] || { echo "ERROR: expose exactly eight GPUs" >&2; exit 1; }
"$PYTHON" -m py_compile smoke/train_foundir_blur_generalist_specialization_ddp.py
bash -n "$RUNNER" "$0"
[[ -f third_party/foundir/premodel/model-2000.pt ]] || { echo "ERROR: pretrained backbone is missing" >&2; exit 1; }
if [[ "${1:-}" == "--check" ]]; then echo "FOUNDIR_BLUR_SPECIALIZATION_V2_LAUNCHER_CHECK_PASS"; exit 0; fi
mkdir -p "$ROOT/logs"
nohup setsid env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" bash "$RUNNER" >/dev/null 2>&1 </dev/null &
pid=$!
printf '%s\n' "$pid" >"$PID_FILE"
sleep 3
kill -0 "$pid" 2>/dev/null || { echo "ERROR: v2 specialization exited; inspect $LOG" >&2; tail -100 "$LOG" >&2 || true; exit 1; }
echo "FOUNDIR_BLUR_SPECIALIZATION_V2_DETACHED process_group=$pid"
echo "Persistent log: $LOG"
echo "Stop safely: kill -- -$pid"
