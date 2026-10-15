#!/usr/bin/env bash
set -euo pipefail
ROOT=/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
PYTHON="$ROOT/.venv/flux2-smoke/bin/python"
CONFIG="$ROOT/configs/foundir_noise_family_integrated_development_v1.json"
SCRIPT="$ROOT/smoke/evaluate_foundir_mixed_restoration_pilot_ddp.py"
BUILDER="$ROOT/smoke/build_foundir_noise_family_integrated_development_v1.py"
RUNNER="$ROOT/smoke/run_foundir_noise_family_integrated_development_v1_8gpu.sh"
LOG="$ROOT/logs/foundir_noise_family_integrated_development_v1.log"
PID_FILE="$ROOT/logs/foundir_noise_family_integrated_development_v1.pid"
RESULT="$ROOT/checkpoints/foundir_noise_family_integrated_development_v1/result.json"
cd "$ROOT"
/usr/bin/python3 smoke/advisor_governance.py check
[[ $# -le 1 && ( $# -eq 0 || "$1" == "--check" ) ]] || { echo "ERROR: only --check supported" >&2; exit 1; }
"$PYTHON" -m py_compile "$SCRIPT" "$BUILDER" lowlight_restoration/noise_family_fail_closed_router.py
bash -n "$RUNNER" "$0"
PYTHONPATH=. "$PYTHON" -m pytest -q tests/test_noise_family_fail_closed_router.py tests/test_foundir_noise_family_integrated_development.py
"$PYTHON" "$BUILDER" >/dev/null
"$PYTHON" "$SCRIPT" --config "$CONFIG" --preflight
echo "TOTAL END-TO-END TIME: H200 25-40 min (3.3-5.3 GPU-hours); H100 45-60 min (6-8 GPU-hours)."
if [[ "${1:-}" == "--check" ]]; then echo "NOISE_FAMILY_INTEGRATED_LAUNCHER_CHECK_PASS"; exit 0; fi
[[ "${CUDA_VISIBLE_DEVICES:-}" == "0,1,2,3,4,5,6,7" ]] || { echo "ERROR: export eight visible GPUs" >&2; exit 1; }
command -v nvidia-smi >/dev/null || { echo "ERROR: nvidia-smi required" >&2; exit 1; }
[[ ! -f "$RESULT" ]] || { echo "ERROR: terminal result already exists" >&2; exit 1; }
mkdir -p "$ROOT/logs"
nohup setsid env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" bash "$RUNNER" >/dev/null 2>&1 </dev/null &
pid=$!
printf '%s\n' "$pid" >"$PID_FILE"
sleep 3
kill -0 "$pid" 2>/dev/null || { echo "ERROR: startup failed; inspect $LOG" >&2; tail -100 "$LOG" >&2 || true; exit 1; }
echo "NOISE_FAMILY_INTEGRATED_DETACHED process_group=$pid"
echo "Persistent log: $LOG"
echo "Monitor: tail -f $LOG"
