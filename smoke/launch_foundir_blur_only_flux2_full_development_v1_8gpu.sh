#!/usr/bin/env bash
set -euo pipefail

ROOT=/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
PYTHON="$ROOT/.venv/flux2-smoke/bin/python"
CONFIG="$ROOT/configs/foundir_blur_only_flux2_full_development_v1.json"
EVALUATOR="$ROOT/smoke/evaluate_foundir_blur_only_flux2_full_development_ddp.py"
RUNNER="$ROOT/smoke/run_foundir_blur_only_flux2_full_development_v1_8gpu.sh"
LOG="$ROOT/logs/foundir_blur_only_flux2_full_development_v1.log"
GPU_LOG="$ROOT/logs/foundir_blur_only_flux2_full_development_v1_gpu.csv"
PID_FILE="$ROOT/logs/foundir_blur_only_flux2_full_development_v1.pid"

cd "$ROOT"
/usr/bin/python3 "$ROOT/smoke/advisor_governance.py" check
[[ $# -le 1 && ( $# -eq 0 || "$1" == "--check" ) ]] || {
  echo "ERROR: only optional --check is supported" >&2
  exit 1
}
[[ "${CUDA_VISIBLE_DEVICES:-}" == "0,1,2,3,4,5,6,7" ]] || {
  echo "ERROR: expose exactly CUDA devices 0,1,2,3,4,5,6,7" >&2
  exit 1
}
for command in setsid flock sha256sum; do
  command -v "$command" >/dev/null || { echo "ERROR: $command is required" >&2; exit 1; }
done
for path in "$PYTHON" "$CONFIG" "$EVALUATOR" "$RUNNER"; do
  [[ -f "$path" ]] || { echo "ERROR: missing dependency $path" >&2; exit 1; }
done

"$PYTHON" -m py_compile "$EVALUATOR"
bash -n "$RUNNER" "$0"
"$PYTHON" -u "$EVALUATOR" --config "$CONFIG" --preflight

if [[ "${1:-}" == "--check" ]]; then
  echo "FOUNDIR_BLUR_ONLY_FULL_DEVELOPMENT_V1_LAUNCHER_CHECK_PASS"
  exit 0
fi
command -v nvidia-smi >/dev/null || { echo "ERROR: nvidia-smi is required" >&2; exit 1; }

mkdir -p "$ROOT/logs"
nohup setsid env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" bash "$RUNNER" \
  >/dev/null 2>&1 </dev/null &
pid=$!
printf '%s\n' "$pid" >"$PID_FILE"
sleep 3
kill -0 "$pid" 2>/dev/null || {
  echo "ERROR: Blur-only full-development evaluation exited during launch; inspect $LOG" >&2
  tail -100 "$LOG" >&2 || true
  exit 1
}
echo "FOUNDIR_BLUR_ONLY_FULL_DEVELOPMENT_V1_DETACHED process_group=$pid"
echo "Persistent log: $LOG"
echo "GPU telemetry: $GPU_LOG"
echo "Stop safely: kill -- -$pid"
