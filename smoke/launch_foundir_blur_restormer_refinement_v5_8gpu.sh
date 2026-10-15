#!/usr/bin/env bash
set -euo pipefail

ROOT=/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
/usr/bin/python3 "$ROOT/smoke/advisor_governance.py" check
RUNNER="$ROOT/smoke/run_foundir_blur_restormer_refinement_v5_8gpu.sh"
TRAINER="$ROOT/smoke/train_foundir_blur_restormer_refinement_ddp.py"
FUSION="$ROOT/lowlight_restoration/blur_restormer_refinement.py"
CONFIG="$ROOT/configs/foundir_blur_restormer_refinement_gate_v5.json"
CHECKPOINT="$ROOT/models/restormer-motion-deblur/hf-pinned/model.safetensors"
LOG="$ROOT/logs/foundir_blur_restormer_refinement_v5.log"
PID_FILE="$ROOT/logs/foundir_blur_restormer_refinement_v5.pid"
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
[[ -x "$PYTHON" && -f "$RUNNER" && -f "$TRAINER" && -f "$FUSION" && -f "$CONFIG" && -f "$CHECKPOINT" ]] || {
  echo "ERROR: Blur Restormer v5 launcher dependency is missing" >&2
  exit 1
}

[[ "$(stat -c %s "$CHECKPOINT")" == "104558296" ]] || {
  echo "ERROR: motion-deblurring checkpoint size changed" >&2
  exit 1
}
[[ "$(sha256sum "$CHECKPOINT" | cut -d' ' -f1)" == "c465f48759cf155253c2c32ba4d9dd54c930fccda9b1aac1f85e2568ce65b65c" ]] || {
  echo "ERROR: motion-deblurring checkpoint SHA-256 changed" >&2
  exit 1
}

"$PYTHON" -m py_compile "$TRAINER" "$FUSION"
bash -n "$RUNNER" "$0"
"$PYTHON" "$TRAINER" --config "$CONFIG" --preflight

if [[ "${1:-}" == "--check" ]]; then
  echo "FOUNDIR_BLUR_RESTORMER_V5_LAUNCHER_CHECK_PASS"
  exit 0
fi

mkdir -p "$ROOT/logs"
nohup setsid env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" bash "$RUNNER" \
  >/dev/null 2>&1 </dev/null &
pid=$!
printf '%s\n' "$pid" >"$PID_FILE"
sleep 3
kill -0 "$pid" 2>/dev/null || {
  echo "ERROR: Blur Restormer v5 exited during launch; inspect $LOG" >&2
  tail -100 "$LOG" >&2 || true
  exit 1
}
echo "FOUNDIR_BLUR_RESTORMER_V5_DETACHED process_group=$pid"
echo "Persistent log: $LOG"
echo "GPU telemetry: $ROOT/logs/foundir_blur_restormer_refinement_v5_gpu.csv"
echo "Stop safely: kill -- -$pid"
