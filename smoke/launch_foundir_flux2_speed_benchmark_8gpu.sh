#!/usr/bin/env bash
set -euo pipefail

ROOT=/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
RUNNER="$ROOT/smoke/run_foundir_flux2_speed_benchmark_8gpu.sh"
LOG="$ROOT/logs/foundir_flux2_speed_benchmark.log"
PID_FILE="$ROOT/logs/foundir_flux2_speed_benchmark.pid"
PYTHON="$ROOT/.venv/flux2-smoke/bin/python"

cd "$ROOT"
[[ $# -le 1 && ( $# -eq 0 || "$1" == "--check" ) ]] || {
  echo "ERROR: only the optional --check argument is supported" >&2; exit 1;
}
[[ "${CUDA_VISIBLE_DEVICES:-}" == "0,1,2,3,4,5,6,7" ]] || {
  echo "ERROR: set CUDA_VISIBLE_DEVICES exactly to 0,1,2,3,4,5,6,7" >&2; exit 1;
}

python_files=(
  smoke/train_foundir_five_type_flux2_full_ddp.py
  smoke/preflight_foundir_flux2_training_geometry.py
  smoke/select_foundir_flux2_speed_benchmark.py
)
"$PYTHON" -m py_compile "${python_files[@]}"
bash -n "$RUNNER" "$0"
"$PYTHON" -m unittest tests.test_foundir_flux2_speed_benchmark

for relative in "${python_files[@]}" \
  smoke/run_foundir_flux2_speed_benchmark_8gpu.sh \
  smoke/launch_foundir_flux2_speed_benchmark_8gpu.sh \
  tests/test_foundir_flux2_speed_benchmark.py; do
  git ls-files --error-unmatch "$relative" >/dev/null 2>&1 || {
    echo "ERROR: required benchmark file is not committed: $relative" >&2; exit 1;
  }
  git diff --quiet HEAD -- "$relative" || {
    echo "ERROR: required benchmark file differs from committed revision: $relative" >&2; exit 1;
  }
done

if [[ "${1:-}" == "--check" ]]; then
  echo "FOUNDIR_FLUX_SPEED_BENCHMARK_LAUNCHER_CHECK_PASS"
  exit 0
fi

mkdir -p "$ROOT/logs"
if [[ -f "$PID_FILE" ]]; then
  old_pid=$(tr -dc '0-9' <"$PID_FILE")
  if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "ERROR: FLUX speed benchmark is already active as PID $old_pid" >&2
    exit 1
  fi
fi
nohup env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" bash "$RUNNER" >/dev/null 2>&1 </dev/null &
pid=$!
printf '%s\n' "$pid" >"$PID_FILE"
sleep 3
if ! kill -0 "$pid" 2>/dev/null; then
  echo "ERROR: speed benchmark exited during launch; inspect $LOG" >&2
  tail -100 "$LOG" >&2 || true
  exit 1
fi
echo "FOUNDIR_FLUX_SPEED_BENCHMARK_DETACHED pid=$pid"
echo "Persistent log: $LOG"
echo "Monitor: tail -f $LOG"
