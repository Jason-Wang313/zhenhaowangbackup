#!/usr/bin/env bash
set -euo pipefail

ROOT=/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
/usr/bin/python3 "$ROOT/smoke/advisor_governance.py" check
RUNNER="$ROOT/smoke/run_foundir_five_type_native_patch_diagnostic_8gpu.sh"
LOG="$ROOT/logs/foundir_five_type_native_patch_diagnostic_v1.log"
PID_FILE="$ROOT/logs/foundir_five_type_native_patch_diagnostic_v1.pid"
PYTHON="$ROOT/.conda/gpu-train-stable/bin/python"

check_only=false
if [[ "${1:-}" == "--check" ]]; then
  [[ "$#" -eq 1 ]] || { echo "ERROR: --check takes no additional arguments" >&2; exit 1; }
  check_only=true
elif [[ "$#" -ne 0 ]]; then
  echo "ERROR: unexpected launcher argument: $1" >&2
  exit 1
fi

cd "$ROOT"
if [[ "${CUDA_VISIBLE_DEVICES:-}" != "0,1,2,3,4,5,6,7" ]]; then
  echo "ERROR: set CUDA_VISIBLE_DEVICES exactly to 0,1,2,3,4,5,6,7" >&2
  exit 1
fi

required=(
  configs/foundir_five_type_native_patch_diagnostic_v1.json
  configs/foundir_five_type_prelaunch_gate_v1.json
  smoke/evaluate_foundir_five_type_qwen_native_patches_ddp.py
  smoke/run_foundir_five_type_native_patch_diagnostic_8gpu.sh
  smoke/launch_foundir_five_type_native_patch_diagnostic_8gpu.sh
  smoke/train_qwen_multidegradation_lora_ddp.py
)
"$PYTHON" -m py_compile smoke/evaluate_foundir_five_type_qwen_native_patches_ddp.py \
  smoke/train_qwen_multidegradation_lora_ddp.py
bash -n "$RUNNER" "$0"
for relative in "${required[@]}"; do
  git ls-files --error-unmatch "$relative" >/dev/null 2>&1 || {
    echo "ERROR: required diagnostic file is not committed: $relative" >&2
    exit 1
  }
  git diff --quiet HEAD -- "$relative" || {
    echo "ERROR: required diagnostic file differs from committed revision: $relative" >&2
    exit 1
  }
done

if [[ "$check_only" == true ]]; then
  echo "FOUNDIR_NATIVE_PATCH_LAUNCHER_CHECK_PASS"
  exit 0
fi

mkdir -p "$ROOT/logs"
if [[ -f "$PID_FILE" ]]; then
  old_pid=$(tr -dc '0-9' <"$PID_FILE")
  if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
    old_command=$(tr '\0' ' ' <"/proc/$old_pid/cmdline" 2>/dev/null || true)
    if [[ "$old_command" == *"$RUNNER"* ]]; then
      echo "ERROR: native-patch diagnostic is already active as PID $old_pid" >&2
      exit 1
    fi
  fi
fi
nohup env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" bash "$RUNNER" \
  >/dev/null 2>&1 </dev/null &
pid=$!
printf '%s\n' "$pid" >"$PID_FILE"
sleep 3
if ! kill -0 "$pid" 2>/dev/null; then
  echo "ERROR: diagnostic exited during launch; inspect $LOG" >&2
  tail -100 "$LOG" >&2 || true
  exit 1
fi
echo "FOUNDIR_NATIVE_PATCH_DETACHED pid=$pid"
echo "Persistent log: $LOG"
echo "Monitor: tail -f $LOG"
