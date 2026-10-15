#!/usr/bin/env bash
set -euo pipefail

ROOT=/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
/usr/bin/python3 "$ROOT/smoke/advisor_governance.py" check
RUNNER="$ROOT/smoke/run_foundir_five_type_stage0_8gpu.sh"
LOG="$ROOT/logs/foundir_five_type_flux2_dit_v1.log"
PID_FILE="$ROOT/logs/foundir_five_type_flux2_dit_v1.pid"
PYTHON="$ROOT/.conda/gpu-train-stable/bin/python"

cd "$ROOT"
if [[ "${CUDA_VISIBLE_DEVICES:-}" != "0,1,2,3,4,5,6,7" ]]; then
  echo "ERROR: set CUDA_VISIBLE_DEVICES exactly to 0,1,2,3,4,5,6,7" >&2
  exit 1
fi

"$PYTHON" -m py_compile \
  lowlight_restoration/foundir_five_type.py lowlight_restoration/flow.py \
  smoke/build_foundir_five_type_manifest.py smoke/foundir_five_type_preflight.py \
  smoke/foundir_five_type_runtime_preflight.py \
  smoke/freeze_foundir_five_type_prelaunch.py \
  smoke/export_foundir_five_type_qwen_conditions.py \
  smoke/evaluate_foundir_five_type_vae_ceiling_ddp.py \
  smoke/train_foundir_five_type_flux2_overfit_ddp.py
bash -n "$RUNNER" "$0"
"$PYTHON" smoke/freeze_foundir_five_type_prelaunch.py \
  --prelaunch-config configs/foundir_five_type_prelaunch_gate_v1.json \
  --verify-existing --require-selected joint

required=(
  configs/foundir_five_type_flux2_dit_v1.json
  configs/foundir_five_type_prelaunch_gate_v1.json
  lowlight_restoration/foundir_five_type.py
  lowlight_restoration/flow.py
  smoke/build_foundir_five_type_manifest.py
  smoke/foundir_five_type_preflight.py
  smoke/foundir_five_type_runtime_preflight.py
  smoke/freeze_foundir_five_type_prelaunch.py
  smoke/export_foundir_five_type_qwen_conditions.py
  smoke/evaluate_foundir_five_type_vae_ceiling_ddp.py
  smoke/train_foundir_five_type_flux2_overfit_ddp.py
  smoke/run_foundir_five_type_stage0_8gpu.sh
  smoke/launch_foundir_five_type_stage0_8gpu.sh
  tests/test_foundir_five_type.py
  tests/test_lowlight_restoration.py
)
for relative in "${required[@]}"; do
  git ls-files --error-unmatch "$relative" >/dev/null 2>&1 || {
    echo "ERROR: required stage-0 file is not committed: $relative" >&2; exit 1;
  }
  git diff --quiet HEAD -- "$relative" || {
    echo "ERROR: required stage-0 file differs from committed revision: $relative" >&2; exit 1;
  }
done

mkdir -p "$ROOT/logs"
if [[ -f "$PID_FILE" ]]; then
  old_pid=$(tr -dc '0-9' <"$PID_FILE")
  if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "ERROR: stage-0 is already active as PID $old_pid" >&2
    exit 1
  fi
fi
nohup env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" bash "$RUNNER" >/dev/null 2>&1 </dev/null &
pid=$!
printf '%s\n' "$pid" >"$PID_FILE"
sleep 2
if ! kill -0 "$pid" 2>/dev/null; then
  echo "ERROR: stage-0 exited during launch; inspect $LOG" >&2
  tail -80 "$LOG" >&2 || true
  exit 1
fi
echo "FOUNDIR_FIVE_TYPE_STAGE0_DETACHED pid=$pid"
echo "Persistent log: $LOG"
echo "Monitor: tail -f $LOG"
