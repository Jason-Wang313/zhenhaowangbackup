#!/usr/bin/env bash
set -euo pipefail

ROOT=/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
RUNNER="$ROOT/smoke/run_foundir_five_type_flux2_main_v3_8gpu.sh"
LOG="$ROOT/logs/foundir_five_type_flux2_main_v3.log"
PID_FILE="$ROOT/logs/foundir_five_type_flux2_main_v3.pid"
PYTHON="$ROOT/.venv/flux2-smoke/bin/python"

cd "$ROOT"
[[ $# -le 1 && ( $# -eq 0 || "$1" == "--check" ) ]] || {
  echo "ERROR: only the optional --check argument is supported" >&2; exit 1;
}
[[ "${CUDA_VISIBLE_DEVICES:-}" == "0,1,2,3,4,5,6,7" ]] || {
  echo "ERROR: set CUDA_VISIBLE_DEVICES exactly to 0,1,2,3,4,5,6,7" >&2; exit 1;
}

required=(
  agents.md
  configs/foundir_five_type_flux2_dit_v3.json
  configs/foundir_five_type_flux2_dit_v2.json
  lowlight_restoration/flow.py
  lowlight_restoration/flux_prompt_cache.py
  lowlight_restoration/foundir_five_type.py
  lowlight_restoration/foundir_flux2_protocol.py
  lowlight_restoration/lora.py
  lowlight_restoration/pixel_refinement.py
  lowlight_restoration/qwen_lora.py
  reports/foundir_five_type_flux2_main_v3_readiness_2026-09-07.md
  reports/foundir_five_type_flux2_sow_metric_correction_2026-09-07.md
  smoke/build_foundir_five_type_physical_v8_conditions.py
  smoke/export_foundir_five_type_physical_v8_conditions_ddp.py
  smoke/foundir_five_type_preflight.py
  smoke/foundir_five_type_runtime_preflight.py
  smoke/run_foundir_lowlight_gate2_teachers.py
  smoke/build_foundir_flux_qwen3_prompt_cache_ddp.py
  smoke/recheck_foundir_five_type_flux2_overfit_sow_ddp.py
  smoke/train_foundir_five_type_flux2_overfit_ddp.py
  smoke/train_foundir_five_type_flux2_full_ddp.py
  smoke/preflight_foundir_flux2_training_geometry.py
  smoke/train_qwen_multidegradation_lora_ddp.py
  smoke/evaluate_foundir_five_type_flux2_ddp.py
  smoke/evaluate_foundir_five_type_flux2_tiled_ddp.py
  smoke/evaluate_foundir_five_type_released_baselines_ddp.py
  smoke/freeze_foundir_five_type_evaluation_contract.py
  smoke/select_foundir_five_type_flux2_checkpoint.py
  smoke/freeze_foundir_five_type_flux2_main_v3_report.py
  smoke/run_foundir_five_type_stage0_v3_8gpu.sh
  smoke/run_foundir_five_type_flux2_main_v3_8gpu.sh
  smoke/launch_foundir_five_type_flux2_main_v3_8gpu.sh
  tests/test_foundir_flux2_protocol.py
  tests/test_foundir_five_type_main_v3.py
  tests/test_foundir_lowlight_gate2_teachers.py
  tests/test_pixel_refinement.py
)
python_files=(
  lowlight_restoration/foundir_flux2_protocol.py
  lowlight_restoration/lora.py
  lowlight_restoration/pixel_refinement.py
  smoke/build_foundir_five_type_physical_v8_conditions.py
  smoke/export_foundir_five_type_physical_v8_conditions_ddp.py
  smoke/foundir_five_type_preflight.py
  smoke/foundir_five_type_runtime_preflight.py
  smoke/run_foundir_lowlight_gate2_teachers.py
  smoke/build_foundir_flux_qwen3_prompt_cache_ddp.py
  smoke/recheck_foundir_five_type_flux2_overfit_sow_ddp.py
  smoke/train_foundir_five_type_flux2_overfit_ddp.py
  smoke/train_foundir_five_type_flux2_full_ddp.py
  smoke/preflight_foundir_flux2_training_geometry.py
  smoke/evaluate_foundir_five_type_flux2_ddp.py
  smoke/evaluate_foundir_five_type_flux2_tiled_ddp.py
  smoke/evaluate_foundir_five_type_released_baselines_ddp.py
  smoke/freeze_foundir_five_type_flux2_main_v3_report.py
  tests/test_foundir_flux2_protocol.py
  tests/test_foundir_five_type_main_v3.py
  tests/test_foundir_lowlight_gate2_teachers.py
  tests/test_pixel_refinement.py
)
"$PYTHON" -m py_compile "${python_files[@]}"
bash -n "$RUNNER" "$ROOT/smoke/run_foundir_five_type_stage0_v3_8gpu.sh" "$0"
"$PYTHON" smoke/foundir_five_type_preflight.py \
  --config configs/foundir_five_type_flux2_dit_v3.json --skip-large-hashes
[[ -f "$ROOT/data/foundir/_extract_stage/five_type_v1/state/install.complete" ]] || {
  echo "ERROR: verified five-type FoundIR training install marker is missing" >&2; exit 1;
}
"$ROOT/.conda/gpu-train-stable/bin/python" \
  smoke/build_foundir_five_type_physical_v8_conditions.py \
  --config configs/foundir_five_type_flux2_dit_v3.json
"$PYTHON" -m unittest \
  tests.test_foundir_flux2_protocol \
  tests.test_foundir_five_type_main_v3 \
  tests.test_foundir_lowlight_gate2_teachers \
  tests.test_pixel_refinement

for relative in "${required[@]}"; do
  git ls-files --error-unmatch "$relative" >/dev/null 2>&1 || {
    echo "ERROR: required v3 file is not committed: $relative" >&2; exit 1;
  }
  git diff --quiet HEAD -- "$relative" || {
    echo "ERROR: required v3 file differs from committed revision: $relative" >&2; exit 1;
  }
done

if [[ "${1:-}" == "--check" ]]; then
  echo "FOUNDIR_FIVE_TYPE_MAIN_V3_LAUNCHER_CHECK_PASS"
  exit 0
fi

mkdir -p "$ROOT/logs"
if [[ -f "$PID_FILE" ]]; then
  old_pid=$(tr -dc '0-9' <"$PID_FILE")
  if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "ERROR: v3 full pipeline is already active as PID $old_pid" >&2
    exit 1
  fi
fi
nohup env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" bash "$RUNNER" >/dev/null 2>&1 </dev/null &
pid=$!
printf '%s\n' "$pid" >"$PID_FILE"
sleep 3
if ! kill -0 "$pid" 2>/dev/null; then
  echo "ERROR: v3 pipeline exited during launch; inspect $LOG" >&2
  tail -100 "$LOG" >&2 || true
  exit 1
fi
echo "FOUNDIR_FIVE_TYPE_MAIN_V3_DETACHED pid=$pid"
echo "Persistent log: $LOG"
echo "Monitor: tail -f $LOG"
