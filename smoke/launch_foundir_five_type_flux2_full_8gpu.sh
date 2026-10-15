#!/usr/bin/env bash
set -euo pipefail

ROOT=/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
/usr/bin/python3 "$ROOT/smoke/advisor_governance.py" check
RUNNER="$ROOT/smoke/run_foundir_five_type_flux2_full_8gpu.sh"
LOG="$ROOT/logs/foundir_five_type_flux2_full_v1.log"
PID_FILE="$ROOT/logs/foundir_five_type_flux2_full_v1.pid"
PYTHON="$ROOT/.venv/flux2-smoke/bin/python"

cd "$ROOT"
if [[ "${CUDA_VISIBLE_DEVICES:-}" != "0,1,2,3,4,5,6,7" ]]; then
  echo "ERROR: set CUDA_VISIBLE_DEVICES exactly to 0,1,2,3,4,5,6,7" >&2
  exit 1
fi

required=(
  configs/foundir_five_type_flux2_dit_v1.json
  configs/foundir_five_type_prelaunch_gate_v1.json
  lowlight_restoration/flux_prompt_cache.py
  lowlight_restoration/foundir_five_type.py
  lowlight_restoration/foundir_flux2_protocol.py
  smoke/build_foundir_flux_qwen3_prompt_cache_ddp.py
  smoke/evaluate_foundir_five_type_flux2_ddp.py
  smoke/evaluate_foundir_five_type_flux2_tiled_ddp.py
  smoke/evaluate_foundir_five_type_released_baselines_ddp.py
  smoke/export_foundir_five_type_qwen_conditions_sharded_ddp.py
  smoke/export_foundir_five_type_qwen_per_type_pilot.py
  smoke/evaluate_foundir_five_type_qwen_training_gate.py
  smoke/freeze_foundir_five_type_prelaunch.py
  smoke/foundir_five_type_distributed_canary.py
  smoke/foundir_five_type_preflight.py
  smoke/foundir_five_type_runtime_preflight.py
  smoke/freeze_foundir_five_type_evaluation_contract.py
  smoke/freeze_foundir_five_type_flux2_final_report.py
  smoke/select_foundir_five_type_flux2_checkpoint.py
  smoke/train_foundir_five_type_flux2_full_ddp.py
  smoke/run_foundir_five_type_flux2_full_8gpu.sh
  smoke/run_foundir_five_type_prelaunch_8gpu.sh
  smoke/launch_foundir_five_type_flux2_full_8gpu.sh
  smoke/launch_foundir_five_type_prelaunch_8gpu.sh
  tests/test_flux_prompt_cache.py
  tests/test_foundir_five_type.py
  tests/test_foundir_flux2_protocol.py
  tests/test_foundir_five_type_prelaunch.py
)
python_files=(
  lowlight_restoration/flux_prompt_cache.py
  lowlight_restoration/foundir_five_type.py
  lowlight_restoration/foundir_flux2_protocol.py
  smoke/build_foundir_flux_qwen3_prompt_cache_ddp.py
  smoke/evaluate_foundir_five_type_flux2_ddp.py
  smoke/evaluate_foundir_five_type_flux2_tiled_ddp.py
  smoke/evaluate_foundir_five_type_released_baselines_ddp.py
  smoke/export_foundir_five_type_qwen_conditions_sharded_ddp.py
  smoke/export_foundir_five_type_qwen_per_type_pilot.py
  smoke/evaluate_foundir_five_type_qwen_training_gate.py
  smoke/freeze_foundir_five_type_prelaunch.py
  smoke/foundir_five_type_distributed_canary.py
  smoke/foundir_five_type_preflight.py
  smoke/foundir_five_type_runtime_preflight.py
  smoke/freeze_foundir_five_type_evaluation_contract.py
  smoke/freeze_foundir_five_type_flux2_final_report.py
  smoke/select_foundir_five_type_flux2_checkpoint.py
  smoke/train_foundir_five_type_flux2_full_ddp.py
)
"$PYTHON" -m py_compile "${python_files[@]}"
bash -n "$RUNNER" "$0"
export TORCH_HOME="$ROOT/models/torch"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR="$ROOT/.cache/triton/foundir_five_type"
export TORCH_EXTENSIONS_DIR="$ROOT/.cache/torch_extensions/foundir_five_type"
mkdir -p "$TRITON_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"
"$PYTHON" smoke/freeze_foundir_five_type_prelaunch.py \
  --prelaunch-config configs/foundir_five_type_prelaunch_gate_v1.json \
  --verify-existing --require-selected joint
for relative in "${required[@]}"; do
  git ls-files --error-unmatch "$relative" >/dev/null 2>&1 || {
    echo "ERROR: required full-pipeline file is not committed: $relative" >&2; exit 1;
  }
  git diff --quiet HEAD -- "$relative" || {
    echo "ERROR: required full-pipeline file differs from committed revision: $relative" >&2; exit 1;
  }
done

mkdir -p "$ROOT/logs"
if [[ -f "$PID_FILE" ]]; then
  old_pid=$(tr -dc '0-9' <"$PID_FILE")
  if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "ERROR: full pipeline is already active as PID $old_pid" >&2
    exit 1
  fi
fi
nohup env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" bash "$RUNNER" >/dev/null 2>&1 </dev/null &
pid=$!
printf '%s\n' "$pid" >"$PID_FILE"
sleep 3
if ! kill -0 "$pid" 2>/dev/null; then
  echo "ERROR: full pipeline exited during launch; inspect $LOG" >&2
  tail -100 "$LOG" >&2 || true
  exit 1
fi
echo "FOUNDIR_FIVE_TYPE_FULL_DETACHED pid=$pid"
echo "Persistent log: $LOG"
echo "Monitor: tail -f $LOG"
