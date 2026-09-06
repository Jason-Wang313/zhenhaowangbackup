#!/usr/bin/env bash
set -euo pipefail

ROOT=/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
CONFIG="$ROOT/configs/foundir_five_type_flux2_dit_v2.json"
QWEN_PYTHON="$ROOT/.conda/gpu-train-stable/bin/python"
FLUX_PYTHON="$ROOT/.venv/flux2-smoke/bin/python"
OUTPUT="$ROOT/checkpoints/foundir_five_type_flux2_dit_v2"
MANIFEST="$ROOT/manifests/foundir_five_type_pairs_v1.csv"
CONDITIONS="$ROOT/manifests/foundir_five_type_all_qwen_conditions_v2.csv"
CONDITIONS_JSON="$ROOT/manifests/foundir_five_type_all_qwen_conditions_v2.json"
PROMPT_CACHE="$OUTPUT/qwen3_prompt_cache"
CONTRACT_CSV="$ROOT/manifests/foundir_five_type_evaluation_contract_v2.csv"
CONTRACT_JSON="$ROOT/manifests/foundir_five_type_evaluation_contract_v2.json"
SELECTION="$OUTPUT/checkpoint_selection.json"
LOG="$ROOT/logs/foundir_five_type_flux2_main_v2.log"
GPU_LOG="$ROOT/logs/foundir_five_type_flux2_main_v2_gpu.csv"
LOCK="$ROOT/logs/foundir_five_type_flux2_main_v2.lock"

cd "$ROOT"
mkdir -p "$ROOT/logs" "$OUTPUT" "$ROOT/.cache/triton/foundir_five_type_v2" \
  "$ROOT/.cache/torch_extensions/foundir_five_type_v2"
exec > >(tee -a "$LOG") 2>&1

telemetry_pid=
finish_run() {
  status=$?
  if [[ -n "$telemetry_pid" ]]; then
    kill "$telemetry_pid" 2>/dev/null || true
    wait "$telemetry_pid" 2>/dev/null || true
  fi
  echo "FOUNDIR_FIVE_TYPE_MAIN_V2_END status=$status utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
trap finish_run EXIT

exec 9>>"$LOCK"
flock -n 9 || { echo "ERROR: another five-type v2 pipeline holds $LOCK" >&2; exit 1; }
[[ "${CUDA_VISIBLE_DEVICES:-}" == "0,1,2,3,4,5,6,7" ]] || {
  echo "ERROR: set CUDA_VISIBLE_DEVICES exactly to 0,1,2,3,4,5,6,7" >&2; exit 1;
}
[[ -x "$QWEN_PYTHON" && -x "$FLUX_PYTHON" ]] || {
  echo "ERROR: workspace Python environments are missing" >&2; exit 1;
}

export TORCH_HOME="$ROOT/models/torch"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR="$ROOT/.cache/triton/foundir_five_type_v2"
export TORCH_EXTENSIONS_DIR="$ROOT/.cache/torch_extensions/foundir_five_type_v2"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

echo "FOUNDIR_FIVE_TYPE_MAIN_V2_START utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "FOUNDIR_FIVE_TYPE_MAIN_V2_SCOPE blur,noise,rain,haze,low_light no_jpeg no_condition_ablation"
echo "FOUNDIR_FIVE_TYPE_MAIN_V2_PROTECTED_OFFICIAL_TEST not_accessed"

bash smoke/run_foundir_five_type_stage0_v2_8gpu.sh

nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw \
  --format=csv -l 5 >"$GPU_LOG" 2>&1 &
telemetry_pid=$!
sleep 1
kill -0 "$telemetry_pid" 2>/dev/null || {
  echo "ERROR: nvidia-smi telemetry failed to stay active" >&2; exit 1;
}

if [[ -f "$CONDITIONS" && -f "$CONDITIONS_JSON" ]]; then
  "$QWEN_PYTHON" smoke/export_foundir_five_type_physical_v8_conditions_ddp.py \
    --config "$CONFIG" --manifest "$MANIFEST" --verify-existing
else
  "$QWEN_PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 \
    smoke/export_foundir_five_type_physical_v8_conditions_ddp.py \
    --config "$CONFIG" --manifest "$MANIFEST"
fi

cache_args=(--config "$CONFIG" --conditions "$CONDITIONS" --output "$PROMPT_CACHE")
if [[ -f "$PROMPT_CACHE/index.json" ]]; then
  "$FLUX_PYTHON" smoke/build_foundir_flux_qwen3_prompt_cache_ddp.py \
    "${cache_args[@]}" --verify-existing
else
  "$FLUX_PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 \
    smoke/build_foundir_flux_qwen3_prompt_cache_ddp.py "${cache_args[@]}"
fi

"$FLUX_PYTHON" smoke/freeze_foundir_five_type_evaluation_contract.py \
  --config "$CONFIG" --conditions "$CONDITIONS" \
  --output-csv "$CONTRACT_CSV" --output-json "$CONTRACT_JSON"

TRAINING_SUMMARY="$OUTPUT/full_curriculum/training_summary.json"
if [[ ! -f "$TRAINING_SUMMARY" ]]; then
  "$FLUX_PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 \
    smoke/train_foundir_five_type_flux2_full_ddp.py --config "$CONFIG"
else
  "$FLUX_PYTHON" - "$CONFIG" "$TRAINING_SUMMARY" <<'PY'
import hashlib,json,pathlib,sys
sha=lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
p=json.loads(pathlib.Path(sys.argv[2]).read_text())
assert p["status"] == "training_complete_pending_development_selection"
assert p["config_sha256"] == sha(sys.argv[1])
checkpoint=pathlib.Path.cwd()/p["final_checkpoint"]
assert sha(checkpoint) == p["final_checkpoint_sha256"]
assert p["protected_foundir_test_accessed"] is False
print("FOUNDIR_FLUX_MAIN_V2_TRAINING_VERIFY_PASS")
PY
fi

mapfile -t candidate_steps < <("$FLUX_PYTHON" - "$CONFIG" <<'PY'
import json,sys
for step in json.load(open(sys.argv[1]))["evaluation"]["candidate_steps"]: print(step)
PY
)
selection_args=(--config "$CONFIG" --output "$SELECTION")
for step in "${candidate_steps[@]}"; do
  checkpoint=$(printf '%s/full_curriculum/step_%06d.safetensors' "$OUTPUT" "$step")
  result_dir=$(printf '%s/development/step_%06d/qwen' "$OUTPUT" "$step")
  [[ -f "$checkpoint" ]] || { echo "ERROR: missing candidate $checkpoint" >&2; exit 1; }
  if [[ ! -f "$result_dir/result.json" ]]; then
    "$FLUX_PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 \
      smoke/evaluate_foundir_five_type_flux2_ddp.py --config "$CONFIG" \
      --checkpoint "$checkpoint" --split validation --arm qwen --output "$result_dir"
  fi
  selection_args+=(--result "$result_dir/result.json")
done
"$FLUX_PYTHON" smoke/select_foundir_five_type_flux2_checkpoint.py "${selection_args[@]}"

readarray -t selected_identity < <("$FLUX_PYTHON" - "$SELECTION" <<'PY'
import json,sys
p=json.load(open(sys.argv[1])); print(p["checkpoint"]); print(p["selected_development_result"])
PY
)
SELECTED_CHECKPOINT="$ROOT/${selected_identity[0]}"

if [[ ! -f "$OUTPUT/native_resolution/result.json" ]]; then
  "$FLUX_PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 \
    smoke/evaluate_foundir_five_type_flux2_tiled_ddp.py --config "$CONFIG" \
    --checkpoint "$SELECTED_CHECKPOINT" --selection "$SELECTION" \
    --output "$OUTPUT/native_resolution"
fi
if [[ ! -f "$OUTPUT/released_baselines/development/result.json" ]]; then
  "$FLUX_PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 \
    smoke/evaluate_foundir_five_type_released_baselines_ddp.py --config "$CONFIG" \
    --split validation --output "$OUTPUT/released_baselines/development"
fi
if [[ ! -f "$OUTPUT/sealed_internal/qwen/result.json" ]]; then
  "$FLUX_PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 \
    smoke/evaluate_foundir_five_type_flux2_ddp.py --config "$CONFIG" \
    --checkpoint "$SELECTED_CHECKPOINT" --selection "$SELECTION" \
    --split internal_test --arm qwen --output "$OUTPUT/sealed_internal/qwen"
fi
if [[ ! -f "$OUTPUT/released_baselines/sealed_internal/result.json" ]]; then
  "$FLUX_PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 \
    smoke/evaluate_foundir_five_type_released_baselines_ddp.py --config "$CONFIG" \
    --split internal_test --output "$OUTPUT/released_baselines/sealed_internal"
fi

set +e
"$FLUX_PYTHON" smoke/freeze_foundir_five_type_flux2_main_v2_report.py \
  --config "$CONFIG" --selection "$SELECTION" \
  --native "$OUTPUT/native_resolution/result.json" \
  --development-baseline "$OUTPUT/released_baselines/development/result.json" \
  --sealed "$OUTPUT/sealed_internal/qwen/result.json" \
  --sealed-baseline "$OUTPUT/released_baselines/sealed_internal/result.json" \
  --output "$OUTPUT/final_internal_freeze.json"
status=$?
set -e
exit "$status"
