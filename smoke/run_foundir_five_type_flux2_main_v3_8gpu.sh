#!/usr/bin/env bash
set -euo pipefail

ROOT=/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
CONFIG="$ROOT/configs/foundir_five_type_flux2_dit_v3.json"
QWEN_PYTHON="$ROOT/.conda/gpu-train-stable/bin/python"
FLUX_PYTHON="$ROOT/.venv/flux2-smoke/bin/python"
OUTPUT="$ROOT/checkpoints/foundir_five_type_flux2_dit_v3"
MANIFEST="$ROOT/manifests/foundir_five_type_pairs_v1.csv"
CONDITIONS="$ROOT/manifests/foundir_five_type_all_qwen_conditions_v3.csv"
CONDITIONS_JSON="$ROOT/manifests/foundir_five_type_all_qwen_conditions_v3.json"
PROMPT_CACHE="$OUTPUT/qwen3_prompt_cache"
CONTRACT_CSV="$ROOT/manifests/foundir_five_type_evaluation_contract_v3.csv"
CONTRACT_JSON="$ROOT/manifests/foundir_five_type_evaluation_contract_v3.json"
SELECTION="$OUTPUT/checkpoint_selection.json"
LOG="$ROOT/logs/foundir_five_type_flux2_main_v3.log"
GPU_LOG="$ROOT/logs/foundir_five_type_flux2_main_v3_gpu.csv"
LOCK="$ROOT/logs/foundir_five_type_flux2_main_v3.lock"

cd "$ROOT"
mkdir -p "$ROOT/logs" "$OUTPUT" "$ROOT/.cache/triton/foundir_five_type_v3" \
  "$ROOT/.cache/torch_extensions/foundir_five_type_v3"
exec > >(tee -a "$LOG") 2>&1

telemetry_pid=
finish_run() {
  status=$?
  if [[ -n "$telemetry_pid" ]]; then
    kill "$telemetry_pid" 2>/dev/null || true
    wait "$telemetry_pid" 2>/dev/null || true
  fi
  echo "FOUNDIR_FIVE_TYPE_MAIN_V3_END status=$status utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
trap finish_run EXIT

exec 9>>"$LOCK"
flock -n 9 || { echo "ERROR: another five-type v3 pipeline holds $LOCK" >&2; exit 1; }
[[ "${CUDA_VISIBLE_DEVICES:-}" == "0,1,2,3,4,5,6,7" ]] || {
  echo "ERROR: set CUDA_VISIBLE_DEVICES exactly to 0,1,2,3,4,5,6,7" >&2; exit 1;
}
[[ -x "$QWEN_PYTHON" && -x "$FLUX_PYTHON" ]] || {
  echo "ERROR: workspace Python environments are missing" >&2; exit 1;
}

export TORCH_HOME="$ROOT/models/torch"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=1048576
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_DESYNC_DEBUG=1
export TRITON_CACHE_DIR="$ROOT/.cache/triton/foundir_five_type_v3"
export TORCH_EXTENSIONS_DIR="$ROOT/.cache/torch_extensions/foundir_five_type_v3"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

echo "FOUNDIR_FIVE_TYPE_MAIN_V3_START utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "FOUNDIR_FIVE_TYPE_MAIN_V3_SCOPE blur,noise,rain,haze,low_light no_jpeg no_condition_ablation"
echo "FOUNDIR_FIVE_TYPE_MAIN_V3_PROTECTED_OFFICIAL_TEST not_accessed"

bash smoke/run_foundir_five_type_stage0_v3_8gpu.sh

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

"$FLUX_PYTHON" smoke/preflight_foundir_flux2_training_geometry.py --config "$CONFIG"

TRAINING_SUMMARY="$OUTPUT/full_curriculum/training_summary.json"
if [[ ! -f "$TRAINING_SUMMARY" ]]; then
  mapfile -t training_gpu_names < <(nvidia-smi --query-gpu=name --format=csv,noheader)
  [[ ${#training_gpu_names[@]} -eq 8 ]] || { echo "ERROR: expected eight training GPUs" >&2; exit 1; }
  case "${training_gpu_names[0]}" in
    *H200*) training_family=H200 ;;
    *H100*) training_family=H100 ;;
    *) echo "ERROR: unsupported training GPU ${training_gpu_names[0]}" >&2; exit 1 ;;
  esac
  for training_gpu_name in "${training_gpu_names[@]}"; do
    [[ "$training_gpu_name" == "${training_gpu_names[0]}" ]] || {
      echo "ERROR: training GPUs are not homogeneous" >&2; exit 1;
    }
  done
  training_args=(--config "$CONFIG")
  speed_selection="$OUTPUT/speed_benchmark_selection_${training_family}.json"
  if [[ -f "$speed_selection" ]]; then
    training_args+=(--approved-speed-selection "$speed_selection")
    echo "FOUNDIR_FLUX_SPEED_SELECTION_APPLY path=$speed_selection"
  else
    echo "FOUNDIR_FLUX_SPEED_SELECTION_DEFAULT family=$training_family"
  fi
  "$FLUX_PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 \
    smoke/train_foundir_five_type_flux2_full_ddp.py "${training_args[@]}"
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
print("FOUNDIR_FLUX_MAIN_V3_TRAINING_VERIFY_PASS")
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
    FOUNDIR_EVALUATION_COORDINATION_TOKEN="development-${step}-$(date +%s%N)-$$" \
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
  FOUNDIR_EVALUATION_COORDINATION_TOKEN="native-$(date +%s%N)-$$" \
    "$FLUX_PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 \
    smoke/evaluate_foundir_five_type_flux2_tiled_ddp.py --config "$CONFIG" \
    --checkpoint "$SELECTED_CHECKPOINT" --selection "$SELECTION" \
    --output "$OUTPUT/native_resolution"
fi
if [[ ! -f "$OUTPUT/released_baselines/development/result.json" ]]; then
  "$FLUX_PYTHON" smoke/evaluate_foundir_five_type_released_baselines_ddp.py \
    --config "$CONFIG" --split validation \
    --output "$OUTPUT/released_baselines/development" \
    --finalize-existing-if-complete
  if [[ ! -f "$OUTPUT/released_baselines/development/result.json" ]]; then
    FOUNDIR_EVALUATION_COORDINATION_TOKEN="development-baseline-$(date +%s%N)-$$" \
      "$FLUX_PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 \
      smoke/evaluate_foundir_five_type_released_baselines_ddp.py --config "$CONFIG" \
      --split validation --output "$OUTPUT/released_baselines/development"
  fi
fi
if [[ ! -f "$OUTPUT/sealed_internal/qwen/result.json" \
      || ! -f "$OUTPUT/sealed_internal_consumption.json" ]]; then
  FOUNDIR_EVALUATION_COORDINATION_TOKEN="sealed-qwen-$(date +%s%N)-$$" \
    "$FLUX_PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 \
    smoke/evaluate_foundir_five_type_flux2_ddp.py --config "$CONFIG" \
    --checkpoint "$SELECTED_CHECKPOINT" --selection "$SELECTION" \
    --split internal_test --arm qwen --output "$OUTPUT/sealed_internal/qwen"
fi
if [[ ! -f "$OUTPUT/released_baselines/sealed_internal/result.json" ]]; then
  "$FLUX_PYTHON" smoke/evaluate_foundir_five_type_released_baselines_ddp.py \
    --config "$CONFIG" --split internal_test \
    --output "$OUTPUT/released_baselines/sealed_internal" \
    --finalize-existing-if-complete
  if [[ ! -f "$OUTPUT/released_baselines/sealed_internal/result.json" ]]; then
    FOUNDIR_EVALUATION_COORDINATION_TOKEN="sealed-baseline-$(date +%s%N)-$$" \
      "$FLUX_PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 \
      smoke/evaluate_foundir_five_type_released_baselines_ddp.py --config "$CONFIG" \
      --split internal_test --output "$OUTPUT/released_baselines/sealed_internal"
  fi
fi

set +e
"$FLUX_PYTHON" smoke/freeze_foundir_five_type_flux2_main_v3_report.py \
  --config "$CONFIG" --selection "$SELECTION" \
  --native "$OUTPUT/native_resolution/result.json" \
  --development-baseline "$OUTPUT/released_baselines/development/result.json" \
  --sealed "$OUTPUT/sealed_internal/qwen/result.json" \
  --sealed-baseline "$OUTPUT/released_baselines/sealed_internal/result.json" \
  --output "$OUTPUT/final_internal_freeze.json"
status=$?
set -e
exit "$status"
