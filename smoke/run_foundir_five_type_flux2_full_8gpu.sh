#!/usr/bin/env bash
set -euo pipefail

ROOT=/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
/usr/bin/python3 "$ROOT/smoke/advisor_governance.py" check
CONFIG="$ROOT/configs/foundir_five_type_flux2_dit_v1.json"
QWEN_CONFIG="$ROOT/configs/qwen_multidegradation_100k_rank64_joint_v1.json"
PRELAUNCH="$ROOT/configs/foundir_five_type_prelaunch_gate_v1.json"
QWEN_PYTHON="$ROOT/.conda/gpu-train-stable/bin/python"
FLUX_PYTHON="$ROOT/.venv/flux2-smoke/bin/python"
OUTPUT="$ROOT/checkpoints/foundir_five_type_flux2_dit_v1"
MANIFEST="$ROOT/manifests/foundir_five_type_pairs_v1.csv"
CONDITIONS="$ROOT/manifests/foundir_five_type_all_qwen_conditions_v1.csv"
CONDITIONS_JSON="$ROOT/manifests/foundir_five_type_all_qwen_conditions_v1.json"
SHARDS="$OUTPUT/qwen_condition_shards"
PROMPT_CACHE="$OUTPUT/qwen3_prompt_cache"
CONTRACT_CSV="$ROOT/manifests/foundir_five_type_evaluation_contract_v1.csv"
CONTRACT_JSON="$ROOT/manifests/foundir_five_type_evaluation_contract_v1.json"
SELECTION="$OUTPUT/checkpoint_selection.json"
LOG="$ROOT/logs/foundir_five_type_flux2_full_v1.log"
GPU_LOG="$ROOT/logs/foundir_five_type_flux2_full_v1_gpu.csv"
LOCK="$ROOT/logs/foundir_five_type_flux2_full_v1.lock"

cd "$ROOT"
mkdir -p "$ROOT/logs" "$OUTPUT" "$ROOT/.cache/triton/foundir_five_type" \
  "$ROOT/.cache/torch_extensions/foundir_five_type"
exec > >(tee -a "$LOG") 2>&1

telemetry_pid=
finish_run() {
  status=$?
  if [[ -n "$telemetry_pid" ]]; then
    kill "$telemetry_pid" 2>/dev/null || true
    wait "$telemetry_pid" 2>/dev/null || true
  fi
  echo "FOUNDIR_FIVE_TYPE_FULL_END status=$status utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
trap finish_run EXIT

exec 9>>"$LOCK"
if ! flock -n 9; then
  echo "ERROR: another five-type full run holds $LOCK" >&2
  exit 1
fi
if [[ "${CUDA_VISIBLE_DEVICES:-}" != "0,1,2,3,4,5,6,7" ]]; then
  echo "ERROR: set CUDA_VISIBLE_DEVICES exactly to 0,1,2,3,4,5,6,7" >&2
  exit 1
fi
if [[ ! -x "$QWEN_PYTHON" || ! -x "$FLUX_PYTHON" ]]; then
  echo "ERROR: workspace Qwen or FLUX Python environment is missing" >&2
  exit 1
fi
for executable in flock nvidia-smi; do
  command -v "$executable" >/dev/null 2>&1 || {
    echo "ERROR: required runtime executable is missing: $executable" >&2
    exit 1
  }
done
export TORCH_HOME="$ROOT/models/torch"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR="$ROOT/.cache/triton/foundir_five_type"
export TORCH_EXTENSIONS_DIR="$ROOT/.cache/torch_extensions/foundir_five_type"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
echo "FOUNDIR_FIVE_TYPE_FULL_START utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "FOUNDIR_FIVE_TYPE_FULL_HARDWARE exactly_8x_homogeneous_H100_80GB_or_H200"
echo "FOUNDIR_FIVE_TYPE_FULL_GIT $(git rev-parse HEAD)"
echo "FOUNDIR_FIVE_TYPE_FULL_PROTECTED_OFFICIAL_TEST not_accessed"

"$QWEN_PYTHON" smoke/freeze_foundir_five_type_prelaunch.py \
  --prelaunch-config "$PRELAUNCH" --verify-existing --require-selected joint
bash smoke/run_foundir_five_type_stage0_8gpu.sh

nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw \
  --format=csv -l 5 >"$GPU_LOG" 2>&1 &
telemetry_pid=$!
sleep 1
if ! kill -0 "$telemetry_pid" 2>/dev/null; then
  echo "ERROR: nvidia-smi telemetry failed to stay active" >&2
  exit 1
fi

export_args=(
  --config "$CONFIG" --qwen-config "$QWEN_CONFIG" --manifest "$MANIFEST"
  --shard-dir "$SHARDS" --output-csv "$CONDITIONS" --output-json "$CONDITIONS_JSON"
)
if [[ -f "$CONDITIONS" && -f "$CONDITIONS_JSON" ]]; then
  "$QWEN_PYTHON" smoke/export_foundir_five_type_qwen_conditions_sharded_ddp.py "${export_args[@]}" --verify-existing
else
  "$QWEN_PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 \
    smoke/export_foundir_five_type_qwen_conditions_sharded_ddp.py "${export_args[@]}"
fi

cache_args=(--config "$CONFIG" --conditions "$CONDITIONS" --output "$PROMPT_CACHE")
if [[ -f "$PROMPT_CACHE/index.json" ]]; then
  "$FLUX_PYTHON" smoke/build_foundir_flux_qwen3_prompt_cache_ddp.py "${cache_args[@]}" --verify-existing
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
root=pathlib.Path.cwd(); payload=json.loads(pathlib.Path(sys.argv[2]).read_text())
assert payload["status"] == "training_complete_pending_development_selection"
assert payload["config_sha256"] == sha(sys.argv[1])
checkpoint=root/payload["final_checkpoint"]
assert sha(checkpoint) == payload["final_checkpoint_sha256"]
assert payload["protected_foundir_test_accessed"] is False
print("FOUNDIR_FLUX_FULL_TRAINING_VERIFY_PASS")
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
  if [[ ! -f "$checkpoint" ]]; then
    echo "ERROR: frozen development candidate checkpoint is missing: $checkpoint" >&2
    exit 1
  fi
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
SELECTED_QWEN_RESULT="$ROOT/${selected_identity[1]}"
for arm in generic shuffled_wrong type_oracle; do
  arm_dir="$OUTPUT/causal/$arm"
  if [[ ! -f "$arm_dir/result.json" ]]; then
    "$FLUX_PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 \
      smoke/evaluate_foundir_five_type_flux2_ddp.py --config "$CONFIG" \
      --checkpoint "$SELECTED_CHECKPOINT" --split validation --arm "$arm" --output "$arm_dir"
  fi
done

if [[ ! -f "$OUTPUT/native_resolution/result.json" ]]; then
  "$FLUX_PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 \
    smoke/evaluate_foundir_five_type_flux2_tiled_ddp.py --config "$CONFIG" \
    --checkpoint "$SELECTED_CHECKPOINT" --selection "$SELECTION" --output "$OUTPUT/native_resolution"
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
"$FLUX_PYTHON" smoke/freeze_foundir_five_type_flux2_final_report.py \
  --config "$CONFIG" --selection "$SELECTION" \
  --generic "$OUTPUT/causal/generic/result.json" \
  --shuffled "$OUTPUT/causal/shuffled_wrong/result.json" \
  --oracle "$OUTPUT/causal/type_oracle/result.json" \
  --native "$OUTPUT/native_resolution/result.json" \
  --development-baseline "$OUTPUT/released_baselines/development/result.json" \
  --sealed "$OUTPUT/sealed_internal/qwen/result.json" \
  --sealed-baseline "$OUTPUT/released_baselines/sealed_internal/result.json" \
  --output "$OUTPUT/final_internal_freeze.json"
final_status=$?
set -e
exit "$final_status"
