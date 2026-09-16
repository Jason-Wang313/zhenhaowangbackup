#!/usr/bin/env bash
set -euo pipefail

ROOT=/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
PYTHON="$ROOT/.venv/flux2-smoke/bin/python"
CONFIG="$ROOT/configs/foundir_four_task_flux2_capacity_sanity_v1.json"
TRAINER="$ROOT/smoke/train_foundir_blur_only_flux2_overfit_ddp.py"
AGGREGATOR="$ROOT/smoke/aggregate_foundir_four_task_flux2_capacity_sanity.py"
OUTPUT="$ROOT/checkpoints/foundir_four_task_flux2_capacity_sanity_v1"
RESULT="$OUTPUT/result.json"
LOG="$ROOT/logs/foundir_four_task_flux2_capacity_sanity_v1.log"
GPU_LOG="$ROOT/logs/foundir_four_task_flux2_capacity_sanity_v1_gpu.csv"
LOCK="$ROOT/logs/foundir_four_task_flux2_capacity_sanity_v1.lock"

cd "$ROOT"
/usr/bin/python3 smoke/advisor_governance.py check
mkdir -p "$ROOT/logs" "$OUTPUT" "$ROOT/.cache/triton/foundir_four_task_capacity_v1"
exec > >(tee -a "$LOG") 2>&1

telemetry_pid=
declare -a task_pids=()
finish_run() {
  status=$?
  if [[ -n "$telemetry_pid" ]]; then
    kill "$telemetry_pid" 2>/dev/null || true
    wait "$telemetry_pid" 2>/dev/null || true
  fi
  for pid in "${task_pids[@]:-}"; do kill "$pid" 2>/dev/null || true; done
  echo "FOUNDIR_FOUR_TASK_CAPACITY_END status=$status utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  return "$status"
}
trap finish_run EXIT

exec 9>>"$LOCK"
flock -n 9 || { echo "ERROR: another four-task capacity run holds $LOCK" >&2; exit 1; }
[[ "${CUDA_VISIBLE_DEVICES:-}" == "0,1,2,3,4,5,6,7" ]] || {
  echo "ERROR: set CUDA_VISIBLE_DEVICES exactly to 0,1,2,3,4,5,6,7" >&2; exit 1;
}
[[ ! -f "$RESULT" ]] || { echo "ERROR: terminal aggregate already exists" >&2; exit 1; }
[[ -x "$PYTHON" ]] || { echo "ERROR: missing workspace FLUX Python: $PYTHON" >&2; exit 1; }

mapfile -t gpu_names < <(nvidia-smi --query-gpu=name --format=csv,noheader | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
[[ "${#gpu_names[@]}" -eq 8 ]] || { echo "ERROR: exactly eight GPUs are required" >&2; exit 1; }
first_name=${gpu_names[0]}
for name in "${gpu_names[@]}"; do
  [[ "$name" == "$first_name" ]] || { echo "ERROR: mixed GPU models are forbidden" >&2; exit 1; }
  [[ "$name" == *H100* || "$name" == *H200* ]] || { echo "ERROR: unsupported GPU $name" >&2; exit 1; }
done

export TORCH_HOME="$ROOT/models/torch" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 TORCH_NCCL_TRACE_BUFFER_SIZE=1048576
export TORCH_NCCL_DUMP_ON_TIMEOUT=1 TORCH_NCCL_DESYNC_DEBUG=1

echo "FOUNDIR_FOUR_TASK_CAPACITY_START utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) git=$(git rev-parse HEAD) gpu=$first_name"
echo "SCOPE four_independent_FLUX2_4B_single_degradation_arms no_Qwen_degradation no_mixed no_pixel_refiner no_sealed no_official_test no_auto_joint"
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw --format=csv -l 5 >"$GPU_LOG" 2>&1 &
telemetry_pid=$!

tasks=(05Noise 08Haze 10Rain 14Lowlight)
devices=(0,1 2,3 4,5 6,7)
slugs=(noise haze rain lowlight)
for i in "${!tasks[@]}"; do
  task=${tasks[$i]}
  visible=${devices[$i]}
  slug=${slugs[$i]}
  task_log="$ROOT/logs/foundir_four_task_flux2_capacity_sanity_v1_${slug}.log"
  cache="$ROOT/.cache/triton/foundir_four_task_capacity_v1_${slug}"
  mkdir -p "$cache"
  echo "FOUNDIR_FOUR_TASK_ARM_LAUNCH task=$task devices=$visible log=$task_log"
  env CUDA_VISIBLE_DEVICES="$visible" TRITON_CACHE_DIR="$cache" \
    "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=2 \
      "$TRAINER" --config "$CONFIG" --task "$task" >"$task_log" 2>&1 &
  task_pids+=("$!")
done

set +e
statuses=()
for i in "${!task_pids[@]}"; do
  wait "${task_pids[$i]}"
  statuses+=("$?")
  echo "FOUNDIR_FOUR_TASK_ARM_END task=${tasks[$i]} status=${statuses[$i]}"
done
set -e

if ! "$PYTHON" "$AGGREGATOR" --config "$CONFIG"; then
  /usr/bin/python3 smoke/advisor_governance.py runtime-failure \
    --gate flux2_four_task_capacity_runtime --evidence reports/foundir_four_task_flux2_capacity_sanity_authorization_v1.json \
    --config "$CONFIG" --model "FLUX.2 Klein Base 4B four independent capacity arms" \
    --status 1 --log "${LOG#$ROOT/}" || true
  exit 1
fi

/usr/bin/python3 smoke/advisor_governance.py emit \
  --gate flux2_four_task_capacity_sanity_result \
  --evidence "$RESULT" --config "$CONFIG" \
  --model "FLUX.2 Klein Base 4B four independent single-degradation LoRA arms" \
  --split "fixed 16 train plus 16 source-group-disjoint descriptive validation per Noise/Haze/Rain/Low-light task" \
  --outcome auto --diagnostic-only --require-decision \
  --root-cause-or-status "All four bounded task arms completed; inspect task-specific train capacity, SOW proxies, descriptive validation, and fixed visuals together. No automatic joint launch." \
  --advisor-decision-question "四类独立 capacity evidence 与视觉审计完成后，是否进入 balanced five-single-degradation joint FLUX baseline design？"

set +e
"$PYTHON" - "$RESULT" <<'PY'
import json, pathlib, sys
p=json.loads(pathlib.Path(sys.argv[1]).read_text())
assert p["status"] == "four_task_capacity_complete_awaiting_joint_visual_audit"
assert p["joint_training_started"] is False
assert p["qwen_degradation_conditioning"] is False and p["mixed_degradations"] is False
raise SystemExit(0 if p["automatic_gate_pass"] else 2)
PY
scientific_status=$?
set -e
if [[ "$scientific_status" -eq 0 ]]; then
  echo "FOUNDIR_FOUR_TASK_CAPACITY_METRIC_PASS_STOP_FOR_VISUAL_AND_JOINT_AUDIT"
else
  echo "FOUNDIR_FOUR_TASK_CAPACITY_SCIENTIFIC_FAILURE_STOP_FOR_AUDIT"
fi
exit "$scientific_status"
