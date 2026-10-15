#!/usr/bin/env bash
set -euo pipefail
ROOT=/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
PYTHON="$ROOT/.venv/flux2-smoke/bin/python"
CONFIG="$ROOT/configs/foundir_composition_whole_image_full_pipeline_v1.json"
EVALUATOR="$ROOT/smoke/evaluate_foundir_composition_full_pipeline_v1.py"
OUTPUT="$ROOT/checkpoints/foundir_composition_whole_image_full_pipeline_v1"
LOG="$ROOT/logs/foundir_composition_whole_image_full_pipeline_v1.log"
GPU_LOG="$ROOT/logs/foundir_composition_whole_image_full_pipeline_v1_gpu.csv"
LOCK="$ROOT/logs/foundir_composition_whole_image_full_pipeline_v1.lock"
cd "$ROOT"
mkdir -p "$ROOT/logs" "$OUTPUT" "$ROOT/.cache/triton/foundir_composition_full_pipeline_v1"
exec > >(tee -a "$LOG") 2>&1
telemetry_pid=
finish_run(){ status=$?; if [[ -n "$telemetry_pid" ]]; then kill "$telemetry_pid" 2>/dev/null || true; wait "$telemetry_pid" 2>/dev/null || true; fi; echo "FOUNDIR_COMPOSITION_FULL_PIPELINE_END status=$status utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"; return "$status"; }
trap finish_run EXIT
exec 9>>"$LOCK"; flock -n 9 || { echo "ERROR: full-pipeline comparison already running" >&2; exit 1; }
[[ "${CUDA_VISIBLE_DEVICES:-}" == "0,1,2,3,4,5,6,7" ]] || { echo "ERROR: export eight visible GPUs" >&2; exit 1; }
[[ ! -f "$OUTPUT/result.json" ]] || { echo "ERROR: terminal result already exists" >&2; exit 1; }
mapfile -t gpu_names < <(nvidia-smi --query-gpu=name --format=csv,noheader | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
[[ "${#gpu_names[@]}" -eq 8 ]] || { echo "ERROR: exactly eight GPUs required" >&2; exit 1; }
for name in "${gpu_names[@]}"; do [[ "$name" == *H100* || "$name" == *H200* ]] || { echo "ERROR: unsupported GPU $name" >&2; exit 1; }; done
export TORCH_HOME="$ROOT/models/torch" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR="$ROOT/.cache/triton/foundir_composition_full_pipeline_v1" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
echo "FOUNDIR_COMPOSITION_FULL_PIPELINE_START utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) gpu=${gpu_names[0]} records=660 specialist_records=360"
echo "SCOPE DIAGNOSTIC_ONLY validation released_FoundIR_Generalist_then_Specialist no_training no_official_test"
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw --format=csv -l 10 >"$GPU_LOG" 2>&1 & telemetry_pid=$!
"$PYTHON" "$EVALUATOR" --config "$CONFIG" --mode preflight
"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 "$EVALUATOR" --config "$CONFIG" --mode weather
"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 "$EVALUATOR" --config "$CONFIG" --mode lowlight
export FOUNDIR_EVALUATION_COORDINATION_TOKEN="full_pipeline_metrics_$(date -u +%Y%m%dT%H%M%SZ)_$$"
"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 "$EVALUATOR" --config "$CONFIG" --mode metrics
"$PYTHON" - <<'PY'
import json
from pathlib import Path
p=Path('checkpoints/foundir_composition_whole_image_full_pipeline_v1/result.json')
d=json.loads(p.read_text())
if d.get('status')!='foundir_composition_full_pipeline_complete' or d.get('records')!=660 or d.get('specialist_records')!=360:
    raise SystemExit('terminal full-pipeline result is invalid')
print(json.dumps(d['overall_metrics'], indent=2, sort_keys=True))
PY
