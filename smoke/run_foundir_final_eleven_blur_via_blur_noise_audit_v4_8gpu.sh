#!/usr/bin/env bash
set -euo pipefail
ROOT=/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
PYTHON="$ROOT/.venv/flux2-smoke/bin/python"
CONFIG="$ROOT/configs/foundir_final_eleven_blur_via_blur_noise_audit_v4.json"
EVALUATOR="$ROOT/smoke/evaluate_foundir_mixed_restoration_pilot_ddp.py"
SUMMARIZER="$ROOT/smoke/summarize_foundir_final_eleven_blur_via_blur_noise_audit_v4.py"
OUTPUT="$ROOT/checkpoints/foundir_final_eleven_blur_via_blur_noise_audit_v4"
TERMINAL="$OUTPUT/terminal_result.json"
LOG="$ROOT/logs/foundir_final_eleven_blur_via_blur_noise_audit_v4.log"
GPU_LOG="$ROOT/logs/foundir_final_eleven_blur_via_blur_noise_audit_v4_gpu.csv"
LOCK="$ROOT/logs/foundir_final_eleven_blur_via_blur_noise_audit_v4.lock"
cd "$ROOT"
/usr/bin/python3 smoke/advisor_governance.py check
mkdir -p "$ROOT/logs" "$ROOT/.cache/triton/foundir_final_eleven_blur_via_blur_noise_audit_v4"
exec > >(tee -a "$LOG") 2>&1
telemetry_pid=
finish_run(){ status=$?; if [[ -n "$telemetry_pid" ]]; then kill "$telemetry_pid" 2>/dev/null || true; wait "$telemetry_pid" 2>/dev/null || true; fi; echo "FINAL_ELEVEN_BLUR_VIA_BLUR_NOISE_V4_END status=$status utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"; return "$status"; }
trap finish_run EXIT
exec 9>>"$LOCK"; flock -n 9 || { echo "ERROR: Blur audit v4 already running" >&2; exit 1; }
[[ "${CUDA_VISIBLE_DEVICES:-}" == "0,1,2,3,4,5,6,7" ]] || { echo "ERROR: set eight visible GPUs" >&2; exit 1; }
[[ ! -f "$TERMINAL" ]] || { echo "ERROR: terminal result already exists" >&2; exit 1; }
mapfile -t gpu_names < <(nvidia-smi --query-gpu=name --format=csv,noheader | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
[[ "${#gpu_names[@]}" -eq 8 ]] || { echo "ERROR: exactly eight GPUs required" >&2; exit 1; }
first_name=${gpu_names[0]}
for name in "${gpu_names[@]}"; do [[ "$name" == "$first_name" ]] || { echo "ERROR: mixed GPU models forbidden" >&2; exit 1; }; [[ "$name" == *H100* || "$name" == *H200* ]] || { echo "ERROR: unsupported GPU $name" >&2; exit 1; }; done
export TORCH_HOME="$ROOT/models/torch" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 TORCH_NCCL_TRACE_BUFFER_SIZE=1048576 TRITON_CACHE_DIR="$ROOT/.cache/triton/foundir_final_eleven_blur_via_blur_noise_audit_v4" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
echo "FINAL_ELEVEN_BLUR_VIA_BLUR_NOISE_V4_START utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) gpu=$first_name"
echo "SCOPE 120_blur_cross_compatibility_records frozen_paired_restormer no_training no_official_or_sealed_test"
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw --format=csv -l 5 >"$GPU_LOG" 2>&1 & telemetry_pid=$!
"$PYTHON" "$EVALUATOR" --config "$CONFIG" --preflight
"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 "$EVALUATOR" --config "$CONFIG" --stage blur_noise
"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 "$EVALUATOR" --config "$CONFIG" --stage metrics
"$PYTHON" "$SUMMARIZER"
[[ -f "$TERMINAL" ]] || { echo "ERROR: terminal result missing" >&2; exit 1; }
