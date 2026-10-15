#!/usr/bin/env bash
set -euo pipefail
ROOT=/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
PYTHON="$ROOT/.venv/flux2-smoke/bin/python"
CONFIG="$ROOT/configs/foundir_lowlight_haze_restormer_v1.json"
TRAINER="$ROOT/smoke/train_foundir_lowlight_haze_restormer_ddp.py"
EVALUATOR="$ROOT/smoke/evaluate_foundir_lowlight_haze_restormer_ddp.py"
OUTPUT="$ROOT/checkpoints/foundir_lowlight_haze_restormer_v1"
RESULT="$OUTPUT/final_comparison/result.json"
LOG="$ROOT/logs/foundir_lowlight_haze_restormer_v1.log"
GPU_LOG="$ROOT/logs/foundir_lowlight_haze_restormer_v1_gpu.csv"
LOCK="$ROOT/logs/foundir_lowlight_haze_restormer_v1.lock"
cd "$ROOT"
mkdir -p "$ROOT/logs"
exec > >(tee -a "$LOG") 2>&1
telemetry_pid=
finish_run() {
  status=$?
  if [[ -n "$telemetry_pid" ]]; then
    kill "$telemetry_pid" 2>/dev/null || true
    wait "$telemetry_pid" 2>/dev/null || true
  fi
  echo "FOUNDIR_LOWLIGHT_HAZE_RESTORMER_V1_END status=$status utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  return "$status"
}
trap finish_run EXIT
exec 9>>"$LOCK"
flock -n 9 || { echo "ERROR: Low-light+Haze Restormer v1 already running" >&2; exit 1; }
[[ "${CUDA_VISIBLE_DEVICES:-}" == "0,1,2,3,4,5,6,7" ]] || { echo "ERROR: set exactly GPUs 0-7 visible" >&2; exit 1; }
[[ ! -f "$RESULT" ]] || { echo "ERROR: terminal comparison already exists" >&2; exit 1; }
mapfile -t names < <(nvidia-smi --query-gpu=name --format=csv,noheader | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
[[ "${#names[@]}" -eq 8 ]] || { echo "ERROR: exactly eight GPUs required" >&2; exit 1; }
profile=
for name in "${names[@]}"; do
  current=
  [[ "$name" == *H100* ]] && current=H100
  [[ "$name" == *H200* ]] && current=H200
  [[ -n "$current" ]] || { echo "ERROR: unsupported GPU $name" >&2; exit 1; }
  [[ -z "$profile" || "$profile" == "$current" ]] || { echo "ERROR: mixed H100/H200 node rejected" >&2; exit 1; }
  profile=$current
done
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TORCH_HOME="$ROOT/models/torch" TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=1048576
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
echo "FOUNDIR_LOWLIGHT_HAZE_RESTORMER_V1_START utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) workspace=$ROOT profile=$profile"
echo "IDENTITY config=$CONFIG config_sha256=$(sha256sum "$CONFIG" | awk '{print $1}') git_revision=$(git rev-parse HEAD) CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "TRANSPORT NCCL_P2P_DISABLE=$NCCL_P2P_DISABLE NCCL_NVLS_ENABLE=$NCCL_NVLS_ENABLE"
"$PYTHON" -c 'import torch; print("ENVIRONMENT torch=" + torch.__version__ + " cuda=" + str(torch.version.cuda) + " available=" + str(torch.cuda.is_available()) + " devices=" + str(torch.cuda.device_count()))'
echo "SCOPE 09Lowlight_Haze direct_Restormer 75370_train 128_full_resolution_development 6000_steps final_60_frozen_comparison no_FoundIR_weights no_official_test"
echo "BUDGET H200_wall=100-160min_center=125min GPU_hours=13.3-21.3_center=16.7; H100_wall=130-200min_center=160min GPU_hours=17.3-26.7_center=21.3"
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw --format=csv -l 5 >"$GPU_LOG" 2>&1 &
telemetry_pid=$!
"$PYTHON" "$ROOT/smoke/advisor_governance.py" check
"$PYTHON" "$TRAINER" --config "$CONFIG" --preflight
"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 "$TRAINER" --config "$CONFIG"
"$PYTHON" "$EVALUATOR" --config "$CONFIG" --preflight
"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 "$EVALUATOR" --config "$CONFIG"
[[ -f "$RESULT" ]] || { echo "ERROR: final Low-light+Haze comparison result missing" >&2; exit 1; }
