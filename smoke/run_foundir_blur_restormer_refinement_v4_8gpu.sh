#!/usr/bin/env bash
set -euo pipefail

ROOT=/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
/usr/bin/python3 "$ROOT/smoke/advisor_governance.py" check
PYTHON="$ROOT/.venv/flux2-smoke/bin/python"
CONFIG="$ROOT/configs/foundir_blur_restormer_refinement_gate_v4.json"
LOG="$ROOT/logs/foundir_blur_restormer_refinement_v4.log"
GPU_LOG="$ROOT/logs/foundir_blur_restormer_refinement_v4_gpu.csv"
LOCK="$ROOT/logs/foundir_blur_restormer_refinement_v4.lock"

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
  echo "FOUNDIR_BLUR_RESTORMER_V4_END status=$status utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
trap finish_run EXIT

exec 9>>"$LOCK"
flock -n 9 || { echo "ERROR: another Blur Restormer v4 gate holds $LOCK" >&2; exit 1; }
[[ "${CUDA_VISIBLE_DEVICES:-}" == "0,1,2,3,4,5,6,7" ]] || {
  echo "ERROR: expose exactly CUDA devices 0,1,2,3,4,5,6,7" >&2
  exit 1
}

export TORCH_HOME="$ROOT/models/torch"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

echo "FOUNDIR_BLUR_RESTORMER_V4_START utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "FOUNDIR_BLUR_RESTORMER_V4_WORKSPACE $ROOT"
echo "FOUNDIR_BLUR_RESTORMER_V4_CONFIG $CONFIG"
echo "FOUNDIR_BLUR_RESTORMER_V4_CONFIG_SHA256 $(sha256sum "$CONFIG" | cut -d' ' -f1)"
echo "FOUNDIR_BLUR_RESTORMER_V4_TRAINER_SHA256 $(sha256sum "$ROOT/smoke/train_foundir_blur_restormer_refinement_ddp.py" | cut -d' ' -f1)"
echo "FOUNDIR_BLUR_RESTORMER_V4_FUSION_SHA256 $(sha256sum "$ROOT/lowlight_restoration/blur_restormer_refinement.py" | cut -d' ' -f1)"
echo "FOUNDIR_BLUR_RESTORMER_V4_GIT $(git rev-parse HEAD)"
echo "FOUNDIR_BLUR_RESTORMER_V4_DEVICES $CUDA_VISIBLE_DEVICES"
echo "FOUNDIR_BLUR_RESTORMER_V4_SCOPE development_only_v3_corrective_l4_l5_psnr_aligned"
echo "FOUNDIR_BLUR_RESTORMER_V4_INTERNAL not_accessed"
echo "FOUNDIR_BLUR_RESTORMER_V4_PROTECTED_OFFICIAL_TEST not_accessed"

nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw \
  --format=csv -l 5 >"$GPU_LOG" 2>&1 &
telemetry_pid=$!

"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 \
  smoke/train_foundir_blur_restormer_refinement_ddp.py --config "$CONFIG"
