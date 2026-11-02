#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPECTED_ROOT=/inspire/hdd/project/multimodal-discrete-diffusion/liuxiaohong-25080/zhenhaowang
[[ "$ROOT" == "$EXPECTED_ROOT" ]] || { echo 'ERROR: use migrated GPU workspace'; exit 2; }
cd "$ROOT"
RUN_ID=${1:-replay8_v1}
GPU_COUNT=${2:-8}
[[ "$RUN_ID" =~ ^[A-Za-z0-9_-]{1,80}$ ]] || { echo 'ERROR: invalid run ID'; exit 2; }
case "$GPU_COUNT" in
  4) export CUDA_VISIBLE_DEVICES=0,1,2,3; WALL_CAP=14400 ;;
  8) export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7; WALL_CAP=7200 ;;
  *) echo 'ERROR: use 4 or 8 GPUs'; exit 2 ;;
esac
# Verified on destination in 39e4575; an explicit override still must exist.
PYTHON="${PROJECT_PYTHON:-$ROOT/.conda/gpu-train-stable/bin/python}"
[[ -x "$PYTHON" ]] || { echo 'ERROR: verified Python unavailable; set PROJECT_PYTHON'; exit 2; }
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NCCL_NVLS_ENABLE=0
export TORCH_HOME="$ROOT/models/torch" PYTHONPATH="$ROOT:$ROOT/third_party/flux2/src"
command -v flock >/dev/null
command -v timeout >/dev/null
OUTPUT="$ROOT/checkpoints/foundir_existing_flux_replay_v1/$RUN_ID"
mkdir -p "$OUTPUT/attempts" logs
exec 9>/tmp/zhenhaowang_existing_flux_replay.lock
flock -n 9 || { echo 'ERROR: replay already active on node'; exit 3; }
exec 8>"$OUTPUT/run.lock"
flock -n 8 || { echo 'ERROR: same run already active'; exit 3; }
LOG="$ROOT/logs/foundir_existing_flux_replay_v1_${RUN_ID}.log"
exec > >(tee -a "$LOG") 2>&1
START=$SECONDS
ATTEMPT_ID="$(date -u +%Y%m%dT%H%M%SZ)_$$"
finish() {
  status=$?
  elapsed=$((SECONDS-START))
  printf 'EXPERT_REPLAY_END status=%s elapsed_seconds=%s allocated_gpu_seconds=%s attempt=%s\n' "$status" "$elapsed" "$((GPU_COUNT*elapsed))" "$ATTEMPT_ID"
  (set -o noclobber; printf '{"status":%s,"seconds":%s,"allocated_gpu_seconds":%s,"gpus":%s}\n' "$status" "$elapsed" "$((GPU_COUNT*elapsed))" "$GPU_COUNT" > "$OUTPUT/attempts/$ATTEMPT_ID.json")
}
trap finish EXIT
echo "EXPERT_REPLAY_START run=$RUN_ID gpus=$GPU_COUNT resume=verified_records_only no_training=true"
echo 'ESTIMATE 8GPU=0.5_to_1.5_hours 4GPU=1_to_3_hours allocated_GPU_hours=4_to_12 unmeasured; hard_cap=16_GPU_hours_per_attempt_plus_30s_cleanup'
SCRIPT=smoke/replay_foundir_existing_flux_experts_v1.py
step() {
  remaining=$((WALL_CAP-SECONDS+START))
  ((remaining > 0)) || return 124
  timeout --signal=TERM --kill-after=30s "${remaining}s" "$@"
}
step "$PYTHON" -u "$SCRIPT" hardware --gpus "$GPU_COUNT"
step "$PYTHON" -u "$SCRIPT" prepare --run-id "$RUN_ID" --gpus "$GPU_COUNT"
# Only GPU0 briefly encodes four fixed texts. Account for all allocated GPUs.
step "$PYTHON" -u "$SCRIPT" prompts --run-id "$RUN_ID" --gpus "$GPU_COUNT"
step "$PYTHON" -u -m torch.distributed.run --standalone --nproc_per_node="$GPU_COUNT" --max_restarts=0 \
  "$SCRIPT" run --run-id "$RUN_ID" --gpus "$GPU_COUNT"
step "$PYTHON" -u "$SCRIPT" verify --run-id "$RUN_ID" --gpus "$GPU_COUNT"
# Never launch Rain training, retention, mixed training, or energy here.
