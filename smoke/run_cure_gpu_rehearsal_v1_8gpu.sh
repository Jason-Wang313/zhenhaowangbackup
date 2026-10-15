#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$ROOT/.venv/cure-repro-py311/bin/python"
RUN_ID="${CURE_REHEARSAL_RUN_ID:-v1}"
if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9_-]+$ ]]; then
  echo "CURE_REHEARSAL_RUN_ID must contain only letters, digits, underscores, or hyphens" >&2
  exit 2
fi
LOG="$ROOT/logs/cure_two_track_gpu_rehearsal_${RUN_ID}.log"
if [[ "${CUDA_VISIBLE_DEVICES:-}" != "0,1,2,3,4,5,6,7" ]]; then
  echo "CUDA_VISIBLE_DEVICES must be 0,1,2,3,4,5,6,7" >&2
  exit 2
fi
if [[ -e "$LOG" ]]; then
  echo "existing rehearsal log requires audit; refusing duplicate: $LOG" >&2
  exit 2
fi
test -s "$ROOT/data/third_party/cure_h5_smoke_v1/shared_cure.h5"
test -s "$ROOT/data/third_party/cure_released/OneRestore_embedder.tar"
test -s "$ROOT/data/third_party/cure_released/OneRestore_restorer.tar"
mkdir -p "$ROOT/logs"
exec > >(tee "$LOG") 2>&1
finish() {
  status=$?
  trap - EXIT
  echo "CURE_GPU_REHEARSAL_END status=$status utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  exit "$status"
}
trap finish EXIT

echo "CURE_GPU_REHEARSAL_START utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) workspace=$ROOT"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "official_commit=$(git -C "$ROOT/third_party/cure_official" rev-parse HEAD)"
sha256sum "$ROOT/data/third_party/cure_released/OneRestore_embedder.tar"
sha256sum "$ROOT/data/third_party/cure_released/OneRestore_restorer.tar"
export NCCL_NVLS_ENABLE=0
export TORCH_HOME="$ROOT/models/torch"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
"$PYTHON" "$ROOT/smoke/advisor_governance.py" check
"$PYTHON" "$ROOT/smoke/check_cure_gpu_profile_v1.py"
"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 \
  "$ROOT/smoke/cure_nccl_rehearsal_v1.py"
"$PYTHON" "$ROOT/smoke/cure_gpu_rehearsal_v1.py" \
  --h5 "$ROOT/data/third_party/cure_h5_smoke_v1/shared_cure.h5" \
  --embedder "$ROOT/data/third_party/cure_released/OneRestore_embedder.tar" \
  --baseline "$ROOT/data/third_party/cure_released/OneRestore_restorer.tar"
