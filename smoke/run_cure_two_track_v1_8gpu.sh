#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
TRACK="${1:-}"
if [[ "$TRACK" != published_style && "$TRACK" != strict_holdout ]]; then
  echo "usage: $0 published_style|strict_holdout" >&2
  exit 2
fi
if [[ "${CUDA_VISIBLE_DEVICES:-}" != "0,1,2,3,4,5,6,7" ]]; then
  echo "CUDA_VISIBLE_DEVICES must be the literal 0,1,2,3,4,5,6,7" >&2
  exit 2
fi

PYTHON="$ROOT/.venv/cure-repro-py311/bin/python"
OFFICIAL="$ROOT/third_party/cure_official"
VIEW="$ROOT/data/third_party/cure_two_track_v1"
OUT="$ROOT/checkpoints/cure_two_track_v1/$TRACK"
LOG="$ROOT/logs/cure_two_track_v1_${TRACK}.log"
mkdir -p "$OUT" "$ROOT/logs"
exec 9>"$OUT/launch.lock"
if ! flock -n 9; then
  echo "another $TRACK launcher owns $OUT/launch.lock" >&2
  exit 2
fi
if [[ -e "$OUT/terminal_end.json" || -e "$OUT/embedder/embedder_200.tar" || -e "$OUT/baseline/model_300.tar" || -e "$OUT/cure/model_300.tar" ]]; then
  echo "existing terminal/stage artifacts require audit; refusing duplicate launch" >&2
  exit 2
fi

set -o pipefail
exec > >(tee -a "$LOG") 2>&1
START="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
finish() {
  status=$?
  trap - EXIT
  "$PYTHON" "$ROOT/smoke/write_cure_terminal_v1.py" --track "$TRACK" --status "$status" --start "$START" || true
  echo "CURE_TWO_TRACK_END track=$TRACK status=$status utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  exit "$status"
}
trap finish EXIT

echo "CURE_TWO_TRACK_START track=$TRACK utc=$START workspace=$ROOT"
echo "official_commit=$(git -C "$OFFICIAL" rev-parse HEAD)"
echo "config_sha256=$(sha256sum "$ROOT/configs/cure_two_track_reproduction_v1.json")"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
"$PYTHON" "$ROOT/smoke/advisor_governance.py" check
"$PYTHON" "$ROOT/smoke/preflight_cure_two_track_v1.py"
"$PYTHON" "$ROOT/smoke/check_cure_gpu_profile_v1.py"

export NCCL_NVLS_ENABLE=0
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export TORCH_HOME="$ROOT/models/torch"
"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 \
  "$ROOT/smoke/cure_nccl_rehearsal_v1.py"
"$PYTHON" "$ROOT/smoke/cure_gpu_rehearsal_v1.py" \
  --h5 "$VIEW/h5/shared_cure.h5" \
  --embedder "$ROOT/data/third_party/cure_released/OneRestore_embedder.tar" \
  --baseline "$ROOT/data/third_party/cure_released/OneRestore_restorer.tar"
if [[ "$TRACK" == published_style ]]; then
  EMBEDDER_TRAIN="$VIEW/paired_train/main_data"
  BASELINE_H5="$VIEW/h5/published_baseline.h5"
else
  EMBEDDER_TRAIN="$VIEW/embedder_strict_train/main_data"
  BASELINE_H5="$VIEW/h5/strict_baseline.h5"
fi
EMBEDDER_DEV="$VIEW/embedder_dev_common/main_data"
CURE_H5="$VIEW/h5/shared_cure.h5"
cd "$OFFICIAL"

echo "STAGE_START embedder track=$TRACK utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 01_train_OneRestore_embedder.py \
  --train-dir "$EMBEDDER_TRAIN" --val-dir "$EMBEDDER_DEV" \
  --glove "$OFFICIAL/assets/glove.6B.300d.txt" --output-dir "$OUT/embedder" \
  --epochs 200 --batch-size 32 --learning-rate 1e-4 --lr-decay-every 50 --seed 124
test -s "$OUT/embedder/embedder_200.tar"
sha256sum "$OUT/embedder/embedder_200.tar"

echo "STAGE_START baseline track=$TRACK utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 02_train_Onerestore_baseline.py \
  --train-h5 "$BASELINE_H5" --embedder-checkpoint "$OUT/embedder/embedder_200.tar" \
  --output-dir "$OUT/baseline" --epochs 300 --batch-size 16 \
  --learning-rate 1e-4 --lr-decay-every 30 --seed 124
test -s "$OUT/baseline/model_300.tar"
sha256sum "$OUT/baseline/model_300.tar"
"$PYTHON" "$ROOT/smoke/probe_cure_checkpoint_v1.py" \
  --checkpoint "$OUT/baseline/model_300.tar" \
  --embedder "$OUT/embedder/embedder_200.tar" \
  --output "$OUT/baseline_probe" --device cuda:0

echo "STAGE_START cure track=$TRACK utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 03_train_CURE.py \
  --train-h5 "$CURE_H5" --embedder-checkpoint "$OUT/embedder/embedder_200.tar" \
  --baseline-checkpoint "$OUT/baseline/model_300.tar" --output-dir "$OUT/cure" \
  --epochs 300 --batch-size 3 --learning-rate 2.5e-5 \
  --lr-decay-every 30 --ratio 0.5 --seed 124
test -s "$OUT/cure/model_300.tar"
sha256sum "$OUT/cure/model_300.tar"
"$PYTHON" "$ROOT/smoke/probe_cure_checkpoint_v1.py" \
  --checkpoint "$OUT/cure/model_300.tar" \
  --embedder "$OUT/embedder/embedder_200.tar" \
  --output "$OUT/cure_probe" --device cuda:0
