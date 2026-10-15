#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
TRACK="${1:-}"
if [[ "$TRACK" != published_style && "$TRACK" != strict_holdout ]]; then
  echo "usage: $0 published_style|strict_holdout" >&2
  exit 2
fi
if [[ "${CUDA_VISIBLE_DEVICES:-}" != "0,1,2,3,4,5,6,7" ]]; then
  echo "CUDA_VISIBLE_DEVICES must be 0,1,2,3,4,5,6,7" >&2
  exit 2
fi

PYTHON="$ROOT/.venv/cure-repro-py311/bin/python"
OUTPUT="$ROOT/checkpoints/cure_two_track_v1/$TRACK/evaluation"
LOG="$ROOT/logs/cure_two_track_v1_${TRACK}_evaluation.log"
mkdir -p "$ROOT/logs"
exec 9>"$ROOT/checkpoints/cure_two_track_v1/$TRACK/eval.lock"
if ! flock -n 9; then
  echo "another $TRACK evaluation owns eval.lock" >&2
  exit 2
fi
if [[ -e "$OUTPUT" ]]; then
  echo "existing evaluation output requires audit; refusing duplicate launch: $OUTPUT" >&2
  exit 2
fi

set -o pipefail
exec > >(tee -a "$LOG") 2>&1
finish() {
  status=$?
  trap - EXIT
  echo "CURE_TWO_TRACK_EVAL_END track=$TRACK status=$status utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  exit "$status"
}
trap finish EXIT

echo "CURE_TWO_TRACK_EVAL_START track=$TRACK utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) workspace=$ROOT"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
export NCCL_NVLS_ENABLE=0
export TORCH_HOME="$ROOT/models/torch"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
"$PYTHON" "$ROOT/smoke/advisor_governance.py" check
"$PYTHON" "$ROOT/smoke/check_cure_gpu_profile_v1.py"
"$PYTHON" "$ROOT/smoke/preflight_cure_two_track_v1.py"
"$PYTHON" "$ROOT/smoke/audit_cure_embedder_text_v1.py"
"$PYTHON" "$ROOT/smoke/build_cure_two_track_eval_manifest_v1.py" --check
"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 \
  "$ROOT/smoke/evaluate_cure_two_track_v1.py" \
  --track "$TRACK" --manifest "$ROOT/manifests/cure_two_track_v1_eval.csv" \
  --output "$OUTPUT"
test -s "$OUTPUT/result.json"
sha256sum "$OUTPUT/result.json"
