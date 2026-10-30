#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
PYTHON=${FLUX_TAIL_V3_PYTHON:-}
RUN_ID=${1:-baseline_parity_2gpu_v1}
[[ "$PYTHON" == /* && -x "$PYTHON" && -f "$PYTHON" ]] || { echo 'Set FLUX_TAIL_V3_PYTHON to the qualified absolute Python path'; exit 2; }
[[ "$RUN_ID" =~ ^[A-Za-z0-9_-]+$ ]] || exit 2
VISIBLE=${CUDA_VISIBLE_DEVICES:-0,1}
[[ "$VISIBLE" =~ ^([0-9]+|GPU-[A-Za-z0-9-]+),([0-9]+|GPU-[A-Za-z0-9-]+)$ ]] || exit 2
[[ "${BASH_REMATCH[1]}" != "${BASH_REMATCH[2]}" ]] || exit 2
COMMAND=("$PYTHON" -u "$ROOT/smoke/diagnose_flux_tail_baseline_parity_v1.py" run --run-id "$RUN_ID")
if [[ "${FLUX_TAIL_PARITY_DRY_RUN:-0}" == 1 ]]; then
  printf 'NODES=1 GPUS=2 WALL_CAP=900 VISIBLE=%s\n' "$VISIBLE"
  printf '%q ' "${COMMAND[@]}"; printf '\n'; exit 0
fi
cd "$ROOT"
unset PYTHONHOME
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
exec 7>/tmp/zhenhaowang_flux_tail_switch_v2.lock
flock -n 7 || { echo 'Another tail job owns this node'; exit 3; }
exec 9>/tmp/zhenhaowang_dual_retry_v1.lock
flock -n 9 || { echo 'Another project job owns this node'; exit 3; }
exec 8>/tmp/zhenhaowang_rain_blur_image_moe_v1.lock
flock -n 8 || { echo 'Another project job owns this node'; exit 3; }
export CUDA_VISIBLE_DEVICES="$VISIBLE" OMP_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCH_HOME="$ROOT/models/torch"
export PYTHONPATH="$ROOT/third_party/flux2/src:$ROOT"
mkdir -p logs
exec > >(tee -a "logs/flux_tail_baseline_parity_${RUN_ID}.log") 2>&1
STARTED=$SECONDS
trap 'status=$?; elapsed=$((SECONDS-STARTED)); echo "BASELINE_PARITY_END status=$status elapsed_seconds=$elapsed local_gpu_seconds=$((2*elapsed)) utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"' EXIT
echo "BASELINE_PARITY_START run_id=$RUN_ID nodes=1 gpus=2 utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo 'BASELINE_PARITY_BUDGET estimate_minutes=3-8 unmeasured wall_cap_seconds=900 used_gpu_hours_cap=0.5 cleanup_seconds=30 no_training no_auto_retry'
timeout --signal=TERM --kill-after=30s 900 "${COMMAND[@]}"
