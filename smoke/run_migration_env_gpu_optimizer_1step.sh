#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
EXPECTED_ROOT=/inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/zhenhaowang
PYTHON=/inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/.envs/gpu-train-stable-hdd3/bin/python

if [[ "$ROOT" != "$EXPECTED_ROOT" ]]; then
  echo "ERROR: unexpected workspace: $ROOT" >&2
  exit 2
fi
if [[ "${CUDA_VISIBLE_DEVICES:-}" != "0" ]]; then
  echo "ERROR: CUDA_VISIBLE_DEVICES must be exactly 0" >&2
  exit 2
fi
if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: new HDD3 Python is missing: $PYTHON" >&2
  exit 2
fi

PARENT="$ROOT/reports/migration_smoke"
RUN="$PARENT/gpu_optimizer_$(date -u +%Y%m%dT%H%M%SZ)_$$"
mkdir -p "$PARENT"
mkdir "$RUN"
printf 'CUDA_VISIBLE_DEVICES=0 bash smoke/run_migration_env_gpu_optimizer_1step.sh\n' > "$RUN/command.txt"

exec > >(tee "$RUN/launcher.log") 2>&1
finish() {
  status=$?
  trap - EXIT
  printf 'HDD3_GPU_FINAL_END status=%s utc=%s log=%s\n' \
    "$status" "$(date -u +%FT%TZ)" "$RUN/launcher.log" | tee "$RUN/terminal_status.txt"
  exit "$status"
}
trap finish EXIT

cd "$ROOT"
echo "pwd=$(pwd -P)"
echo "python=$PYTHON"
echo "config=$ROOT/configs/cure_two_track_reproduction_v1.json"
echo "dataset=$ROOT/data/third_party/cure_h5_smoke_v1/shared_cure.h5"
echo "embedder=$ROOT/data/third_party/cure_released/OneRestore_embedder.tar"
echo "restorer=$ROOT/data/third_party/cure_released/OneRestore_restorer.tar"
echo "vgg=$ROOT/models/torch/hub/checkpoints/vgg16-397923af.pth"
echo "output=$RUN"

env -i \
  PATH="$(dirname "$PYTHON"):/usr/bin:/bin" \
  HOME="$RUN/home" \
  TMPDIR=/tmp \
  CUDA_VISIBLE_DEVICES=0 \
  TORCH_HOME="$ROOT/models/torch" \
  PYTHONNOUSERSITE=1 \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONUNBUFFERED=1 \
  OMP_NUM_THREADS=1 \
  "$PYTHON" "$ROOT/smoke/migration_env_gpu_optimizer_save_reload.py" \
  --output-dir "$RUN" 2>&1 | tee "$RUN/run.log"

grep -Fq 'HDD3_NEW_ENV_GPU_OPTIMIZER_SAVE_RELOAD_OK' "$RUN/run.log"
test -s "$RUN/temporary_step_000001.pt"
test -s "$RUN/reload_verification.json"
test -s "$RUN/result.json"
printf 'HDD3_NEW_ENV_GPU_OPTIMIZER_SAVE_RELOAD_OK\n' | tee "$RUN/SUCCESS"
