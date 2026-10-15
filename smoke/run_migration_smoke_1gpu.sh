#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
EXPECTED_ROOT=/inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/zhenhaowang

if [[ "$ROOT" != "$EXPECTED_ROOT" ]]; then
  echo "ERROR: unexpected workspace: $ROOT" >&2
  exit 2
fi
if [[ "${CUDA_VISIBLE_DEVICES:-}" != "0" ]]; then
  echo "ERROR: CUDA_VISIBLE_DEVICES must be exactly 0" >&2
  exit 2
fi

PARENT="$ROOT/reports/migration_smoke"
RUN="$PARENT/cure_$(date -u +%Y%m%dT%H%M%SZ)_$$"
mkdir -p "$PARENT"
mkdir "$RUN"

exec > >(tee "$RUN/launcher.log") 2>&1
finish() {
  status=$?
  trap - EXIT
  printf 'MIGRATION_SMOKE_END status=%s utc=%s evidence_dir=%s\n' \
    "$status" "$(date -u +%FT%TZ)" "$RUN" | tee "$RUN/terminal_status.txt"
  exit "$status"
}
trap finish EXIT

cd "$ROOT"
export TORCH_HOME="$ROOT/models/torch"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export PYTHONPATH="$ROOT/.venv/cure-repro-py311/lib/python3.11/site-packages"

printf '%s  %s\n' \
  3fbecfadf3b6c630fa57b4cc3cbb90ccab0c71db6f4274b99f8a3cf595b6e042 "$ROOT/smoke/cure_gpu_rehearsal_v1.py" \
  01a428b7eced044cc26e5a6a939010d23eade49627d6e776d19938acef823119 "$ROOT/configs/cure_two_track_reproduction_v1.json" \
  f394e7fd180b062eaa268d1f16f85b33cdb02a97e65fa9caacd8f3824539ac7d "$ROOT/data/third_party/cure_h5_smoke_v1/shared_cure.h5" \
  0a852fd2f0dadfb71f0d1970f2cba9d0ee3f851ccfd123941c3511a2d08ae8d2 "$ROOT/data/third_party/cure_released/OneRestore_embedder.tar" \
  746365e9d25bc0861b2bc7d7bd63bbb9c978661bee2634fddb0b68def2c81b01 "$ROOT/data/third_party/cure_released/OneRestore_restorer.tar" \
  397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0 "$ROOT/models/torch/hub/checkpoints/vgg16-397923af.pth" \
  | sha256sum -c - | tee "$RUN/sha256_check.log"

/lib64/ld-linux-x86-64.so.2 \
  "$ROOT/.conda/gpu-train-clean/bin/python" \
  "$ROOT/smoke/cure_gpu_rehearsal_v1.py" \
  --h5 "$ROOT/data/third_party/cure_h5_smoke_v1/shared_cure.h5" \
  --embedder "$ROOT/data/third_party/cure_released/OneRestore_embedder.tar" \
  --baseline "$ROOT/data/third_party/cure_released/OneRestore_restorer.tar" \
  2>&1 | tee "$RUN/run.log"

grep -Fq '"status": "no_update_cuda_rehearsal_pass"' "$RUN/run.log"
printf 'MIGRATION_SMOKE_PASS utc=%s\n' "$(date -u +%FT%TZ)" | tee "$RUN/SUCCESS"
