#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FLUX_REPO="$ROOT/third_party/flux2"
PATCH_FILE="$ROOT/patches/flux2-offline-smoke.patch"
VENV="$ROOT/.venv/flux2-smoke"
BASE_PYTHON="${BASE_PYTHON:-$ROOT/.conda/gpu-train-stable/bin/python}"
EXPECTED_FLUX_REVISION="50fe5162777813d869182b139e83b10743caef15"

if [[ ! -x "$BASE_PYTHON" ]]; then
  echo "ERROR: base Python is unavailable: $BASE_PYTHON" >&2
  exit 1
fi
if ! git -C "$FLUX_REPO" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "ERROR: FLUX.2 checkout is unavailable: $FLUX_REPO" >&2
  exit 1
fi
if [[ "$(git -C "$FLUX_REPO" rev-parse HEAD)" != "$EXPECTED_FLUX_REVISION" ]]; then
  echo "ERROR: FLUX.2 revision differs from the validated revision" >&2
  exit 1
fi

if git -C "$FLUX_REPO" apply --check "$PATCH_FILE"; then
  git -C "$FLUX_REPO" apply "$PATCH_FILE"
elif git -C "$FLUX_REPO" apply --reverse --check "$PATCH_FILE"; then
  echo "FLUX.2 offline smoke patch is already applied"
else
  echo "ERROR: FLUX.2 checkout is neither clean nor exactly patched as expected" >&2
  exit 1
fi

"$BASE_PYTHON" -m venv --system-site-packages "$VENV"
"$VENV/bin/python" -m pip install --upgrade pip setuptools wheel
"$VENV/bin/python" -m pip install --requirement "$ROOT/smoke/flux2_smoke_requirements.txt"
"$VENV/bin/python" -m pip install --editable "$FLUX_REPO" --no-deps --no-build-isolation
"$VENV/bin/python" -c "import cv2, fire, flux2, openai, safetensors, transformers; print('FLUX.2 smoke environment imports PASS')"

echo "Environment ready: $VENV"
echo "Run the GPU smoke in the GPU terminal with: CUDA_VISIBLE_DEVICES=0 smoke/run_flux2_gpu_smoke.sh"
