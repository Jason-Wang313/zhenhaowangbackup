#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${LOW_LIGHT_PYTHON:-$ROOT/.venv/flux2-smoke/bin/python}"
OUTPUT_ROOT="${LOW_LIGHT_SMOKE_OUTPUT_ROOT:-$ROOT/data/low_light_pilot_smoke_100}"
MANIFEST="${LOW_LIGHT_SMOKE_MANIFEST:-$ROOT/manifests/low_light_pilot_smoke_500.csv}"
METADATA="${LOW_LIGHT_SMOKE_METADATA:-$ROOT/manifests/low_light_pilot_smoke_500.json}"

if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: validated workspace Python is unavailable: $PYTHON" >&2
  exit 1
fi

cd "$ROOT"
"$PYTHON" -c "import cv2, numpy; assert numpy.__version__.startswith('1.'); print('CPU dependency preflight PASS')"
"$PYTHON" "$ROOT/generate_low_light_pilot.py" \
  --source-root "$ROOT/data/clean" \
  --allow-fewer-sources \
  --num-sources 100 \
  --output-root "$OUTPUT_ROOT" \
  --manifest "$MANIFEST" \
  --metadata "$METADATA"
"$PYTHON" "$ROOT/smoke/validate_low_light_manifest.py" "$MANIFEST" \
  --expected-pairs 500 \
  --expected-sources 100 \
  --expected-dataset-count smoke=100 \
  --allow-smoke-license
