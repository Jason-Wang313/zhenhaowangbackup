#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${LOW_LIGHT_PYTHON:-$ROOT/.venv/flux2-smoke/bin/python}"
SOURCE_ROOT="${LOW_LIGHT_SOURCE_ROOT:-$ROOT/data/clean_pool}"
DATASET_METADATA="${LOW_LIGHT_DATASET_METADATA:-$ROOT/manifests/clean_source_datasets.json}"
SOURCE_MANIFEST="${LOW_LIGHT_SOURCE_MANIFEST:-$ROOT/manifests/low_light_source_pool_2000.json}"
PAIR_MANIFEST="${LOW_LIGHT_PAIR_MANIFEST:-$ROOT/manifests/low_light_pilot_10000.csv}"
PAIR_METADATA="${LOW_LIGHT_PAIR_METADATA:-$ROOT/manifests/low_light_pilot_10000.json}"
OUTPUT_ROOT="${LOW_LIGHT_OUTPUT_ROOT:-$ROOT/data/low_light_pilot}"
WORKERS="${LOW_LIGHT_WORKERS:-2}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: validated workspace Python is unavailable: $PYTHON" >&2
  exit 1
fi
if [[ ! "$WORKERS" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: LOW_LIGHT_WORKERS must be a positive integer: $WORKERS" >&2
  exit 1
fi
overlap_args=()
if [[ -n "${FOUNDIR_SHA256_INDEX:-}" ]]; then
  overlap_args=(--foundir-sha256-index "$FOUNDIR_SHA256_INDEX")
fi
metadata_args=()
if [[ -f "$DATASET_METADATA" ]]; then
  metadata_args=(--dataset-metadata "$DATASET_METADATA")
fi

cd "$ROOT"
"$PYTHON" -c "import cv2, numpy, PIL; assert numpy.__version__ == '1.26.4'; print('CPU dependency preflight PASS')"
"$PYTHON" "$ROOT/smoke/check_low_light_source_pool.py" \
  --source-root "$SOURCE_ROOT" \
  --output "$SOURCE_MANIFEST" \
  "${metadata_args[@]}" \
  "${overlap_args[@]}"
"$PYTHON" "$ROOT/generate_low_light_pilot.py" \
  --source-manifest "$SOURCE_MANIFEST" \
  --output-root "$OUTPUT_ROOT" \
  --manifest "$PAIR_MANIFEST" \
  --metadata "$PAIR_METADATA" \
  --workers "$WORKERS" \
  --resume
"$PYTHON" "$ROOT/smoke/validate_low_light_manifest.py" "$PAIR_MANIFEST" \
  --expected-pairs 10000 \
  --expected-sources 2000 \
  --expected-dataset-count mio100=100 \
  --expected-dataset-count div2k=800 \
  --expected-dataset-count lsdir=1100 \
  --workers "$WORKERS"

echo "LOW-LIGHT 10,000-PAIR PILOT PASS: $PAIR_MANIFEST"
