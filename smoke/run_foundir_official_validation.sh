#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${FOUNDIR_PYTHON:-$ROOT/.venv/flux2-smoke/bin/python}"
RUNTIME_ROOT="${FOUNDIR_RUNTIME_ROOT:-$ROOT/downloads/foundir_official}"
GT_ARCHIVE="$RUNTIME_ROOT/GT.zip"
LQ_ARCHIVE="$RUNTIME_ROOT/LQ.zip"
GT_ROOT="$RUNTIME_ROOT/extracted_gt"
LQ_ROOT="$RUNTIME_ROOT/extracted_lq"

if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: workspace Python is unavailable: $PYTHON" >&2
  exit 1
fi
if [[ ! -f "$GT_ARCHIVE" || ! -f "$LQ_ARCHIVE" ]]; then
  echo "ERROR: official archives are unavailable under $RUNTIME_ROOT" >&2
  exit 1
fi
if [[ -e "$GT_ROOT.partial" || -e "$LQ_ROOT.partial" ]]; then
  echo "ERROR: partial extraction exists; refusing to continue" >&2
  exit 1
fi

cd "$ROOT"
if [[ ! -d "$GT_ROOT" ]]; then
  "$PYTHON" smoke/extract_foundir_archive.py "$GT_ARCHIVE" "$GT_ROOT"
fi
if [[ ! -d "$LQ_ROOT" ]]; then
  "$PYTHON" smoke/extract_foundir_archive.py "$LQ_ARCHIVE" "$LQ_ROOT"
fi

"$PYTHON" smoke/validate_foundir_official.py \
  --gt-root "$GT_ROOT" \
  --lq-root "$LQ_ROOT" \
  --gt-archive "$GT_ARCHIVE" \
  --lq-archive "$LQ_ARCHIVE"

echo "FOUNDIR OFFICIAL ACQUISITION AND VALIDATION PASS"
