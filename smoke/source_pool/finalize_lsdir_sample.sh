#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ARCHIVE="$ROOT/downloads/source_pool/lsdir/shard-00.tar.gz"
DESTINATION="$ROOT/data/clean_pool/lsdir"
TARGET=1100
CANDIDATES="${LSDIR_CANDIDATES:-1300}"
PYTHON="${LOW_LIGHT_PYTHON:-$ROOT/.venv/flux2-smoke/bin/python}"

if [[ ! -f "$ARCHIVE" ]]; then
  echo "ERROR: missing LSDIR shard: $ARCHIVE" >&2
  exit 1
fi
tar -tzf "$ARCHIVE" >/dev/null
if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: validated workspace Python is unavailable: $PYTHON" >&2
  exit 1
fi
mkdir -p "$DESTINATION"
"$PYTHON" "$ROOT/smoke/source_pool/stream_lsdir_candidates.py" \
  --archive "$ARCHIVE" --count "$CANDIDATES" --output "$DESTINATION"

count="$(find "$DESTINATION" -maxdepth 1 -type f ! -name '.*.partial' | wc -l)"
if [[ "$count" -lt "$CANDIDATES" ]]; then
  echo "ERROR: expected at least $CANDIDATES staged LSDIR candidates, found $count" >&2
  exit 1
fi
echo "LSDIR TRAIN CANDIDATE STAGING PASS: $count images available from shard-00"
