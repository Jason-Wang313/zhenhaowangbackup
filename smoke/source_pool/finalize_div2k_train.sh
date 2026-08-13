#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ARCHIVE="$ROOT/downloads/source_pool/DIV2K_train_HR.zip"
STAGING="$ROOT/downloads/source_pool/div2k_extract"
DESTINATION="$ROOT/data/clean_pool/div2k"
EXPECTED_BYTES=3530603713

if [[ ! -f "$ARCHIVE" ]]; then
  echo "ERROR: missing DIV2K archive: $ARCHIVE" >&2
  exit 1
fi
actual_bytes="$(stat -c %s "$ARCHIVE")"
if [[ "$actual_bytes" -ne "$EXPECTED_BYTES" ]]; then
  echo "ERROR: incomplete DIV2K archive: $actual_bytes/$EXPECTED_BYTES bytes" >&2
  exit 1
fi
unzip -tq "$ARCHIVE" >/dev/null
mkdir -p "$STAGING" "$DESTINATION"
unzip -qn "$ARCHIVE" -d "$STAGING"

mapfile -t images < <(find "$STAGING/DIV2K_train_HR" -maxdepth 1 -type f -name '*.png' | sort)
if [[ "${#images[@]}" -ne 800 ]]; then
  echo "ERROR: expected 800 DIV2K train-HR images, found ${#images[@]}" >&2
  exit 1
fi
for source in "${images[@]}"; do
  name="$(basename "$source")"
  if [[ ! "$name" =~ ^0([0-7][0-9]{2}|800)\.png$ ]]; then
    echo "ERROR: unexpected DIV2K train filename: $name" >&2
    exit 1
  fi
  cp --update=none "$source" "$DESTINATION/$name"
done

count="$(find "$DESTINATION" -maxdepth 1 -type f -name '*.png' | wc -l)"
if [[ "$count" -ne 800 ]]; then
  echo "ERROR: expected 800 staged DIV2K images, found $count" >&2
  exit 1
fi
echo "DIV2K TRAIN-HR STAGING PASS: $count images"
