#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ARCHIVE="$ROOT/downloads/source_pool/shard-00.tar.gz"
STAGING="$ROOT/downloads/source_pool/lsdir_shard_00"
DESTINATION="$ROOT/data/clean_pool/lsdir"
TARGET=1100

if [[ ! -f "$ARCHIVE" ]]; then
  echo "ERROR: missing LSDIR shard: $ARCHIVE" >&2
  exit 1
fi
tar -tzf "$ARCHIVE" >/dev/null
mkdir -p "$STAGING" "$DESTINATION"
tar -xzf "$ARCHIVE" -C "$STAGING"

mapfile -t images < <(find "$STAGING" -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \) | sort)
if [[ "${#images[@]}" -lt "$TARGET" ]]; then
  echo "ERROR: shard-00 contains only ${#images[@]} images; need $TARGET" >&2
  exit 1
fi
for ((index=0; index<TARGET; index++)); do
  source="${images[$index]}"
  name="$(basename "$source")"
  cp --update=none "$source" "$DESTINATION/$name"
done

count="$(find "$DESTINATION" -maxdepth 1 -type f | wc -l)"
if [[ "$count" -ne "$TARGET" ]]; then
  echo "ERROR: expected $TARGET staged LSDIR images, found $count" >&2
  exit 1
fi
echo "LSDIR TRAIN SAMPLE STAGING PASS: $count images from shard-00"
