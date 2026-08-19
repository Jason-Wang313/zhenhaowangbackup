#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
URL="https://drive.usercontent.google.com/download?id=1dzuLCk9_gE2bFF222n3-7GVUlSVHpMYC&export=download&confirm=t"
SOURCE_DIR="$ROOT/data/real_lowlight_validation/sources"
INVALID_CANDIDATE="$SOURCE_DIR/LOL-v2.zip"
RECOVERY_DIR="$SOURCE_DIR/lolv2_official_recovery"
VERIFIED_ARCHIVE="$SOURCE_DIR/LOL-v2.verified.zip"
TOTAL_BYTES=1046491030
GOOD_PREFIX_BYTES=999565536
GOOD_SUFFIX_START=1010548496
CONNECTIONS=4

if [[ ! -f "$INVALID_CANDIDATE" || "$(stat -c '%s' "$INVALID_CANDIDATE")" -ne "$TOTAL_BYTES" ]]; then
  echo "ERROR: the preserved invalid LOL-v2 candidate has an unexpected size" >&2
  exit 1
fi
if [[ -e "$VERIFIED_ARCHIVE" ]]; then
  if [[ ! -f "$VERIFIED_ARCHIVE" || "$(stat -c '%s' "$VERIFIED_ARCHIVE")" -ne "$TOTAL_BYTES" ]]; then
    echo "ERROR: existing immutable recovered LOL-v2 archive differs" >&2
    exit 1
  fi
  unzip -tq "$VERIFIED_ARCHIVE"
  sha256sum "$VERIFIED_ARCHIVE"
  echo "LOLV2_OFFICIAL_RECOVERY_PASS existing_verified_archive=$VERIFIED_ARCHIVE"
  exit 0
fi

mkdir -p "$RECOVERY_DIR"
recovery_bytes=$((GOOD_SUFFIX_START - GOOD_PREFIX_BYTES))
chunk_bytes=$(((recovery_bytes + CONNECTIONS - 1) / CONNECTIONS))
pids=()
parts=()

download_recovery_range() {
  start="$1"
  end="$2"
  part="$3"
  expected_bytes=$((end - start + 1))
  existing_bytes=0
  if [[ -e "$part" ]]; then
    if [[ ! -f "$part" ]]; then
      echo "ERROR: LOL-v2 recovery path is not a file: $part" >&2
      return 1
    fi
    existing_bytes="$(stat -c '%s' "$part")"
  fi
  if [[ "$existing_bytes" -gt "$expected_bytes" ]]; then
    echo "ERROR: LOL-v2 recovery range is oversized: $part" >&2
    return 1
  fi
  if [[ "$existing_bytes" -lt "$expected_bytes" ]]; then
    resume_start=$((start + existing_bytes))
    echo "LOLV2_RECOVERY_RANGE_BEGIN start=$resume_start end=$end part=$part"
    curl \
      --location \
      --fail \
      --silent \
      --show-error \
      --range "$resume_start-$end" \
      --output - \
      "$URL" >>"$part"
  fi
  actual_bytes="$(stat -c '%s' "$part")"
  if [[ "$actual_bytes" -ne "$expected_bytes" ]]; then
    echo "ERROR: LOL-v2 recovery range size differs: $part" >&2
    return 1
  fi
  echo "LOLV2_RECOVERY_RANGE_PASS start=$start end=$end bytes=$actual_bytes"
}

for index in $(seq 0 $((CONNECTIONS - 1))); do
  start=$((GOOD_PREFIX_BYTES + index * chunk_bytes))
  end=$((start + chunk_bytes - 1))
  if [[ "$end" -ge "$GOOD_SUFFIX_START" ]]; then
    end=$((GOOD_SUFFIX_START - 1))
  fi
  part="$RECOVERY_DIR/range_${start}_${end}.part"
  parts+=("$part")
  download_recovery_range "$start" "$end" "$part" &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if [[ "$failed" -ne 0 ]]; then
  echo "ERROR: one or more LOL-v2 recovery ranges failed; rerun to resume" >&2
  exit 1
fi

dd \
  if="$INVALID_CANDIDATE" \
  of="$VERIFIED_ARCHIVE" \
  iflag=count_bytes \
  count="$GOOD_PREFIX_BYTES" \
  status=none
for part in "${parts[@]}"; do
  dd if="$part" of="$VERIFIED_ARCHIVE" oflag=append conv=notrunc status=none
done
dd \
  if="$INVALID_CANDIDATE" \
  of="$VERIFIED_ARCHIVE" \
  iflag=skip_bytes \
  skip="$GOOD_SUFFIX_START" \
  oflag=append \
  conv=notrunc \
  status=none

actual_bytes="$(stat -c '%s' "$VERIFIED_ARCHIVE")"
if [[ "$actual_bytes" -ne "$TOTAL_BYTES" ]]; then
  echo "ERROR: recovered LOL-v2 archive size differs: $actual_bytes" >&2
  exit 1
fi
unzip -tq "$VERIFIED_ARCHIVE"
sha256sum "$VERIFIED_ARCHIVE"
echo "LOLV2_OFFICIAL_RECOVERY_PASS archive=$VERIFIED_ARCHIVE bytes=$actual_bytes"
