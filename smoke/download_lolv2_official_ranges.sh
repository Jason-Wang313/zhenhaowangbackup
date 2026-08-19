#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
URL="https://drive.usercontent.google.com/download?id=1dzuLCk9_gE2bFF222n3-7GVUlSVHpMYC&export=download&confirm=t"
SOURCE_DIR="$ROOT/data/real_lowlight_validation/sources"
PREFIX="$SOURCE_DIR/LOL-v2.zip.partial"
PART_DIR="$SOURCE_DIR/lolv2_official_ranges"
ARCHIVE="$SOURCE_DIR/LOL-v2.zip"
TOTAL_BYTES=1046491030
PREFIX_BYTES=213614592
CONNECTIONS=8
RESUME_CONNECTIONS=4

if [[ ! -f "$PREFIX" ]]; then
  echo "ERROR: preserved LOL-v2 prefix is missing: $PREFIX" >&2
  exit 1
fi
actual_prefix_bytes="$(stat -c '%s' "$PREFIX")"
if [[ "$actual_prefix_bytes" -ne "$PREFIX_BYTES" ]]; then
  echo "ERROR: preserved LOL-v2 prefix size changed: $actual_prefix_bytes" >&2
  exit 1
fi
if [[ -e "$ARCHIVE" ]]; then
  if [[ ! -f "$ARCHIVE" || "$(stat -c '%s' "$ARCHIVE")" -ne "$TOTAL_BYTES" ]]; then
    echo "ERROR: existing immutable LOL-v2 archive differs: $ARCHIVE" >&2
    exit 1
  fi
  unzip -tq "$ARCHIVE"
  echo "LOLV2_OFFICIAL_RANGE_DOWNLOAD_PASS existing_verified_archive=$ARCHIVE"
  exit 0
fi

mkdir -p "$PART_DIR"
remaining_bytes=$((TOTAL_BYTES - PREFIX_BYTES))
chunk_bytes=$(((remaining_bytes + CONNECTIONS - 1) / CONNECTIONS))
pids=()
parts=()

download_range() {
  start="$1"
  end="$2"
  part="$3"
  expected_bytes=$((end - start + 1))
  existing_bytes=0
  if [[ -e "$part" ]]; then
    if [[ ! -f "$part" ]]; then
      echo "ERROR: LOL-v2 range path is not a file: $part" >&2
      return 1
    fi
    existing_bytes="$(stat -c '%s' "$part")"
    if [[ "$existing_bytes" -gt "$expected_bytes" ]]; then
      echo "ERROR: LOL-v2 range is oversized: $part" >&2
      return 1
    fi
  fi
  if [[ "$existing_bytes" -lt "$expected_bytes" ]]; then
    resume_start=$((start + existing_bytes))
    if [[ "$existing_bytes" -eq 0 ]]; then
      echo "LOLV2_RANGE_BEGIN start=$resume_start end=$end part=$part"
      curl \
        --location \
        --fail \
        --silent \
        --show-error \
        --retry 10 \
        --retry-all-errors \
        --range "$resume_start-$end" \
        --output - \
        "$URL" >>"$part"
    else
      resume_bytes=$((end - resume_start + 1))
      resume_chunk_bytes=$(((resume_bytes + RESUME_CONNECTIONS - 1) / RESUME_CONNECTIONS))
      resume_pids=()
      resume_parts=()
      for resume_index in $(seq 0 $((RESUME_CONNECTIONS - 1))); do
        sub_start=$((resume_start + resume_index * resume_chunk_bytes))
        sub_end=$((sub_start + resume_chunk_bytes - 1))
        if [[ "$sub_start" -gt "$end" ]]; then
          break
        fi
        if [[ "$sub_end" -gt "$end" ]]; then
          sub_end="$end"
        fi
        resume_part="${part}.resume_${sub_start}_${sub_end}"
        realized_resume_bytes=0
        if [[ -e "$resume_part" ]]; then
          if [[ ! -f "$resume_part" ]]; then
            echo "ERROR: LOL-v2 resume path is not a file: $resume_part" >&2
            return 1
          fi
          realized_resume_bytes="$(stat -c '%s' "$resume_part")"
        fi
        expected_resume_bytes=$((sub_end - sub_start + 1))
        if [[ "$realized_resume_bytes" -gt "$expected_resume_bytes" ]]; then
          echo "ERROR: LOL-v2 resume range is oversized: $resume_part" >&2
          return 1
        fi
        resume_parts+=("$resume_part")
        realized_sub_start=$((sub_start + realized_resume_bytes))
        echo "LOLV2_RANGE_RESUME_BEGIN start=$realized_sub_start end=$sub_end part=$resume_part"
        (
          if [[ "$realized_resume_bytes" -lt "$expected_resume_bytes" ]]; then
            curl \
              --location \
              --fail \
              --silent \
              --show-error \
              --retry 10 \
              --retry-all-errors \
              --range "$realized_sub_start-$sub_end" \
              --output - \
              "$URL" >>"$resume_part"
          fi
          if [[ "$(stat -c '%s' "$resume_part")" -ne "$expected_resume_bytes" ]]; then
            echo "ERROR: LOL-v2 resume range size differs: $resume_part" >&2
            exit 1
          fi
        ) &
        resume_pids+=("$!")
      done
      resume_failed=0
      for resume_pid in "${resume_pids[@]}"; do
        if ! wait "$resume_pid"; then
          resume_failed=1
        fi
      done
      if [[ "$resume_failed" -ne 0 ]]; then
        return 1
      fi
      for resume_part in "${resume_parts[@]}"; do
        dd if="$resume_part" of="$part" oflag=append conv=notrunc status=none
      done
    fi
  fi
  actual_bytes="$(stat -c '%s' "$part")"
  if [[ "$actual_bytes" -ne "$expected_bytes" ]]; then
    echo "ERROR: LOL-v2 range size differs expected=$expected_bytes actual=$actual_bytes part=$part" >&2
    return 1
  fi
  echo "LOLV2_RANGE_PASS start=$start end=$end bytes=$actual_bytes part=$part"
}

for index in $(seq 0 $((CONNECTIONS - 1))); do
  start=$((PREFIX_BYTES + index * chunk_bytes))
  end=$((start + chunk_bytes - 1))
  if [[ "$end" -ge "$TOTAL_BYTES" ]]; then
    end=$((TOTAL_BYTES - 1))
  fi
  part="$PART_DIR/range_${start}_${end}.part"
  parts+=("$part")
  download_range "$start" "$end" "$part" &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if [[ "$failed" -ne 0 ]]; then
  echo "ERROR: one or more LOL-v2 ranges failed; rerun to resume" >&2
  exit 1
fi

cp --reflink=auto "$PREFIX" "$ARCHIVE"
for part in "${parts[@]}"; do
  dd if="$part" of="$ARCHIVE" oflag=append conv=notrunc status=none
done
actual_archive_bytes="$(stat -c '%s' "$ARCHIVE")"
if [[ "$actual_archive_bytes" -ne "$TOTAL_BYTES" ]]; then
  echo "ERROR: assembled LOL-v2 archive size differs: $actual_archive_bytes" >&2
  exit 1
fi
unzip -tq "$ARCHIVE"
sha256sum "$ARCHIVE"
echo "LOLV2_OFFICIAL_RANGE_DOWNLOAD_PASS archive=$ARCHIVE bytes=$actual_archive_bytes"
