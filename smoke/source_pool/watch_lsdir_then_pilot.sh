#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DOWNLOAD_DIR="$ROOT/downloads/source_pool/lsdir"
ARCHIVE="$DOWNLOAD_DIR/shard-00.tar.gz"
EXPECTED_BYTES=8979223633
STATE_DIR="$ROOT/logs/lsdir_pilot_watchdog"
LOG="$STATE_DIR/watchdog.log"
COMPLETE_MARKER="$STATE_DIR/low_light_pilot_10000.complete"
SOURCE_MANIFEST="$ROOT/manifests/low_light_source_pool_2000.json"
PAIR_MANIFEST="$ROOT/manifests/low_light_pilot_10000.csv"
PAIR_METADATA="$ROOT/manifests/low_light_pilot_10000.json"
OUTPUT_ROOT="$ROOT/data/low_light_pilot"
PYTHON="${LOW_LIGHT_PYTHON:-$ROOT/.venv/flux2-smoke/bin/python}"
CHECK_INTERVAL_SECONDS="${LSDIR_CHECK_INTERVAL_SECONDS:-300}"

mkdir -p "$DOWNLOAD_DIR" "$STATE_DIR"
exec 9>"$STATE_DIR/watchdog.lock"
if ! flock -n 9; then
  echo "Another LSDIR watchdog is already running." >&2
  exit 0
fi

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$LOG"
}

archive_is_complete() {
  [[ -f "$ARCHIVE" ]] && [[ "$(stat -c %s "$ARCHIVE")" -eq "$EXPECTED_BYTES" ]]
}

download_is_running() {
  pgrep -f "hf download ofsoundof/LSDIR shard-00.tar.gz" >/dev/null
}

resume_download() {
  log "No active downloader; resuming the official Hugging Face Xet download."
  if HF_XET_HIGH_PERFORMANCE=1 HF_XET_RECONSTRUCT_WRITE_SEQUENTIALLY=1 \
    hf download ofsoundof/LSDIR shard-00.tar.gz \
      --repo-type model --local-dir "$DOWNLOAD_DIR" >>"$LOG" 2>&1; then
    log "Hugging Face download command completed."
  else
    log "Download command exited unsuccessfully; cached data is preserved and will be retried."
  fi
}

if [[ -f "$COMPLETE_MARKER" ]]; then
  log "Completion marker already exists; no work needed."
  exit 0
fi

log "Watchdog started; checks do not consume model tokens."
while ! archive_is_complete; do
  if [[ -f "$ARCHIVE" ]]; then
    actual_bytes="$(stat -c %s "$ARCHIVE")"
    if ! download_is_running; then
      log "Final archive exists with unexpected size $actual_bytes; refusing to overwrite it."
      exit 1
    fi
  fi

  if download_is_running; then
    log "Download process is active; leaving it uninterrupted."
    sleep "$CHECK_INTERVAL_SECONDS"
  else
    resume_download
    sleep 5
  fi
done

log "Archive reached the expected size; validating gzip/tar integrity."
if ! tar -tzf "$ARCHIVE" >/dev/null; then
  log "Archive integrity validation failed; preserving all files for diagnosis."
  exit 1
fi
log "Archive integrity PASS."

ready=0
for candidates in 1300 1600 2000 2500 3000; do
  log "Staging the first $candidates LSDIR training candidates."
  if ! LSDIR_CANDIDATES="$candidates" "$ROOT/smoke/source_pool/finalize_lsdir_sample.sh" >>"$LOG" 2>&1; then
    log "LSDIR candidate staging failed; preserving current outputs."
    exit 1
  fi
  if "$PYTHON" "$ROOT/smoke/check_low_light_source_pool.py" \
      --source-root "$ROOT/data/clean_pool" \
      --output "$SOURCE_MANIFEST" >>"$LOG" 2>&1; then
    ready=1
    log "2,000-source quality and deduplication gate PASS."
    break
  fi
  log "$candidates candidates were insufficient after quality/deduplication; expanding safely."
done

if [[ "$ready" -ne 1 ]]; then
  log "Unable to obtain 1,100 accepted LSDIR sources from 3,000 candidates."
  exit 1
fi

log "Starting the CPU-only 10,000-pair low-light pilot."
if ! LOW_LIGHT_SOURCE_MANIFEST="$SOURCE_MANIFEST" \
    LOW_LIGHT_PAIR_MANIFEST="$PAIR_MANIFEST" \
    LOW_LIGHT_PAIR_METADATA="$PAIR_METADATA" \
    LOW_LIGHT_OUTPUT_ROOT="$OUTPUT_ROOT" \
    "$ROOT/smoke/run_low_light_pilot_10000.sh" >>"$LOG" 2>&1; then
  log "10,000-pair pilot failed; preserving all generated data and logs."
  exit 1
fi

date -u +%Y-%m-%dT%H:%M:%SZ >"$COMPLETE_MARKER"
log "10,000-pair low-light pilot PASS."
