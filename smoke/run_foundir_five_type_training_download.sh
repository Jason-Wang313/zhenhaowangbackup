#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
BIN="$ROOT/.tools/baidupcs-go-v4.0.2/verified/BaiduPCS-Go-v4.0.2-linux-amd64/BaiduPCS-Go"
CFG="$ROOT/.tools/baidupcs-go-v4.0.2/private-config"
MANIFEST="$ROOT/manifests/foundir_five_type_training_archives_baidupan_v1.tsv"
REMOTE_BASE='/我的资源/FoundIR-Training Set'
STAGE="$ROOT/data/foundir/_download_stage/five_type_archives_v1"
STATE="$ROOT/data/foundir/_download_stage/state_v1"
LOCK="$STATE/download.lock"
COMPLETE="$STATE/download.complete"
INSTALLER="$ROOT/smoke/install_foundir_five_type_training_archives.sh"
EXPECTED_FILES=27
EXPECTED_BYTES=3737165269315
DOWNLOAD_WORKERS=${FOUNDIR_DOWNLOAD_WORKERS:-12}
CONNECTIONS_PER_WORKER=${FOUNDIR_DOWNLOAD_CONNECTIONS:-15}

mkdir -p "$STAGE" "$STATE"
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "FOUNDIR_FIVE_TYPE_DOWNLOAD_ALREADY_RUNNING"
  exit 75
fi

on_exit() {
  local status=$?
  echo "FOUNDIR_FIVE_TYPE_DOWNLOAD_EXIT utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) status=$status"
}
trap on_exit EXIT

run_installer_when_available() {
  local installer_status
  while true; do
    if "$INSTALLER"; then
      return 0
    else
      installer_status=$?
    fi
    if [[ $installer_status -ne 75 ]]; then
      return "$installer_status"
    fi
    echo "FOUNDIR_FIVE_TYPE_INSTALL_LOCKED retry_in_seconds=60"
    sleep 60
  done
}

if [[ ! -x "$BIN" ]]; then
  echo "ERROR: verified BaiduPCS-Go binary is missing or not executable: $BIN" >&2
  exit 1
fi
if [[ ! -d "$CFG" ]] || [[ "$(stat -c '%a' "$CFG")" != 700 ]]; then
  echo "ERROR: private BaiduPCS-Go config is absent or not mode 700: $CFG" >&2
  exit 1
fi
if [[ ! -f "$MANIFEST" ]]; then
  echo "ERROR: archive manifest is missing: $MANIFEST" >&2
  exit 1
fi
if [[ -f "$COMPLETE" ]]; then
  if grep -qx 'FOUNDIR_FIVE_TYPE_DOWNLOAD_COMPLETE' "$COMPLETE"; then
    echo "FOUNDIR_FIVE_TYPE_DOWNLOAD_REUSE marker=$COMPLETE"
    if [[ ! -x "$INSTALLER" ]]; then
      echo "ERROR: FoundIR five-type archive installer is missing or not executable: $INSTALLER" >&2
      exit 1
    fi
    run_installer_when_available
    exit 0
  fi
  echo "ERROR: existing FoundIR download marker is inconsistent: $COMPLETE" >&2
  exit 1
fi
if [[ ! "$DOWNLOAD_WORKERS" =~ ^[1-9][0-9]*$ ]] || [[ "$DOWNLOAD_WORKERS" -gt "$EXPECTED_FILES" ]]; then
  echo "ERROR: FOUNDIR_DOWNLOAD_WORKERS must be between 1 and $EXPECTED_FILES" >&2
  exit 1
fi
if [[ ! "$CONNECTIONS_PER_WORKER" =~ ^[1-9][0-9]*$ ]] || [[ "$CONNECTIONS_PER_WORKER" -gt 20 ]]; then
  echo "ERROR: FOUNDIR_DOWNLOAD_CONNECTIONS must be between 1 and 20" >&2
  exit 1
fi

mapfile -t records < <(tail -n +2 "$MANIFEST")
if [[ ${#records[@]} -ne $EXPECTED_FILES ]]; then
  echo "ERROR: expected $EXPECTED_FILES manifest records, found ${#records[@]}" >&2
  exit 1
fi

declare -a remote_paths=()
declare -a filenames=()
declare -a sizes=()
manifest_bytes=0
for record in "${records[@]}"; do
  IFS=$'\t' read -r filename size_bytes provider_md5 category <<<"$record"
  if [[ -z "$filename" || ! "$size_bytes" =~ ^[0-9]+$ || ! "$provider_md5" =~ ^[0-9a-f]{32}$ || -z "$category" ]]; then
    echo "ERROR: malformed manifest row: $record" >&2
    exit 1
  fi
  remote_paths+=("$REMOTE_BASE/$filename")
  filenames+=("$filename")
  sizes+=("$size_bytes")
  manifest_bytes=$((manifest_bytes + size_bytes))
done
if [[ $manifest_bytes -ne $EXPECTED_BYTES ]]; then
  echo "ERROR: expected $EXPECTED_BYTES manifest bytes, found $manifest_bytes" >&2
  exit 1
fi

complete_files=0
complete_bytes=0
for i in "${!filenames[@]}"; do
  local_path="$STAGE/${filenames[$i]}"
  if [[ -e "$local_path" ]]; then
    if [[ ! -f "$local_path" ]]; then
      echo "ERROR: expected a regular file at $local_path" >&2
      exit 1
    fi
    actual_size=$(stat -c '%s' "$local_path")
    if [[ "$actual_size" -ne "${sizes[$i]}" ]]; then
      resume_sidecar="$local_path.BaiduPCS-Go-downloading"
      if [[ "$actual_size" -lt "${sizes[$i]}" && -f "$resume_sidecar" ]]; then
        echo "FOUNDIR_FIVE_TYPE_RESUMABLE_PARTIAL file=${filenames[$i]} bytes=$actual_size"
        continue
      fi
      echo "ERROR: existing final-name file has wrong size without a BaiduPCS-Go resume sidecar: $local_path expected=${sizes[$i]} actual=$actual_size" >&2
      exit 1
    fi
    complete_files=$((complete_files + 1))
    complete_bytes=$((complete_bytes + actual_size))
  fi
done

echo "FOUNDIR_FIVE_TYPE_DOWNLOAD_START utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "remote_base=$REMOTE_BASE"
echo "stage=$STAGE"
echo "expected_files=$EXPECTED_FILES expected_bytes=$EXPECTED_BYTES already_complete_files=$complete_files already_complete_bytes=$complete_bytes"
echo "download_workers=$DOWNLOAD_WORKERS connections_per_worker=$CONNECTIONS_PER_WORKER"

export BAIDUPCS_GO_CONFIG_DIR="$CFG"
"$BIN" config set -cache_size 256KB -max_parallel "$CONNECTIONS_PER_WORKER" -max_download_load 1 -savedir "$STAGE"

download_worker() {
  local worker=$1
  local i
  for ((i=worker; i<EXPECTED_FILES; i+=DOWNLOAD_WORKERS)); do
    echo "FOUNDIR_FIVE_TYPE_WORKER_START worker=$worker file=${filenames[$i]} utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    "$BIN" download --saveto "$STAGE" -p "$CONNECTIONS_PER_WORKER" -l 1 --retry 100 --mtime "${remote_paths[$i]}"
    echo "FOUNDIR_FIVE_TYPE_WORKER_DONE worker=$worker file=${filenames[$i]} utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  done
}

declare -a worker_pids=()
for ((worker=0; worker<DOWNLOAD_WORKERS; worker++)); do
  download_worker "$worker" &
  worker_pids+=("$!")
done

download_status=0
for worker in "${!worker_pids[@]}"; do
  if ! wait "${worker_pids[$worker]}"; then
    echo "ERROR: download worker failed: worker=$worker pid=${worker_pids[$worker]}" >&2
    download_status=1
  fi
done
if [[ $download_status -ne 0 ]]; then
  exit 1
fi

verified_files=0
verified_bytes=0
for i in "${!filenames[@]}"; do
  local_path="$STAGE/${filenames[$i]}"
  if [[ ! -f "$local_path" ]]; then
    echo "ERROR: missing completed archive: $local_path" >&2
    exit 1
  fi
  actual_size=$(stat -c '%s' "$local_path")
  if [[ "$actual_size" -ne "${sizes[$i]}" ]]; then
    echo "ERROR: completed archive has wrong size: $local_path expected=${sizes[$i]} actual=$actual_size" >&2
    exit 1
  fi
  if [[ -e "$local_path.BaiduPCS-Go-downloading" ]]; then
    echo "ERROR: resume sidecar remains after completed download: $local_path.BaiduPCS-Go-downloading" >&2
    exit 1
  fi
  verified_files=$((verified_files + 1))
  verified_bytes=$((verified_bytes + actual_size))
done

if [[ $verified_files -ne $EXPECTED_FILES || $verified_bytes -ne $EXPECTED_BYTES ]]; then
  echo "ERROR: final archive accounting mismatch files=$verified_files bytes=$verified_bytes" >&2
  exit 1
fi

printf 'FOUNDIR_FIVE_TYPE_DOWNLOAD_COMPLETE\nutc=%s\nfiles=%s\nbytes=%s\nmanifest=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$verified_files" "$verified_bytes" "$MANIFEST" >"$COMPLETE"
echo "FOUNDIR_FIVE_TYPE_DOWNLOAD_PASS files=$verified_files bytes=$verified_bytes marker=$COMPLETE"
if [[ ! -x "$INSTALLER" ]]; then
  echo "ERROR: FoundIR five-type archive installer is missing or not executable: $INSTALLER" >&2
  exit 1
fi
run_installer_when_available
