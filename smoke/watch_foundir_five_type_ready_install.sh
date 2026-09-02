#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
INSTALLER="$ROOT/smoke/install_foundir_five_type_training_archives.sh"
DOWNLOAD_MARKER="$ROOT/data/foundir/_download_stage/state_v1/download.complete"
DOWNLOAD_PIDFILE="$ROOT/data/foundir/_download_stage/state_v1/download.pid"
POLL_SECONDS=${FOUNDIR_READY_INSTALL_POLL_SECONDS:-900}

if [[ ! "$POLL_SECONDS" =~ ^[0-9]+$ ]] || [[ "$POLL_SECONDS" -lt 60 || "$POLL_SECONDS" -gt 3600 ]]; then
  echo "ERROR: FOUNDIR_READY_INSTALL_POLL_SECONDS must be between 60 and 3600" >&2
  exit 1
fi
if [[ ! -x "$INSTALLER" ]]; then
  echo "ERROR: incremental FoundIR installer is unavailable: $INSTALLER" >&2
  exit 1
fi

echo "FOUNDIR_FIVE_TYPE_READY_WATCH_START utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) poll_seconds=$POLL_SECONDS official_test_accessed=false"
while [[ ! -f "$DOWNLOAD_MARKER" ]]; do
  if [[ ! -f "$DOWNLOAD_PIDFILE" ]]; then
    echo "ERROR: FoundIR download PID file disappeared before completion" >&2
    exit 1
  fi
  download_pid=$(awk -F= '$1 == "pid" {print $2}' "$DOWNLOAD_PIDFILE")
  if [[ ! "$download_pid" =~ ^[0-9]+$ ]] || ! kill -0 "$download_pid" 2>/dev/null; then
    echo "ERROR: FoundIR downloader stopped before publishing its completion marker" >&2
    exit 1
  fi

  if env FOUNDIR_INSTALL_READY_ONLY=1 FOUNDIR_EXTRACT_WORKERS=4 "$INSTALLER"; then
    echo "FOUNDIR_FIVE_TYPE_READY_WATCH_CYCLE_PASS utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  else
    installer_status=$?
    if [[ $installer_status -eq 75 ]]; then
      echo "FOUNDIR_FIVE_TYPE_READY_WATCH_INSTALL_BUSY utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    else
      echo "ERROR: incremental FoundIR installation failed with status $installer_status" >&2
      exit "$installer_status"
    fi
  fi
  sleep "$POLL_SECONDS"
done
echo "FOUNDIR_FIVE_TYPE_READY_WATCH_DOWNLOAD_COMPLETE utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) final_installer_owned_by_downloader=true"
