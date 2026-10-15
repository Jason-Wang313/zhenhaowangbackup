#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPECTED_ROOT="/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang"
PYTHON="$ROOT/.venv/flux2-smoke/bin/python"
MODEL_DIR="$ROOT/models/qwen2.5-vl-7b-instruct"
CONFIG="$ROOT/configs/qwen_lowlight_smoke25.json"
LOG="$ROOT/logs/qwen_download_monitor.log"
DOWNLOAD_PID="${1:-}"

if [[ "$ROOT" != "$EXPECTED_ROOT" ]]; then
  echo "ERROR: unexpected workspace: $ROOT" >&2
  exit 1
fi
if [[ ! "$DOWNLOAD_PID" =~ ^[1-9][0-9]*$ ]]; then
  echo "Usage: $0 DOWNLOAD_PID" >&2
  exit 2
fi
if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: validated workspace Python is unavailable: $PYTHON" >&2
  exit 1
fi

mkdir -p "$ROOT/logs"
exec >>"$LOG" 2>&1

echo "QWEN_DOWNLOAD_MONITOR_START $(date -u +%FT%TZ) pid=$DOWNLOAD_PID"
ticks=0
download_command=""
if [[ -r "/proc/$DOWNLOAD_PID/cmdline" ]]; then
  download_command="$(tr '\0' ' ' <"/proc/$DOWNLOAD_PID/cmdline")"
fi
while [[ "$download_command" == *"hf download Qwen/Qwen2.5-VL-7B-Instruct"* ]]; do
  sleep 60
  ticks=$((ticks + 1))
  if (( ticks % 5 == 0 )); then
    echo "QWEN_DOWNLOAD_MONITOR_WAIT $(date -u +%FT%TZ)"
    du -sh "$MODEL_DIR"
  fi
  download_command=""
  if [[ -r "/proc/$DOWNLOAD_PID/cmdline" ]]; then
    download_command="$(tr '\0' ' ' <"/proc/$DOWNLOAD_PID/cmdline")"
  fi
done

echo "QWEN_DOWNLOAD_PROCESS_EXITED $(date -u +%FT%TZ)"
if find "$MODEL_DIR" -type f -name '*.incomplete' -print -quit | grep -q .; then
  echo "QWEN_DOWNLOAD_MONITOR_FAILED incomplete files remain"
  find "$MODEL_DIR" -type f -name '*.incomplete' -printf '%P %s bytes\n' | sort
  exit 1
fi

cd "$ROOT"
"$PYTHON" smoke/qwen_lowlight_preflight.py --config "$CONFIG"
"$PYTHON" -m unittest discover -s tests -v
"$PYTHON" -m py_compile \
  predict_qwen_lowlight.py \
  lowlight_restoration/qwen_condition.py \
  lowlight_restoration/condition_adapter.py \
  smoke/qwen_lowlight_preflight.py
bash -n smoke/run_qwen_lowlight_smoke25.sh
git diff --check
sha256sum "$CONFIG"
echo "QWEN_DOWNLOAD_MONITOR_COMPLETE $(date -u +%FT%TZ)"
