#!/usr/bin/env bash
set -u

ROOT="/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang"
MODEL="$ROOT/models/flux2-klein-base-4b/flux-2-klein-base-4b.safetensors"
LOG="$ROOT/logs/flux2_qwen_download_watch.log"
EXPECTED_MODEL_BYTES=7751105712
QWEN_DIR="$ROOT/models/qwen3-4b-fp8"

mkdir -p "$ROOT/logs" "$QWEN_DIR"
exec >>"$LOG" 2>&1
echo "[$(date -u +%FT%TZ)] watcher started; preserving active FLUX.2 downloader"

last_hour=""
QWEN_PID_FILE="$QWEN_DIR/.download.pid"
while true; do
    now=$(date -u +%FT%TZ)
    size=0
    [[ -f "$MODEL" ]] && size=$(stat -c %s "$MODEL")
    hour=$(date -u +%Y-%m-%dT%H)
    if [[ "$hour" != "$last_hour" ]]; then
        echo "[$now] hourly status: FLUX.2 bytes=$size/$EXPECTED_MODEL_BYTES"
        last_hour="$hour"
    fi

    qwen_size=0
    [[ -d "$QWEN_DIR" ]] && qwen_size=$(du -sb "$QWEN_DIR" 2>/dev/null | awk '{print $1}')
    if [[ "$hour" != "$last_hour" ]]; then
        echo "[$now] hourly status: FLUX.2 bytes=$size/$EXPECTED_MODEL_BYTES; Qwen directory bytes=$qwen_size"
        last_hour="$hour"
    fi

    if [[ "$size" -eq "$EXPECTED_MODEL_BYTES" ]]; then
        echo "[$now] expected FLUX.2 size reached; validating SafeTensors header"
        MODEL_PATH="$MODEL" "$ROOT/.conda/gpu-train-stable/bin/python" - <<'PY'
from os import environ
from safetensors import safe_open
path = environ["MODEL_PATH"]
with safe_open(path, framework="pt", device="cpu") as f:
    print(f"validated FLUX.2 SafeTensors: {len(f.keys())} tensors")
PY
        if [[ $? -ne 0 ]]; then
            echo "[$now] FLUX.2 validation failed; watcher will retry"
            sleep 60
            continue
        fi

        if [[ ! -f "$QWEN_DIR/.download_complete" && ! -f "$QWEN_PID_FILE" ]]; then
            echo "[$now] starting Qwen/Qwen3-4B-FP8 download"
            (
            ROOT_PATH="$QWEN_DIR" "$ROOT/.conda/gpu-train-stable/bin/python" - <<'PY'
from os import environ
from pathlib import Path
from huggingface_hub import snapshot_download

out = Path(environ["ROOT_PATH"])
snapshot_download(
    repo_id="Qwen/Qwen3-4B-FP8",
    local_dir=str(out),
    local_dir_use_symlinks=False,
    resume_download=True,
)
(out / ".download_complete").write_text("Qwen/Qwen3-4B-FP8\n")
print("Qwen download complete")
PY
            status=$?
            rm -f "$QWEN_PID_FILE"
            echo "[$(date -u +%FT%TZ)] Qwen downloader exited with status $status"
            ) >>"$LOG" 2>&1 &
            echo $! > "$QWEN_PID_FILE"
        fi
        if [[ -f "$QWEN_DIR/.download_complete" ]]; then
            echo "[$now] FLUX.2 and Qwen download completion markers present"
        fi
    fi
    sleep 60
done
