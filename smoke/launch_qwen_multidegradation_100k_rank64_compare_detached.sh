#!/usr/bin/env bash
set -euo pipefail

ROOT=/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
RUNNER="$ROOT/smoke/launch_qwen_multidegradation_100k_rank64_compare.sh"
LOG="$ROOT/logs/qwen_multidegradation_100k_rank64_compare_detached.log"
visible=${CUDA_VISIBLE_DEVICES:-}
schedule=${QWEN_COMPARE_SCHEDULING:-sequential}

IFS=, read -r -a devices <<<"$visible"
case ${#devices[@]} in
  2|8) ;;
  *) echo "ERROR: set CUDA_VISIBLE_DEVICES to exactly 2 or 8 GPU IDs" >&2; exit 1 ;;
esac
if [[ $(printf '%s\n' "${devices[@]}" | sort -u | wc -l) -ne ${#devices[@]} ]]; then
  echo "ERROR: CUDA_VISIBLE_DEVICES contains duplicates" >&2
  exit 1
fi

mkdir -p "$ROOT/logs"
system_tmux=$(command -v tmux 2>/dev/null || true)
if [[ -n "$system_tmux" && -x "$system_tmux" ]]; then
  session=qwen_100k_rank64_compare
  "$system_tmux" has-session -t "$session" 2>/dev/null && {
    echo "ERROR: tmux session already exists: $session" >&2
    exit 1
  }
  "$system_tmux" new-session -d -s "$session" \
    "cd '$ROOT' && CUDA_VISIBLE_DEVICES='$visible' QWEN_COMPARE_SCHEDULING='$schedule' bash '$RUNNER' >> '$LOG' 2>&1"
  echo "QWEN_100K_RANK64_COMPARE_DETACHED_TMUX session=$session"
else
  nohup env CUDA_VISIBLE_DEVICES="$visible" QWEN_COMPARE_SCHEDULING="$schedule" bash "$RUNNER" >> "$LOG" 2>&1 </dev/null &
  pid=$!
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "ERROR: detached comparison process did not start" >&2
    exit 1
  fi
  echo "QWEN_100K_RANK64_COMPARE_DETACHED_NOHUP pid=$pid"
fi
echo "Scheduling: $schedule"
echo "Monitor: tail -f $LOG"
