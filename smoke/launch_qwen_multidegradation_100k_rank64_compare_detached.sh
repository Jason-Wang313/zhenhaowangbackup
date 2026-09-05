#!/usr/bin/env bash
set -euo pipefail

ROOT=/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
RUNNER="$ROOT/smoke/launch_qwen_multidegradation_100k_rank64_compare.sh"
visible=${CUDA_VISIBLE_DEVICES:-}
schedule=${QWEN_COMPARE_SCHEDULING:-sequential}
version=${QWEN_100K_VERSION:-v1}
case "$version" in
  v1) LOG="$ROOT/logs/qwen_multidegradation_100k_rank64_compare_detached_v1.log" ;;
  v2) LOG="$ROOT/logs/qwen_multidegradation_100k_rank64_compare_detached_v2.log" ;;
  *) echo "ERROR: QWEN_100K_VERSION must be v1 or v2" >&2; exit 1 ;;
esac

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
if [[ -z "$system_tmux" || ! -x "$system_tmux" ]]; then
  workspace_tmux="$ROOT/.tools/tmux-env/bin/tmux"
  if [[ -x "$workspace_tmux" ]]; then
    system_tmux=$workspace_tmux
  fi
fi
if [[ -n "$system_tmux" && -x "$system_tmux" ]]; then
  session=qwen_100k_rank64_compare_$version
  "$system_tmux" has-session -t "$session" 2>/dev/null && {
    echo "ERROR: tmux session already exists: $session" >&2
    exit 1
  }
  "$system_tmux" new-session -d -s "$session" \
    "cd '$ROOT' && CUDA_VISIBLE_DEVICES='$visible' QWEN_100K_VERSION='$version' QWEN_COMPARE_SCHEDULING='$schedule' bash '$RUNNER' >> '$LOG' 2>&1; exec bash"
  "$system_tmux" set-option -t "$session" remain-on-exit on
  echo "QWEN_100K_RANK64_COMPARE_DETACHED_TMUX session=$session binary=$system_tmux"
else
  nohup env CUDA_VISIBLE_DEVICES="$visible" QWEN_100K_VERSION="$version" QWEN_COMPARE_SCHEDULING="$schedule" bash "$RUNNER" >> "$LOG" 2>&1 </dev/null &
  pid=$!
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "ERROR: detached comparison process did not start" >&2
    exit 1
  fi
  echo "QWEN_100K_RANK64_COMPARE_DETACHED_NOHUP pid=$pid"
fi
echo "Scheduling: $schedule"
echo "Version: $version"
echo "Monitor: tail -f $LOG"
