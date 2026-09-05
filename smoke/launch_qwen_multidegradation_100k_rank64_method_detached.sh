#!/usr/bin/env bash
set -euo pipefail

ROOT=/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
RUNNER="$ROOT/smoke/run_qwen_multidegradation_100k_rank64_method.sh"
METHOD=${QWEN_100K_METHOD:-}
VERSION=${QWEN_100K_VERSION:-v1}
VISIBLE=${CUDA_VISIBLE_DEVICES:-}

case "$VERSION:$METHOD" in
  v1:joint|v1:per_type|v2:joint|v2:per_type) ;;
  *) echo "ERROR: set QWEN_100K_VERSION to v1 or v2 and QWEN_100K_METHOD to joint or per_type" >&2; exit 1 ;;
esac
IFS=, read -r -a devices <<<"$VISIBLE"
case ${#devices[@]} in
  1|2|4|8) ;;
  *) echo "ERROR: set CUDA_VISIBLE_DEVICES to 1, 2, 4, or 8 unique GPU IDs" >&2; exit 1 ;;
esac
if [[ $(printf '%s\n' "${devices[@]}" | sort -u | wc -l) -ne ${#devices[@]} ]]; then
  echo "ERROR: CUDA_VISIBLE_DEVICES contains duplicates" >&2
  exit 1
fi

session="qwen_100k_rank64_${METHOD}_${VERSION}"
launcher_log="$ROOT/logs/qwen_multidegradation_100k_rank64_${METHOD}_${VERSION}_launcher.log"
mkdir -p "$ROOT/logs"
system_tmux=$(command -v tmux 2>/dev/null || true)
if [[ -z "$system_tmux" || ! -x "$system_tmux" ]]; then
  workspace_tmux="$ROOT/.tools/tmux-env/bin/tmux"
  if [[ -x "$workspace_tmux" ]]; then
    system_tmux=$workspace_tmux
  fi
fi
if [[ -n "$system_tmux" && -x "$system_tmux" ]]; then
  "$system_tmux" has-session -t "$session" 2>/dev/null && {
    echo "ERROR: tmux session already exists: $session" >&2
    exit 1
  }
  "$system_tmux" new-session -d -s "$session" \
    "cd '$ROOT' && CUDA_VISIBLE_DEVICES='$VISIBLE' QWEN_100K_VERSION='$VERSION' QWEN_100K_METHOD='$METHOD' bash '$RUNNER' >> '$launcher_log' 2>&1; exec bash"
  "$system_tmux" set-option -t "$session" remain-on-exit on
  echo "QWEN_100K_RANK64_METHOD_DETACHED_TMUX session=$session binary=$system_tmux"
else
  nohup env CUDA_VISIBLE_DEVICES="$VISIBLE" QWEN_100K_VERSION="$VERSION" \
    QWEN_100K_METHOD="$METHOD" bash "$RUNNER" >>"$launcher_log" 2>&1 </dev/null &
  pid=$!
  sleep 2
  kill -0 "$pid" 2>/dev/null || {
    echo "ERROR: detached Qwen method exited during launch; inspect $launcher_log" >&2
    exit 1
  }
  echo "QWEN_100K_RANK64_METHOD_DETACHED_NOHUP pid=$pid"
fi
echo "Method: $METHOD"
echo "Version: $VERSION"
echo "Monitor: tail -f $launcher_log"
