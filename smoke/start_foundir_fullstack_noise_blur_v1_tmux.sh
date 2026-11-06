#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ $# == 1 && "$1" =~ ^[0-9a-f]{40}$ ]] || { echo 'usage: bash smoke/start_foundir_fullstack_noise_blur_v1_tmux.sh PUBLISHED_FULL_COMMIT'; exit 2; }
cd "$ROOT"
git merge-base --is-ancestor "$1" origin/main
TMUX_BIN="$ROOT/.tools/tmux-env/bin/tmux"
[[ -x "$TMUX_BIN" ]] || { echo 'ERROR: real tmux missing'; exit 3; }
export TMUX_TMPDIR=/tmp
session=noise_blur_bounded_repair_v1
if "$TMUX_BIN" -L zhenhaowang_jobs has-session -t "$session" 2>/dev/null; then
 echo "NOISE_BLUR_MONITOR_ALREADY_RUNNING session=$session"; exit 0
fi
mkdir -p logs
printf -v command 'cd %q && CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 PYTHONPATH=%q %q -S -u smoke/watch_foundir_fullstack_noise_blur_v1_cpu.py --commit %q >> logs/foundir_fullstack_noise_blur_v1_cpu_monitor.log 2>&1' \
 "$ROOT" "$ROOT/.conda/gpu-train-stable/lib/python3.12/site-packages:$ROOT" "$ROOT/.conda/gpu-train-stable/bin/python" "$1"
"$TMUX_BIN" -L zhenhaowang_jobs new-session -d -s "$session" "$command"
echo "NOISE_BLUR_MONITOR_CREATED session=$session; job completion requires logs/results/exit codes"
