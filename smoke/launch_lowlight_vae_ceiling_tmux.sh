#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMUX_HELPERS="$ROOT/smoke/tmux_launcher_lib.sh"
RUNNER="$ROOT/smoke/run_lowlight_vae_ceiling.sh"
SESSION="lowlight_vae_ceiling_v2"
TMUX_SERVER_LABEL="zhenhaowang_jobs"
LOG="$ROOT/logs/lowlight_vae_ceiling_v2.log"
OUTPUT="$ROOT/checkpoints/lowlight_vae_ceiling_v2"
DEVICE_SET="${CUDA_VISIBLE_DEVICES:-}"

if [[ "$DEVICE_SET" != "0,1" ]]; then
  echo "ERROR: set CUDA_VISIBLE_DEVICES exactly to 0,1" >&2
  exit 1
fi
if [[ ! -f "$TMUX_HELPERS" || ! -f "$RUNNER" ]]; then
  echo "ERROR: tmux helpers or VAE-ceiling runner are unavailable" >&2
  exit 1
fi
source "$TMUX_HELPERS"
TMUX_BIN_PATH="$(select_tmux_binary)"
tmux_job_server -V
bash "$RUNNER" --check

if [[ "${1:-}" == "--check" ]]; then
  if [[ "$#" -ne 1 ]]; then
    echo "ERROR: --check takes no additional arguments" >&2
    exit 1
  fi
  echo "LOWLIGHT_VAE_CEILING_TMUX_CHECK_PASS session=$SESSION"
  exit 0
fi
if [[ "$#" -ne 0 ]]; then
  echo "ERROR: unexpected VAE-ceiling launcher argument: $1" >&2
  exit 1
fi
if tmux_job_server has-session -t "$SESSION" 2>/dev/null; then
  echo "LOWLIGHT_VAE_CEILING_SESSION_EXISTS session=$SESSION"
  tmux_attach_or_report "$SESSION"
  exit 0
fi

bash "$RUNNER" --preflight
launch_command="exec env CUDA_VISIBLE_DEVICES=$DEVICE_SET bash smoke/run_lowlight_vae_ceiling.sh"
tmux_job_server new-session -d -s "$SESSION" -c "$ROOT" "$launch_command"
if ! tmux_job_server has-session -t "$SESSION" 2>/dev/null; then
  echo "ERROR: tmux did not retain the VAE-ceiling session" >&2
  exit 1
fi

echo "LOWLIGHT_VAE_CEILING_TMUX_LAUNCHED session=$SESSION"
printf 'Reattach from a separate terminal: '
tmux_attach_hint "$SESSION"
printf '\n'
echo "Persistent log: tail -n 100 -f $LOG"
echo "Immutable output: $OUTPUT"
tmux_attach_or_report "$SESSION"
