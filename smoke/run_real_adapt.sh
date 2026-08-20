#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$ROOT/smoke/launch_qwen_lowlight_real_adaptation_tmux.sh" "$@"
