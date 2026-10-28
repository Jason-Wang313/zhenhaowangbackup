#!/usr/bin/env bash
# Historical entrypoint; canonical launcher supports an explicit 4/8 GPU count.
set -euo pipefail
exec bash "$(dirname "${BASH_SOURCE[0]}")/run_foundir_flux_tail_oracle_v1_gpu.sh" "$@"
