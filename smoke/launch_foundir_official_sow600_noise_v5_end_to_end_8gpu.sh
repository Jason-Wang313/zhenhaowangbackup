#!/usr/bin/env bash
set -euo pipefail

ROOT=/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
cd "$ROOT"
bash smoke/launch_foundir_official_sow600_noise_v5_8gpu.sh
bash smoke/launch_foundir_official_sow600_metrics_v5_8gpu.sh
