#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 5 || "$1" != "--id" || "$3" != "--continue" || "$4" != "--output" ]]; then
  echo "usage: gdown_legacy_id_compat.sh --id ID --continue --output PATH" >&2
  exit 2
fi

workspace="/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang"
exec "$workspace/.conda/gpu-train-stable/bin/gdown" "$2" --continue --output "$5"
