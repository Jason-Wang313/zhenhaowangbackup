#!/usr/bin/env bash
set -euo pipefail

ROOT="/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang"
cd "$ROOT"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" || "$CUDA_VISIBLE_DEVICES" == *,* ]]; then
  echo "ERROR: expose exactly one GPU to the Gate 2 bundle" >&2
  exit 2
fi

export PYTHONPATH="$ROOT/.tools/foundir-runtime${PYTHONPATH:+:$PYTHONPATH}"

"$ROOT/.conda/gpu-train-stable/bin/python" -u -m smoke.run_foundir_lowlight_gate2_teachers
"$ROOT/.venv/flux2-smoke/bin/python" -u -m smoke.run_foundir_lowlight_gate2_historical_flux
"$ROOT/.venv/foundir-metrics/bin/python" -u -m smoke.run_foundir_lowlight_gate2_metric_blend

echo "FOUNDIR_LOWLIGHT_GATE2_GPU_BUNDLE_PASS"
