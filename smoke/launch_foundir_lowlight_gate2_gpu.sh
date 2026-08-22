#!/usr/bin/env bash
set -euo pipefail

ROOT="/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang"
ATTEMPTS="$ROOT/logs/foundir_lowlight_gate2_gpu_v1_attempts.tsv"
RESULT="$ROOT/reports/foundir_lowlight_gate2_proposal_lattice_result_v1.json"
cd "$ROOT"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" || "$CUDA_VISIBLE_DEVICES" == *,* ]]; then
  echo "ERROR: set CUDA_VISIBLE_DEVICES to exactly one GPU index, for example 0" >&2
  exit 2
fi

for executable in \
  "$ROOT/.conda/gpu-train-stable/bin/python" \
  "$ROOT/.venv/flux2-smoke/bin/python" \
  "$ROOT/.venv/foundir-metrics/bin/python"; do
  if [[ ! -x "$executable" ]]; then
    echo "ERROR: missing executable: $executable" >&2
    exit 2
  fi
done

if [[ -f "$RESULT" ]]; then
  echo "FOUNDIR_LOWLIGHT_GATE2_ALREADY_COMPLETE"
  exit 0
fi

if [[ -f "$ATTEMPTS" ]]; then
  while IFS=$'\t' read -r existing_pid existing_log; do
    if [[ -n "$existing_pid" ]] && kill -0 "$existing_pid" 2>/dev/null; then
      echo "FOUNDIR_LOWLIGHT_GATE2_ALREADY_RUNNING pid=$existing_pid log=$existing_log"
      exit 0
    fi
  done < "$ATTEMPTS"
fi

export PYTHONPATH="$ROOT/.tools/foundir-runtime${PYTHONPATH:+:$PYTHONPATH}"

"$ROOT/.conda/gpu-train-stable/bin/python" - <<'PY'
import importlib
import torch
assert torch.cuda.is_available() and torch.cuda.device_count() == 1
major, minor = torch.cuda.get_device_capability(0)
assert (major, minor) == (9, 0), (major, minor)
importlib.import_module("ema_pytorch")
importlib.import_module("causal_conv1d")
importlib.import_module("mamba_ssm")
print("GATE2_TEACHER_GPU_PREFLIGHT_PASS", torch.cuda.get_device_name(0))
PY

"$ROOT/.venv/flux2-smoke/bin/python" - <<'PY'
import torch
assert torch.cuda.is_available() and torch.cuda.device_count() == 1
print("GATE2_FLUX_GPU_PREFLIGHT_PASS", torch.cuda.get_device_name(0))
PY

"$ROOT/.venv/foundir-metrics/bin/python" - <<'PY'
import lpips
import pyiqa
import torch
assert torch.cuda.is_available() and torch.cuda.device_count() == 1
print("GATE2_METRIC_GPU_PREFLIGHT_PASS", torch.cuda.get_device_name(0))
PY

mkdir -p "$ROOT/logs"
log="$(mktemp "$ROOT/logs/foundir_lowlight_gate2_gpu_v1_XXXXXX.log")"
nohup bash "$ROOT/smoke/run_foundir_lowlight_gate2_gpu_bundle.sh" >> "$log" 2>&1 &
pid=$!
printf '%s\t%s\n' "$pid" "$log" >> "$ATTEMPTS"
echo "FOUNDIR_LOWLIGHT_GATE2_DETACHED pid=$pid log=$log"
