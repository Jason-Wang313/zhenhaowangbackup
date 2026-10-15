#!/usr/bin/env bash
set -euo pipefail

ROOT="/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang"
/usr/bin/python3 "$ROOT/smoke/advisor_governance.py" check
ATTEMPTS="$ROOT/logs/foundir_lowlight_gate2_gpu_resume_v1b_attempts.tsv"
RESULT="$ROOT/reports/foundir_lowlight_gate2_proposal_lattice_result_v1b.json"
cd "$ROOT"

if [[ ! "${CUDA_VISIBLE_DEVICES:-}" =~ ^[0-9]+(,[0-9]+)?$ ]]; then
  echo "ERROR: set CUDA_VISIBLE_DEVICES to one or two GPU indices, for example 0,1" >&2
  exit 2
fi
IFS=',' read -r -a devices <<< "$CUDA_VISIBLE_DEVICES"
teacher_device="${devices[0]}"

if [[ -f "$RESULT" ]]; then
  echo "FOUNDIR_LOWLIGHT_GATE2_ALREADY_COMPLETE"
  exit 0
fi
if [[ ! -f "$ROOT/manifests/foundir_lowlight_gate2_generalist_v1b.csv" ]]; then
  echo "ERROR: completed generalist manifest is missing; resume is unsafe" >&2
  exit 2
fi
for directory in \
  "$ROOT/.tools/foundir-tokenizers-0.21.0" \
  "$ROOT/.tools/foundir-huggingface-hub-0.27.1" \
  "$ROOT/.tools/foundir-transformers-4.47.1-v2" \
  "$ROOT/.tools/foundir-runtime"; do
  if [[ ! -d "$directory" ]]; then
    echo "ERROR: missing specialist dependency directory: $directory" >&2
    exit 2
  fi
done

if [[ -f "$ATTEMPTS" ]]; then
  while IFS=$'\t' read -r existing_pid existing_log; do
    if [[ -n "$existing_pid" ]] && kill -0 "$existing_pid" 2>/dev/null; then
      echo "FOUNDIR_LOWLIGHT_GATE2_RESUME_ALREADY_RUNNING pid=$existing_pid log=$existing_log"
      exit 0
    fi
  done < "$ATTEMPTS"
fi

export PYTHONPATH="$ROOT/.tools/foundir-runtime${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONPATH="$ROOT/.tools/foundir-transformers-4.47.1-v2:$PYTHONPATH"
export PYTHONPATH="$ROOT/.tools/foundir-huggingface-hub-0.27.1:$PYTHONPATH"
export PYTHONPATH="$ROOT/.tools/foundir-tokenizers-0.21.0:$PYTHONPATH"

mkdir -p "$ROOT/logs"
log="$(mktemp "$ROOT/logs/foundir_lowlight_gate2_gpu_resume_v1b_XXXXXX.log")"
set +e
(
set -euo pipefail
echo "FOUNDIR_LOWLIGHT_GATE2_RESUME_PREFLIGHT_START utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) device=$teacher_device"
CUDA_VISIBLE_DEVICES="$teacher_device" "$ROOT/.conda/gpu-train-stable/bin/python" - <<'PY'
import importlib
from pathlib import Path

import huggingface_hub
import tokenizers
import torch
import transformers

root = Path.cwd()
assert torch.cuda.is_available() and torch.cuda.device_count() == 1
assert torch.cuda.get_device_capability(0) == (9, 0)
assert transformers.__version__ == "4.47.1", transformers.__version__
assert tokenizers.__version__ == "0.21.0", tokenizers.__version__
assert huggingface_hub.__version__ == "0.27.1", huggingface_hub.__version__
importlib.import_module("causal_conv1d")
importlib.import_module("mamba_ssm")
from smoke.run_foundir_lowlight_gate2_teachers import import_specialist_architecture
WaveMamba = import_specialist_architecture(root / "third_party/foundir/specialist_model")
print("GATE2_SPECIALIST_RESUME_GPU_PREFLIGHT_PASS", torch.cuda.get_device_name(0), WaveMamba.__name__)
PY
CUDA_VISIBLE_DEVICES="$teacher_device" "$ROOT/.venv/foundir-metrics/bin/python" - <<'PY'
import lpips
import pyiqa
import torch
assert torch.cuda.is_available() and torch.cuda.device_count() == 1
print("GATE2_RESUME_METRIC_GPU_PREFLIGHT_PASS", torch.cuda.get_device_name(0))
PY
echo "FOUNDIR_LOWLIGHT_GATE2_RESUME_PREFLIGHT_PASS utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
) 2>&1 | tee -a "$log"
preflight_status=${PIPESTATUS[0]}
set -e
if [[ "$preflight_status" -ne 0 ]]; then
  echo "FOUNDIR_LOWLIGHT_GATE2_RESUME_PREFLIGHT_FAILED status=$preflight_status log=$log" | tee -a "$log" >&2
  exit "$preflight_status"
fi

nohup bash "$ROOT/smoke/run_foundir_lowlight_gate2_gpu_resume_logged.sh" >> "$log" 2>&1 &
pid=$!
printf '%s\t%s\n' "$pid" "$log" >> "$ATTEMPTS"
echo "FOUNDIR_LOWLIGHT_GATE2_RESUME_DETACHED pid=$pid log=$log"
