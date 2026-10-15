#!/usr/bin/env bash
set -euo pipefail

ROOT=/inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/zhenhaowang
PY=/inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/.envs/gpu-train-stable-hdd3/bin/python
RUN_ID=${EBT_PHASE0_RUN_ID:-gpu_smoke_v1}
OUT_DIR="$ROOT/checkpoints/foundir_compositional_ebt_phase0_v1/$RUN_ID"
LOG="$ROOT/logs/foundir_compositional_ebt_phase0_v1_${RUN_ID}.log"

if [[ "$RUN_ID" == *[!a-zA-Z0-9_.-]* || -z "$RUN_ID" ]]; then
  echo "invalid EBT_PHASE0_RUN_ID: $RUN_ID" >&2
  exit 2
fi
if [[ ! -x "$PY" ]]; then
  echo "missing project interpreter: $PY" >&2
  exit 2
fi
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "CUDA_VISIBLE_DEVICES must name exactly one GPU" >&2
  exit 2
fi
if [[ "${CUDA_VISIBLE_DEVICES}" == *,* ]]; then
  echo "Phase-0 requires exactly one visible GPU" >&2
  exit 2
fi
if [[ -e "$LOG" || -e "$OUT_DIR/result.json" ]]; then
  echo "refusing duplicate Phase-0 run id: $RUN_ID" >&2
  exit 2
fi

mkdir -p "$OUT_DIR" "$ROOT/logs"
cd "$ROOT"
set +e
{
  echo "FOUNDIR_COMPOSITIONAL_EBT_PHASE0_START utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "workspace=$ROOT"
  echo "python=$PY"
  echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  sha256sum configs/foundir_compositional_ebt_phase0_v1.json
  git -C third_party/ebt_official rev-parse HEAD
  "$PY" -u smoke/advisor_governance.py check
  "$PY" - <<'PY'
import torch
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise RuntimeError("expected exactly one visible CUDA device")
name = torch.cuda.get_device_name(0)
memory = torch.cuda.get_device_properties(0).total_memory
if "H100" not in name and "H200" not in name:
    raise RuntimeError(f"unsupported Phase-0 GPU: {name}")
print({"gpu": name, "memory_bytes": memory, "torch": torch.__version__})
PY
  "$PY" -u smoke/compositional_ebt_phase0_smoke.py \
    --device cuda:0 \
    --result "$OUT_DIR/result.json"
  echo "FOUNDIR_COMPOSITIONAL_EBT_PHASE0_END status=0 utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} 2>&1 | tee "$LOG"
status=${PIPESTATUS[0]}
set -e
if [[ $status -ne 0 ]]; then
  echo "FOUNDIR_COMPOSITIONAL_EBT_PHASE0_END status=$status utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG"
  "$PY" -u smoke/advisor_governance.py runtime-failure \
    --gate foundir_compositional_ebt_phase0_v1_runtime \
    --evidence reports/advisor_governance/foundir_compositional_ebt_phase0_v1_design_evidence.json \
    --config configs/foundir_compositional_ebt_phase0_v1.json \
    --model "VLM-activated atomic/pairwise compositional Energy-Based Transformer Phase-0" \
    --status "$status" \
    --log "${LOG#$ROOT/}" \
    --diagnostic-only || true
fi
exit "$status"
