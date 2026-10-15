#!/usr/bin/env bash
set -euo pipefail

ROOT=/inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/zhenhaowang
PY=/inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/.envs/gpu-train-stable-hdd3/bin/python
RUN_ID=${EBT_LOWER_ORDER_RUN_ID:-pilot_v1}
OUT_DIR="$ROOT/checkpoints/foundir_compositional_ebt_lower_order_pilot_v1/$RUN_ID"
RESULT="$OUT_DIR/result.json"
LOG="$ROOT/logs/foundir_compositional_ebt_lower_order_pilot_v1_${RUN_ID}.log"
CONFIG=configs/foundir_compositional_ebt_lower_order_pilot_v1.json

if [[ -z "$RUN_ID" || "$RUN_ID" == *[!a-zA-Z0-9_.-]* ]]; then
  echo "invalid EBT_LOWER_ORDER_RUN_ID: $RUN_ID" >&2
  exit 2
fi
if [[ ! -x "$PY" ]]; then
  echo "missing project interpreter: $PY" >&2
  exit 2
fi
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" || "$CUDA_VISIBLE_DEVICES" == *,* ]]; then
  echo "CUDA_VISIBLE_DEVICES must name exactly one H100/H200" >&2
  exit 2
fi
if [[ -e "$LOG" || -e "$OUT_DIR" ]]; then
  echo "refusing duplicate run id: $RUN_ID" >&2
  exit 2
fi

mkdir -p "$OUT_DIR" "$ROOT/logs"
cd "$ROOT"
set +e
{
  echo "FOUNDIR_COMPOSITIONAL_EBT_LOWER_ORDER_START utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "workspace=$ROOT"
  echo "python=$PY"
  echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  echo "PRELAUNCH_ESTIMATE wall_time=10_to_30_minutes gpu_hours=0.17_to_0.50 gpus=1"
  sha256sum "$CONFIG" \
    manifests/foundir_triple_counterfactual_reproduction_r0_v1.jsonl \
    manifests/foundir_triple_counterfactual_reproduction_r0_v1.json \
    data/foundir_triple_onerestore_r0_v1/manifest.jsonl
  "$PY" -u smoke/advisor_governance.py check
  "$PY" - <<'PY'
import torch
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise RuntimeError("expected exactly one visible CUDA device")
name = torch.cuda.get_device_name(0)
if "H100" not in name and "H200" not in name:
    raise RuntimeError(f"unsupported GPU: {name}")
print({"gpu": name, "memory_bytes": torch.cuda.get_device_properties(0).total_memory, "torch": torch.__version__})
PY
  "$PY" -u smoke/train_foundir_compositional_ebt_lower_order_pilot_v1.py \
    --device cuda:0 \
    --output-root "$OUT_DIR" \
    --result "$RESULT"
} 2>&1 | tee "$LOG"
train_status=${PIPESTATUS[0]}
set -e

if [[ -s "$RESULT" ]]; then
  result_status=$($PY -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$RESULT")
  if [[ "$result_status" == "lower_order_learnability_pass" || "$result_status" == "lower_order_learnability_fail" ]]; then
    emit_args=(
      emit
      --gate foundir_compositional_ebt_lower_order_pilot_v1
      --evidence "${RESULT#$ROOT/}"
      --config "$CONFIG"
      --model "matched Plain versus atomic/pairwise Compositional Energy-Based Transformer"
      --split "source-disjoint singleton/double development only"
      --outcome auto
      --root-cause-or-status "$result_status; held-out triples remained untouched"
      --advisor-decision-question "请审阅 lower-order learnability gate；只有 pass 后才考虑另行批准一次 frozen held-out-triple evaluation。"
      --diagnostic-only
      --hold-on-failure
    )
    "$PY" -u smoke/advisor_governance.py "${emit_args[@]}"
    echo "FOUNDIR_COMPOSITIONAL_EBT_LOWER_ORDER_END status=0 scientific_gate=$result_status utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG"
    exit 0
  fi
fi

echo "FOUNDIR_COMPOSITIONAL_EBT_LOWER_ORDER_END status=$train_status utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG"
"$PY" -u smoke/advisor_governance.py runtime-failure \
  --gate foundir_compositional_ebt_lower_order_pilot_v1_runtime \
  --evidence reports/advisor_governance/foundir_compositional_ebt_lower_order_pilot_v1_design_evidence.json \
  --config "$CONFIG" \
  --model "matched Plain versus atomic/pairwise Compositional Energy-Based Transformer" \
  --status "$train_status" \
  --log "${LOG#$ROOT/}" \
  --diagnostic-only || true
exit "${train_status:-3}"
