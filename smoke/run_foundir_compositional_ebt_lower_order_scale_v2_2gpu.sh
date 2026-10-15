#!/usr/bin/env bash
set -euo pipefail

ROOT=/inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/zhenhaowang
PY=/inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/.envs/gpu-train-stable-hdd3/bin/python
RUN_ID=${EBT_LOWER_ORDER_SCALE_RUN_ID:-scale10k_v2}
OUT_DIR="$ROOT/checkpoints/foundir_compositional_ebt_lower_order_scale_v2/$RUN_ID"
MAIN_LOG="$ROOT/logs/foundir_compositional_ebt_lower_order_scale_v2_${RUN_ID}.log"
PLAIN_LOG="$ROOT/logs/foundir_compositional_ebt_lower_order_scale_v2_${RUN_ID}_plain.log"
COMP_LOG="$ROOT/logs/foundir_compositional_ebt_lower_order_scale_v2_${RUN_ID}_compositional.log"
RESULT="$OUT_DIR/result.json"
CONFIG=configs/foundir_compositional_ebt_lower_order_scale_v2.json

if [[ -z "$RUN_ID" || "$RUN_ID" == *[!a-zA-Z0-9_.-]* ]]; then
  echo "invalid EBT_LOWER_ORDER_SCALE_RUN_ID: $RUN_ID" >&2
  exit 2
fi
if [[ ! -x "$PY" ]]; then
  echo "missing project interpreter: $PY" >&2
  exit 2
fi
IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES:-}"
if [[ ${#GPU_IDS[@]} -ne 2 || -z "${GPU_IDS[0]}" || -z "${GPU_IDS[1]}" || "${GPU_IDS[0]}" == "${GPU_IDS[1]}" ]]; then
  echo "CUDA_VISIBLE_DEVICES must name exactly two distinct GPUs" >&2
  exit 2
fi
if [[ -e "$OUT_DIR" || -e "$MAIN_LOG" || -e "$PLAIN_LOG" || -e "$COMP_LOG" ]]; then
  echo "refusing duplicate v2 run id: $RUN_ID" >&2
  exit 2
fi

mkdir -p "$OUT_DIR" "$ROOT/logs"
cd "$ROOT"
exec > >(tee "$MAIN_LOG") 2>&1
echo "FOUNDIR_COMPOSITIONAL_EBT_LOWER_ORDER_SCALE_V2_START utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "workspace=$ROOT"
echo "python=$PY"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "PRELAUNCH_ESTIMATE wall_time=8_to_20_minutes gpu_hours=0.27_to_0.67 gpus=2"
sha256sum "$CONFIG" smoke/train_foundir_compositional_ebt_lower_order_scale_v2.py
"$PY" -u smoke/advisor_governance.py check
"$PY" - <<'PY'
import torch
if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
    raise RuntimeError("expected exactly two visible CUDA devices")
names = [torch.cuda.get_device_name(i) for i in range(2)]
families = ["H200" if "H200" in n else "H100" if "H100" in n else "unsupported" for n in names]
if len(set(families)) != 1 or families[0] == "unsupported":
    raise RuntimeError(f"requires two homogeneous H100/H200 GPUs, got {names}")
print({"gpus": names, "memory_bytes": [torch.cuda.get_device_properties(i).total_memory for i in range(2)], "torch": torch.__version__})
PY

set +e
CUDA_VISIBLE_DEVICES="${GPU_IDS[0]}" "$PY" -u smoke/train_foundir_compositional_ebt_lower_order_scale_v2.py \
  --device cuda:0 --arm plain --output-root "$OUT_DIR" \
  --result "$OUT_DIR/plain_train_result.json" >"$PLAIN_LOG" 2>&1 &
plain_pid=$!
CUDA_VISIBLE_DEVICES="${GPU_IDS[1]}" "$PY" -u smoke/train_foundir_compositional_ebt_lower_order_scale_v2.py \
  --device cuda:0 --arm compositional --output-root "$OUT_DIR" \
  --result "$OUT_DIR/compositional_train_result.json" >"$COMP_LOG" 2>&1 &
comp_pid=$!
echo "parallel_arms_started plain_pid=$plain_pid gpu=${GPU_IDS[0]} compositional_pid=$comp_pid gpu=${GPU_IDS[1]}"
wait "$plain_pid"; plain_status=$?
wait "$comp_pid"; comp_status=$?
set -e
echo "parallel_arms_finished plain_status=$plain_status compositional_status=$comp_status utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
tail -n 8 "$PLAIN_LOG" || true
tail -n 8 "$COMP_LOG" || true

if [[ $plain_status -ne 0 || $comp_status -ne 0 ]]; then
  status=$plain_status
  [[ $status -eq 0 ]] && status=$comp_status
  echo "FOUNDIR_COMPOSITIONAL_EBT_LOWER_ORDER_SCALE_V2_END status=$status stage=parallel_training utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  "$PY" -u smoke/advisor_governance.py runtime-failure \
    --gate foundir_compositional_ebt_lower_order_scale_v2_runtime \
    --evidence reports/advisor_governance/foundir_compositional_ebt_lower_order_scale_v2_design_evidence.json \
    --config "$CONFIG" \
    --model "matched Plain versus atomic/pairwise Compositional EBT 10k scale" \
    --status "$status" --log "${MAIN_LOG#$ROOT/}" --diagnostic-only || true
  exit "$status"
fi

set +e
CUDA_VISIBLE_DEVICES="${GPU_IDS[0]}" "$PY" -u smoke/train_foundir_compositional_ebt_lower_order_scale_v2.py \
  --device cuda:0 --finalize --output-root "$OUT_DIR" --result "$RESULT"
finalize_status=$?
set -e
if [[ -s "$RESULT" ]]; then
  result_status=$($PY -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$RESULT")
  if [[ "$result_status" == "lower_order_learnability_pass" || "$result_status" == "lower_order_learnability_fail" ]]; then
    emit_args=(
      emit
      --gate foundir_compositional_ebt_lower_order_scale_v2
      --evidence "${RESULT#$ROOT/}"
      --config "$CONFIG"
      --model "matched Plain versus atomic/pairwise Compositional EBT 10k steps-only scale"
      --split "source-disjoint singleton/double development only"
      --outcome auto
      --root-cause-or-status "$result_status after matched 10000-step arms; held-out triples untouched"
      --advisor-decision-question "请审阅 10k step-only result，并决定停止、扩大 model/data，或另行批准后续；不得自动进入 100k/triple。"
      --diagnostic-only
      --hold-on-failure
    )
    "$PY" -u smoke/advisor_governance.py "${emit_args[@]}"
    echo "FOUNDIR_COMPOSITIONAL_EBT_LOWER_ORDER_SCALE_V2_END status=0 scientific_gate=$result_status utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    exit 0
  fi
fi

echo "FOUNDIR_COMPOSITIONAL_EBT_LOWER_ORDER_SCALE_V2_END status=$finalize_status stage=finalize utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
"$PY" -u smoke/advisor_governance.py runtime-failure \
  --gate foundir_compositional_ebt_lower_order_scale_v2_runtime \
  --evidence reports/advisor_governance/foundir_compositional_ebt_lower_order_scale_v2_design_evidence.json \
  --config "$CONFIG" \
  --model "matched Plain versus atomic/pairwise Compositional EBT 10k scale" \
  --status "$finalize_status" --log "${MAIN_LOG#$ROOT/}" --diagnostic-only || true
exit "${finalize_status:-3}"
