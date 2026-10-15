#!/usr/bin/env bash
set -euo pipefail

ROOT=/inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/zhenhaowang
PY=/inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/.envs/gpu-train-stable-hdd3/bin/python
RUN_ID=${EBT_SCALED_LEARNING_SMOKE_RUN_ID:-real500_v1}
OUT_DIR="$ROOT/checkpoints/foundir_compositional_ebt_scaled_learning_smoke_v1/$RUN_ID"
MAIN_LOG="$ROOT/logs/foundir_compositional_ebt_scaled_learning_smoke_v1_${RUN_ID}.log"
CONFIG=configs/foundir_compositional_ebt_scaled_learning_smoke_v1.json
SCRIPT=smoke/train_foundir_compositional_ebt_scaled_learning_smoke_v1.py

if [[ -z "$RUN_ID" || "$RUN_ID" == *[!a-zA-Z0-9_.-]* ]]; then echo "invalid run id" >&2; exit 2; fi
if [[ ! -x "$PY" ]]; then echo "missing project interpreter" >&2; exit 2; fi
IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES:-}"
if [[ ${#GPU_IDS[@]} -ne 2 || "${GPU_IDS[0]}" == "${GPU_IDS[1]}" ]]; then
  echo "CUDA_VISIBLE_DEVICES must name exactly two distinct GPUs" >&2; exit 2
fi
if [[ -e "$OUT_DIR" || -e "$MAIN_LOG" ]]; then echo "refusing duplicate run id: $RUN_ID" >&2; exit 2; fi

mkdir -p "$OUT_DIR" "$ROOT/logs"
cd "$ROOT"
exec > >(tee "$MAIN_LOG") 2>&1
echo "FOUNDIR_EBT_SCALED_LEARNING_SMOKE_START utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "workspace=$ROOT"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "PRELAUNCH_ESTIMATE wall_time_hours=0.10_to_0.75 gpu_hours=0.20_to_1.50 allocated_gpus=2"
sha256sum "$CONFIG" "$SCRIPT"
"$PY" - <<'PY'
import torch
if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
    raise RuntimeError("expected exactly two visible GPUs")
names=[torch.cuda.get_device_name(i) for i in range(2)]
families=["H200" if "H200" in n else "H100" if "H100" in n else "unsupported" for n in names]
if len(set(families)) != 1 or families[0] == "unsupported":
    raise RuntimeError(f"requires two homogeneous H100/H200 GPUs, got {names}")
print({"gpus":names,"memory_bytes":[torch.cuda.get_device_properties(i).total_memory for i in range(2)]})
PY

set +e
CUDA_VISIBLE_DEVICES="${GPU_IDS[0]}" "$PY" -u "$SCRIPT" --arm plain --output "$OUT_DIR/plain.json" >"$ROOT/logs/foundir_compositional_ebt_scaled_learning_smoke_v1_${RUN_ID}_plain.log" 2>&1 &
plain_pid=$!
CUDA_VISIBLE_DEVICES="${GPU_IDS[1]}" "$PY" -u "$SCRIPT" --arm compositional --output "$OUT_DIR/compositional.json" >"$ROOT/logs/foundir_compositional_ebt_scaled_learning_smoke_v1_${RUN_ID}_compositional.log" 2>&1 &
comp_pid=$!
echo "arms_started plain_pid=$plain_pid compositional_pid=$comp_pid"
wait "$plain_pid"; plain_status=$?
wait "$comp_pid"; comp_status=$?
set -e
echo "arms_finished plain_status=$plain_status compositional_status=$comp_status"
tail -n 20 "$ROOT/logs/foundir_compositional_ebt_scaled_learning_smoke_v1_${RUN_ID}_plain.log" || true
tail -n 20 "$ROOT/logs/foundir_compositional_ebt_scaled_learning_smoke_v1_${RUN_ID}_compositional.log" || true
if [[ $plain_status -ne 0 || $comp_status -ne 0 ]]; then
  status=$plain_status; [[ $status -eq 0 ]] && status=$comp_status
  echo "FOUNDIR_EBT_SCALED_LEARNING_SMOKE_END status=$status utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  exit "$status"
fi
CUDA_VISIBLE_DEVICES="${GPU_IDS[0]}" "$PY" -u "$SCRIPT" --finalize --run-id "$RUN_ID"
echo "FOUNDIR_EBT_SCALED_LEARNING_SMOKE_END status=0 utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) result=$OUT_DIR/result.json"
