#!/usr/bin/env bash
set -euo pipefail
ROOT=/inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/zhenhaowang
PY=/inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/.envs/gpu-train-stable-hdd3/bin/python
RUN_ID=${EBT_TENSOR_SHARD_100K_RUN_ID:-shard100k_v5}
OUT_DIR="$ROOT/checkpoints/foundir_compositional_ebt_tensor_shard_100k_v5/$RUN_ID"
MAIN_LOG="$ROOT/logs/foundir_compositional_ebt_tensor_shard_100k_v5_${RUN_ID}.log"
CONFIG=configs/foundir_compositional_ebt_tensor_shard_100k_v5.json
SCRIPT=smoke/train_foundir_compositional_ebt_tensor_shard_100k_v5.py
if [[ -z "$RUN_ID" || "$RUN_ID" == *[!a-zA-Z0-9_.-]* ]]; then echo "invalid run id" >&2; exit 2; fi
IFS=',' read -r -a G <<< "${CUDA_VISIBLE_DEVICES:-}"
if [[ ${#G[@]} -ne 2 || "${G[0]}" == "${G[1]}" ]]; then echo "CUDA_VISIBLE_DEVICES must name two distinct GPUs" >&2; exit 2; fi
if [[ -e "$OUT_DIR" || -e "$MAIN_LOG" ]]; then echo "refusing duplicate run id" >&2; exit 2; fi
mkdir -p "$OUT_DIR" "$ROOT/logs"; cd "$ROOT"; exec > >(tee "$MAIN_LOG") 2>&1
echo "FOUNDIR_EBT_TENSOR_SHARD_100K_V5_START utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "PRELAUNCH_ESTIMATE wall_time_hours=1.50_to_3.50 gpu_hours=3.00_to_7.00 allocated_gpus=2 includes_matched_660=true"
sha256sum "$CONFIG" "$SCRIPT" \
  smoke/evaluate_foundir_compositional_ebt_100k_v5.py \
  smoke/train_foundir_compositional_ebt_scaled_learning_smoke_v1.py \
  smoke/train_foundir_compositional_ebt_scaled_learning_cached_v2.py \
  lowlight_restoration/compositional_ebt.py
"$PY" - <<'PY'
import torch
if not torch.cuda.is_available() or torch.cuda.device_count()!=2: raise RuntimeError("expected two visible GPUs")
n=[torch.cuda.get_device_name(i) for i in range(2)]; f=["H200" if "H200" in x else "H100" if "H100" in x else "bad" for x in n]
if len(set(f))!=1 or f[0]=="bad": raise RuntimeError(f"requires homogeneous H100/H200, got {n}")
print({"gpus":n})
PY
set +e
CUDA_VISIBLE_DEVICES="${G[0]}" "$PY" -u "$SCRIPT" --arm plain --output "$OUT_DIR/plain.json" >"$ROOT/logs/foundir_compositional_ebt_tensor_shard_100k_v5_${RUN_ID}_plain.log" 2>&1 & p0=$!
CUDA_VISIBLE_DEVICES="${G[1]}" "$PY" -u "$SCRIPT" --arm compositional --output "$OUT_DIR/compositional.json" >"$ROOT/logs/foundir_compositional_ebt_tensor_shard_100k_v5_${RUN_ID}_compositional.log" 2>&1 & p1=$!
echo "arms_started plain_pid=$p0 compositional_pid=$p1"; wait "$p0"; s0=$?; wait "$p1"; s1=$?; set -e
echo "arms_finished plain_status=$s0 compositional_status=$s1"
tail -n 20 "$ROOT/logs/foundir_compositional_ebt_tensor_shard_100k_v5_${RUN_ID}_plain.log" || true
tail -n 20 "$ROOT/logs/foundir_compositional_ebt_tensor_shard_100k_v5_${RUN_ID}_compositional.log" || true
if [[ $s0 -ne 0 || $s1 -ne 0 ]]; then status=$s0; [[ $status -eq 0 ]] && status=$s1; echo "FOUNDIR_EBT_TENSOR_SHARD_100K_V5_END status=$status utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"; exit "$status"; fi
CUDA_VISIBLE_DEVICES="${G[0]}" "$PY" -u "$SCRIPT" --finalize --run-id "$RUN_ID"
sha256sum "$OUT_DIR/result.json" "$OUT_DIR/plain_step_100000.pt" "$OUT_DIR/compositional_step_100000.pt"
set +e
CUDA_VISIBLE_DEVICES="${G[0]}" TORCH_HOME="$ROOT/models/torch" "$PY" -u smoke/evaluate_foundir_compositional_ebt_100k_v5.py --arm plain --run-id "$RUN_ID" >"$ROOT/logs/foundir_compositional_ebt_tensor_shard_100k_v5_${RUN_ID}_plain_eval.log" 2>&1 & e0=$!
CUDA_VISIBLE_DEVICES="${G[1]}" TORCH_HOME="$ROOT/models/torch" "$PY" -u smoke/evaluate_foundir_compositional_ebt_100k_v5.py --arm compositional --run-id "$RUN_ID" >"$ROOT/logs/foundir_compositional_ebt_tensor_shard_100k_v5_${RUN_ID}_compositional_eval.log" 2>&1 & e1=$!
echo "matched_660_evaluation_started plain_pid=$e0 compositional_pid=$e1"; wait "$e0"; es0=$?; wait "$e1"; es1=$?; set -e
echo "matched_660_evaluation_finished plain_status=$es0 compositional_status=$es1"
tail -n 20 "$ROOT/logs/foundir_compositional_ebt_tensor_shard_100k_v5_${RUN_ID}_plain_eval.log" || true
tail -n 20 "$ROOT/logs/foundir_compositional_ebt_tensor_shard_100k_v5_${RUN_ID}_compositional_eval.log" || true
if [[ $es0 -ne 0 || $es1 -ne 0 ]]; then status=$es0; [[ $status -eq 0 ]] && status=$es1; echo "FOUNDIR_EBT_TENSOR_SHARD_100K_V5_END status=$status stage=matched_660 utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"; exit "$status"; fi
CUDA_VISIBLE_DEVICES="${G[0]}" "$PY" -u smoke/evaluate_foundir_compositional_ebt_100k_v5.py --finalize --run-id "$RUN_ID"
sha256sum "$OUT_DIR/matched_660_result.json"
echo "FOUNDIR_EBT_TENSOR_SHARD_100K_V5_END status=0 utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) train_result=$OUT_DIR/result.json matched_660_result=$OUT_DIR/matched_660_result.json"
