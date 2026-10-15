#!/usr/bin/env bash
set -euo pipefail
ROOT=/inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/zhenhaowang
PY=/inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/.envs/gpu-train-stable-hdd3/bin/python
RUN_ID=${EBT_TENSOR_SHARD_100K_RUN_ID:-shard100k_v5}
OUT_DIR="$ROOT/checkpoints/foundir_compositional_ebt_tensor_shard_100k_v5/$RUN_ID"
MAIN_LOG="$ROOT/logs/foundir_compositional_ebt_tensor_shard_100k_v5_${RUN_ID}_matched_660_retry1.log"
SCRIPT=smoke/evaluate_foundir_compositional_ebt_100k_v5.py
IFS=',' read -r -a G <<< "${CUDA_VISIBLE_DEVICES:-}"
if [[ ${#G[@]} -ne 2 || "${G[0]}" == "${G[1]}" ]]; then echo "CUDA_VISIBLE_DEVICES must name two distinct GPUs" >&2; exit 2; fi
if [[ ! -s "$OUT_DIR/plain_step_100000.pt" || ! -s "$OUT_DIR/compositional_step_100000.pt" || ! -s "$OUT_DIR/result.json" ]]; then echo "missing frozen 100k artifacts" >&2; exit 2; fi
if [[ -e "$OUT_DIR/matched_660_result.json" || -e "$MAIN_LOG" ]]; then echo "refusing duplicate completed/retry artifact" >&2; exit 2; fi
cd "$ROOT"; exec > >(tee "$MAIN_LOG") 2>&1
echo "FOUNDIR_EBT_100K_MATCHED_660_RETRY1_START utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "PRELAUNCH_ESTIMATE wall_time_hours=0.70_to_1.50 gpu_hours=1.40_to_3.00 allocated_gpus=2 existing_records_per_arm=60"
sha256sum configs/foundir_compositional_ebt_tensor_shard_100k_v5.json "$SCRIPT" "$OUT_DIR/result.json" "$OUT_DIR/plain_step_100000.pt" "$OUT_DIR/compositional_step_100000.pt"
"$PY" - <<'PY'
import csv
import torch
from smoke.evaluate_foundir_compositional_ebt_100k_v5 import routed_atoms
if not torch.cuda.is_available() or torch.cuda.device_count()!=2: raise RuntimeError("expected two visible GPUs")
n=[torch.cuda.get_device_name(i) for i in range(2)]; f=["H200" if "H200" in x else "H100" if "H100" in x else "bad" for x in n]
if len(set(f))!=1 or f[0]=="bad": raise RuntimeError(f"requires homogeneous H100/H200, got {n}")
rows=list(csv.DictReader(open("manifests/foundir_composition_whole_image_v1.csv")))
atoms=[routed_atoms(row["expert_order_json"]) for row in rows]
if len(rows)!=660 or len(atoms)!=660: raise RuntimeError("route-vocabulary preflight failed")
print({"gpus":n,"route_vocabulary_preflight":"pass","records":len(rows)})
PY
set +e
CUDA_VISIBLE_DEVICES="${G[0]}" TORCH_HOME="$ROOT/models/torch" "$PY" -u "$SCRIPT" --arm plain --run-id "$RUN_ID" >"$ROOT/logs/foundir_compositional_ebt_tensor_shard_100k_v5_${RUN_ID}_plain_eval_retry1.log" 2>&1 & p0=$!
CUDA_VISIBLE_DEVICES="${G[1]}" TORCH_HOME="$ROOT/models/torch" "$PY" -u "$SCRIPT" --arm compositional --run-id "$RUN_ID" >"$ROOT/logs/foundir_compositional_ebt_tensor_shard_100k_v5_${RUN_ID}_compositional_eval_retry1.log" 2>&1 & p1=$!
echo "evaluation_resumed plain_pid=$p0 compositional_pid=$p1"; wait "$p0"; s0=$?; wait "$p1"; s1=$?; set -e
echo "evaluation_finished plain_status=$s0 compositional_status=$s1"
tail -n 20 "$ROOT/logs/foundir_compositional_ebt_tensor_shard_100k_v5_${RUN_ID}_plain_eval_retry1.log" || true
tail -n 20 "$ROOT/logs/foundir_compositional_ebt_tensor_shard_100k_v5_${RUN_ID}_compositional_eval_retry1.log" || true
if [[ $s0 -ne 0 || $s1 -ne 0 ]]; then status=$s0; [[ $status -eq 0 ]] && status=$s1; echo "FOUNDIR_EBT_100K_MATCHED_660_RETRY1_END status=$status utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"; exit "$status"; fi
CUDA_VISIBLE_DEVICES="${G[0]}" "$PY" -u "$SCRIPT" --finalize --run-id "$RUN_ID"
sha256sum "$OUT_DIR/matched_660_result.json"
echo "FOUNDIR_EBT_100K_MATCHED_660_RETRY1_END status=0 utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) result=$OUT_DIR/matched_660_result.json"
