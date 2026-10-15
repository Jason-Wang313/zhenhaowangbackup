#!/usr/bin/env bash
set -euo pipefail
ROOT=/inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/zhenhaowang
PY=/inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/.envs/gpu-train-stable-hdd3/bin/python
OUT="$ROOT/checkpoints/foundir_compositional_ebt_tensor_shard_100k_v5/shard100k_v5/calibration_v1"
LOG="$ROOT/logs/foundir_ebt_100k_inference_calibration_v1.log"
IFS=',' read -r -a G <<< "${CUDA_VISIBLE_DEVICES:-}"
if [[ ${#G[@]} -ne 2 || "${G[0]}" == "${G[1]}" ]]; then echo "CUDA_VISIBLE_DEVICES must name two distinct GPUs" >&2; exit 2; fi
if [[ -e "$OUT" || -e "$LOG" ]]; then echo "refusing duplicate calibration" >&2; exit 2; fi
mkdir -p "$OUT"; cd "$ROOT"; exec > >(tee "$LOG") 2>&1
echo "FOUNDIR_EBT_100K_CALIBRATION_START utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "PRELAUNCH_ESTIMATE wall_time_hours=0.08_to_0.50 gpu_hours=0.16_to_1.00 allocated_gpus=2"
sha256sum configs/foundir_ebt_100k_inference_calibration_v1.json smoke/calibrate_foundir_ebt_100k_inference_v1.py
"$PY" - <<'PY'
import torch
if not torch.cuda.is_available() or torch.cuda.device_count()!=2: raise RuntimeError("expected two visible GPUs")
print({"gpus":[torch.cuda.get_device_name(i) for i in range(2)]})
PY
set +e
CUDA_VISIBLE_DEVICES="${G[0]}" TORCH_HOME="$ROOT/models/torch" "$PY" -u smoke/calibrate_foundir_ebt_100k_inference_v1.py --arm plain >"$ROOT/logs/foundir_ebt_100k_inference_calibration_v1_plain.log" 2>&1 & p0=$!
CUDA_VISIBLE_DEVICES="${G[1]}" TORCH_HOME="$ROOT/models/torch" "$PY" -u smoke/calibrate_foundir_ebt_100k_inference_v1.py --arm compositional >"$ROOT/logs/foundir_ebt_100k_inference_calibration_v1_compositional.log" 2>&1 & p1=$!
echo "calibration_started plain_pid=$p0 compositional_pid=$p1"; wait "$p0"; s0=$?; wait "$p1"; s1=$?; set -e
echo "calibration_finished plain_status=$s0 compositional_status=$s1"
tail -n 20 "$ROOT/logs/foundir_ebt_100k_inference_calibration_v1_plain.log" || true
tail -n 20 "$ROOT/logs/foundir_ebt_100k_inference_calibration_v1_compositional.log" || true
if [[ $s0 -ne 0 || $s1 -ne 0 ]]; then status=$s0; [[ $status -eq 0 ]] && status=$s1; echo "FOUNDIR_EBT_100K_CALIBRATION_END status=$status utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"; exit "$status"; fi
CUDA_VISIBLE_DEVICES="${G[0]}" "$PY" -u smoke/calibrate_foundir_ebt_100k_inference_v1.py --finalize
sha256sum "$OUT/result.json"
echo "FOUNDIR_EBT_100K_CALIBRATION_END status=0 utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) result=$OUT/result.json"
