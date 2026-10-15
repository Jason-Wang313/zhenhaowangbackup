#!/usr/bin/env bash
set -u
cd /inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
exec env HF_HUB_DOWNLOAD_TIMEOUT=60 HF_HUB_ETAG_TIMEOUT=30 \
  .conda/gpu-train-stable/bin/python -u -c '
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="Qwen/Qwen3-4B-FP8", filename="model-00001-of-00002.safetensors", local_dir="models/qwen3-4b-fp8")
print("part1 complete", flush=True)
'
