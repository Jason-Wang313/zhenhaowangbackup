#!/usr/bin/env python3
"""Check homogeneous eight-H100/H200 execution without changing batch semantics."""

from __future__ import annotations

import json


LOCAL_BATCH = {"embedder": 32, "baseline": 16, "cure": 3}
GLOBAL_BATCH = {"embedder": 256, "baseline": 128, "cure": 24}


def validate_profile(names: list[str], memories: list[int]) -> dict:
    if len(names) != 8 or len(memories) != 8:
        raise RuntimeError("exactly eight visible GPUs required")
    if len(set(names)) != 1:
        raise RuntimeError(f"mixed GPUs not supported: {names}")
    model = "H200" if "H200" in names[0] else "H100" if "H100" in names[0] else None
    if model is None or min(memories) < 75 * 1024**3:
        raise RuntimeError(f"unsupported GPU name/memory: {names} {memories}")
    if any(8 * LOCAL_BATCH[stage] != GLOBAL_BATCH[stage] for stage in LOCAL_BATCH):
        raise RuntimeError("profile changes frozen global batch")
    return {
        "gpu_model": model,
        "names": names,
        "memory_bytes": memories,
        "local_batch": LOCAL_BATCH,
        "global_batch": GLOBAL_BATCH,
    }


def main() -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    memories = [torch.cuda.get_device_properties(i).total_memory for i in range(torch.cuda.device_count())]
    print("CURE_GPU_PROFILE_PASS " + json.dumps(validate_profile(names, memories)), flush=True)


if __name__ == "__main__":
    main()
