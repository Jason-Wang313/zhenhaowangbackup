#!/usr/bin/env python3
"""Eight-GPU zero-update preflight for the intersection model."""

from __future__ import annotations

import hashlib
import json
import os
import random
import socket
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.vlm_energy_objectives import (  # noqa: E402
    edge_reconstruction,
    trajectory_calibrated_objectives,
)
from lowlight_restoration.vlm_energy_restoration import (  # noqa: E402
    DEGRADATIONS,
    VLMConditionedEnergyRestorer,
    count_trainable_parameters,
)


CONFIG_PATH = ROOT / "configs/foundir_vlm_energy_intersection_v1.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_model(config: dict[str, Any]) -> VLMConditionedEnergyRestorer:
    model = config["model"]
    torch.manual_seed(int(model["initialization_seed"]))
    result = VLMConditionedEnergyRestorer(
        condition_dim=int(model["condition_dim"]),
        proposal_width=int(model["proposal_width"]),
        proposal_encoder_blocks=tuple(model["proposal_encoder_blocks"]),
        proposal_middle_blocks=int(model["proposal_middle_blocks"]),
        proposal_residual_bound=float(model["proposal_residual_bound"]),
        energy_width=int(model["energy_width"]),
        energy_blocks_per_scale=tuple(model["energy_blocks_per_scale"]),
        maximum_descent_steps=int(model["maximum_descent_steps"]),
        initial_step_size=float(model["initial_step_size"]),
    )
    actual = count_trainable_parameters(result)
    if actual != int(model["trainable_parameters"]):
        raise RuntimeError(f"model parameter count changed: {actual}")
    return result


def load_pair(row: dict[str, Any], target: dict[str, Any], crop: int, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with Image.open(ROOT / str(row["output_path"])) as handle:
        degraded = np.asarray(handle.convert("RGB"), dtype=np.uint8)
    with Image.open(ROOT / str(target["target_path"])) as handle:
        clean = np.asarray(handle.convert("RGB"), dtype=np.uint8)
    if degraded.shape != clean.shape or min(degraded.shape[:2]) < crop:
        raise RuntimeError(f"bad preflight pair: {row['record_id']}")
    rng = random.Random(seed)
    y = rng.randrange(degraded.shape[0] - crop + 1)
    x = rng.randrange(degraded.shape[1] - crop + 1)
    degraded = np.ascontiguousarray(degraded[y:y+crop, x:x+crop])
    clean = np.ascontiguousarray(clean[y:y+crop, x:x+crop])
    present = set(map(str, row["degradations"]))
    q = torch.tensor([float(name in present) for name in DEGRADATIONS])
    d = torch.from_numpy(degraded).permute(2, 0, 1).float().div_(127.5).sub_(1)
    c = torch.from_numpy(clean).permute(2, 0, 1).float().div_(127.5).sub_(1)
    return d, c, q


def main() -> None:
    dist.init_process_group("nccl")
    rank, world, local_rank = dist.get_rank(), dist.get_world_size(), int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    config = json.loads(CONFIG_PATH.read_text())
    if world != int(config["preflight"]["required_visible_gpus"]):
        raise RuntimeError(f"preflight requires exactly 8 ranks, got {world}")
    gpu = torch.cuda.get_device_name(local_rank)
    if not any(name in gpu for name in ("H100", "H200")):
        raise RuntimeError(f"unsupported GPU: {gpu}")
    for key in ("manifest", "metadata", "target_manifest", "target_metadata"):
        path = ROOT / config["data"][key]
        expected = config["data"][key + "_sha256"]
        if sha256(path) != expected:
            raise RuntimeError(f"data identity changed: {path}")
    expected_torch_home = (ROOT / "models/torch").resolve()
    actual_torch_home = Path(os.environ.get("TORCH_HOME", "")).resolve()
    if actual_torch_home != expected_torch_home:
        raise RuntimeError(f"TORCH_HOME must be {expected_torch_home}, got {actual_torch_home}")
    alexnet = ROOT / config["loss"]["alexnet_weights"]
    if sha256(alexnet) != config["loss"]["alexnet_weights_sha256"]:
        raise RuntimeError("offline AlexNet LPIPS weights changed")
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        if os.environ.get(key) != "1":
            raise RuntimeError(f"offline environment missing {key}=1")

    rows = [row for row in load_jsonl(ROOT / config["data"]["manifest"]) if row["split"] == "train"]
    targets = {str(row["record_id"]): row for row in load_jsonl(ROOT / config["data"]["target_manifest"])}
    order = rank % 3 + 1
    selected = next(row for row in rows if int(row["composition_size"]) == order)
    degraded, clean, descriptor = load_pair(
        selected, targets[str(selected["record_id"])], int(config["data"]["crop_size"]),
        int(config["model"]["initialization_seed"]) + rank,
    )
    degraded = degraded[None].to(device)
    clean = clean[None].to(device)
    descriptor = descriptor[None].to(device)

    model = build_model(config).to(device)
    distributed = DDP(model, device_ids=[local_rank], broadcast_buffers=False, find_unused_parameters=False)
    # TORCH_HOME is pinned to the workspace cache above.  Construction must
    # resolve the immutable local AlexNet weights and is forbidden to access
    # the network.
    perceptual = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=False).to(device).eval()
    perceptual.requires_grad_(False)
    torch.cuda.reset_peak_memory_stats(device)
    dist.barrier()
    started = time.perf_counter()
    trajectory = distributed(
        degraded,
        descriptor,
        steps=int(config["preflight"]["descent_steps"]),
        create_graph=True,
        step_scale=1.0,
    )
    terms = trajectory_calibrated_objectives(
        distributed.module,
        degraded,
        clean,
        descriptor,
        trajectory,
        ranking_margin_scale=float(config["landscape_training"]["ranking_margin_scale"]),
    )
    weights = config["loss"]
    lpips_loss = perceptual(trajectory.output.float(), clean.float())
    loss = (
        float(weights["multi_scale_reconstruction"]) * terms.reconstruction
        + float(weights["edge_reconstruction"]) * edge_reconstruction(trajectory.output, clean)
        + float(weights["lpips_alex"]) * lpips_loss
        + float(weights["vector_direction"]) * terms.vector_direction
        + float(weights["vector_progress"]) * terms.vector_progress
        + float(weights["quality_ranking"]) * terms.quality_ranking
    )
    if not bool(torch.isfinite(loss)):
        raise RuntimeError("non-finite preflight loss")
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(distributed.parameters(), 1.0)
    if not bool(torch.isfinite(norm)):
        raise RuntimeError("non-finite preflight gradient")
    dist.barrier()
    elapsed = time.perf_counter() - started
    payload = {
        "rank": rank,
        "host": socket.gethostname(),
        "gpu": gpu,
        "composition_size": order,
        "record_id": selected["record_id"],
        "loss": float(loss.detach()),
        "lpips_loss": float(lpips_loss.detach()),
        "gradient_norm": float(norm),
        "elapsed_seconds": elapsed,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
    }
    gathered: list[dict[str, Any] | None] = [None] * world if rank == 0 else []
    dist.gather_object(payload, gathered if rank == 0 else None, dst=0)
    if rank == 0:
        result = {
            "schema_version": 1,
            "status": "foundir_vlm_energy_intersection_v1_preflight_pass",
            "config_sha256": sha256(CONFIG_PATH),
            "world_size": world,
            "optimizer_steps": 0,
            "ranks": gathered,
            "maximum_elapsed_seconds": max(float(row["elapsed_seconds"]) for row in gathered if row),
            "maximum_peak_allocated_bytes": max(int(row["peak_allocated_bytes"]) for row in gathered if row),
            "parent_test_accessed": False,
            "official_foundir_test_accessed": False,
            "sealed_internal_accessed": False,
        }
        output = ROOT / config["runtime"]["output_root"] / "preflight" / "result.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(output.name + f".tmp.{os.getpid()}")
        temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, output)
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
