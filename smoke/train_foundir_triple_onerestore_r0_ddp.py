#!/usr/bin/env python3
"""Train the three frozen OneRestore-style R0 arms and export final outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.onerestore_r0 import (  # noqa: E402
    CATEGORY_TO_INDEX,
    OneRestoreR0,
    VGG16FeatureLoss,
    category_name,
    ms_ssim,
    parameter_count,
)


CONFIG_PATH = ROOT / "configs/foundir_triple_onerestore_r0_v1.json"
EXPECTED_CONFIG_SHA = "415ffaa126393e0125ca070830596c413645919836e290b7a16970b2f27c2c64"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != encoded:
            raise RuntimeError(f"refusing to overwrite differing artifact: {path}")
        return
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_bytes(encoded)
    os.replace(temporary, path)


def save_png_atomic(path: Path, pixels: np.ndarray) -> str:
    image = Image.fromarray(pixels, mode="RGB")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    image.save(temporary, format="PNG")
    payload = temporary.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"refusing to overwrite differing output: {path}")
        temporary.unlink()
    else:
        os.replace(temporary, path)
    return digest


def load_rgb(path: Path) -> torch.Tensor:
    with Image.open(path) as handle:
        array = np.asarray(handle.convert("RGB"), dtype=np.uint8).copy()
    return torch.from_numpy(array).permute(2, 0, 1).float().div_(255.0)


def node_name(order: list[str]) -> str:
    return "clean" if not order else "+".join(order)


class FrozenRecords:
    def __init__(self, config: dict[str, Any]) -> None:
        lattices = load_jsonl(ROOT / config["parent_protocol"]["manifest"])
        cache_rows = load_jsonl(ROOT / config["data"]["cache_root"] / "manifest.jsonl")
        self.cache = {
            (str(row["parent_record_id"]), str(row["node_id"])): ROOT / row["path"]
            for row in cache_rows
        }
        self.by_role: dict[str, list[dict[str, Any]]] = {}
        for role in ("train", "development", "heldout_triple"):
            self.by_role[role] = sorted(
                (row for row in lattices if row["protocol_role"] == role),
                key=lambda row: str(row["parent_record_id"]),
            )

    def paths(self, record: dict[str, Any]) -> tuple[Path, Path, Path]:
        parent = str(record["parent_record_id"])
        full = node_name(list(map(str, record["application_order"])))
        nonclean = [
            node_name(node["application_order"])
            for node in record["nodes"]
            if node["application_order"] and node_name(node["application_order"]) != full
        ]
        negative = sorted(nonclean)[0] if nonclean else full
        return self.cache[(parent, full)], self.cache[(parent, "clean")], self.cache[(parent, negative)]


def deterministic_crop_augment(
    images: tuple[torch.Tensor, ...], crop: int, seed: int
) -> tuple[torch.Tensor, ...]:
    generator = random.Random(seed)
    height, width = images[0].shape[-2:]
    y = generator.randrange(height - crop + 1)
    x = generator.randrange(width - crop + 1)
    result = [image[:, y : y + crop, x : x + crop] for image in images]
    if generator.randrange(2):
        result = [image.flip(-1) for image in result]
    if generator.randrange(2):
        result = [image.flip(-2) for image in result]
    if generator.randrange(2):
        result = [image.transpose(-2, -1) for image in result]
    return tuple(result)


def validate(config: dict[str, Any]) -> FrozenRecords:
    if sha256_file(CONFIG_PATH) != EXPECTED_CONFIG_SHA:
        raise RuntimeError("frozen config hash mismatch")
    for key in ("config", "manifest", "metadata"):
        path = ROOT / config["parent_protocol"][key]
        if sha256_file(path) != config["parent_protocol"][f"{key}_sha256"]:
            raise RuntimeError(f"parent protocol {key} hash mismatch")
    weight = ROOT / config["frozen_losses"]["vgg16_weight"]
    if sha256_file(weight) != config["frozen_losses"]["vgg16_weight_sha256"]:
        raise RuntimeError("VGG16 weight hash mismatch")
    cache_metadata = json.loads((ROOT / config["data"]["cache_root"] / "metadata.json").read_text())
    if cache_metadata["records"] != 620 or cache_metadata["config_sha256"] != EXPECTED_CONFIG_SHA:
        raise RuntimeError("cache metadata mismatch")
    records = FrozenRecords(config)
    expected = {"train": 60, "development": 30, "heldout_triple": 40}
    if {role: len(rows) for role, rows in records.by_role.items()} != expected:
        raise RuntimeError("role parent counts changed")
    if any(int(row["composition_size"]) > 2 for row in records.by_role["train"]):
        raise RuntimeError("triple gradient leakage")
    if any(int(row["composition_size"]) > 2 for row in records.by_role["development"]):
        raise RuntimeError("triple checkpoint-selection leakage")
    if any(int(row["composition_size"]) != 3 for row in records.by_role["heldout_triple"]):
        raise RuntimeError("heldout role is not triple-only")
    model = OneRestoreR0(
        channel=int(config["model"]["channel"]),
        descriptor_dim=int(config["model"]["descriptor_dim"]),
        descriptor_classes=int(config["model"]["descriptor_classes"]),
        descriptor_seed=int(config["model"]["descriptor_seed"]),
    )
    if parameter_count(model) != int(config["model"]["parameter_count"]):
        raise RuntimeError("model parameter count changed")
    return records


def category_index(record: dict[str, Any]) -> int:
    name = category_name(record["degradations"])
    return CATEGORY_TO_INDEX[name] if name in CATEGORY_TO_INDEX else CATEGORY_TO_INDEX["clean"]


def init_distributed() -> tuple[int, int, torch.device]:
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)
    return rank, world, device


def hardware_guard(world: int, device: torch.device) -> str:
    if world != 8:
        raise RuntimeError("frozen R0 requires exactly eight ranks")
    name = torch.cuda.get_device_name(device)
    family = "H200" if "H200" in name else "H100" if "H100" in name else ""
    families: list[str | None] = [None] * world
    dist.all_gather_object(families, family)
    if not family or len(set(families)) != 1:
        raise RuntimeError(f"unsupported or mixed GPU families: {families}")
    return family


def train_arm(
    config: dict[str, Any],
    records: FrozenRecords,
    arm: dict[str, Any],
    rank: int,
    world: int,
    device: torch.device,
) -> dict[str, Any]:
    seed = int(config["training"]["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    model = OneRestoreR0(
        channel=int(config["model"]["channel"]),
        descriptor_dim=int(config["model"]["descriptor_dim"]),
        descriptor_classes=int(config["model"]["descriptor_classes"]),
        descriptor_seed=int(config["model"]["descriptor_seed"]),
    ).to(device)
    ddp = DDP(model, device_ids=[device.index], output_device=device.index, broadcast_buffers=True)
    optimizer = torch.optim.AdamW(
        ddp.parameters(),
        lr=float(config["training"]["learning_rate"]),
        betas=tuple(map(float, config["training"]["betas"])),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    cdrl: nn.Module | None = None
    if arm["cdrl"]:
        cdrl = VGG16FeatureLoss(ROOT / config["frozen_losses"]["vgg16_weight"]).to(device)
    train_records = records.by_role["train"]
    cached = [tuple(load_rgb(path) for path in records.paths(record)) for record in train_records]
    steps = int(config["training"]["total_steps_per_arm"])
    minimum_lr = 1e-6
    metrics = []
    started = time.time()
    for step in range(1, steps + 1):
        position = ((step - 1) * world + rank) % len(train_records)
        record = train_records[position]
        degraded, clean, negative = deterministic_crop_augment(
            cached[position], int(config["data"]["crop_size"]), seed + (step - 1) * world + rank
        )
        degraded = degraded.unsqueeze(0).to(device, non_blocking=True)
        clean = clean.unsqueeze(0).to(device, non_blocking=True)
        negative = negative.unsqueeze(0).to(device, non_blocking=True)
        index = torch.tensor([category_index(record)], device=device)
        progress = (step - 1) / max(steps - 1, 1)
        lr = minimum_lr + 0.5 * (float(config["training"]["learning_rate"]) - minimum_lr) * (
            1.0 + math.cos(math.pi * progress)
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            restored = ddp(
                degraded,
                index,
                descriptor_enabled=bool(arm["descriptor"]),
            )
            smooth = F.smooth_l1_loss(restored, clean)
            structural = 1.0 - ms_ssim(restored.float(), clean.float())
            contrast = restored.new_zeros(())
            if cdrl is not None:
                contrast = cdrl(degraded.float(), clean.float(), negative.float(), restored.float())
            weights = arm["loss_weights"]
            loss = (
                float(weights["smooth_l1"]) * smooth
                + float(weights["ms_ssim"]) * structural
                + float(weights["cdrl"]) * contrast
            )
        if not torch.isfinite(loss):
            raise RuntimeError(f"nonfinite loss arm={arm['name']} step={step}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(ddp.parameters(), float(config["training"]["gradient_clip_norm"]))
        if not torch.isfinite(grad_norm):
            raise RuntimeError(f"nonfinite gradient arm={arm['name']} step={step}")
        optimizer.step()
        if step % 25 == 0 or step == 1 or step == steps:
            row = {
                "arm": arm["name"],
                "step": step,
                "loss": float(loss.detach()),
                "smooth_l1": float(smooth.detach()),
                "ms_ssim_loss": float(structural.detach()),
                "cdrl": float(contrast.detach()),
                "grad_norm": float(grad_norm.detach()),
                "lr": lr,
            }
            if rank == 0:
                print("FOUNDIR_TRIPLE_R0_TRAIN " + json.dumps(row, sort_keys=True), flush=True)
                metrics.append(row)
    dist.barrier()
    output_root = ROOT / config["runtime"]["output_root"] / str(arm["name"])
    checkpoint = output_root / f"step_{steps:06d}.pt"
    if rank == 0:
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        temporary = checkpoint.with_name(checkpoint.name + f".tmp.{os.getpid()}")
        torch.save(
            {
                "schema_version": 1,
                "config_sha256": EXPECTED_CONFIG_SHA,
                "arm": arm,
                "step": steps,
                "model": ddp.module.state_dict(),
            },
            temporary,
        )
        os.replace(temporary, checkpoint)
    dist.barrier()
    return {
        "arm": arm["name"],
        "checkpoint": checkpoint.relative_to(ROOT).as_posix(),
        "checkpoint_sha256": sha256_file(checkpoint) if rank == 0 else None,
        "elapsed_seconds": time.time() - started,
        "training_metrics": metrics if rank == 0 else [],
    }


@torch.no_grad()
def export_arm_outputs(
    config: dict[str, Any],
    records: FrozenRecords,
    arm: dict[str, Any],
    checkpoint_path: Path,
    rank: int,
    world: int,
    device: torch.device,
) -> int:
    model = OneRestoreR0(
        channel=int(config["model"]["channel"]),
        descriptor_dim=int(config["model"]["descriptor_dim"]),
        descriptor_classes=int(config["model"]["descriptor_classes"]),
        descriptor_seed=int(config["model"]["descriptor_seed"]),
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    count = 0
    for role in ("development", "heldout_triple"):
        for index_record, record in enumerate(records.by_role[role]):
            if index_record % world != rank:
                continue
            degraded_path, clean_path, _ = records.paths(record)
            degraded = load_rgb(degraded_path).unsqueeze(0).to(device)
            class_index = torch.tensor([category_index(record)], device=device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                restored = model(
                    degraded,
                    class_index,
                    descriptor_enabled=bool(arm["descriptor"]),
                    degradations=[list(map(str, record["degradations"]))],
                )
            pixels = (
                restored[0].float().clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).cpu().numpy()
            )
            output = (
                ROOT
                / config["runtime"]["output_root"]
                / str(arm["name"])
                / "outputs"
                / role
                / f"{record['parent_record_id']}.png"
            )
            output_sha = save_png_atomic(output, pixels)
            write_json_atomic(
                output.with_suffix(".json"),
                {
                    "schema_version": 1,
                    "status": "complete",
                    "config_sha256": EXPECTED_CONFIG_SHA,
                    "arm": arm["name"],
                    "role": role,
                    "parent_record_id": record["parent_record_id"],
                    "source_id": record["source_id"],
                    "degradations": record["degradations"],
                    "composition_size": record["composition_size"],
                    "input_path": degraded_path.relative_to(ROOT).as_posix(),
                    "input_sha256": sha256_file(degraded_path),
                    "gt_path": clean_path.relative_to(ROOT).as_posix(),
                    "gt_sha256": sha256_file(clean_path),
                    "output_path": output.relative_to(ROOT).as_posix(),
                    "output_sha256": output_sha,
                    "official_foundir_test_accessed": False,
                    "foundir_60_image_triple_accessed": False,
                },
            )
            count += 1
    dist.barrier()
    return count


def preflight(config: dict[str, Any], records: FrozenRecords) -> None:
    model = OneRestoreR0(channel=8)
    image = torch.rand(1, 3, 64, 64)
    index = torch.tensor([CATEGORY_TO_INDEX["blur+noise"]])
    output = model(image, index, descriptor_enabled=True)
    loss = F.smooth_l1_loss(output, image) + 0.1 * (1.0 - ms_ssim(output, image))
    loss.backward()
    if not torch.isfinite(loss):
        raise RuntimeError("CPU forward/backward preflight failed")
    # Exercise the frozen CDRL implementation and the exact serialized VGG16
    # weights before consuming an eight-GPU allocation.  A small spatial size
    # is sufficient to validate loading, feature capture, gradients to the
    # restored image, and finite loss without making CPU preflight expensive.
    cdrl = VGG16FeatureLoss(ROOT / config["frozen_losses"]["vgg16_weight"])
    degraded = torch.rand(1, 3, 32, 32)
    clean = torch.rand(1, 3, 32, 32)
    negative = torch.rand(1, 3, 32, 32)
    restored = torch.rand(1, 3, 32, 32, requires_grad=True)
    cdrl_loss = cdrl(degraded, clean, negative, restored)
    cdrl_loss.backward()
    if not torch.isfinite(cdrl_loss) or restored.grad is None or not torch.isfinite(restored.grad).all():
        raise RuntimeError("CPU CDRL forward/backward preflight failed")
    print(
        "FOUNDIR_TRIPLE_ONERESTORE_R0_PREFLIGHT_PASS "
        f"train={len(records.by_role['train'])} development={len(records.by_role['development'])} "
        f"heldout_triple={len(records.by_role['heldout_triple'])} config_sha256={EXPECTED_CONFIG_SHA}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if args.config.resolve() != CONFIG_PATH:
        raise RuntimeError("only the frozen R0 config is accepted")
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    records = validate(config)
    if args.preflight:
        preflight(config, records)
        return
    rank, world, device = init_distributed()
    family = hardware_guard(world, device)
    summaries = []
    started = time.time()
    # Freeze all three final checkpoints before the first held-out-triple
    # access.  Exporting inside this training loop would violate the protocol
    # even if those metrics were not used for selection.
    for arm in config["arms"]:
        summary = train_arm(config, records, arm, rank, world, device)
        if rank == 0:
            summaries.append(summary)
    dist.barrier()
    for arm_index, arm in enumerate(config["arms"]):
        checkpoint = ROOT / config["runtime"]["output_root"] / str(arm["name"]) / f"step_{int(config['training']['total_steps_per_arm']):06d}.pt"
        exported = export_arm_outputs(config, records, arm, checkpoint, rank, world, device)
        if rank == 0:
            summaries[arm_index]["rank0_exported_outputs"] = exported
    dist.barrier()
    if rank == 0:
        write_json_atomic(
            ROOT / config["runtime"]["output_root"] / "training_result.json",
            {
                "schema_version": 1,
                "status": "foundir_triple_onerestore_r0_v1_training_complete",
                "config_sha256": EXPECTED_CONFIG_SHA,
                "gpu_family": family,
                "world_size": world,
                "arms": summaries,
                "elapsed_seconds": time.time() - started,
                "official_foundir_test_accessed": False,
                "foundir_60_image_triple_accessed": False,
            },
        )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
