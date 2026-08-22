#!/usr/bin/env python3
"""Train and cache the frozen Gate 3 compact deterministic control."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from PIL import Image
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.compact_residual import CompactResidualControl, parameter_count


DEFAULT_CONFIG = ROOT / "configs/foundir_lowlight_gate3_v1.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def workspace_path(raw: str) -> Path:
    path = (ROOT / raw).resolve()
    path.relative_to(ROOT)
    return path


def read_config(path: Path) -> tuple[dict, str]:
    config = json.loads(path.read_text())
    if (
        config.get("schema_version") != 1
        or config.get("status") != "frozen_before_gate3_training"
        or config.get("protected_foundir_pixels_allowed") is not False
    ):
        raise RuntimeError("Gate 3 contract is not frozen or permits protected pixels")
    for key in (
        "development_manifest",
        "development_contract",
        "gate2_result",
        "cascade_manifest",
        "qwen_exposure_contract",
    ):
        dependency = workspace_path(config["parents"][key])
        if sha256(dependency) != config["parents"][f"{key}_sha256"]:
            raise RuntimeError(f"frozen Gate 3 dependency changed: {key}")
    for key in ("model", "trainer", "evaluator", "test"):
        identity = config["implementation_identities"]
        if sha256(workspace_path(identity[key])) != identity[f"{key}_sha256"]:
            raise RuntimeError(f"frozen Gate 3 implementation changed: {key}")
    training = config["training"]
    if (
        training["global_batch_size"]
        != training["world_size"] * training["batch_size_per_rank"] * training["gradient_accumulation_steps"]
        or training["sample_presentations"]
        != training["optimizer_steps"] * training["global_batch_size"]
    ):
        raise RuntimeError("Gate 3 optimizer/sample semantics changed")
    return config, sha256(path)


def read_manifest(config: Mapping[str, object]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    with workspace_path(config["parents"]["development_manifest"]).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if any(row["protected_exact_overlap"].lower() != "false" for row in rows):
        raise RuntimeError("development contract contains a protected overlap")
    training = [row for row in rows if row["role"] == config["data"]["training_role"]]
    outer = [row for row in rows if row["role"] == config["data"]["outer_role"]]
    if len(training) != config["data"]["training_records"] or len(outer) != config["data"]["outer_records"]:
        raise RuntimeError("Gate 3 development record counts changed")
    if sorted({row["family"] for row in training}) != sorted(config["data"]["training_families"]):
        raise RuntimeError("Gate 3 training families changed")
    if sorted({row["family"] for row in outer}) != sorted(config["data"]["outer_families"]):
        raise RuntimeError("Gate 3 outer families changed")
    identities = {(row["input_sha256"], row["target_sha256"]) for row in training}
    if any((row["input_sha256"], row["target_sha256"]) in identities for row in outer):
        raise RuntimeError("an exact pair crosses Gate 3 training and outer roles")
    return training, outer


def load_rgb(row: Mapping[str, str], key: str) -> np.ndarray:
    path = workspace_path(row[key])
    if path.suffix.lower() == ".npy":
        image = np.load(path, allow_pickle=False, mmap_mode="r")
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise RuntimeError(f"invalid RGB array: {path}")
        if row["stored_color_order"] == "BGR":
            image = image[..., ::-1]
        return np.asarray(image)
    with Image.open(path) as opened:
        return np.asarray(opened.convert("RGB"), dtype=np.uint8)


def to_tensor(image: np.ndarray) -> Tensor:
    return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float().div_(255.0)


def resize_view(image: np.ndarray, size: int) -> Tensor:
    resized = Image.fromarray(np.ascontiguousarray(image), mode="RGB").resize(
        (size, size), Image.Resampling.BILINEAR
    )
    return to_tensor(np.asarray(resized, dtype=np.uint8))


class PresentationDataset(Dataset):
    """Index-addressable, family/group-balanced frozen crop presentations."""

    def __init__(self, rows: Sequence[dict[str, str]], config: Mapping[str, object], indices: range) -> None:
        self.config = config
        self.indices = indices
        grouped: dict[str, dict[str, list[dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
        for row in rows:
            grouped[row["family"]][row["group_id"]].append(row)
        self.families = tuple(config["data"]["training_families"])
        self.groups = {
            family: tuple((group, tuple(sorted(records, key=lambda item: item["record_id"]))) for group, records in sorted(group_rows.items()))
            for family, group_rows in grouped.items()
        }

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, local_index: int) -> dict[str, Tensor]:
        presentation = self.indices[local_index]
        seed = int(self.config["training"]["seed"]) + presentation * 104729
        rng = np.random.default_rng(seed)
        family = self.families[presentation % len(self.families)]
        family_groups = self.groups[family]
        _, records = family_groups[int(rng.integers(0, len(family_groups)))]
        row = records[int(rng.integers(0, len(records)))]
        source = load_rgb(row, "input_path")
        target = load_rgb(row, "target_path")
        if source.shape != target.shape:
            raise RuntimeError(f"training pair dimensions differ: {row['record_id']}")
        identity = bool(rng.random() < float(self.config["data"]["identity_presentation_fraction"]))
        if identity:
            source = target
        crop = int(self.config["data"]["crop_size"])
        height, width = source.shape[:2]
        if height < crop or width < crop:
            scale = max(crop / height, crop / width)
            resized_size = (max(crop, round(width * scale)), max(crop, round(height * scale)))
            source = np.asarray(Image.fromarray(source).resize(resized_size, Image.Resampling.BICUBIC))
            target = np.asarray(Image.fromarray(target).resize(resized_size, Image.Resampling.BICUBIC))
            height, width = source.shape[:2]
        top = int(rng.integers(0, height - crop + 1))
        left = int(rng.integers(0, width - crop + 1))
        source_crop = np.ascontiguousarray(source[top : top + crop, left : left + crop])
        target_crop = np.ascontiguousarray(target[top : top + crop, left : left + crop])
        if self.config["data"]["d4_augmentation"]:
            transform = int(rng.integers(0, 8))
            turns, reflect = transform % 4, transform // 4
            source_crop = np.rot90(source_crop, turns)
            target_crop = np.rot90(target_crop, turns)
            if reflect:
                source_crop = source_crop[:, ::-1]
                target_crop = target_crop[:, ::-1]
        global_view = resize_view(source, int(self.config["data"]["global_view_size"]))
        return {
            "source": to_tensor(source_crop),
            "target": to_tensor(target_crop),
            "global_view": global_view,
            "identity": torch.tensor(identity, dtype=torch.bool),
        }


def build_model(config: Mapping[str, object]) -> CompactResidualControl:
    model = config["compact_control"]
    return CompactResidualControl(
        width=int(model["width"]),
        encoder_blocks=model["encoder_blocks"],
        middle_blocks=int(model["middle_blocks"]),
        decoder_blocks=model["decoder_blocks"],
        global_width=int(model["global_width"]),
        condition_dim=int(model["condition_dim"]),
        local_residual_bound=float(model["local_residual_bound"]),
    )


def learning_rate(step: int, config: Mapping[str, object]) -> float:
    training = config["training"]
    peak = float(training["learning_rate"])
    floor = float(training["minimum_learning_rate"])
    warmup = int(training["warmup_steps"])
    total = int(training["optimizer_steps"])
    if step <= warmup:
        return peak * step / warmup
    progress = (step - warmup) / (total - warmup)
    return floor + (peak - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))


def compute_loss(prediction: Tensor, target: Tensor, source: Tensor, identity: Tensor, config: Mapping[str, object]) -> tuple[Tensor, dict[str, float]]:
    contract = config["training"]["loss"]
    error = prediction - target
    mse = error.square().mean()
    charbonnier = torch.sqrt(error.square() + float(contract["charbonnier_epsilon"]) ** 2).mean()
    kernel = int(contract["low_frequency_kernel"])
    low_frequency = (F.avg_pool2d(prediction, kernel) - F.avg_pool2d(target, kernel)).square().mean()
    if identity.any():
        identity_error = (prediction[identity] - source[identity]).abs().mean()
    else:
        identity_error = prediction.new_zeros(())
    loss = (
        mse * float(contract["rgb_mse_weight"])
        + charbonnier * float(contract["charbonnier_weight"])
        + low_frequency * float(contract["low_frequency_mse_weight"])
        + identity_error * float(contract["identity_residual_weight"])
    )
    return loss, {
        "loss": float(loss.detach()),
        "mse": float(mse.detach()),
        "charbonnier": float(charbonnier.detach()),
        "low_frequency_mse": float(low_frequency.detach()),
        "identity_l1": float(identity_error.detach()),
    }


def atomic_checkpoint(path: Path, payload: Mapping[str, object]) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite immutable Gate 3 checkpoint: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".rank0-{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def latest_checkpoint(output: Path, allowed_steps: Sequence[int]) -> Path | None:
    realized = [output / f"checkpoint_step_{step:06d}.pt" for step in allowed_steps]
    existing = [path for path in realized if path.exists()]
    return existing[-1] if existing else None


def train(config: dict, config_sha: str, rows: Sequence[dict[str, str]], rank: int, world_size: int, device: torch.device) -> tuple[CompactResidualControl, list[dict[str, float]], int]:
    model = build_model(config).to(device=device, memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    output = workspace_path(config["outputs"]["directory"])
    checkpoints = tuple(int(step) for step in config["training"]["checkpoint_steps"])
    resume = latest_checkpoint(output, checkpoints)
    start_step = 0
    history: list[dict[str, float]] = []
    if resume is not None:
        state = torch.load(resume, map_location="cpu", weights_only=False)
        if state["config_sha256"] != config_sha or int(state["world_size"]) != world_size:
            raise RuntimeError("Gate 3 resume checkpoint contract differs")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_step = int(state["step"])
        history = list(state["history"])
    wrapped = DistributedDataParallel(model, device_ids=[device.index], broadcast_buffers=False)
    total_presentations = int(config["training"]["sample_presentations"])
    global_batch = int(config["training"]["global_batch_size"])
    start_presentation = start_step * global_batch + rank
    indices = range(start_presentation, total_presentations, world_size)
    dataset = PresentationDataset(rows, config, indices)
    loader = DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size_per_rank"]),
        shuffle=False,
        num_workers=int(config["runtime"]["num_data_workers_per_rank"]),
        pin_memory=True,
        persistent_workers=len(dataset) > 0,
        drop_last=True,
    )
    wrapped.train()
    started = time.perf_counter()
    for local_step, batch in enumerate(loader, start=start_step + 1):
        source = batch["source"].to(device, non_blocking=True, memory_format=torch.channels_last)
        target = batch["target"].to(device, non_blocking=True, memory_format=torch.channels_last)
        global_view = batch["global_view"].to(device, non_blocking=True, memory_format=torch.channels_last)
        identity = batch["identity"].to(device, non_blocking=True)
        rate = learning_rate(local_step, config)
        for group in optimizer.param_groups:
            group["lr"] = rate
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            prediction = wrapped(source, global_view)
            loss, values = compute_loss(prediction.float(), target.float(), source.float(), identity, config)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(wrapped.parameters(), float(config["training"]["gradient_clip_norm"]))
        optimizer.step()
        if local_step % 20 == 0 or local_step == start_step + 1:
            reduced = torch.tensor([values["loss"], values["mse"], values["charbonnier"], values["low_frequency_mse"], values["identity_l1"]], device=device)
            dist.all_reduce(reduced, op=dist.ReduceOp.AVG)
            if rank == 0:
                record = {
                    "step": local_step,
                    "loss": float(reduced[0]),
                    "mse": float(reduced[1]),
                    "charbonnier": float(reduced[2]),
                    "low_frequency_mse": float(reduced[3]),
                    "identity_l1": float(reduced[4]),
                    "learning_rate": rate,
                    "gradient_norm": float(gradient_norm),
                    "elapsed_seconds": time.perf_counter() - started,
                }
                history.append(record)
                print(f"GATE3_COMPACT_TRAIN step={local_step}/{config['training']['optimizer_steps']} loss={record['loss']:.7f} lr={rate:.8f}", flush=True)
        if local_step in checkpoints:
            dist.barrier()
            if rank == 0:
                atomic_checkpoint(
                    output / f"checkpoint_step_{local_step:06d}.pt",
                    {
                        "schema_version": 1,
                        "config_sha256": config_sha,
                        "step": local_step,
                        "world_size": world_size,
                        "parameter_count": parameter_count(model),
                        "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                        "optimizer": optimizer.state_dict(),
                        "history": history,
                    },
                )
            dist.barrier()
        if local_step >= int(config["training"]["optimizer_steps"]):
            break
    return model, history, start_step


def pad_multiple(value: Tensor, multiple: int) -> tuple[Tensor, int, int]:
    height, width = value.shape[-2:]
    bottom, right = (-height) % multiple, (-width) % multiple
    if bottom or right:
        mode = "reflect" if height > bottom and width > right else "replicate"
        value = F.pad(value, (0, right, 0, bottom), mode=mode)
    return value, height, width


@torch.inference_mode()
def restore(model: CompactResidualControl, image: np.ndarray, config: Mapping[str, object], device: torch.device) -> np.ndarray:
    model.eval()
    source = to_tensor(image).unsqueeze(0).to(device=device, memory_format=torch.channels_last)
    view = resize_view(image, int(config["data"]["global_view_size"])).unsqueeze(0).to(device=device, memory_format=torch.channels_last)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        tone = model.tone_parameters(view)
    height, width = image.shape[:2]
    inference = config["native_inference"]
    if height * width <= int(inference["whole_image_max_pixels"]):
        padded, native_height, native_width = pad_multiple(source, int(inference["padding_multiple"]))
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            restored = model(padded, tone_parameters=tone)[..., :native_height, :native_width]
    else:
        core, halo = int(inference["tile_core"]), int(inference["tile_halo"])
        padded_height = math.ceil(height / core) * core
        padded_width = math.ceil(width / core) * core
        bottom, right = padded_height - height, padded_width - width
        mode = "reflect" if height > bottom + halo and width > right + halo else "replicate"
        padded = F.pad(source, (halo, right + halo, halo, bottom + halo), mode=mode)
        output = torch.empty((1, 3, padded_height, padded_width), device=device, dtype=torch.float32)
        tiles: list[Tensor] = []
        locations: list[tuple[int, int]] = []
        tile_batch = 8
        for top in range(0, padded_height, core):
            for left in range(0, padded_width, core):
                tiles.append(padded[..., top : top + core + 2 * halo, left : left + core + 2 * halo])
                locations.append((top, left))
                if len(tiles) == tile_batch or (top + core >= padded_height and left + core >= padded_width):
                    batch = torch.cat(tiles, dim=0).contiguous(memory_format=torch.channels_last)
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        predicted = model(batch, tone_parameters=tone.expand(len(tiles), -1)).float()
                    for tile, (tile_top, tile_left) in zip(predicted, locations):
                        output[..., tile_top : tile_top + core, tile_left : tile_left + core] = tile[None, :, halo : halo + core, halo : halo + core]
                    tiles.clear()
                    locations.clear()
        restored = output[..., :height, :width]
    return restored[0].clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).cpu().numpy()


def png_bytes(image: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(image, mode="RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def save_once(path: Path, image: np.ndarray) -> str:
    content = png_bytes(image)
    if path.exists():
        if path.read_bytes() != content:
            raise RuntimeError(f"immutable Gate 3 output differs: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            handle.write(content)
    return hashlib.sha256(content).hexdigest()


def infer_outer(model: CompactResidualControl, rows: Sequence[dict[str, str]], config: Mapping[str, object], rank: int, world_size: int, device: torch.device) -> list[dict[str, object]]:
    results = []
    output = workspace_path(config["outputs"]["prediction_directory"])
    for index in range(rank, len(rows), world_size):
        row = rows[index]
        image = load_rgb(row, "input_path")
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        prediction = restore(model, image, config, device)
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        if prediction.shape != image.shape:
            raise RuntimeError(f"compact control changed native dimensions: {row['record_id']}")
        name = f"{index:04d}_{row['family']}_{hashlib.sha256(row['record_id'].encode()).hexdigest()[:16]}.png"
        path = output / name
        digest = save_once(path, prediction)
        results.append(
            {
                "candidate_id": config["compact_control"]["candidate_id"],
                "family": row["family"],
                "group_id": row["group_id"],
                "record_id": row["record_id"],
                "target_path": row["target_path"],
                "output_path": str(path.relative_to(ROOT)),
                "output_sha256": digest,
                "width": image.shape[1],
                "height": image.shape[0],
                "seconds": elapsed,
            }
        )
        print(f"GATE3_COMPACT_INFER rank={rank} records={len(results)} record_index={index + 1}/{len(rows)} seconds={elapsed:.4f}", flush=True)
    return results


def write_once(path: Path, content: bytes) -> None:
    if path.exists():
        if path.read_bytes() != content:
            raise RuntimeError(f"immutable Gate 3 artifact differs: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            handle.write(content)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--contract-only", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config, config_sha = read_config(config_path)
    training_rows, outer_rows = read_manifest(config)
    probe = build_model(config)
    count = parameter_count(probe)
    if args.contract_only:
        print(f"FOUNDIR_LOWLIGHT_GATE3_COMPACT_CONTRACT_PASS training={len(training_rows)} outer={len(outer_rows)} parameters={count}")
        return
    if not torch.cuda.is_available():
        raise SystemExit("Gate 3 compact training requires CUDA")
    dist.init_process_group("nccl")
    rank, world_size = dist.get_rank(), dist.get_world_size()
    if world_size != int(config["training"]["world_size"]) or torch.cuda.device_count() != world_size:
        raise RuntimeError("Gate 3 compact world size or visible device count changed")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    random.seed(int(config["training"]["seed"]) + rank)
    np.random.seed(int(config["training"]["seed"]) + rank)
    torch.manual_seed(int(config["training"]["seed"]))
    torch.cuda.manual_seed_all(int(config["training"]["seed"]))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    started = time.perf_counter()
    model, history, resumed_from = train(config, config_sha, training_rows, rank, world_size, device)
    dist.barrier()
    local_results = infer_outer(model, outer_rows, config, rank, world_size, device)
    gathered: list[list[dict[str, object]] | None] = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, local_results)
    dist.barrier()
    if rank == 0:
        results = sorted((record for part in gathered if part for record in part), key=lambda row: row["record_id"])
        if len(results) != len(outer_rows) or len({row["record_id"] for row in results}) != len(outer_rows):
            raise RuntimeError("Gate 3 compact inference result coverage differs")
        fields = list(results[0])
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(results)
        manifest_content = buffer.getvalue().encode()
        manifest_path = workspace_path(config["outputs"]["prediction_manifest"])
        write_once(manifest_path, manifest_content)
        checkpoint_path = workspace_path(config["outputs"]["checkpoint"])
        payload = {
            "schema_version": 1,
            "status": "compact_training_and_outer_inference_complete_awaiting_metrics",
            "protected_foundir_pixels_used": False,
            "config": str(config_path.relative_to(ROOT)),
            "config_sha256": config_sha,
            "runner_sha256": sha256(Path(__file__)),
            "implementation_sha256": sha256(ROOT / config["compact_control"]["implementation"]),
            "checkpoint": str(checkpoint_path.relative_to(ROOT)),
            "checkpoint_sha256": sha256(checkpoint_path),
            "parameter_count": count,
            "optimizer_steps": int(config["training"]["optimizer_steps"]),
            "sample_presentations": int(config["training"]["sample_presentations"]),
            "resumed_from_step": resumed_from,
            "history": history,
            "outer_records": len(results),
            "manifest": str(manifest_path.relative_to(ROOT)),
            "manifest_sha256": hashlib.sha256(manifest_content).hexdigest(),
            "runtime_seconds": time.perf_counter() - started,
            "mean_inference_seconds": float(np.mean([row["seconds"] for row in results])),
        }
        result_path = workspace_path(config["outputs"]["training_result"])
        write_once(result_path, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode())
        print(f"FOUNDIR_LOWLIGHT_GATE3_COMPACT_TRAINING_PASS records={len(results)} parameters={count} runtime_seconds={payload['runtime_seconds']:.3f}", flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
