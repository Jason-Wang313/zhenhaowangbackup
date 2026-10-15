#!/usr/bin/env python3
"""Run frozen experts on Qwen-routed official SOW images at exact original size."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
import time
import uuid
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party/flux2/src"))

from lowlight_restoration.task_native_restormer import (  # noqa: E402
    Restormer,
    load_task_native_restormer,
)
from smoke.evaluate_foundir_mixed_restoration_pilot_ddp import load_flux_expert  # noqa: E402
from smoke.evaluate_foundir_blur_only_flux2_full_development_ddp import restore_batch  # noqa: E402
from smoke.export_foundir_official_sow600_qwen_features_ddp import (  # noqa: E402
    load_contract,
    workspace,
)
from smoke.run_foundir_qwen_five_class_gate_ddp import sha256  # noqa: E402


TASKS = ("rain", "noise", "haze", "low_light", "blur")


def stable_seed(*parts: object) -> int:
    payload = "\0".join(map(str, parts)).encode()
    return int(hashlib.sha256(payload).hexdigest()[:16], 16)


def load_rows(config: dict) -> tuple[list[dict[str, str]], dict[str, dict[str, str]]]:
    manifest = workspace(config["data"]["manifest"])
    decisions_path = workspace(config["runtime"]["router_decisions"])
    with manifest.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    with decisions_path.open(newline="") as handle:
        decisions = list(csv.DictReader(handle))
    by_id = {row["record_id"]: row for row in decisions}
    if len(rows) != 600 or len(by_id) != 600 or set(by_id) != {row["record_id"] for row in rows}:
        raise RuntimeError("official manifest and frozen Qwen decisions are not aligned")
    if any(row["status"] == "wrong" for row in decisions):
        # Wrong routes are scientifically reportable, but must never be silently hidden.
        print("WARNING: official Qwen decisions contain wrong routes", flush=True)
    return rows, by_id


def load_restoration_config(config: dict) -> dict:
    path = workspace(config["frozen_restoration"]["config"])
    if sha256(path) != config["frozen_restoration"]["config_sha256"]:
        raise RuntimeError("frozen restoration config changed")
    return json.loads(path.read_text())


def origins(length: int, tile_size: int, overlap: int) -> tuple[int, ...]:
    if length <= tile_size:
        return (0,)
    stride = tile_size - overlap
    values = list(range(0, length - tile_size + 1, stride))
    final = length - tile_size
    if values[-1] != final:
        values.append(final)
    return tuple(values)


def window(height: int, width: int) -> np.ndarray:
    vertical = np.hanning(height).astype(np.float32) if height > 2 else np.ones(height, np.float32)
    horizontal = np.hanning(width).astype(np.float32) if width > 2 else np.ones(width, np.float32)
    return np.maximum(np.sqrt(np.outer(vertical, horizontal)), np.float32(1e-3))


def tensor_tile(array: np.ndarray, x: int, y: int, tile_size: int) -> tuple[torch.Tensor, int, int]:
    cropped = array[y : min(y + tile_size, array.shape[0]), x : min(x + tile_size, array.shape[1])]
    height, width = cropped.shape[:2]
    value = torch.from_numpy(cropped.copy()).permute(2, 0, 1).float().div_(127.5).sub_(1)
    pad_right, pad_bottom = tile_size - width, tile_size - height
    if pad_right or pad_bottom:
        mode = "reflect" if height > 1 and width > 1 else "replicate"
        value = F.pad(value.unsqueeze(0), (0, pad_right, 0, pad_bottom), mode=mode).squeeze(0)
    return value, height, width


def save_png_once(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with Image.open(path) as handle:
            current = np.asarray(handle.convert("RGB"), dtype=np.uint8)
        if not np.array_equal(current, array):
            raise RuntimeError(f"existing official output differs: {path}")
        return
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    Image.fromarray(array, mode="RGB").save(temporary, format="PNG")
    try:
        os.link(temporary, path)
    except FileExistsError:
        with Image.open(path) as handle:
            current = np.asarray(handle.convert("RGB"), dtype=np.uint8)
        if not np.array_equal(current, array):
            raise RuntimeError(f"racing official output differs: {path}")
    finally:
        temporary.unlink(missing_ok=True)


def write_json_once(path: Path, payload: dict) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != encoded:
            raise RuntimeError(f"existing official sidecar differs: {path}")
        return
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    temporary.write_bytes(encoded)
    try:
        os.link(temporary, path)
    except FileExistsError:
        if path.read_bytes() != encoded:
            raise RuntimeError(f"racing official sidecar differs: {path}")
    finally:
        temporary.unlink(missing_ok=True)


def restore_image(
    *,
    array: np.ndarray,
    task: str,
    record_id: str,
    model,
    expert: dict,
    device: torch.device,
    profile: str,
    restoration: dict,
    flux_state,
    tile_size: int,
    overlap: int,
) -> tuple[np.ndarray, int]:
    height, width = array.shape[:2]
    positions = [(x, y) for y in origins(height, tile_size, overlap) for x in origins(width, tile_size, overlap)]
    accumulation = np.zeros((height, width, 3), dtype=np.float32)
    normalization = np.zeros((height, width), dtype=np.float32)
    batch_size = int(restoration["distributed"]["batch_size_per_rank"][profile])
    alpha = float(expert["alpha"])
    for start in range(0, len(positions), batch_size):
        batch_positions = positions[start : start + batch_size]
        materialized = [tensor_tile(array, x, y, tile_size) for x, y in batch_positions]
        source = torch.stack([item[0] for item in materialized]).to(device)
        if expert["kind"] == "flux2":
            flow, autoencoder, context, ids, pack_image = flux_state
            seeds = [stable_seed(restoration["inference"]["seed"], task, record_id, x, y) for x, y in batch_positions]
            restored = restore_batch(
                flow,
                autoencoder,
                source,
                context,
                ids,
                pack_image,
                seeds=seeds,
                steps=int(restoration["inference"]["num_steps"]),
                reference_time=int(restoration["inference"]["reference_time_coordinate"]),
            )
        else:
            with torch.inference_mode():
                restored = model((source + 1) * 0.5).float().clamp(0, 1).mul(2).sub(1)
        blended = source.float().add((restored.float() - source.float()) * alpha).clamp(-1, 1)
        blended = blended.add(1).mul(127.5).permute(0, 2, 3, 1).cpu().numpy()
        for (x, y), (_, tile_h, tile_w), tile in zip(batch_positions, materialized, blended, strict=True):
            cropped = tile[:tile_h, :tile_w]
            weights = window(tile_h, tile_w)
            accumulation[y : y + tile_h, x : x + tile_w] += cropped * weights[..., None]
            normalization[y : y + tile_h, x : x + tile_w] += weights
    if np.any(normalization <= 0):
        raise RuntimeError(f"tile stitching left uncovered pixels: {record_id}")
    output = np.round(accumulation / normalization[..., None]).clip(0, 255).astype(np.uint8)
    return output, len(positions)


def identity_stage(config: dict, rows: list[dict[str, str]], decisions: dict[str, dict[str, str]]) -> None:
    output = workspace(config["runtime"]["output_dir"]) / "restored"
    selected = [row for row in rows if decisions[row["record_id"]]["expert_plan"] == "identity"]
    for row in selected:
        source = workspace(row["lq_path"])
        target = output / f"{row['record_id']}.png"
        with Image.open(source) as handle:
            array = np.asarray(handle.convert("RGB"), dtype=np.uint8)
        save_png_once(target, array)
    print(f"FOUNDIR_OFFICIAL_IDENTITY_COMPLETE records={len(selected)}", flush=True)


def expert_stage(
    config: dict,
    rows: list[dict[str, str]],
    decisions: dict[str, dict[str, str]],
    task: str,
    *,
    shard_count: int,
    shard_index: int,
) -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    # Every rank restores disjoint records and only uses barriers.  A CPU
    # process group avoids making independent inference depend on NVLink/
    # NCCL fabric state; image tensors and models remain on the local GPU.
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    try:
        if world != 8 or torch.cuda.device_count() != 8:
            raise RuntimeError("official restoration requires exactly eight visible GPUs")
        name = torch.cuda.get_device_name(device)
        profile = "H200" if "H200" in name else "H100" if "H100" in name else ""
        if not profile:
            raise RuntimeError(f"unsupported official restoration GPU: {name}")
        restoration = load_restoration_config(config)
        expert = restoration["experts"][task]
        complete_selection = [
            row for row in rows if decisions[row["record_id"]]["expert_plan"] == task
        ]
        selected = complete_selection[shard_index::shard_count]
        if expert["kind"] == "flux2":
            flux_state = load_flux_expert(task, expert, restoration, device)
            model = None
        else:
            if expert["base_checkpoint_name"] == "motion_deblurring":
                # The pinned MLX-converted motion-deblurring checkpoint uses
                # the same Restormer graph with WithBias layer normalization.
                # Adaptation checkpoints already contain the complete PyTorch
                # state, so no upstream weights are loaded at inference time.
                model = Restormer(layer_norm_type="WithBias")
            else:
                model, _ = load_task_native_restormer(
                    expert["base_checkpoint_name"], workspace(expert["base_checkpoint_dir"])
                )
            payload = torch.load(workspace(expert["checkpoint"]), map_location="cpu", weights_only=False)
            if int(payload.get("step", -1)) != int(expert["checkpoint_step"]):
                raise RuntimeError(f"{task} checkpoint metadata changed")
            model.load_state_dict(payload["model"], strict=True)
            model = model.to(device).eval().requires_grad_(False)
            flux_state = None
        output = workspace(config["runtime"]["output_dir"])
        tile_size = int(config["frozen_restoration"]["tile_size"])
        overlap = int(config["frozen_restoration"]["tile_overlap"])
        started = time.monotonic()
        for index in range(rank, len(selected), world):
            row = selected[index]
            target = output / "restored" / f"{row['record_id']}.png"
            sidecar = output / "restoration_sidecars" / f"{row['record_id']}.json"
            if target.exists() and sidecar.exists():
                continue
            with Image.open(workspace(row["lq_path"])) as handle:
                array = np.asarray(handle.convert("RGB"), dtype=np.uint8)
            restored, tile_count = restore_image(
                array=array,
                task=task,
                record_id=row["record_id"],
                model=model,
                expert=expert,
                device=device,
                profile=profile,
                restoration=restoration,
                flux_state=flux_state,
                tile_size=tile_size,
                overlap=overlap,
            )
            if restored.shape != array.shape:
                raise RuntimeError(f"official output dimensions changed: {row['record_id']}")
            save_png_once(target, restored)
            write_json_once(sidecar, {
                "schema_version": 1,
                "status": "complete",
                "record_id": row["record_id"],
                "category": row["category"],
                "task": task,
                "checkpoint_sha256": expert["checkpoint_sha256"],
                "alpha": float(expert["alpha"]),
                "width": int(row["width"]),
                "height": int(row["height"]),
                "tile_size": tile_size,
                "tile_overlap": overlap,
                "tile_count": tile_count,
                "output_sha256": sha256(target),
            })
            print(f"FOUNDIR_OFFICIAL_RESTORE_PROGRESS task={task} rank={rank} complete={index + 1}/{len(selected)} tiles={tile_count}", flush=True)
        dist.barrier()
        if rank == 0:
            targets = [output / "restored" / f"{row['record_id']}.png" for row in selected]
            if any(not path.is_file() for path in targets):
                raise RuntimeError(f"official {task} outputs incomplete")
            marker_name = (
                f"{task}.json"
                if shard_count == 1
                else f"{task}.shard_{shard_index}_of_{shard_count}.json"
            )
            write_json_once(output / "stages" / marker_name, {
                "schema_version": 1,
                "status": "complete",
                "task": task,
                "records": len(selected),
                "complete_task_records": len(complete_selection),
                "shard_count": shard_count,
                "shard_index": shard_index,
                "checkpoint_sha256": expert["checkpoint_sha256"],
                "elapsed_seconds": time.monotonic() - started,
            })
            print(f"FOUNDIR_OFFICIAL_RESTORE_STAGE_COMPLETE task={task} records={len(selected)}", flush=True)
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=("identity", *TASKS), required=True)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    config, config_sha, _ = load_contract(args.config.resolve())
    rows, decisions = load_rows(config)
    counts = Counter(decisions[row["record_id"]]["expert_plan"] for row in rows)
    allowed = {"identity", *TASKS}
    if not set(counts).issubset(allowed) or sum(counts.values()) != 600:
        raise RuntimeError(f"official singleton plan escaped supported scope: {dict(counts)}")
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        raise RuntimeError("invalid official restoration shard geometry")
    if args.stage == "identity" and (args.shard_count != 1 or args.shard_index != 0):
        raise RuntimeError("identity stage is unsharded and must run exactly once")
    if args.check_only:
        restoration = load_restoration_config(config)
        for task, expert in restoration["experts"].items():
            if sha256(workspace(expert["checkpoint"])) != expert["checkpoint_sha256"]:
                raise RuntimeError(f"frozen expert checkpoint changed: {task}")
            if expert["kind"] == "flux2" and sha256(workspace(expert["parent_config"])) != expert["parent_config_sha256"]:
                raise RuntimeError(f"frozen expert parent config changed: {task}")
        tile_size = int(config["frozen_restoration"]["tile_size"])
        overlap = int(config["frozen_restoration"]["tile_overlap"])
        tiles = Counter()
        shard_tiles = Counter()
        for row in rows:
            plan = decisions[row["record_id"]]["expert_plan"]
            if plan != "identity":
                tiles[plan] += len(origins(int(row["height"]), tile_size, overlap)) * len(
                    origins(int(row["width"]), tile_size, overlap)
                )
        for task in TASKS:
            task_rows = [row for row in rows if decisions[row["record_id"]]["expert_plan"] == task]
            for row in task_rows[args.shard_index::args.shard_count]:
                shard_tiles[task] += len(origins(int(row["height"]), tile_size, overlap)) * len(
                    origins(int(row["width"]), tile_size, overlap)
                )
        print("FOUNDIR_OFFICIAL_RESTORATION_PREFLIGHT_PASS " + json.dumps({
            "config_sha256": config_sha,
            "plans": dict(counts),
            "tiles": dict(tiles),
            "shard_count": args.shard_count,
            "shard_index": args.shard_index,
            "shard_tiles": dict(shard_tiles),
        }, sort_keys=True))
        return
    if args.stage == "identity":
        identity_stage(config, rows, decisions)
    else:
        expert_stage(
            config,
            rows,
            decisions,
            args.stage,
            shard_count=args.shard_count,
            shard_index=args.shard_index,
        )


if __name__ == "__main__":
    main()
