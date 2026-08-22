#!/usr/bin/env python3
"""Export frozen Qwen soft and visual conditions for Gate 3 development inputs."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.causal_condition import qwen_condition_from_scores
from lowlight_restoration.data import PairRecord
from lowlight_restoration.qwen_constrained import mean_shifted_target_token_nll
from lowlight_restoration.qwen_real_validation import load_real_validation_config, validate_real_validation_contract
from lowlight_restoration.qwen_supervised import prepare_supervised_batch
from smoke.evaluate_qwen_lowlight_real_validation import _load_frozen_model
from smoke.qwen_lowlight_supervised_preflight import sha256, workspace_path
from smoke.train_qwen_lowlight_supervised import atomic_write_new_json, seed_everything


def read_contract(path: Path) -> tuple[dict[str, object], str, list[dict[str, str]], dict[str, object]]:
    config = json.loads(path.read_text())
    if (
        config.get("schema_version") != 1
        or config.get("status") != "frozen_before_gate3_causal_matrix"
        or config.get("protected_foundir_pixels_allowed") is not False
    ):
        raise RuntimeError("causal contract is not frozen or permits protected pixels")
    parent = config["parents"]
    for key in (
        "development_manifest", "gate3_config", "gate3_checkpoint", "gate3_result",
        "qwen_exposure_config", "qwen_visual_checkpoint", "qwen_ordinal_result",
    ):
        if sha256(workspace_path(parent[key])) != parent[f"{key}_sha256"]:
            raise RuntimeError(f"causal dependency changed: {key}")
    for key, identity in config["implementation_identities"].items():
        if sha256(workspace_path(identity["path"])) != identity["sha256"]:
            raise RuntimeError(f"causal implementation changed: {key}")
    with workspace_path(parent["development_manifest"]).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected = int(config["data"]["training_records"]) + int(config["data"]["outer_records"])
    if len(rows) != expected or any(row["protected_exact_overlap"].lower() != "false" for row in rows):
        raise RuntimeError("causal development coverage or protected exclusion changed")
    exposure = json.loads(workspace_path(parent["qwen_exposure_config"]).read_text())
    base_path = workspace_path(exposure["model"]["base_contract"])
    if sha256(base_path) != exposure["model"]["base_contract_sha256"]:
        raise RuntimeError("Qwen base model contract changed")
    model_contract, _, _ = load_real_validation_config(base_path)
    validate_real_validation_contract(model_contract)
    model_contract = copy.deepcopy(model_contract)
    model_contract["model"].update(
        selected_checkpoint_step=exposure["model"]["selected_checkpoint_step"],
        selected_visual_checkpoint=exposure["model"]["selected_visual_checkpoint"],
        selected_visual_sha256=exposure["model"]["selected_visual_sha256"],
    )
    return config, sha256(path), rows, model_contract


def records(rows: Sequence[Mapping[str, str]], seed: int) -> list[PairRecord]:
    return [
        PairRecord(
            pair_id=row["record_id"],
            source_id=row["group_id"],
            clean_path=workspace_path(row["input_path"]),
            degraded_path=workspace_path(row["input_path"]),
            project_level="L1",
            seed=seed,
            split=row["role"],
            clean_sha256=row["input_sha256"],
            degraded_sha256=row["input_sha256"],
        )
        for row in rows
    ]


def open_input_images(batch_records: Sequence[PairRecord]) -> list[Any]:
    """Open RGB inputs, including the frozen SDSD BGR uint8 NPY storage."""

    from PIL import Image

    images = []
    for record in batch_records:
        if record.degraded_path.suffix.lower() == ".npy":
            array = np.load(record.degraded_path, allow_pickle=False, mmap_mode="r")
            if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
                raise RuntimeError(f"invalid causal Qwen NPY input: {record.degraded_path}")
            images.append(Image.fromarray(np.ascontiguousarray(array[..., ::-1]), mode="RGB"))
        else:
            with Image.open(record.degraded_path) as opened:
                images.append(opened.convert("RGB"))
    return images


def visual_projection(visual: torch.Tensor, counts: Sequence[int], output_dim: int, seed: int) -> list[list[float]]:
    if visual.ndim != 2 or sum(counts) != visual.shape[0]:
        raise RuntimeError(f"Qwen visual-token segmentation changed: shape={tuple(visual.shape)} counts={counts}")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    projection = torch.randn(visual.shape[1] * 2, output_dim, generator=generator, dtype=torch.float32)
    projection.mul_(1.0 / math.sqrt(visual.shape[1] * 2))
    projection = projection.to(visual.device)
    rows, start = [], 0
    for count in counts:
        tokens = visual[start : start + count].float()
        pooled = torch.cat((tokens.mean(0), tokens.std(0, unbiased=False)))
        rows.append((pooled @ projection).cpu().tolist())
        start += count
    return rows


def score_batch(
    model: torch.nn.Module,
    processor: Any,
    batch_records: Sequence[PairRecord],
    *,
    device: torch.device,
    output_dim: int,
    projection_seed: int,
    temperature: float,
) -> list[dict[str, object]]:
    images = open_input_images(batch_records)
    score_rows: list[dict[str, float]] = [dict() for _ in batch_records]
    visual_rows: list[list[float]] | None = None
    captured: list[torch.Tensor] = []

    def hook(_module: torch.nn.Module, _inputs: object, output: object) -> None:
        value = output[0] if isinstance(output, tuple) else output
        if not isinstance(value, torch.Tensor):
            raise RuntimeError("Qwen visual output is not a tensor")
        captured.append(value.detach())

    handle = model.visual.register_forward_hook(hook)
    try:
        for level in range(1, 6):
            candidates = [replace(record, project_level=f"L{level}") for record in batch_records]
            batch, expected_counts = prepare_supervised_batch(processor, candidates, images)
            labels = batch.pop("labels").to(device)
            grids = batch["image_grid_thw"]
            merge = int(processor.image_processor.merge_size)
            token_counts = [int(grid.prod().item()) // (merge * merge) for grid in grids]
            inputs = {key: value.to(device) for key, value in batch.items() if isinstance(value, torch.Tensor)}
            captured.clear()
            with torch.inference_mode():
                output = model(**inputs, use_cache=False, return_dict=True)
                means, realized = mean_shifted_target_token_nll(output.logits, labels)
            if realized != expected_counts:
                raise RuntimeError("Qwen canonical token counts changed")
            if level == 1:
                if len(captured) != 1:
                    raise RuntimeError(f"expected one Qwen visual call, observed {len(captured)}")
                visual_rows = visual_projection(captured[0], token_counts, output_dim, projection_seed)
            for index, mean_nll in enumerate(means):
                score_rows[index][str(level)] = mean_nll
            del output, inputs, labels, batch
    finally:
        handle.remove()
        for image in images:
            image.close()
    if visual_rows is None:
        raise RuntimeError("Qwen visual summaries were not captured")
    return [
        {
            "record_id": record.pair_id,
            "input_sha256": record.degraded_sha256,
            "candidate_mean_nll": scores,
            "visual_summary": visual,
            "condition": qwen_condition_from_scores(scores, visual, temperature).tolist(),
        }
        for record, scores, visual in zip(batch_records, score_rows, visual_rows, strict=True)
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--contract-only", action="store_true")
    args = parser.parse_args()
    config, config_sha, rows, model_contract = read_contract(args.config.resolve())
    if args.contract_only:
        print(f"FOUNDIR_GATE3_CAUSAL_QWEN_CONTRACT_PASS records={len(rows)}")
        return 0
    if not torch.cuda.is_available():
        raise RuntimeError("causal Qwen export requires CUDA")
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    runtime = config["runtime"]
    if world != int(runtime["world_size"]) or torch.cuda.device_count() != world:
        raise RuntimeError("causal Qwen world size changed")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    seed_everything(int(runtime["seed"]) + rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    model, processor, evidence = _load_frozen_model(model_contract, device)
    model.config.use_cache = False
    selected = records(rows, int(runtime["seed"]))[rank::world]
    output: list[dict[str, object]] = []
    batch_size = int(config["qwen_features"]["batch_size_per_rank"])
    for start in range(0, len(selected), batch_size):
        output.extend(
            score_batch(
                model,
                processor,
                selected[start : start + batch_size],
                device=device,
                output_dim=int(config["qwen_features"]["visual_projection_dim"]),
                projection_seed=int(config["qwen_features"]["visual_projection_seed"]),
                temperature=float(config["qwen_features"]["posterior_temperature"]),
            )
        )
        print(f"GATE3_CAUSAL_QWEN rank={rank} records={len(output)}/{len(selected)}", flush=True)
    gathered: list[object] = [None for _ in range(world)]
    dist.all_gather_object(gathered, output)
    if rank == 0:
        combined = sorted((item for part in gathered for item in part), key=lambda item: item["record_id"])
        if len(combined) != len(rows) or len({item["record_id"] for item in combined}) != len(rows):
            raise RuntimeError("causal Qwen coverage changed")
        payload = {
            "schema_version": 1,
            "status": "frozen_qwen_causal_features_complete",
            "protected_foundir_pixels_used": False,
            "config_sha256": config_sha,
            "qwen_visual_sha256": config["parents"]["qwen_visual_checkpoint_sha256"],
            "model": evidence,
            "records": combined,
        }
        atomic_write_new_json(workspace_path(config["outputs"]["qwen_features"]), payload)
        print(f"FOUNDIR_GATE3_CAUSAL_QWEN_COMPLETE records={len(combined)}", flush=True)
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
