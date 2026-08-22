#!/usr/bin/env python3
"""Train and infer Gate 3 structured-Qwen interface correction 1."""

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

from lowlight_restoration.causal_condition import CausalToneAdapter, group_derangement
from lowlight_restoration.structured_qwen_residual import (
    STRUCTURED_QWEN_DIM,
    StructuredQwenToneResidual,
    fit_measured_posterior_ridge,
    predict_measured_posterior,
    structured_qwen,
    support_reference,
    support_shrink,
)
from smoke.train_foundir_lowlight_gate3_compact import (
    build_model,
    load_rgb,
    pad_multiple,
    resize_view,
    sha256,
    to_tensor,
    workspace_path,
)


VARIANTS = ("measured_only", "oracle_plus_measured", "measured_plus_qwen", "measured_plus_shuffled_qwen")


def write_once(path: Path, content: bytes) -> None:
    if path.exists():
        if path.read_bytes() != content:
            raise RuntimeError(f"immutable correction-1 artifact differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(content)


def read_contract(path: Path) -> tuple[dict[str, object], str, list[dict[str, str]], dict[str, object]]:
    config = json.loads(path.read_text())
    if (
        config.get("schema_version") != 1
        or config.get("status") != "frozen_before_gate3_qwen_correction1"
        or config.get("protected_foundir_pixels_allowed") is not False
        or int(config["training"]["correction_index"]) != 1
    ):
        raise RuntimeError("Gate 3 correction-1 contract is not frozen")
    for key, identity in config["implementation_identities"].items():
        if sha256(workspace_path(identity["path"])) != identity["sha256"]:
            raise RuntimeError(f"correction-1 implementation changed: {key}")
    parent = config["parents"]
    for key in (
        "development_manifest", "gate3_config", "gate3_checkpoint", "causal_config",
        "causal_adapter_checkpoint", "causal_qwen_features", "causal_image_features", "causal_result",
        "causal_prediction_manifest",
    ):
        if sha256(workspace_path(parent[key])) != parent[f"{key}_sha256"]:
            raise RuntimeError(f"correction-1 dependency changed: {key}")
    with workspace_path(parent["development_manifest"]).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected = int(config["data"]["training_records"]) + int(config["data"]["outer_records"])
    if len(rows) != expected or any(row["protected_exact_overlap"].lower() != "false" for row in rows):
        raise RuntimeError("correction-1 development coverage or protected exclusion changed")
    causal_config = json.loads(workspace_path(parent["causal_config"]).read_text())
    return config, sha256(path), rows, causal_config


def load_feature_maps(config: Mapping[str, object], causal_config: Mapping[str, object], rows: Sequence[dict[str, str]]):
    parent = config["parents"]
    image_payload = json.loads(workspace_path(parent["causal_image_features"]).read_text())
    qwen_payload = json.loads(workspace_path(parent["causal_qwen_features"]).read_text())
    causal_sha = parent["causal_config_sha256"]
    if image_payload.get("config_sha256") != causal_sha or qwen_payload.get("config_sha256") != causal_sha:
        raise RuntimeError("parent causal features belong to another contract")
    image = {row["record_id"]: row for row in image_payload["records"]}
    qwen = {row["record_id"]: row for row in qwen_payload["records"]}
    expected = {row["record_id"] for row in rows}
    if set(image) != expected or set(qwen) != expected:
        raise RuntimeError("correction-1 feature coverage changed")
    if int(causal_config["adapter"]["measured_dim"]) != len(next(iter(image.values()))["measured"]):
        raise RuntimeError("parent measured feature dimension changed")
    return image, qwen


def build_metadata(config, rows, training_rows, image_features, qwen_features):
    measured = np.asarray([image_features[row["record_id"]]["measured"] for row in training_rows], dtype=np.float32)
    training_structured = np.asarray(
        [structured_qwen(qwen_features[row["record_id"]]["condition"]) for row in training_rows], dtype=np.float32
    )
    ridge = fit_measured_posterior_ridge(
        measured, training_structured[:, :5], float(config["structured_residual"]["measured_ridge_penalty"])
    )
    support = support_reference(
        training_structured,
        [row["group_id"] for row in training_rows],
        float(config["structured_residual"]["support_radius_quantile"]),
    )
    all_measured = np.asarray([image_features[row["record_id"]]["measured"] for row in rows], dtype=np.float32)
    all_structured = np.asarray(
        [structured_qwen(qwen_features[row["record_id"]]["condition"]) for row in rows], dtype=np.float32
    )
    baselines = predict_measured_posterior(all_measured, ridge)
    shrink = support_shrink(all_structured, support)
    metadata = {
        row["record_id"]: {
            "structured": all_structured[index],
            "measured_posterior": baselines[index],
            "support_shrink": float(shrink[index]),
        }
        for index, row in enumerate(rows)
    }
    training_ids = {row["record_id"] for row in training_rows}
    training_shrink = support_shrink(
        training_structured, support,
        query_group_ids=[row["group_id"] for row in training_rows], exclude_same_group=True,
    )
    for index, row in enumerate(training_rows):
        if row["record_id"] not in training_ids:
            raise RuntimeError("training metadata identity changed")
        metadata[row["record_id"]]["support_shrink"] = float(training_shrink[index])
    return metadata, ridge, support


class CorrectionDataset(Dataset):
    def __init__(
        self,
        rows: Sequence[dict[str, str]],
        config: Mapping[str, object],
        indices: range,
        image_features: Mapping[str, Mapping[str, object]],
        metadata: Mapping[str, Mapping[str, object]],
        measured_normalizer: tuple[np.ndarray, np.ndarray],
    ) -> None:
        self.config, self.indices = config, indices
        self.image_features, self.metadata = image_features, metadata
        self.measured_normalizer = measured_normalizer
        grouped: dict[str, dict[str, list[dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
        for row in rows:
            grouped[row["family"]][row["group_id"]].append(row)
        self.families = tuple(config["data"]["training_families"])
        self.groups = {
            family: tuple(tuple(sorted(records, key=lambda item: item["record_id"])) for _, records in sorted(groups.items()))
            for family, groups in grouped.items()
        }
        self.shuffle = group_derangement(rows, int(config["evaluation"]["training_shuffle_seed"]))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, local_index: int) -> dict[str, Tensor]:
        presentation = self.indices[local_index]
        rng = np.random.default_rng(int(self.config["training"]["seed"]) + presentation * 104729)
        family = self.families[presentation % len(self.families)]
        records = self.groups[family][int(rng.integers(0, len(self.groups[family])))]
        row = records[int(rng.integers(0, len(records)))]
        source, target = load_rgb(row, "input_path"), load_rgb(row, "target_path")
        crop = int(self.config["data"]["crop_size"])
        height, width = source.shape[:2]
        if height < crop or width < crop:
            scale = max(crop / height, crop / width)
            size = (max(crop, round(width * scale)), max(crop, round(height * scale)))
            source = np.asarray(Image.fromarray(source).resize(size, Image.Resampling.BICUBIC))
            target = np.asarray(Image.fromarray(target).resize(size, Image.Resampling.BICUBIC))
            height, width = source.shape[:2]
        top, left = int(rng.integers(0, height - crop + 1)), int(rng.integers(0, width - crop + 1))
        source_crop = np.ascontiguousarray(source[top : top + crop, left : left + crop])
        target_crop = np.ascontiguousarray(target[top : top + crop, left : left + crop])
        transform = int(rng.integers(0, 8))
        turns, reflect = transform % 4, transform // 4
        source_crop, target_crop = np.rot90(source_crop, turns), np.rot90(target_crop, turns)
        if reflect:
            source_crop, target_crop = source_crop[:, ::-1], target_crop[:, ::-1]
        record_id = row["record_id"]
        shuffled_id = self.shuffle[record_id]
        current, shuffled = self.metadata[record_id], self.metadata[shuffled_id]
        mean, scale = self.measured_normalizer
        measured = (np.asarray(self.image_features[record_id]["measured"], dtype=np.float32) - mean) / scale
        return {
            "source": to_tensor(source_crop),
            "target": to_tensor(target_crop),
            "global_view": resize_view(source, int(self.config["data"]["global_view_size"])),
            "measured": torch.from_numpy(measured),
            "correct_structured": torch.from_numpy(np.asarray(current["structured"], dtype=np.float32)),
            "shuffled_structured": torch.from_numpy(np.asarray(shuffled["structured"], dtype=np.float32)),
            "measured_posterior": torch.from_numpy(np.asarray(current["measured_posterior"], dtype=np.float32)),
            "correct_support": torch.tensor(float(current["support_shrink"]), dtype=torch.float32),
            "shuffled_support": torch.tensor(float(shuffled["support_shrink"]), dtype=torch.float32),
        }


def load_frozen_models(config, causal_config, device):
    parent_config = json.loads(workspace_path(config["parents"]["gate3_config"]).read_text())
    base = build_model(parent_config)
    base_state = torch.load(workspace_path(config["parents"]["gate3_checkpoint"]), map_location="cpu", weights_only=False)
    base.load_state_dict(base_state["model"], strict=True)
    causal = CausalToneAdapter(
        int(causal_config["adapter"]["measured_dim"]),
        int(causal_config["adapter"]["qwen_dim"]),
        int(causal_config["adapter"]["oracle_dim"]),
        int(causal_config["adapter"]["width"]),
    )
    causal_state = torch.load(
        workspace_path(config["parents"]["causal_adapter_checkpoint"]), map_location="cpu", weights_only=False
    )
    causal.load_state_dict(causal_state["adapter"], strict=True)
    normalizers = {
        name: (np.asarray(value["mean"], dtype=np.float32), np.asarray(value["scale"], dtype=np.float32))
        for name, value in causal_state["normalizers"].items()
    }
    base.requires_grad_(False).eval().to(device=device, memory_format=torch.channels_last)
    causal.requires_grad_(False).eval().to(device)
    return base, causal, parent_config, normalizers


def reconstruction_loss(prediction: Tensor, target: Tensor, config: Mapping[str, object]) -> Tensor:
    error = prediction.float() - target.float()
    result = error.square().mean() + float(config["training"]["charbonnier_weight"]) * torch.sqrt(
        error.square() + float(config["training"]["charbonnier_epsilon"]) ** 2
    ).mean()
    kernel = int(config["training"]["low_frequency_kernel"])
    return result + float(config["training"]["low_frequency_weight"]) * (
        F.avg_pool2d(prediction.float(), kernel) - F.avg_pool2d(target.float(), kernel)
    ).square().mean()


def train_correction(config, config_sha, causal_config, rows, training_rows, image_features, qwen_features, rank, world, device):
    base, causal, parent_config, normalizers = load_frozen_models(config, causal_config, device)
    metadata, ridge, support = build_metadata(config, rows, training_rows, image_features, qwen_features)
    model = StructuredQwenToneResidual(
        maximum_gain_parameter_delta=float(config["structured_residual"]["maximum_gain_parameter_delta"]),
        maximum_gamma_parameter_delta=float(config["structured_residual"]["maximum_gamma_parameter_delta"]),
        entropy_deficit_anchor=float(support["entropy_deficit_anchor"]),
        margin_anchor=float(support["margin_anchor"]),
        posterior_center=float(np.mean([metadata[row["record_id"]]["structured"][5] for row in training_rows])),
        maximum_knot_adjustment=float(config["structured_residual"]["maximum_knot_adjustment"]),
        initial_raw_slope=float(config["structured_residual"]["initial_raw_slope"]),
        initial_raw_residual_scale=float(config["structured_residual"]["initial_raw_residual_scale"]),
    ).to(device)
    checkpoint_path = workspace_path(config["outputs"]["correction_checkpoint"])
    if checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if state.get("config_sha256") != config_sha:
            raise RuntimeError("existing correction-1 checkpoint differs")
        model.load_state_dict(state["correction"], strict=True)
        print(f"GATE3_CORRECTION1_CHECKPOINT_REUSED rank={rank}", flush=True)
        return base, causal, model.eval(), parent_config, normalizers, metadata
    wrapped = DistributedDataParallel(model, device_ids=[device.index], broadcast_buffers=False)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    global_batch = int(config["training"]["batch_size_per_rank"]) * world
    indices = range(rank, int(config["training"]["optimizer_steps"]) * global_batch, world)
    dataset = CorrectionDataset(training_rows, config, indices, image_features, metadata, normalizers["measured"])
    loader = DataLoader(
        dataset, batch_size=int(config["training"]["batch_size_per_rank"]), shuffle=False,
        num_workers=int(config["runtime"]["data_workers_per_rank"]), pin_memory=True, drop_last=True,
    )
    zero_qwen = torch.zeros((int(config["training"]["batch_size_per_rank"]), int(causal_config["adapter"]["qwen_dim"])), device=device)
    zero_oracle = torch.zeros((int(config["training"]["batch_size_per_rank"]), int(causal_config["adapter"]["oracle_dim"])), device=device)
    started = time.perf_counter()
    for step, batch in enumerate(loader, start=1):
        source = batch["source"].to(device, non_blocking=True, memory_format=torch.channels_last)
        target = batch["target"].to(device, non_blocking=True, memory_format=torch.channels_last)
        view = batch["global_view"].to(device, non_blocking=True, memory_format=torch.channels_last)
        measured = batch["measured"].to(device, non_blocking=True)
        correct_structured = batch["correct_structured"].to(device, non_blocking=True)
        shuffled_structured = batch["shuffled_structured"].to(device, non_blocking=True)
        measured_posterior = batch["measured_posterior"].to(device, non_blocking=True)
        correct_support = batch["correct_support"].to(device, non_blocking=True)
        shuffled_support = batch["shuffled_support"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            with torch.no_grad():
                measured_tone = base.tone_parameters(view) + causal(measured, zero_qwen, zero_oracle)
                measured_prediction = base(source, tone_parameters=measured_tone)
            correct_delta = wrapped(correct_structured, measured_posterior, correct_support)
            shuffled_delta = wrapped(shuffled_structured, measured_posterior, shuffled_support)
            paired_source = torch.cat((source, source), dim=0)
            paired_tone = torch.cat((measured_tone + correct_delta, measured_tone + shuffled_delta), dim=0)
            paired_prediction = base(paired_source, tone_parameters=paired_tone)
            correct_prediction, shuffled_prediction = paired_prediction.chunk(2)
            correct_loss = reconstruction_loss(correct_prediction, target, config)
            shuffle_tether = (shuffled_prediction.float() - measured_prediction.float()).square().mean()
            correct_error = (correct_prediction.float() - target.float()).square().mean((1, 2, 3))
            shuffled_error = (shuffled_prediction.float() - target.float()).square().mean((1, 2, 3))
            measured_error = (measured_prediction.float() - target.float()).square().mean((1, 2, 3))
            margin = float(config["training"]["ranking_margin_mse"])
            ranking = F.relu(correct_error - shuffled_error + margin).mean()
            measured_ranking = F.relu(correct_error - measured_error + margin).mean()
            regularization = correct_delta.square().mean() + shuffled_delta.square().mean()
            loss = (
                correct_loss
                + float(config["training"]["shuffle_tether_weight"]) * shuffle_tether
                + float(config["training"]["correct_vs_shuffled_ranking_weight"]) * ranking
                + float(config["training"]["correct_vs_measured_ranking_weight"]) * measured_ranking
                + float(config["training"]["tone_delta_regularization_weight"]) * regularization
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["training"]["gradient_clip_norm"]))
        optimizer.step()
        if step % 20 == 0 or step == 1:
            values = torch.stack((loss.detach(), correct_loss.detach(), shuffle_tether.detach(), ranking.detach(), measured_ranking.detach()))
            dist.all_reduce(values, op=dist.ReduceOp.AVG)
            if rank == 0:
                print(
                    f"GATE3_CORRECTION1_TRAIN step={step}/{config['training']['optimizer_steps']} "
                    f"loss={float(values[0]):.7f} correct={float(values[1]):.7f} "
                    f"tether={float(values[2]):.7f} rank={float(values[3]):.7f}", flush=True,
                )
        if step >= int(config["training"]["optimizer_steps"]):
            break
    dist.barrier()
    if rank == 0:
        serial_support = {key: (value.tolist() if isinstance(value, np.ndarray) else value) for key, value in support.items()}
        serial_ridge = {key: (value.tolist() if isinstance(value, np.ndarray) else value) for key, value in ridge.items()}
        payload = {
            "schema_version": 1, "status": "structured_qwen_correction1_training_complete",
            "protected_foundir_pixels_used": False, "config_sha256": config_sha,
            "base_checkpoint_sha256": config["parents"]["gate3_checkpoint_sha256"],
            "measured_adapter_checkpoint_sha256": config["parents"]["causal_adapter_checkpoint_sha256"],
            "correction": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "ridge": serial_ridge, "support_reference": serial_support,
            "optimizer_steps": int(config["training"]["optimizer_steps"]),
            "runtime_seconds": time.perf_counter() - started,
        }
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = checkpoint_path.with_suffix(f".tmp-{os.getpid()}")
        torch.save(payload, temporary)
        os.replace(temporary, checkpoint_path)
    dist.barrier()
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["correction"], strict=True)
    return base, causal, model.eval(), parent_config, normalizers, metadata


@torch.inference_mode()
def restore_qwen_variants(base, causal, correction, image, conditions, parent_config, device):
    source = to_tensor(image).unsqueeze(0).to(device=device, memory_format=torch.channels_last)
    view = resize_view(image, int(parent_config["data"]["global_view_size"])).unsqueeze(0).to(device=device, memory_format=torch.channels_last)
    measured = conditions["measured"].to(device)
    zero_qwen = torch.zeros((1, conditions["qwen_dim"]), device=device)
    zero_oracle = torch.zeros((1, conditions["oracle_dim"]), device=device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        measured_tone = base.tone_parameters(view) + causal(measured, zero_qwen, zero_oracle)
        qwen_delta = correction(
            conditions["qwen"].to(device), conditions["measured_posterior"].to(device), conditions["qwen_support"].to(device)
        )
        shuffled_delta = correction(
            conditions["shuffled_qwen"].to(device), conditions["measured_posterior"].to(device),
            conditions["shuffled_support"].to(device),
        )
        tone = torch.cat((measured_tone + qwen_delta, measured_tone + shuffled_delta), dim=0)
    height, width = image.shape[:2]
    inference = parent_config["native_inference"]
    if height * width <= int(inference["whole_image_max_pixels"]):
        padded, native_height, native_width = pad_multiple(source, int(inference["padding_multiple"]))
        batch = padded.expand(2, -1, -1, -1).contiguous(memory_format=torch.channels_last)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = base(batch, tone_parameters=tone).float()[..., :native_height, :native_width]
    else:
        core, halo = int(inference["tile_core"]), int(inference["tile_halo"])
        padded_height, padded_width = math.ceil(height / core) * core, math.ceil(width / core) * core
        bottom, right = padded_height - height, padded_width - width
        mode = "reflect" if height > bottom + halo and width > right + halo else "replicate"
        padded = F.pad(source, (halo, right + halo, halo, bottom + halo), mode=mode)
        full = torch.empty((2, 3, padded_height, padded_width), device=device, dtype=torch.float32)
        for top in range(0, padded_height, core):
            for left in range(0, padded_width, core):
                tile = padded[..., top : top + core + 2 * halo, left : left + core + 2 * halo]
                tile = tile.expand(2, -1, -1, -1).contiguous(memory_format=torch.channels_last)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    predicted = base(tile, tone_parameters=tone).float()
                full[..., top : top + core, left : left + core] = predicted[..., halo : halo + core, halo : halo + core]
        outputs = full[..., :height, :width]
    arrays = outputs.clamp(0, 1).mul(255).round().byte().permute(0, 2, 3, 1).cpu().numpy()
    return dict(zip(VARIANTS[2:], arrays, strict=True))


def standardize(value, normalizer):
    mean, scale = normalizer
    array = (np.asarray(value, dtype=np.float32) - mean) / scale
    return torch.from_numpy(array).unsqueeze(0)


def infer_outer(config, config_sha, causal_config, outer_rows, image_features, metadata, base, causal, correction, parent_config, normalizers, rank, world, device):
    shuffle = group_derangement(outer_rows, int(config["evaluation"]["outer_shuffle_seed"]))
    with workspace_path(config["parents"]["causal_prediction_manifest"]).open(newline="") as handle:
        parent_predictions = {
            (row["variant"], row["record_id"]): row for row in csv.DictReader(handle)
            if row["variant"] in VARIANTS[:2]
        }
    if len(parent_predictions) != len(outer_rows) * 2:
        raise RuntimeError("frozen measured/oracle parent coverage changed")
    results = []
    for index in range(rank, len(outer_rows), world):
        row = outer_rows[index]
        record_id = row["record_id"]
        shuffled_id = shuffle[record_id]
        source = load_rgb(row, "input_path")
        current, shuffled = metadata[record_id], metadata[shuffled_id]
        conditions = {
            "measured": standardize(image_features[record_id]["measured"], normalizers["measured"]),
            "qwen": torch.from_numpy(np.asarray(current["structured"], dtype=np.float32)).unsqueeze(0),
            "shuffled_qwen": torch.from_numpy(np.asarray(shuffled["structured"], dtype=np.float32)).unsqueeze(0),
            "measured_posterior": torch.from_numpy(np.asarray(current["measured_posterior"], dtype=np.float32)).unsqueeze(0),
            "qwen_support": torch.tensor([float(current["support_shrink"])], dtype=torch.float32),
            "shuffled_support": torch.tensor([float(shuffled["support_shrink"])], dtype=torch.float32),
            "qwen_dim": int(causal_config["adapter"]["qwen_dim"]),
            "oracle_dim": int(causal_config["adapter"]["oracle_dim"]),
        }
        stems = {
            variant: workspace_path(config["outputs"]["prediction_root"]) / variant /
            f"{index:04d}_{row['family']}_{hashlib.sha256(record_id.encode()).hexdigest()[:16]}.png"
            for variant in VARIANTS[2:]
        }
        reusable = all(path.is_file() for path in stems.values())
        if reusable:
            predictions = {variant: np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8) for variant, path in stems.items()}
            elapsed = 0.0
        else:
            started = time.perf_counter()
            predictions = restore_qwen_variants(base, causal, correction, source, conditions, parent_config, device)
            elapsed = time.perf_counter() - started
        for variant, prediction in predictions.items():
            buffer = io.BytesIO()
            Image.fromarray(prediction, mode="RGB").save(buffer, format="PNG")
            content = buffer.getvalue()
            write_once(stems[variant], content)
            results.append({
                "variant": variant, "family": row["family"], "group_id": row["group_id"], "record_id": record_id,
                "target_path": row["target_path"], "output_path": str(stems[variant].relative_to(ROOT)),
                "output_sha256": hashlib.sha256(content).hexdigest(), "width": source.shape[1], "height": source.shape[0],
                "seconds_all_four": elapsed, "shuffled_qwen_source_record_id": shuffled_id if "shuffled" in variant else "",
            })
        for variant in VARIANTS[:2]:
            parent_row = parent_predictions[(variant, record_id)]
            parent_path = workspace_path(parent_row["output_path"])
            if sha256(parent_path) != parent_row["output_sha256"]:
                raise RuntimeError(f"frozen {variant} output changed: {record_id}")
            results.append({
                "variant": variant, "family": row["family"], "group_id": row["group_id"], "record_id": record_id,
                "target_path": row["target_path"], "output_path": parent_row["output_path"],
                "output_sha256": parent_row["output_sha256"], "width": source.shape[1], "height": source.shape[0],
                "seconds_all_four": 0.0, "shuffled_qwen_source_record_id": "",
            })
        marker = "REUSED" if reusable else "GENERATED"
        print(f"GATE3_CORRECTION1_INFER_{marker} rank={rank} records={index + 1}/{len(outer_rows)} seconds={elapsed:.3f}", flush=True)
    gathered: list[object] = [None for _ in range(world)]
    dist.all_gather_object(gathered, results)
    if rank == 0:
        combined = sorted((item for part in gathered for item in part), key=lambda row: (row["variant"], row["record_id"]))
        if len(combined) != len(outer_rows) * len(VARIANTS):
            raise RuntimeError("correction-1 inference coverage changed")
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=list(combined[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(combined)
        write_once(workspace_path(config["outputs"]["prediction_manifest"]), buffer.getvalue().encode())
        print(f"FOUNDIR_GATE3_CORRECTION1_INFERENCE_COMPLETE outputs={len(combined)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--contract-only", action="store_true")
    args = parser.parse_args()
    config, config_sha, rows, causal_config = read_contract(args.config.resolve())
    training_rows = [row for row in rows if row["role"] == config["data"]["training_role"]]
    outer_rows = [row for row in rows if row["role"] == config["data"]["outer_role"]]
    image_features, qwen_features = load_feature_maps(config, causal_config, rows)
    if args.contract_only:
        metadata, _, support = build_metadata(config, rows, training_rows, image_features, qwen_features)
        if len(metadata) != len(rows) or not 0.0 < float(support["radius"]):
            raise RuntimeError("correction-1 metadata preflight failed")
        print(f"FOUNDIR_GATE3_CORRECTION1_CONTRACT_PASS training={len(training_rows)} outer={len(outer_rows)}")
        return 0
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    if world != int(config["runtime"]["world_size"]) or torch.cuda.device_count() != world:
        raise RuntimeError("correction-1 world size changed")
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
    base, causal, correction, parent_config, normalizers, metadata = train_correction(
        config, config_sha, causal_config, rows, training_rows, image_features, qwen_features, rank, world, device
    )
    infer_outer(
        config, config_sha, causal_config, outer_rows, image_features, metadata, base, causal, correction,
        parent_config, normalizers, rank, world, device,
    )
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
