#!/usr/bin/env python3
"""Prepare features, train one shared causal adapter, and infer all four arms."""

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
from concurrent.futures import ThreadPoolExecutor
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

from lowlight_restoration.causal_condition import (
    CausalToneAdapter,
    group_derangement,
    measured_image_features,
    normalization,
    oracle_pair_features,
)
from smoke.export_foundir_lowlight_gate3_causal_qwen import read_contract
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
            raise RuntimeError(f"immutable causal artifact differs: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            handle.write(content)


def prepare_one(row: Mapping[str, str]) -> dict[str, object]:
    source, target = load_rgb(row, "input_path"), load_rgb(row, "target_path")
    return {
        "record_id": row["record_id"],
        "input_sha256": row["input_sha256"],
        "target_sha256": row["target_sha256"],
        "measured": measured_image_features(source).tolist(),
        "oracle": oracle_pair_features(source, target).tolist(),
    }


def prepare_features(config: Mapping[str, object], config_sha: str, rows: Sequence[dict[str, str]]) -> None:
    output = workspace_path(config["outputs"]["image_features"])
    if output.is_file():
        payload = json.loads(output.read_text())
        if payload.get("config_sha256") != config_sha or len(payload.get("records", [])) != len(rows):
            raise RuntimeError("existing causal image features differ")
        print(f"FOUNDIR_GATE3_CAUSAL_IMAGE_FEATURES_REUSED records={len(rows)}")
        return
    with ThreadPoolExecutor(max_workers=int(config["runtime"]["cpu_feature_workers"])) as executor:
        features = list(executor.map(prepare_one, rows))
    payload = {
        "schema_version": 1,
        "status": "causal_measured_and_oracle_features_complete",
        "protected_foundir_pixels_used": False,
        "config_sha256": config_sha,
        "records": sorted(features, key=lambda row: row["record_id"]),
    }
    write_once(output, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode())
    print(f"FOUNDIR_GATE3_CAUSAL_IMAGE_FEATURES_COMPLETE records={len(features)}")


def load_feature_maps(config: Mapping[str, object], config_sha: str, rows: Sequence[dict[str, str]]):
    image_payload = json.loads(workspace_path(config["outputs"]["image_features"]).read_text())
    qwen_payload = json.loads(workspace_path(config["outputs"]["qwen_features"]).read_text())
    if image_payload.get("config_sha256") != config_sha or qwen_payload.get("config_sha256") != config_sha:
        raise RuntimeError("causal feature bundle belongs to another contract")
    image = {row["record_id"]: row for row in image_payload["records"]}
    qwen = {row["record_id"]: row for row in qwen_payload["records"]}
    expected = {row["record_id"] for row in rows}
    if set(image) != expected or set(qwen) != expected:
        raise RuntimeError("causal feature coverage changed")
    return image, qwen


class AdapterDataset(Dataset):
    def __init__(
        self,
        rows: Sequence[dict[str, str]],
        config: Mapping[str, object],
        indices: range,
        image_features: Mapping[str, Mapping[str, object]],
        qwen_features: Mapping[str, Mapping[str, object]],
        normalizers: Mapping[str, tuple[np.ndarray, np.ndarray]],
    ) -> None:
        self.config, self.indices = config, indices
        self.image_features, self.qwen_features, self.normalizers = image_features, qwen_features, normalizers
        grouped: dict[str, dict[str, list[dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
        for row in rows:
            grouped[row["family"]][row["group_id"]].append(row)
        self.families = tuple(config["data"]["training_families"])
        self.groups = {
            family: tuple(tuple(sorted(records, key=lambda item: item["record_id"])) for _, records in sorted(groups.items()))
            for family, groups in grouped.items()
        }

    def __len__(self) -> int:
        return len(self.indices)

    def standardized(self, name: str, value: Sequence[float]) -> Tensor:
        mean, scale = self.normalizers[name]
        return torch.from_numpy((np.asarray(value, dtype=np.float32) - mean) / scale)

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
        image_row, qwen_row = self.image_features[row["record_id"]], self.qwen_features[row["record_id"]]
        mode = presentation % 3
        qwen = self.standardized("qwen", qwen_row["condition"]) if mode == 2 else torch.zeros(len(qwen_row["condition"]))
        oracle = self.standardized("oracle", image_row["oracle"]) if mode == 1 else torch.zeros(len(image_row["oracle"]))
        return {
            "source": to_tensor(source_crop),
            "target": to_tensor(target_crop),
            "global_view": resize_view(source, int(self.config["data"]["global_view_size"])),
            "measured": self.standardized("measured", image_row["measured"]),
            "qwen": qwen,
            "oracle": oracle,
        }


def load_base(config: Mapping[str, object], device: torch.device):
    parent_config = json.loads(workspace_path(config["parents"]["gate3_config"]).read_text())
    model = build_model(parent_config)
    checkpoint = torch.load(workspace_path(config["parents"]["gate3_checkpoint"]), map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.requires_grad_(False).eval()
    return model.to(device=device, memory_format=torch.channels_last), parent_config


def build_normalizers(training_rows, image_features, qwen_features):
    result = {}
    for name, source, field in (
        ("measured", image_features, "measured"),
        ("oracle", image_features, "oracle"),
        ("qwen", qwen_features, "condition"),
    ):
        matrix = np.asarray([source[row["record_id"]][field] for row in training_rows], dtype=np.float32)
        result[name] = normalization(matrix)
    return result


def adapter_loss(prediction: Tensor, target: Tensor, config: Mapping[str, object]) -> Tensor:
    error = prediction.float() - target.float()
    loss = error.square().mean() + float(config["training"]["charbonnier_weight"]) * torch.sqrt(
        error.square() + float(config["training"]["charbonnier_epsilon"]) ** 2
    ).mean()
    kernel = int(config["training"]["low_frequency_kernel"])
    return loss + float(config["training"]["low_frequency_weight"]) * (
        F.avg_pool2d(prediction.float(), kernel) - F.avg_pool2d(target.float(), kernel)
    ).square().mean()


def train_adapter(config, config_sha, training_rows, image_features, qwen_features, rank, world, device):
    base, parent_config = load_base(config, device)
    normalizers = build_normalizers(training_rows, image_features, qwen_features)
    adapter = CausalToneAdapter(
        int(config["adapter"]["measured_dim"]), int(config["adapter"]["qwen_dim"]),
        int(config["adapter"]["oracle_dim"]), int(config["adapter"]["width"]),
    ).to(device)
    checkpoint_path = workspace_path(config["outputs"]["adapter_checkpoint"])
    if checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if state.get("config_sha256") != config_sha or state.get("base_checkpoint_sha256") != config["parents"]["gate3_checkpoint_sha256"]:
            raise RuntimeError("existing causal adapter checkpoint differs")
        adapter.load_state_dict(state["adapter"], strict=True)
        frozen_normalizers = {
            name: (np.asarray(value["mean"], dtype=np.float32), np.asarray(value["scale"], dtype=np.float32))
            for name, value in state["normalizers"].items()
        }
        print(f"GATE3_CAUSAL_ADAPTER_REUSED rank={rank}", flush=True)
        return base, adapter.eval(), parent_config, frozen_normalizers
    wrapped = DistributedDataParallel(adapter, device_ids=[device.index], broadcast_buffers=False)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=float(config["training"]["learning_rate"]), weight_decay=float(config["training"]["weight_decay"]))
    global_batch = int(config["training"]["batch_size_per_rank"]) * world
    indices = range(rank, int(config["training"]["optimizer_steps"]) * global_batch, world)
    dataset = AdapterDataset(training_rows, config, indices, image_features, qwen_features, normalizers)
    loader = DataLoader(dataset, batch_size=int(config["training"]["batch_size_per_rank"]), shuffle=False,
                        num_workers=int(config["runtime"]["data_workers_per_rank"]), pin_memory=True, drop_last=True)
    started = time.perf_counter()
    for step, batch in enumerate(loader, start=1):
        source = batch["source"].to(device, non_blocking=True, memory_format=torch.channels_last)
        target = batch["target"].to(device, non_blocking=True, memory_format=torch.channels_last)
        view = batch["global_view"].to(device, non_blocking=True, memory_format=torch.channels_last)
        measured, qwen, oracle = (batch[name].to(device, non_blocking=True) for name in ("measured", "qwen", "oracle"))
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            with torch.no_grad():
                base_tone = base.tone_parameters(view)
            delta = wrapped(measured, qwen, oracle)
            prediction = base(source, tone_parameters=base_tone + delta)
            loss = adapter_loss(prediction, target, config)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), float(config["training"]["gradient_clip_norm"]))
        optimizer.step()
        if step % 20 == 0 or step == 1:
            reduced = loss.detach().clone()
            dist.all_reduce(reduced, op=dist.ReduceOp.AVG)
            if rank == 0:
                print(f"GATE3_CAUSAL_ADAPTER_TRAIN step={step}/{config['training']['optimizer_steps']} loss={float(reduced):.7f}", flush=True)
        if step >= int(config["training"]["optimizer_steps"]):
            break
    dist.barrier()
    if rank == 0:
        payload = {
            "schema_version": 1, "status": "causal_adapter_training_complete", "config_sha256": config_sha,
            "base_checkpoint_sha256": config["parents"]["gate3_checkpoint_sha256"],
            "adapter": {key: value.detach().cpu() for key, value in adapter.state_dict().items()},
            "normalizers": {name: {"mean": mean.tolist(), "scale": scale.tolist()} for name, (mean, scale) in normalizers.items()},
            "optimizer_steps": int(config["training"]["optimizer_steps"]), "runtime_seconds": time.perf_counter() - started,
        }
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = checkpoint_path.with_suffix(f".tmp-{os.getpid()}")
        torch.save(payload, temporary)
        os.replace(temporary, checkpoint_path)
    dist.barrier()
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    adapter.load_state_dict(state["adapter"], strict=True)
    return base, adapter.eval(), parent_config, {
        name: (np.asarray(value["mean"], dtype=np.float32), np.asarray(value["scale"], dtype=np.float32))
        for name, value in state["normalizers"].items()
    }


@torch.inference_mode()
def restore_variants(base, adapter, image, conditions, parent_config, device):
    source = to_tensor(image).unsqueeze(0).to(device=device, memory_format=torch.channels_last)
    view = resize_view(image, int(parent_config["data"]["global_view_size"])).unsqueeze(0).to(device=device, memory_format=torch.channels_last)
    measured = conditions["measured"].to(device).expand(4, -1)
    qwen = torch.stack((conditions["zero_qwen"], conditions["zero_qwen"], conditions["qwen"], conditions["shuffled_qwen"])).to(device)
    oracle = torch.stack((conditions["zero_oracle"], conditions["oracle"], conditions["zero_oracle"], conditions["zero_oracle"])).to(device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        tone = base.tone_parameters(view).expand(4, -1) + adapter(measured, qwen, oracle)
    height, width = image.shape[:2]
    inference = parent_config["native_inference"]
    outputs = torch.empty((4, 3, height, width), device=device, dtype=torch.float32)
    if height * width <= int(inference["whole_image_max_pixels"]):
        padded, native_height, native_width = pad_multiple(source, int(inference["padding_multiple"]))
        batch = padded.expand(4, -1, -1, -1).contiguous(memory_format=torch.channels_last)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = base(batch, tone_parameters=tone).float()[..., :native_height, :native_width]
    else:
        core, halo = int(inference["tile_core"]), int(inference["tile_halo"])
        padded_height, padded_width = math.ceil(height / core) * core, math.ceil(width / core) * core
        bottom, right = padded_height - height, padded_width - width
        mode = "reflect" if height > bottom + halo and width > right + halo else "replicate"
        padded = F.pad(source, (halo, right + halo, halo, bottom + halo), mode=mode)
        full = torch.empty((4, 3, padded_height, padded_width), device=device, dtype=torch.float32)
        for top in range(0, padded_height, core):
            for left in range(0, padded_width, core):
                tile = padded[..., top : top + core + 2 * halo, left : left + core + 2 * halo]
                tile = tile.expand(4, -1, -1, -1).contiguous(memory_format=torch.channels_last)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    predicted = base(tile, tone_parameters=tone).float()
                full[..., top : top + core, left : left + core] = predicted[..., halo : halo + core, halo : halo + core]
        outputs = full[..., :height, :width]
    arrays = outputs.clamp(0, 1).mul(255).round().byte().permute(0, 2, 3, 1).cpu().numpy()
    return dict(zip(VARIANTS, arrays, strict=True))


def standardize(value, normalizer):
    mean, scale = normalizer
    return torch.from_numpy((np.asarray(value, dtype=np.float32) - mean) / scale)


def infer_outer(config, config_sha, outer_rows, image_features, qwen_features, base, adapter, parent_config, normalizers, rank, world, device):
    shuffle = group_derangement(outer_rows, int(config["evaluation"]["shuffle_seed"]))
    results = []
    for index in range(rank, len(outer_rows), world):
        row = outer_rows[index]
        record_id = row["record_id"]
        source = load_rgb(row, "input_path")
        image_row, qwen_row = image_features[record_id], qwen_features[record_id]
        conditions = {
            "measured": standardize(image_row["measured"], normalizers["measured"]),
            "oracle": standardize(image_row["oracle"], normalizers["oracle"]),
            "qwen": standardize(qwen_row["condition"], normalizers["qwen"]),
            "shuffled_qwen": standardize(qwen_features[shuffle[record_id]]["condition"], normalizers["qwen"]),
            "zero_qwen": torch.zeros(int(config["adapter"]["qwen_dim"])),
            "zero_oracle": torch.zeros(int(config["adapter"]["oracle_dim"])),
        }
        stems = {
            variant: workspace_path(config["outputs"]["prediction_root"]) / variant /
            f"{index:04d}_{row['family']}_{hashlib.sha256(record_id.encode()).hexdigest()[:16]}.png"
            for variant in VARIANTS
        }
        reusable = all(path.is_file() for path in stems.values())
        if reusable:
            predictions = {variant: np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8) for variant, path in stems.items()}
            elapsed = 0.0
        else:
            started = time.perf_counter()
            predictions = restore_variants(base, adapter, source, conditions, parent_config, device)
            elapsed = time.perf_counter() - started
        for variant, prediction in predictions.items():
            path = stems[variant]
            buffer = io.BytesIO()
            Image.fromarray(prediction, mode="RGB").save(buffer, format="PNG")
            content = buffer.getvalue()
            write_once(path, content)
            results.append({
                "variant": variant, "family": row["family"], "group_id": row["group_id"], "record_id": record_id,
                "target_path": row["target_path"], "output_path": str(path.relative_to(ROOT)),
                "output_sha256": hashlib.sha256(content).hexdigest(), "width": source.shape[1], "height": source.shape[0],
                "seconds_all_four": elapsed, "shuffled_qwen_source_record_id": shuffle[record_id] if "shuffled" in variant else "",
            })
        marker = "REUSED" if reusable else "GENERATED"
        print(f"GATE3_CAUSAL_INFER_{marker} rank={rank} records={index + 1}/{len(outer_rows)} seconds={elapsed:.3f}", flush=True)
    gathered: list[object] = [None for _ in range(world)]
    dist.all_gather_object(gathered, results)
    if rank == 0:
        combined = sorted((item for part in gathered for item in part), key=lambda row: (row["variant"], row["record_id"]))
        if len(combined) != len(outer_rows) * len(VARIANTS):
            raise RuntimeError("causal inference coverage changed")
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=list(combined[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(combined)
        write_once(workspace_path(config["outputs"]["prediction_manifest"]), buffer.getvalue().encode())
        print(f"FOUNDIR_GATE3_CAUSAL_INFERENCE_COMPLETE outputs={len(combined)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prepare-features", action="store_true")
    parser.add_argument("--contract-only", action="store_true")
    args = parser.parse_args()
    config, config_sha, rows, _ = read_contract(args.config.resolve())
    training_rows = [row for row in rows if row["role"] == config["data"]["training_role"]]
    outer_rows = [row for row in rows if row["role"] == config["data"]["outer_role"]]
    if args.contract_only:
        probe = np.zeros((16, 16, 3), dtype=np.uint8)
        if measured_image_features(probe).shape != (int(config["adapter"]["measured_dim"]),):
            raise RuntimeError("measured feature dimension changed")
        if oracle_pair_features(probe, probe).shape != (int(config["adapter"]["oracle_dim"]),):
            raise RuntimeError("oracle feature dimension changed")
        print(f"FOUNDIR_GATE3_CAUSAL_ADAPTER_CONTRACT_PASS training={len(training_rows)} outer={len(outer_rows)}")
        return 0
    if args.prepare_features:
        prepare_features(config, config_sha, rows)
        return 0
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    if world != int(config["runtime"]["world_size"]) or torch.cuda.device_count() != world:
        raise RuntimeError("causal adapter world size changed")
    local_rank = int(os.environ["LOCAL_RANK"]); torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    random.seed(int(config["training"]["seed"]) + rank); np.random.seed(int(config["training"]["seed"]) + rank)
    torch.manual_seed(int(config["training"]["seed"])); torch.cuda.manual_seed_all(int(config["training"]["seed"]))
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True; torch.backends.cudnn.benchmark = True
    image_features, qwen_features = load_feature_maps(config, config_sha, rows)
    base, adapter, parent_config, normalizers = train_adapter(
        config, config_sha, training_rows, image_features, qwen_features, rank, world, device
    )
    infer_outer(config, config_sha, outer_rows, image_features, qwen_features, base, adapter, parent_config, normalizers, rank, world, device)
    dist.barrier(); dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
