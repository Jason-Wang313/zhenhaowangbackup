#!/usr/bin/env python3
"""Extract frozen Qwen visual features for the authorized official SOW600 manifest."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.distributed as dist
from PIL import Image
from safetensors import safe_open
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.qwen_five_class_gate import image_embedding_sequence  # noqa: E402
from lowlight_restoration.qwen_lora import load_qwen_adapter  # noqa: E402
from smoke.run_foundir_qwen_five_class_gate_ddp import sha256  # noqa: E402
from smoke.train_qwen_multidegradation_lora_ddp import load_model  # noqa: E402


AUTHORIZED = {"01Blur": 150, "05Noise": 100, "08Haze": 200, "10Rain": 100, "14Lowlight": 50}


def workspace(value: str | Path) -> Path:
    path = Path(value)
    result = (path if path.is_absolute() else ROOT / path).resolve()
    result.relative_to(ROOT.resolve())
    return result


def load_contract(path: Path) -> tuple[dict, str, list[dict[str, str]]]:
    config = json.loads(path.read_text())
    config_sha = sha256(path)
    authorization = workspace(config["authorization"])
    manifest = workspace(config["data"]["manifest"])
    qwen_config = workspace(config["qwen"]["training_config"])
    adapter = workspace(config["qwen"]["adapter"])
    if (
        sha256(authorization) != config["authorization_sha256"]
        or json.loads(authorization.read_text()).get("status") != "authorized"
    ):
        raise RuntimeError("official SOW600 authorization is not active")
    if sha256(manifest) != config["data"]["manifest_sha256"]:
        raise RuntimeError("official SOW600 manifest changed")
    if sha256(qwen_config) != config["qwen"]["training_config_sha256"]:
        raise RuntimeError("Qwen training config changed")
    if sha256(adapter) != config["qwen"]["adapter_sha256"]:
        raise RuntimeError("Qwen adapter changed")
    restoration = workspace(config["frozen_restoration"]["config"])
    if sha256(restoration) != config["frozen_restoration"]["config_sha256"]:
        raise RuntimeError("frozen restoration configuration changed")
    with manifest.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    counts = Counter(row["category"] for row in rows)
    if len(rows) != 600 or dict(counts) != AUTHORIZED:
        raise RuntimeError("official manifest escaped the authorized five-category scope")
    if len({row["record_id"] for row in rows}) != len(rows):
        raise RuntimeError("official record IDs are not unique")
    return config, config_sha, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    config, config_sha, rows = load_contract(args.config.resolve())
    if args.check_only:
        print(f"FOUNDIR_OFFICIAL_SOW600_QWEN_PREFLIGHT_PASS records={len(rows)} config_sha256={config_sha}")
        return
    if any(name not in os.environ for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK")):
        raise RuntimeError("launch Qwen extraction with torchrun")
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if world != int(config["qwen"]["world_size"]) or torch.cuda.device_count() != world:
        raise RuntimeError("official Qwen extraction requires exactly eight visible GPUs")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    try:
        qwen_config = json.loads(workspace(config["qwen"]["training_config"]).read_text())
        model, processor, _ = load_model(qwen_config, variant="merger_trainable", device=device)
        load_qwen_adapter(model, workspace(config["qwen"]["adapter"]))
        model.requires_grad_(False).eval()
        output = workspace(config["runtime"]["feature_dir"])
        output.mkdir(parents=True, exist_ok=True)
        assigned = rows[rank::world]
        batch_size = int(config["qwen"]["batch_size_per_gpu"])
        for start in range(0, len(assigned), batch_size):
            batch = assigned[start : start + batch_size]
            images = []
            for row in batch:
                path = workspace(row["lq_path"])
                if sha256(path) != row["lq_sha256"]:
                    raise RuntimeError(f"official input changed: {row['record_id']}")
                with Image.open(path) as handle:
                    images.append(handle.convert("RGB"))
            inputs = processor.image_processor(images=images, return_tensors="pt")
            with torch.inference_mode():
                vision = model.get_image_features(
                    inputs.pixel_values.to(device), inputs.image_grid_thw.to(device)
                )
                embeddings = image_embedding_sequence(vision)
            for row, embedding in zip(batch, embeddings, strict=True):
                feature = torch.cat(
                    (embedding.float().mean(0), embedding.float().std(0, unbiased=False)), dim=0
                ).cpu().contiguous()
                if feature.shape != (int(config["qwen"]["feature_dim"]),):
                    raise RuntimeError("Qwen feature shape changed")
                path = output / f"{row['record_id']}.safetensors"
                metadata = {
                    "schema_version": "1",
                    "config_sha256": config_sha,
                    "manifest_sha256": config["data"]["manifest_sha256"],
                    "adapter_sha256": config["qwen"]["adapter_sha256"],
                    "record_id": row["record_id"],
                    "lq_sha256": row["lq_sha256"],
                    "pooling": "qwen_visual_token_mean_and_std",
                    "feature_dim": str(config["qwen"]["feature_dim"]),
                }
                if path.exists():
                    with safe_open(path, framework="pt", device="cpu") as handle:
                        if handle.metadata() != metadata or set(handle.keys()) != {"feature"}:
                            raise RuntimeError(f"existing official Qwen feature changed: {row['record_id']}")
                        existing = handle.get_tensor("feature")
                    if existing.shape != feature.shape or not torch.equal(existing, feature):
                        raise RuntimeError(f"existing official Qwen feature differs: {row['record_id']}")
                    continue
                temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
                save_file({"feature": feature}, temporary, metadata=metadata)
                os.link(temporary, path)
                temporary.unlink()
            print(
                f"FOUNDIR_OFFICIAL_QWEN_PROGRESS rank={rank} "
                f"complete={min(start + len(batch), len(assigned))}/{len(assigned)}",
                flush=True,
            )
        dist.barrier()
        if rank == 0:
            expected = [output / f"{row['record_id']}.safetensors" for row in rows]
            if any(not path.is_file() for path in expected):
                raise RuntimeError("official Qwen feature export is incomplete")
            print("FOUNDIR_OFFICIAL_SOW600_QWEN_FEATURES_COMPLETE records=600", flush=True)
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
