#!/usr/bin/env python3
"""Fail-fast CPU/GPU preflight for the FLUX.2 low-light baseline smoke."""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import importlib
import json
import os
import re
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

from PIL import Image
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party/flux2/src"))

from lowlight_restoration.config import load_train_config
from lowlight_restoration.authorization import validate_run_authorization
from lowlight_restoration.data import load_pair_manifest

EXPECTED_FLUX_REVISION = "50fe5162777813d869182b139e83b10743caef15"
EXPECTED_FILES = {
    "flow_weights": (
        7_751_105_712,
        "9c5fed22b76baea749d88fc2abe3ad53245e7b21a0d353a762665eea00043b92",
    ),
    "autoencoder_weights": (
        336_211_292,
        "868fe7b343cc8f3a19dbcfcafbc3d5f888802be3f89bd81b65b3621a066ce8f3",
    ),
}
EXPECTED_TEXT_ENCODER_FILES = {
    "model-00001-of-00002.safetensors": (
        4_412_140_424,
        "30b1bebf7b81f39f280b09c74f90feed00835a73145955a8d2fb69543b8371fb",
    ),
    "model-00002-of-00002.safetensors": (
        777_912_448,
        "79f82dbc594a0237e24982800665eb0ddbe8c7bbdf3ac49c2a0005682aca15dc",
    ),
}
EXPECTED_ALEXNET = (
    ROOT / "models/torch/hub/checkpoints/alexnet-owt-7be5be79.pth",
    244_408_911,
    "7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/lowlight_flux2_overfit16.json",
    )
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--skip-large-hashes", action="store_true")
    args = parser.parse_args()

    config = load_train_config(args.config)
    try:
        authorized_scope = validate_run_authorization(config)
    except RuntimeError as exc:
        raise SystemExit(f"RUN_NOT_AUTHORIZED {exc}") from exc
    expected_prefix = ROOT / ".venv/flux2-smoke"
    if Path(sys.prefix).resolve() != expected_prefix.resolve():
        raise SystemExit(f"WRONG_PYTHON expected={expected_prefix} actual={sys.prefix}")

    flux_repo = ROOT / "third_party/flux2"
    revision = subprocess.run(
        ["git", "-C", str(flux_repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != EXPECTED_FLUX_REVISION:
        raise SystemExit(f"BAD_FLUX2_REVISION {revision}")
    patch_check = subprocess.run(
        [
            "git",
            "-C",
            str(flux_repo),
            "apply",
            "--reverse",
            "--check",
            str(ROOT / "patches/flux2-offline-smoke.patch"),
        ],
        capture_output=True,
        text=True,
    )
    if patch_check.returncode:
        raise SystemExit("FLUX2_OFFLINE_PATCH_NOT_EXACTLY_APPLIED")

    for module in (
        "torch",
        "torchvision",
        "einops",
        "transformers",
        "safetensors",
        "PIL",
        "numpy",
        "skimage",
        "flux2",
    ):
        try:
            importlib.import_module(module)
        except Exception as exc:
            raise SystemExit(f"MISSING_OR_BROKEN_IMPORT {module}: {exc}") from exc

    if authorized_scope == "full_baseline":
        os.environ["TORCH_HOME"] = str(ROOT / "models/torch")
        try:
            importlib.import_module("torchmetrics")
        except Exception as exc:
            raise SystemExit(f"MISSING_OR_BROKEN_IMPORT torchmetrics: {exc}") from exc
        alexnet_path, expected_size, expected_digest = EXPECTED_ALEXNET
        if not alexnet_path.is_file() or alexnet_path.stat().st_size != expected_size:
            raise SystemExit(f"BAD_LPIPS_ALEXNET_FILE {alexnet_path}")
        if sha256(alexnet_path) != expected_digest:
            raise SystemExit(f"BAD_LPIPS_ALEXNET_SHA256 {alexnet_path}")
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True)
        del lpips
        print(f"OK_LPIPS alexnet_sha256={expected_digest}")

    model = config.section("model")
    flow_shapes: dict[str, tuple[int, ...]] = {}
    for key, (expected_size, expected_digest) in EXPECTED_FILES.items():
        path = config.workspace_path("model", key)
        if not path.is_file() or path.stat().st_size != expected_size:
            raise SystemExit(f"BAD_MODEL_FILE {path}")
        if not args.skip_large_hashes and sha256(path) != expected_digest:
            raise SystemExit(f"BAD_MODEL_SHA256 {path}")
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            print(f"OK_MODEL {path.name} tensors={len(handle.keys())}")
            if key == "flow_weights":
                flow_shapes = {
                    tensor_key: tuple(handle.get_slice(tensor_key).get_shape())
                    for tensor_key in handle.keys()
                }

    text_encoder = config.workspace_path("model", "text_encoder")
    index = text_encoder / "model.safetensors.index.json"
    if not index.is_file():
        raise SystemExit("MISSING_FLUX_TEXT_ENCODER_INDEX")
    parts = set(json.loads(index.read_text())["weight_map"].values())
    if parts != {"model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"}:
        raise SystemExit(f"BAD_FLUX_TEXT_ENCODER_PARTS {parts}")
    for part, (expected_size, expected_digest) in EXPECTED_TEXT_ENCODER_FILES.items():
        path = text_encoder / part
        if not path.is_file() or path.stat().st_size != expected_size:
            raise SystemExit(f"BAD_FLUX_TEXT_ENCODER_PART {path}")
        if not args.skip_large_hashes and sha256(path) != expected_digest:
            raise SystemExit(f"BAD_FLUX_TEXT_ENCODER_SHA256 {path}")

    records = load_pair_manifest(config.workspace_path("data", "manifest"))
    data = config.section("data")
    split_records = [record for record in records if record.split == data["split"]]
    expected_records = int(data["expected_records"])
    expected_sources = int(data["expected_unique_sources"])
    if len(split_records) != expected_records:
        raise SystemExit(f"BAD_RECORD_COUNT {len(split_records)} != {expected_records}")
    sources = {record.source_id for record in split_records}
    if len(sources) != expected_sources:
        raise SystemExit(f"BAD_SOURCE_COUNT {len(sources)} != {expected_sources}")
    levels = Counter(record.project_level for record in split_records)
    if set(levels) != {"L1", "L2", "L3", "L4", "L5"}:
        raise SystemExit(f"BAD_LEVEL_COVERAGE {levels}")
    crop_size = int(data["crop_size"])
    checked_paths: set[Path] = set()
    for record in split_records:
        for path, expected_digest in (
            (record.clean_path, record.clean_sha256),
            (record.degraded_path, record.degraded_sha256),
        ):
            if path in checked_paths:
                continue
            if not path.is_file() or sha256(path) != expected_digest:
                raise SystemExit(f"BAD_PAIR_FILE {path}")
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                if image.mode != "RGB" or min(image.size) < crop_size:
                    raise SystemExit(f"BAD_PAIR_IMAGE {path} mode={image.mode} size={image.size}")
            checked_paths.add(path)
    print(f"OK_DATA records={len(split_records)} sources={len(sources)} levels={dict(levels)}")

    lora = config.section("lora")
    patterns = list(lora["target_patterns"])
    names_and_shapes = []
    for key, shape in flow_shapes.items():
        if not key.endswith(".weight") or len(shape) != 2:
            continue
        name = key.removesuffix(".weight")
        if any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns):
            names_and_shapes.append((name, shape))
    names = [name for name, _ in names_and_shapes]
    if len(names) != int(lora["expected_module_count"]):
        raise SystemExit(f"BAD_LORA_TARGET_COUNT {len(names)}")
    rank = int(lora["rank"])
    trainable = sum(rank * (shape[0] + shape[1]) for _, shape in names_and_shapes)
    if trainable != 8_110_080:
        raise SystemExit(f"BAD_LORA_PARAMETER_COUNT {trainable}")
    print(f"OK_LORA modules={len(names)} trainable_parameters={trainable}")

    if shutil.disk_usage(ROOT).free < 20 * 1024**3:
        raise SystemExit("LESS_THAN_20_GIB_FREE_DISK")

    import torch

    if not args.cpu_only:
        if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
            raise SystemExit("CUDA_NOT_AVAILABLE")
        properties = torch.cuda.get_device_properties(0)
        required = int(config.section("runtime")["minimum_vram_gib"]) * 1024**3
        if properties.total_memory < required:
            raise SystemExit(f"INSUFFICIENT_VRAM {properties.total_memory}")
        if not re.search(r"(cu|nv)", torch.__version__, re.IGNORECASE):
            raise SystemExit(f"TORCH_IS_NOT_A_CUDA_BUILD {torch.__version__}")
        print(f"OK_CUDA name={properties.name} bytes={properties.total_memory}")
    print(
        "LOWLIGHT_BASELINE_PREFLIGHT_PASS "
        f"scope={authorized_scope} cpu_only={args.cpu_only} config_sha256={config.sha256}"
    )


if __name__ == "__main__":
    main()
