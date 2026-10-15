#!/usr/bin/env python3
"""CPU-only preflight for the independently prefixed HDD3 migration environment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import h5py
import torch
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ENV = ROOT.parent / ".envs/gpu-train-stable-hdd3"
OLD_ROOT = "/" + "/".join(("inspire", "hdd", "global_user", "liuxiaohong-25080", "zhenhaowang"))
CONFIG = ROOT / "configs/cure_two_track_reproduction_v1.json"
H5_PATH = ROOT / "data/third_party/cure_h5_smoke_v1/shared_cure.h5"
EMBEDDER = ROOT / "data/third_party/cure_released/OneRestore_embedder.tar"
RESTORER = ROOT / "data/third_party/cure_released/OneRestore_restorer.tar"
VGG = ROOT / "models/torch/hub/checkpoints/vgg16-397923af.pth"

EXPECTED_HASHES = {
    CONFIG: "01a428b7eced044cc26e5a6a939010d23eade49627d6e776d19938acef823119",
    H5_PATH: "f394e7fd180b062eaa268d1f16f85b33cdb02a97e65fa9caacd8f3824539ac7d",
    EMBEDDER: "0a852fd2f0dadfb71f0d1970f2cba9d0ee3f851ccfd123941c3511a2d08ae8d2",
    RESTORER: "746365e9d25bc0861b2bc7d7bd63bbb9c978661bee2634fddb0b68def2c81b01",
    VGG: "397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class RealH5Dataset(Dataset):
    def __init__(self, path: Path) -> None:
        self.path = path
        with h5py.File(path, "r") as handle:
            self.keys = sorted(handle.keys(), key=int)
            self.names = tuple(json.loads(handle.attrs["degradation_names"]))

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        with h5py.File(self.path, "r") as handle:
            row = handle[self.keys[index]]
            result: dict[str, torch.Tensor | str] = {
                field: torch.from_numpy(row[field][()])
                for field in ("zero", "gt", "lr", "type1", "type2", "half")
            }
        result["record_key"] = self.keys[index]
        return result


def state_dict(checkpoint: object) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    if not isinstance(checkpoint, dict):
        raise TypeError(f"unsupported checkpoint type: {type(checkpoint)!r}")
    return {str(key): value for key, value in checkpoint.items() if isinstance(value, torch.Tensor)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    executable = Path(sys.executable).resolve()
    if executable != (EXPECTED_ENV / "bin/python").resolve():
        raise RuntimeError(f"wrong Python: {executable}")
    if Path(sys.prefix).resolve() != EXPECTED_ENV.resolve() or Path(sys.base_prefix).resolve() != EXPECTED_ENV.resolve():
        raise RuntimeError(f"wrong prefix: prefix={sys.prefix} base_prefix={sys.base_prefix}")
    active_strings = list(sys.path) + [f"{key}={value}" for key, value in os.environ.items()]
    old_active = [value for value in active_strings if OLD_ROOT in value]
    if old_active:
        raise RuntimeError(f"old project path is active: {old_active}")

    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "third_party/cure_official"))
    import lowlight_restoration  # noqa: F401
    from cure.checkpoint import load_model
    from cure.data import sample_cure_batch
    from cure.embeddings import PromptEncoder
    from cure.losses import CURELoss
    from cure.models import OneRestore

    config = json.loads(CONFIG.read_text())
    if config.get("official_code", {}).get("path") != "third_party/cure_official":
        raise RuntimeError("unexpected CURE config")
    if config.get("dataset", {}).get("prohibit") != [
        "FoundIR official sealed test",
        "FoundIR 60-image triple endpoint",
    ]:
        raise RuntimeError("protected-test contract changed")

    hashes = {}
    for path, expected in EXPECTED_HASHES.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"hash mismatch: {path} {actual} != {expected}")
        hashes[str(path)] = actual

    dataset = RealH5Dataset(H5_PATH)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    if batch["zero"].shape != (1, 3, 256, 256):
        raise RuntimeError(f"unexpected real batch shape: {batch['zero'].shape}")
    if not all(torch.isfinite(batch[field]).all() for field in ("zero", "gt", "lr", "type1", "type2", "half")):
        raise RuntimeError("nonfinite real HDF5 batch")

    embedder_state = state_dict(torch.load(EMBEDDER, map_location="cpu", weights_only=False))
    restorer_state = state_dict(torch.load(RESTORER, map_location="cpu", weights_only=False))
    encoder = PromptEncoder(EMBEDDER)
    model = OneRestore()
    load_model(model, RESTORER)
    objective = CURELoss(pretrained_vgg=False)
    sampled = sample_cure_batch({field: batch[field] for field in ("zero", "gt", "lr", "type1", "type2", "half")}, dataset.names)
    if len(sampled["prompts"]) != 1:
        raise RuntimeError("CPU preflight must sample exactly one real record")

    probe = output / "write_probe.txt"
    probe.write_text("HDD3 report directory is writable\n")
    result = {
        "status": "HDD3_NEW_ENV_CPU_PREFLIGHT_OK",
        "workspace": str(ROOT),
        "environment": str(EXPECTED_ENV),
        "sys_executable": sys.executable,
        "sys_prefix": sys.prefix,
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available_expected_false_on_cpu_node": torch.cuda.is_available(),
        "config": str(CONFIG),
        "resolved_paths": {
            "dataset": str(H5_PATH),
            "embedder": str(EMBEDDER),
            "restorer": str(RESTORER),
            "vgg": str(VGG),
            "output": str(output),
        },
        "hashes": hashes,
        "dataset_records": len(dataset),
        "dataloader_batch_size": 1,
        "real_batch_shapes": {field: list(batch[field].shape) for field in ("zero", "gt", "lr", "type1", "type2", "half")},
        "sampled_prompt": sampled["prompts"][0],
        "embedder_checkpoint_tensor_count": len(embedder_state),
        "restorer_checkpoint_tensor_count": len(restorer_state),
        "constructed_modules": [
            f"{type(encoder).__module__}.{type(encoder).__name__}",
            f"{type(model).__module__}.{type(model).__name__}",
            f"{type(objective).__module__}.{type(objective).__name__}",
        ],
        "old_active_dependency_count": 0,
        "official_foundir_test_accessed": False,
        "write_probe": str(probe),
    }
    (output / "cpu_preflight.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    print("HDD3_NEW_ENV_CPU_PREFLIGHT_OK", flush=True)


if __name__ == "__main__":
    main()
