#!/usr/bin/env python3
"""Fail fast on offline/runtime faults before the five-type GPU pipeline."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party/flux2/src"))

ALEXNET_RELATIVE = Path("models/torch/hub/checkpoints/alexnet-owt-7be5be79.pth")
ALEXNET_SHA256 = "7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02"
ALEXNET_BYTES = 244_408_911
MINIMUM_FREE_BYTES = 100 * 1024**3
MINIMUM_FREE_INODES = 1_000_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def workspace_path(value: str | Path) -> Path:
    path = (ROOT / value).resolve()
    path.relative_to(ROOT.resolve())
    return path


def verify_environment() -> dict[str, object]:
    expected_torch_home = workspace_path("models/torch")
    actual_torch_home = Path(os.environ.get("TORCH_HOME", "")).resolve()
    if actual_torch_home != expected_torch_home:
        raise RuntimeError(
            f"TORCH_HOME must be the workspace cache: {expected_torch_home}; got {actual_torch_home}"
        )
    required_offline = {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
    }
    changed = {
        key: os.environ.get(key)
        for key, expected in required_offline.items()
        if os.environ.get(key) != expected
    }
    if changed:
        raise RuntimeError(f"offline environment is incomplete: {changed}")
    cache = workspace_path(ALEXNET_RELATIVE)
    if (
        not cache.is_file()
        or cache.stat().st_size != ALEXNET_BYTES
        or sha256(cache) != ALEXNET_SHA256
    ):
        raise RuntimeError(f"workspace AlexNet LPIPS cache changed: {cache}")
    return {
        "torch_home": str(expected_torch_home),
        "alexnet": str(cache.relative_to(ROOT)),
        "alexnet_sha256": ALEXNET_SHA256,
    }


def verify_storage() -> dict[str, int]:
    usage = shutil.disk_usage(ROOT)
    stats = os.statvfs(ROOT)
    free_inodes = int(stats.f_favail)
    if usage.free < MINIMUM_FREE_BYTES:
        raise RuntimeError(f"workspace has insufficient free bytes: {usage.free}")
    if free_inodes < MINIMUM_FREE_INODES:
        raise RuntimeError(f"workspace has insufficient free inodes: {free_inodes}")
    runtime_root = workspace_path("logs/.foundir_five_type_runtime_preflight")
    runtime_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=runtime_root) as directory:
        directory_path = Path(directory)
        source = directory_path / "source"
        linked = directory_path / "linked"
        source.write_bytes(b"foundir-five-type-runtime-preflight\n")
        os.link(source, linked)
        if source.stat().st_ino != linked.stat().st_ino or linked.read_bytes() != source.read_bytes():
            raise RuntimeError("workspace atomic-hardlink capability failed")
    return {"free_bytes": int(usage.free), "free_inodes": free_inodes}


def verify_versions() -> dict[str, str]:
    expected = {
        "torch": "2.7.0+cu128",
        "torchvision": "0.22.0+cu128",
        "torchmetrics": "1.9.0",
        "numpy": "1.26.4",
        "Pillow": "11.1.0",
    }
    actual = {name: importlib.metadata.version(name) for name in expected}
    if actual != expected:
        raise RuntimeError(f"FLUX metric/runtime versions changed: {actual}")
    return actual


def verify_metric_cache() -> None:
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    metric = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True)
    metric.eval().requires_grad_(False)
    del metric


def verify_imports(*, require_cuda: bool) -> dict[str, object]:
    modules = (
        "smoke.evaluate_foundir_five_type_vae_ceiling_ddp",
        "smoke.train_foundir_five_type_flux2_overfit_ddp",
        "smoke.export_foundir_five_type_qwen_conditions_sharded_ddp",
        "smoke.export_foundir_five_type_physical_v8_conditions_ddp",
        "smoke.build_foundir_flux_qwen3_prompt_cache_ddp",
        "smoke.freeze_foundir_five_type_evaluation_contract",
        "smoke.train_foundir_five_type_flux2_full_ddp",
        "smoke.evaluate_foundir_five_type_flux2_ddp",
        "smoke.select_foundir_five_type_flux2_checkpoint",
        "smoke.evaluate_foundir_five_type_flux2_tiled_ddp",
        "smoke.evaluate_foundir_five_type_released_baselines_ddp",
        "smoke.freeze_foundir_five_type_flux2_final_report",
        "smoke.freeze_foundir_five_type_flux2_main_v2_report",
    )
    for name in modules:
        importlib.import_module(name)

    import torch

    inventory: list[dict[str, object]] = []
    if require_cuda:
        if not torch.cuda.is_available() or torch.cuda.device_count() != 8:
            raise RuntimeError(
                f"runtime preflight requires exactly eight visible GPUs; got {torch.cuda.device_count()}"
            )
        families = []
        for index in range(8):
            properties = torch.cuda.get_device_properties(index)
            family = "H200" if "H200" in properties.name else "H100" if "H100" in properties.name else None
            if family is None or properties.total_memory < 75 * 1024**3:
                raise RuntimeError(f"unsupported GPU {index}: {properties.name} {properties.total_memory}")
            families.append(family)
            inventory.append(
                {
                    "index": index,
                    "name": properties.name,
                    "memory_bytes": int(properties.total_memory),
                }
            )
        if len(set(families)) != 1:
            raise RuntimeError(f"GPU family is not homogeneous: {families}")

        # The released specialist imports Triton/CUDA extensions at module load.
        # Do this on the actual GPU host before the long run, not days later.
        torch.cuda.set_device(0)
        extra = workspace_path(".tools/foundir-runtime")
        if str(extra) not in sys.path:
            sys.path.insert(0, str(extra))
        from smoke.run_foundir_lowlight_gate2_teachers import (
            SPECIALIST_DEPENDENCY_DIRECTORIES,
            import_specialist_architecture,
        )

        for dependency in reversed(SPECIALIST_DEPENDENCY_DIRECTORIES):
            if not dependency.is_dir():
                raise RuntimeError(f"missing FoundIR specialist dependency: {dependency}")
            if str(dependency) not in sys.path:
                sys.path.insert(0, str(dependency))
        versions = {
            name: importlib.import_module(name).__version__
            for name in ("transformers", "tokenizers", "huggingface_hub")
        }
        expected = {
            "transformers": "4.47.1",
            "tokenizers": "0.21.0",
            "huggingface_hub": "0.27.1",
        }
        if versions != expected:
            raise RuntimeError(f"FoundIR specialist dependency versions changed: {versions}")
        specialist = import_specialist_architecture(
            workspace_path("third_party/foundir/specialist_model")
        )
        if specialist.__name__ != "WaveMamba":
            raise RuntimeError("released specialist architecture identity changed")

    revision = subprocess.run(
        ["git", "-C", str(workspace_path("third_party/flux2")), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != "50fe5162777813d869182b139e83b10743caef15":
        raise RuntimeError(f"FLUX.2 source revision changed: {revision}")
    return {"gpu_inventory": inventory, "flux2_revision": revision}


def verify_artifact_deserialization(config_path: Path, *, full: bool) -> dict[str, object]:
    from safetensors import safe_open

    config = json.loads(config_path.read_text())
    checked: dict[str, object] = {}
    for key in ("flow_weights", "autoencoder_weights"):
        path = workspace_path(config["model"][key])
        expected = str(config["model"][f"{key}_sha256"])
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"frozen model changed before deserialization: {path}: {actual}")
        with safe_open(path, framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
        if not keys:
            raise RuntimeError(f"empty SafeTensors model: {path}")
        checked[key] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": actual,
            "tensor_keys": len(keys),
        }
    if full:
        import torch

        for key in ("generalist", "lowlight_specialist", "weather_specialist"):
            path = workspace_path(config["released_baseline"][key])
            expected = str(config["released_baseline"][f"{key}_sha256"])
            actual = sha256(path)
            if actual != expected:
                raise RuntimeError(
                    f"released checkpoint changed before deserialization: {path}: {actual}"
                )
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if not isinstance(payload, dict) or not payload:
                raise RuntimeError(f"released checkpoint cannot be deserialized: {path}")
            checked[key] = {
                "path": str(path.relative_to(ROOT)),
                "sha256": actual,
                "top_level_keys": sorted(map(str, payload)),
            }
            del payload
    return checked


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/foundir_five_type_flux2_dit_v1.json",
    )
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--full-deserialization", action="store_true")
    args = parser.parse_args()
    config = workspace_path(args.config)
    payload = {
        "schema_version": 1,
        "status": "pass",
        "environment": verify_environment(),
        "storage": verify_storage(),
        "versions": verify_versions(),
    }
    verify_metric_cache()
    payload.update(verify_imports(require_cuda=args.require_cuda))
    payload["artifacts"] = verify_artifact_deserialization(
        config, full=args.full_deserialization
    )
    payload["protected_foundir_test_accessed"] = False
    print("FOUNDIR_FIVE_TYPE_RUNTIME_PREFLIGHT_PASS " + json.dumps(payload, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
