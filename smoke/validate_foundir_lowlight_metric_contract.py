#!/usr/bin/env python3
"""Validate the finite FoundIR-Lowlight metric contract without test images."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import math
import os
import sys
from pathlib import Path
from typing import Mapping

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/foundir_lowlight_metric_contract_v1.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def workspace_path(raw: str, *, label: str) -> Path:
    path = (ROOT / raw).resolve()
    try:
        path.relative_to(ROOT)
    except ValueError as exc:
        raise RuntimeError(f"{label} escapes workspace: {path}") from exc
    return path


def literal_psnr(reference: np.ndarray, prediction: np.ndarray) -> float:
    if (
        reference.dtype != np.uint8
        or prediction.dtype != np.uint8
        or reference.shape != prediction.shape
        or reference.ndim != 3
        or reference.shape[2] != 3
    ):
        raise ValueError("PSNR canary inputs must be aligned uint8 RGB")
    difference = reference.astype(np.float64) - prediction.astype(np.float64)
    mse = float(np.mean((difference / 255.0) ** 2))
    return 100.0 if mse == 0.0 else 10.0 * math.log10(1.0 / mse)


def make_canaries(seed: int, height: int, width: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    if height < 64 or width < 64:
        raise ValueError("LPIPS canaries must be at least 64 by 64")
    generator = np.random.default_rng(seed)
    reference = generator.integers(0, 256, (height, width, 3), dtype=np.uint8)

    one_code = reference.copy()
    value = int(one_code[7, 11, 1])
    one_code[7, 11, 1] = value - 1 if value == 255 else value + 1

    local_patch = reference.copy()
    patch = local_patch[5:10, 7:12, 0].astype(np.int16) + 17
    local_patch[5:10, 7:12, 0] = np.clip(patch, 0, 255).astype(np.uint8)

    channel_swap = np.ascontiguousarray(reference[..., [2, 1, 0]])
    clipped = reference.copy()
    clipped[clipped < 32] = 0
    clipped[clipped > 223] = 255

    return {
        "identity": (reference, reference.copy()),
        "one_code": (reference, one_code),
        "local_patch": (reference, local_patch),
        "channel_swap": (reference, channel_swap),
        "clipping": (reference, clipped),
    }


def png_roundtrip(image: np.ndarray) -> tuple[np.ndarray, str]:
    stream = io.BytesIO()
    Image.fromarray(image, mode="RGB").save(stream, format="PNG")
    encoded = stream.getvalue()
    with Image.open(io.BytesIO(encoded)) as decoded:
        result = np.asarray(decoded.convert("RGB"), dtype=np.uint8)
    return result, hashlib.sha256(encoded).hexdigest()


def tensor_from_uint8(image: np.ndarray):
    import torch

    return (
        torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))
        .unsqueeze(0)
        .to(dtype=torch.float32)
        / 255.0
    )


def load_and_validate_config(path: Path) -> dict[str, object]:
    config = json.loads(path.read_text())
    required = {
        "schema_version",
        "environment",
        "official_reference",
        "weights",
        "metric_semantics",
        "canaries",
        "protected_data",
        "output",
    }
    if set(config) != required or config["schema_version"] != 1:
        raise RuntimeError("metric contract schema changed")
    if config["protected_data"] != {
        "foundir_pixels_allowed": False,
        "canaries_are_generated_in_memory": True,
    }:
        raise RuntimeError("metric contract permits protected pixels")
    return config


def validate_identities(config: Mapping[str, object]) -> dict[str, str]:
    environment = config["environment"]
    reference = config["official_reference"]
    weights = config["weights"]
    assert isinstance(environment, dict)
    assert isinstance(reference, dict)
    assert isinstance(weights, dict)

    expected_prefix = workspace_path(str(environment["python_prefix"]), label="python prefix")
    if Path(sys.prefix).resolve() != expected_prefix:
        raise RuntimeError(f"wrong metric environment: {sys.prefix} != {expected_prefix}")
    packages = environment["packages"]
    assert isinstance(packages, dict)
    for name, version in packages.items():
        actual = importlib.metadata.version(str(name))
        if actual != str(version):
            raise RuntimeError(f"package identity changed: {name} {actual} != {version}")

    identities: dict[str, str] = {}
    for key in ("evaluator", "environment"):
        path = workspace_path(str(reference[key]), label=key)
        actual = sha256(path)
        if actual != str(reference[f"{key}_sha256"]):
            raise RuntimeError(f"official {key} identity changed")
        identities[f"official_{key}_sha256"] = actual

    alexnet = weights["alexnet"]
    lpips_weight = weights["lpips_v0_1_alex"]
    assert isinstance(alexnet, dict)
    assert isinstance(lpips_weight, dict)
    alexnet_path = workspace_path(str(alexnet["path"]), label="AlexNet weight")
    if sha256(alexnet_path) != str(alexnet["sha256"]):
        raise RuntimeError("AlexNet weight identity changed")
    identities["alexnet_sha256"] = str(alexnet["sha256"])
    for key in ("pyiqa_path", "direct_path"):
        path = workspace_path(str(lpips_weight[key]), label=key)
        if sha256(path) != str(lpips_weight["sha256"]):
            raise RuntimeError(f"LPIPS weight identity changed: {key}")
    identities["lpips_v0_1_alex_sha256"] = str(lpips_weight["sha256"])
    return identities


def write_once_or_verify(path: Path, payload: Mapping[str, object]) -> None:
    content = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    if path.exists():
        if path.read_bytes() != content:
            raise RuntimeError(f"existing immutable result differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(content)


def run(config_path: Path) -> dict[str, object]:
    os.environ.setdefault("TORCH_HOME", str(ROOT / "models/torch"))
    os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache/foundir-metrics"))
    config = load_and_validate_config(config_path)
    identities = validate_identities(config)

    import lpips
    import pyiqa
    import torch

    pyiqa_psnr = pyiqa.create_metric("psnr", device="cpu")
    pyiqa_lpips = pyiqa.create_metric("lpips", device="cpu")
    direct_lpips = lpips.LPIPS(net="alex", version="0.1", verbose=False).eval()

    canary_config = config["canaries"]
    assert isinstance(canary_config, dict)
    canaries = make_canaries(
        int(canary_config["seed"]),
        int(canary_config["height"]),
        int(canary_config["width"]),
    )
    results: dict[str, object] = {}
    gate_pass = True
    for name, (raw_reference, raw_prediction) in canaries.items():
        reference, reference_png_sha256 = png_roundtrip(raw_reference)
        prediction, prediction_png_sha256 = png_roundtrip(raw_prediction)
        roundtrip_exact = bool(
            np.array_equal(reference, raw_reference)
            and np.array_equal(prediction, raw_prediction)
        )
        reference_tensor = tensor_from_uint8(reference)
        prediction_tensor = tensor_from_uint8(prediction)
        with torch.inference_mode():
            pyiqa_lpips_value = float(
                pyiqa_lpips(prediction_tensor, reference_tensor).item()
            )
            direct_lpips_value = float(
                direct_lpips(
                    prediction_tensor * 2.0 - 1.0,
                    reference_tensor * 2.0 - 1.0,
                ).item()
            )
            pyiqa_psnr_value = (
                None
                if name == "identity"
                else float(pyiqa_psnr(prediction_tensor, reference_tensor).item())
            )
        literal_psnr_value = literal_psnr(reference, prediction)
        psnr_difference = (
            0.0
            if pyiqa_psnr_value is None
            else abs(pyiqa_psnr_value - literal_psnr_value)
        )
        lpips_difference = abs(pyiqa_lpips_value - direct_lpips_value)
        identity_ok = name != "identity" or (
            abs(pyiqa_lpips_value) <= float(canary_config["identity_absolute_tolerance"])
            and abs(direct_lpips_value) <= float(canary_config["identity_absolute_tolerance"])
        )
        canary_pass = bool(
            roundtrip_exact
            and psnr_difference <= float(canary_config["psnr_absolute_tolerance_db"])
            and lpips_difference <= float(canary_config["lpips_path_absolute_tolerance"])
            and identity_ok
            and math.isfinite(pyiqa_lpips_value)
            and math.isfinite(direct_lpips_value)
        )
        gate_pass = gate_pass and canary_pass
        results[name] = {
            "reference_png_sha256": reference_png_sha256,
            "prediction_png_sha256": prediction_png_sha256,
            "save_reload_exact": roundtrip_exact,
            "literal_psnr_db": literal_psnr_value,
            "pyiqa_psnr_db": pyiqa_psnr_value,
            "psnr_absolute_difference_db": psnr_difference,
            "pyiqa_lpips": pyiqa_lpips_value,
            "direct_lpips": direct_lpips_value,
            "lpips_absolute_difference": lpips_difference,
            "worse_lpips": max(pyiqa_lpips_value, direct_lpips_value),
            "pass": canary_pass,
        }

    result: dict[str, object] = {
        "schema_version": 1,
        "status": "pass" if gate_pass else "failed",
        "automatic_gate_pass": gate_pass,
        "config_sha256": sha256(config_path),
        "evaluator_sha256": sha256(Path(__file__)),
        "identities": identities,
        "canaries": results,
        "protected_foundir_pixels_used": False,
    }
    output = workspace_path(str(config["output"]), label="output")
    write_once_or_verify(output, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    arguments = parser.parse_args()
    result = run(arguments.config.resolve())
    print(
        "FOUNDIR_LOWLIGHT_METRIC_CONTRACT_"
        + ("PASS" if result["automatic_gate_pass"] else "FAILED")
    )
    return 0 if result["automatic_gate_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
