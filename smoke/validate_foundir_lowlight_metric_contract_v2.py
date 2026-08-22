#!/usr/bin/env python3
"""Validate both frozen FoundIR metric paths without protected pixels."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Mapping

import numpy as np

from smoke.validate_foundir_lowlight_metric_contract import (
    ROOT,
    literal_psnr,
    make_canaries,
    png_roundtrip,
    sha256,
    tensor_from_uint8,
    validate_identities,
    workspace_path,
    write_once_or_verify,
)


DEFAULT_CONFIG = ROOT / "configs/foundir_lowlight_metric_contract_v2.json"


def load_and_validate_config(path: Path) -> dict[str, object]:
    config = json.loads(path.read_text())
    required = {
        "schema_version",
        "supersedes",
        "supersession_reason",
        "environment",
        "official_reference",
        "weights",
        "metric_semantics",
        "canaries",
        "protected_data",
        "output",
    }
    if set(config) != required or config["schema_version"] != 2:
        raise RuntimeError("metric contract v2 schema changed")
    if config["protected_data"] != {
        "foundir_pixels_allowed": False,
        "canaries_are_generated_in_memory": True,
    }:
        raise RuntimeError("metric contract permits protected pixels")
    return config


def torch_literal_psnr(reference, prediction) -> float:
    import torch

    difference = reference.to(dtype=torch.float64) - prediction.to(dtype=torch.float64)
    mse = float(torch.mean(difference * difference).item())
    return 100.0 if mse == 0.0 else 10.0 * math.log10(1.0 / mse)


def expected_pyiqa_psnr(reference, prediction, eps: float) -> float:
    # This intentionally mirrors the frozen evaluator's float32 arithmetic.
    difference = reference - prediction
    mse = float((difference * difference).mean().item())
    return 10.0 * math.log10(1.0 / (mse + eps))


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

    settings = config["canaries"]
    assert isinstance(settings, dict)
    canaries = make_canaries(
        int(settings["seed"]), int(settings["height"]), int(settings["width"])
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
            evaluator_psnr = float(
                pyiqa_psnr(prediction_tensor, reference_tensor).item()
            )
            pyiqa_lpips_value = float(
                pyiqa_lpips(prediction_tensor, reference_tensor).item()
            )
            direct_lpips_value = float(
                direct_lpips(
                    prediction_tensor * 2.0 - 1.0,
                    reference_tensor * 2.0 - 1.0,
                ).item()
            )
        literal_numpy = literal_psnr(reference, prediction)
        literal_torch = torch_literal_psnr(reference_tensor, prediction_tensor)
        expected_evaluator = expected_pyiqa_psnr(
            reference_tensor,
            prediction_tensor,
            float(settings["pyiqa_psnr_eps"]),
        )
        evaluator_formula_difference = abs(evaluator_psnr - expected_evaluator)
        literal_difference = abs(literal_numpy - literal_torch)
        lpips_difference = abs(pyiqa_lpips_value - direct_lpips_value)
        identity_lpips_ok = name != "identity" or (
            abs(pyiqa_lpips_value)
            <= float(settings["identity_lpips_absolute_tolerance"])
            and abs(direct_lpips_value)
            <= float(settings["identity_lpips_absolute_tolerance"])
        )
        canary_pass = bool(
            roundtrip_exact
            and evaluator_formula_difference
            <= float(settings["psnr_formula_absolute_tolerance_db"])
            and literal_difference
            <= float(settings["literal_crosscheck_absolute_tolerance_db"])
            and lpips_difference <= float(settings["lpips_path_absolute_tolerance"])
            and identity_lpips_ok
            and all(
                math.isfinite(value)
                for value in (
                    evaluator_psnr,
                    literal_numpy,
                    literal_torch,
                    pyiqa_lpips_value,
                    direct_lpips_value,
                )
            )
        )
        gate_pass = gate_pass and canary_pass
        results[name] = {
            "reference_png_sha256": reference_png_sha256,
            "prediction_png_sha256": prediction_png_sha256,
            "save_reload_exact": roundtrip_exact,
            "pyiqa_psnr_db": evaluator_psnr,
            "expected_pyiqa_psnr_db": expected_evaluator,
            "pyiqa_formula_absolute_difference_db": evaluator_formula_difference,
            "literal_numpy_psnr_db": literal_numpy,
            "literal_torch_psnr_db": literal_torch,
            "literal_absolute_difference_db": literal_difference,
            "conservative_psnr_db": min(evaluator_psnr, literal_numpy),
            "pyiqa_lpips": pyiqa_lpips_value,
            "direct_lpips": direct_lpips_value,
            "lpips_absolute_difference": lpips_difference,
            "conservative_lpips": max(pyiqa_lpips_value, direct_lpips_value),
            "pass": canary_pass,
        }

    result: dict[str, object] = {
        "schema_version": 2,
        "status": "pass" if gate_pass else "failed",
        "automatic_gate_pass": gate_pass,
        "config_sha256": sha256(config_path),
        "evaluator_sha256": sha256(Path(__file__)),
        "identities": identities,
        "canaries": results,
        "protected_foundir_pixels_used": False,
        "v1_result_preserved": "reports/foundir_lowlight_metric_contract_result_v1.json",
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
        "FOUNDIR_LOWLIGHT_METRIC_CONTRACT_V2_"
        + ("PASS" if result["automatic_gate_pass"] else "FAILED")
    )
    return 0 if result["automatic_gate_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
