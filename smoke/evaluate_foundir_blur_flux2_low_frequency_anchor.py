#!/usr/bin/env python3
"""Select a low-frequency Blur anchor on 384 cached calibration records, then test 128."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_STATUS = "user_authorized_one_shot_cached_low_frequency_anchor_gate"


def workspace_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def image_tensor(path: Path, *, device: torch.device, crop_xy: tuple[int, int] | None = None) -> torch.Tensor:
    with Image.open(path) as handle:
        array = np.asarray(handle.convert("RGB"), dtype=np.uint8)
    if crop_xy is not None:
        x, y = crop_xy
        array = array[y:y + 512, x:x + 512]
    if array.shape != (512, 512, 3):
        raise RuntimeError(f"invalid image shape {array.shape}: {path}")
    return torch.from_numpy(array.copy()).permute(2, 0, 1).float().div(255.0).to(device)


def quantize(value: torch.Tensor) -> torch.Tensor:
    return value.mul(255.0).round().clamp(0, 255).div(255.0)


def lowpass(value: torch.Tensor, sigma: float) -> torch.Tensor:
    radius = int(round(3.0 * sigma))
    coordinate = torch.arange(-radius, radius + 1, dtype=value.dtype, device=value.device)
    kernel = torch.exp(-coordinate.square() / (2.0 * sigma * sigma))
    kernel = kernel / kernel.sum()
    channels = value.shape[1]
    horizontal = kernel.view(1, 1, 1, -1).repeat(channels, 1, 1, 1)
    vertical = kernel.view(1, 1, -1, 1).repeat(channels, 1, 1, 1)
    value = F.pad(value, (radius, radius, 0, 0), mode="reflect")
    value = F.conv2d(value, horizontal, groups=channels)
    value = F.pad(value, (0, 0, radius, radius), mode="reflect")
    return F.conv2d(value, vertical, groups=channels)


def psnr(reference: torch.Tensor, prediction: torch.Tensor) -> float:
    mse = float(torch.mean((reference - prediction) ** 2))
    return 100.0 if mse <= 1e-12 else -10.0 * math.log10(mse)


def key(sigma: float, strength: float) -> str:
    return f"sigma_{sigma:.1f}_lambda_{strength:.2f}"


def average(rows: list[dict[str, Any]], system: str, metric: str) -> float:
    return sum(float(row["systems"][system][metric]) for row in rows) / len(rows)


def write_once(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"refusing to overwrite differing artifact: {path}")
        return
    path.write_bytes(payload)


def cached_flux(row: dict[str, str], source: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    record_id = row["record_id"]
    if int(row["selection_rank"]) <= 383:
        directory = workspace_path(source["calibration_export_dir"])
        sidecar = load_json(directory / "per_record" / f"{record_id}.json")
        if (
            sidecar.get("checkpoint_sha256") != source["checkpoint_sha256"]
            or int(sidecar.get("selection_rank", -1)) != int(row["selection_rank"])
            or sidecar.get("development_accessed") is not False
            or sidecar.get("sealed_internal_accessed") is not False
            or sidecar.get("official_foundir_test_accessed") is not False
        ):
            raise RuntimeError(f"calibration cache identity changed: {record_id}")
    else:
        directory = workspace_path(source["cached_output_dir"])
        sidecar = load_json(directory / "per_record" / f"{record_id}.json")
        if (
            sidecar.get("config_sha256") != source["cached_config_sha256"]
            or sidecar.get("checkpoint_sha256") != source["checkpoint_sha256"]
            or sidecar.get("role") != "validation"
            or int(sidecar.get("selection_rank", -1)) != int(row["selection_rank"])
            or sidecar.get("development_accessed") is not False
            or sidecar.get("sealed_internal_accessed") is not False
            or sidecar.get("official_foundir_test_accessed") is not False
        ):
            raise RuntimeError(f"validation cache identity changed: {record_id}")
    path = directory / "base_flux_png" / f"{record_id}.png"
    with Image.open(path) as handle:
        array = np.asarray(handle.convert("RGB"), dtype=np.uint8)
    if hashlib.sha256(array.tobytes()).hexdigest() != sidecar["base_flux_png_uint8_sha256"]:
        raise RuntimeError(f"cached FLUX PNG identity changed: {record_id}")
    return path, sidecar


def evaluate(
    rows: Iterable[dict[str, str]],
    *,
    source: dict[str, Any],
    candidates: list[tuple[float, float]],
    gate: dict[str, Any],
    device: torch.device,
    metric: LearnedPerceptualImagePatchSimilarity,
    label: str,
) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    rows = list(rows)
    for index, row in enumerate(rows):
        base_path, old_sidecar = cached_flux(row, source)
        crop_xy = (int(row["crop_x"]), int(row["crop_y"]))
        lq = image_tensor(workspace_path(row["lq_path"]), device=device, crop_xy=crop_xy)
        gt = image_tensor(workspace_path(row["gt_path"]), device=device, crop_xy=crop_xy)
        base = image_tensor(base_path, device=device)
        scalar = quantize(lq + 0.75 * (base - lq))
        residual = (lq - base).unsqueeze(0)
        by_sigma = {sigma: lowpass(residual, sigma)[0] for sigma in sorted({item[0] for item in candidates})}
        names = ["base", "scalar"] + [key(sigma, strength) for sigma, strength in candidates]
        predictions = [base, scalar] + [quantize(base + strength * by_sigma[sigma]) for sigma, strength in candidates]
        batch = torch.stack(predictions)
        references = gt.unsqueeze(0).expand_as(batch)
        with torch.inference_mode():
            lpips_values = metric(batch, references).detach().float().cpu().tolist()
            metric.reset()
        systems = {
            name: {"psnr": psnr(gt, prediction), "lpips": float(lpips_value)}
            for name, prediction, lpips_value in zip(names, predictions, lpips_values, strict=True)
        }
        if int(row["selection_rank"]) >= 384:
            stored_references = {}
            for name, stored_key in (("scalar", "0.750000"), ("base", "1.000000")):
                stored = old_sidecar["alpha_metrics"][stored_key]
                stored_references[name] = {
                    "psnr": float(stored["psnr"]),
                    "lpips": float(stored["lpips"]),
                }
        else:
            stored_references = None
        output = {
            "record_id": row["record_id"],
            "source_group": row["source_group"],
            "selection_rank": int(row["selection_rank"]),
            "systems": systems,
        }
        if stored_references is not None:
            output["stored_references"] = stored_references
        outputs.append(output)
        if index == 0 or (index + 1) % 16 == 0:
            print(f"BLUR_LOW_FREQUENCY_ANCHOR {label} {index + 1}/{len(rows)}", flush=True)
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    config_path = workspace_path(args.config)
    config = load_json(config_path)
    if config.get("status") != EXPECTED_STATUS:
        raise RuntimeError("low-frequency gate authorization changed")
    scope, source, correction, gate = config["scope"], config["source"], config["correction"], config["gate"]
    required_false = (
        "model_weight_updates", "parameter_search", "development_accessed",
        "sealed_internal_accessed", "official_foundir_test_accessed",
    )
    if scope.get("category") != "01Blur" or any(scope.get(name) is not False for name in required_false):
        raise RuntimeError("low-frequency gate scope changed")
    sigmas = [float(value) for value in correction["gaussian_sigma_grid_pixels"]]
    strengths = [float(value) for value in correction["lambda_grid"]]
    if sigmas != [4.0, 8.0, 16.0] or strengths != [0.15, 0.25, 0.35]:
        raise RuntimeError("predeclared candidate grid changed")
    candidates = [(sigma, strength) for sigma in sigmas for strength in strengths]
    manifest_path = workspace_path(source["manifest"])
    if sha256(manifest_path) != source["manifest_sha256"]:
        raise RuntimeError("source manifest identity changed")
    with manifest_path.open(newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle) if row["role"] == "pilot_train"]
    rows.sort(key=lambda row: int(row["selection_rank"]))
    if len(rows) != 512 or len({row["source_group"] for row in rows}) != 512:
        raise RuntimeError("training-only 384/128 split changed")
    calibration, validation = rows[:384], rows[384:]
    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    metric = LearnedPerceptualImagePatchSimilarity(net_type="alex", reduction="none", normalize=True).to(device).eval()

    calibration_rows = evaluate(
        calibration, source=source, candidates=candidates, gate=gate,
        device=device, metric=metric, label="calibration",
    )
    scalar_calibration = {
        "psnr": average(calibration_rows, "scalar", "psnr"),
        "lpips": average(calibration_rows, "scalar", "lpips"),
    }
    calibration_metrics: dict[str, dict[str, float]] = {}
    eligible: list[tuple[float, float]] = []
    for sigma, strength in candidates:
        name = key(sigma, strength)
        values = {"psnr": average(calibration_rows, name, "psnr"), "lpips": average(calibration_rows, name, "lpips")}
        calibration_metrics[name] = values
        if values["lpips"] - scalar_calibration["lpips"] <= float(gate["maximum_lpips_regression_over_scalar"]):
            eligible.append((sigma, strength))
    if not eligible:
        raise RuntimeError("no calibration candidate satisfies the frozen LPIPS constraint")
    selected_sigma, selected_strength = max(
        eligible,
        key=lambda values: (calibration_metrics[key(*values)]["psnr"], -values[1], -values[0]),
    )
    selected_name = key(selected_sigma, selected_strength)
    print(f"BLUR_LOW_FREQUENCY_ANCHOR_FROZEN {selected_name}", flush=True)

    validation_rows = evaluate(
        validation, source=source, candidates=[(selected_sigma, selected_strength)], gate=gate,
        device=device, metric=metric, label="validation",
    )
    aggregates = {
        name: {metric_name: average(validation_rows, name, metric_name) for metric_name in ("psnr", "lpips")}
        for name in ("base", "scalar", selected_name)
    }
    deltas = [row["systems"][selected_name]["psnr"] - row["systems"]["scalar"]["psnr"] for row in validation_rows]
    comparison = {
        "psnr_gain_over_scalar_db": aggregates[selected_name]["psnr"] - aggregates["scalar"]["psnr"],
        "lpips_regression_over_scalar": aggregates[selected_name]["lpips"] - aggregates["scalar"]["lpips"],
        "psnr_win_fraction_over_scalar": sum(delta > 0 for delta in deltas) / len(deltas),
        "maximum_per_record_psnr_regression_over_scalar_db": max(-delta for delta in deltas),
    }
    replay = {
        f"{name}_{metric_name}_aggregate_delta": (
            aggregates[name][metric_name]
            - sum(row["stored_references"][name][metric_name] for row in validation_rows) / len(validation_rows)
        )
        for name in ("base", "scalar")
        for metric_name in ("psnr", "lpips")
    }
    checks = {
        "reference_aggregate_psnr_replay": max(
            abs(replay["base_psnr_aggregate_delta"]), abs(replay["scalar_psnr_aggregate_delta"])
        ) <= float(gate["reference_aggregate_replay_psnr_tolerance_db"]),
        "reference_aggregate_lpips_replay": max(
            abs(replay["base_lpips_aggregate_delta"]), abs(replay["scalar_lpips_aggregate_delta"])
        ) <= float(gate["reference_aggregate_replay_lpips_tolerance"]),
        "minimum_psnr_gain_over_scalar": comparison["psnr_gain_over_scalar_db"] >= float(gate["minimum_psnr_gain_over_scalar_db"]),
        "maximum_lpips_regression_over_scalar": comparison["lpips_regression_over_scalar"] <= float(gate["maximum_lpips_regression_over_scalar"]),
        "minimum_psnr_win_fraction_over_scalar": comparison["psnr_win_fraction_over_scalar"] >= float(gate["minimum_psnr_win_fraction_over_scalar"]),
        "maximum_per_record_psnr_regression_over_scalar": comparison["maximum_per_record_psnr_regression_over_scalar_db"] <= float(gate["maximum_per_record_psnr_regression_over_scalar_db"]),
    }
    passed = all(checks.values())
    payload = {
        "schema_version": 1,
        "status": "pass_stop_before_full_development" if passed else "failed_stop",
        "automatic_gate_pass": passed,
        "config": config_path.relative_to(ROOT).as_posix(),
        "config_sha256": sha256(config_path),
        "split": "training-only 384 calibration plus frozen 128 validation",
        "records": 512,
        "source_groups": 512,
        "selected": {"name": selected_name, "gaussian_sigma_pixels": selected_sigma, "lambda": selected_strength},
        "calibration_scalar": scalar_calibration,
        "calibration_metrics": calibration_metrics,
        "validation_aggregates": aggregates,
        "validation_comparison": comparison,
        "validation_reference_aggregate_replay": replay,
        "checks": checks,
        "model_inference": False,
        "model_weight_updates": False,
        "parameter_search": False,
        "development_accessed": False,
        "sealed_internal_accessed": False,
        "official_foundir_test_accessed": False,
        "full_development_automatic": False,
        "device": str(device),
        "wall_seconds": time.monotonic() - started,
        "calibration_per_record": calibration_rows,
        "validation_per_record": validation_rows,
    }
    output = workspace_path(config["runtime"]["output_dir"])
    write_once(output / "result.json", (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode())
    print("BLUR_LOW_FREQUENCY_ANCHOR_RESULT " + json.dumps({
        "pass": passed, "selected": payload["selected"], "validation_aggregates": aggregates,
        "validation_comparison": comparison, "checks": checks,
    }, sort_keys=True), flush=True)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
