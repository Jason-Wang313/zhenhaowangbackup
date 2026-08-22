#!/usr/bin/env python3
"""Measure FLUX.2 VAE crop and full-resolution tiled fidelity on two GPUs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Mapping

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party/flux2/src"))

from lowlight_restoration.paired_development import (  # noqa: E402
    clustered_metric_summary,
    peak_signal_noise_ratio,
    rgb_structural_similarity,
)
from lowlight_restoration.representation_ceiling import (  # noqa: E402
    centered_crop,
    high_frequency_diagnostic,
    reconstruct_tiled,
    seam_error_ratio,
    tile_origins,
)


DEFAULT_CONFIG = ROOT / "configs/lowlight_vae_ceiling_v2.json"
REQUIRED_MANIFEST_FIELDS = {
    "development_id",
    "dataset",
    "scene_id",
    "target_path",
    "target_sha256",
    "width",
    "height",
    "exact_foundir_overlap",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def workspace_path(raw_path: str, *, label: str) -> Path:
    path = (ROOT / raw_path).resolve()
    try:
        path.relative_to(ROOT)
    except ValueError as exc:
        raise RuntimeError(f"{label} escapes workspace: {path}") from exc
    return path


def write_once_or_verify(path: Path, content: bytes) -> None:
    if path.exists():
        if path.read_bytes() != content:
            raise RuntimeError(f"existing immutable output differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(content)


def load_config(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text())
    required = {
        "schema_version",
        "input",
        "model",
        "tiling",
        "metrics",
        "runtime",
        "reporting",
        "decision",
        "output",
    }
    if set(payload) != required or payload["schema_version"] != 1:
        raise RuntimeError("VAE-ceiling config schema changed")
    tiling = payload["tiling"]
    runtime = payload["runtime"]
    if not isinstance(tiling, dict) or not isinstance(runtime, dict):
        raise RuntimeError("VAE-ceiling runtime/tiling config must be objects")
    if (
        int(tiling["tile_size"]) <= int(tiling["overlap"])
        or int(tiling["tile_size"]) % int(tiling["alignment_multiple"]) != 0
        or int(runtime["world_size"]) != 2
    ):
        raise RuntimeError("invalid VAE-ceiling distributed tiling contract")
    return payload


def load_manifest(config: Mapping[str, object]) -> list[dict[str, str]]:
    input_config = config["input"]
    assert isinstance(input_config, dict)
    path = workspace_path(str(input_config["manifest"]), label="input manifest")
    if sha256(path) != str(input_config["manifest_sha256"]):
        raise RuntimeError("paired-development manifest identity changed")
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not REQUIRED_MANIFEST_FIELDS.issubset(
            reader.fieldnames
        ):
            raise RuntimeError("paired-development manifest schema changed")
        rows = list(reader)
    counts = (
        len(rows),
        len({row["scene_id"] for row in rows}),
        len({row["target_sha256"] for row in rows}),
    )
    expected = (
        int(input_config["expected_records"]),
        int(input_config["expected_scenes"]),
        int(input_config["expected_unique_targets"]),
    )
    if counts != expected:
        raise RuntimeError(f"paired-development counts changed: {counts} != {expected}")
    if any(row["exact_foundir_overlap"] != "false" for row in rows):
        raise RuntimeError("paired-development row has FoundIR overlap")
    return rows


def contract_check(
    config_path: Path, *, skip_autoencoder_hash: bool
) -> tuple[dict[str, object], list[dict[str, str]]]:
    expected_environment = ROOT / ".venv/flux2-smoke"
    if Path(sys.prefix).resolve() != expected_environment.resolve():
        raise RuntimeError(
            f"wrong Python environment: {sys.prefix} != {expected_environment}"
        )
    config = load_config(config_path)
    rows = load_manifest(config)
    model = config["model"]
    assert isinstance(model, dict)
    weights = workspace_path(
        str(model["autoencoder_weights"]), label="autoencoder weights"
    )
    if not weights.is_file() or weights.stat().st_size != int(
        model["autoencoder_bytes"]
    ):
        raise RuntimeError("FLUX.2 autoencoder size changed")
    if not skip_autoencoder_hash and sha256(weights) != str(
        model["autoencoder_sha256"]
    ):
        raise RuntimeError("FLUX.2 autoencoder hash changed")
    flux_revision = subprocess.run(
        ["git", "-C", str(ROOT / "third_party/flux2"), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if flux_revision != str(model["flux2_revision"]):
        raise RuntimeError(f"FLUX.2 revision changed: {flux_revision}")
    for row in rows:
        path = workspace_path(row["target_path"], label="target image")
        if not path.is_file() or path.stat().st_size <= 0:
            raise RuntimeError(f"missing paired target: {path}")
        if "foundir_official" in str(path).lower():
            raise RuntimeError("protected FoundIR path entered VAE diagnostic")
    return config, rows


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def quantize(image: np.ndarray) -> np.ndarray:
    return (
        np.rint(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8).astype(np.float32)
        / 255.0
    )


class AutoencoderTileProcessor:
    def __init__(self, autoencoder, device, alignment_multiple: int):
        self.autoencoder = autoencoder
        self.device = device
        self.alignment_multiple = alignment_multiple

    def __call__(self, tile: np.ndarray) -> np.ndarray:
        import torch
        import torch.nn.functional as functional

        height, width = tile.shape[:2]
        pad_height = (-height) % self.alignment_multiple
        pad_width = (-width) % self.alignment_multiple
        tensor = torch.from_numpy(np.ascontiguousarray(tile.transpose(2, 0, 1)))
        tensor = tensor.unsqueeze(0).to(device=self.device, dtype=torch.float32)
        tensor = tensor * 2.0 - 1.0
        if pad_height or pad_width:
            tensor = functional.pad(
                tensor, (0, pad_width, 0, pad_height), mode="reflect"
            )
        with torch.inference_mode():
            latent = self.autoencoder.encode(tensor)
            restored = self.autoencoder.decode(latent).float().clamp(-1.0, 1.0)
        restored = (restored[0, :, :height, :width] + 1.0) * 0.5
        return restored.permute(1, 2, 0).cpu().numpy()


def resize_for_lpips(image: np.ndarray, maximum_dimension: int) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(1.0, maximum_dimension / float(max(height, width)))
    if scale == 1.0:
        return image
    return cv2.resize(
        image,
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    )


def lpips_value(metric, reference: np.ndarray, prediction: np.ndarray, device) -> float:
    import torch

    # OpenCV resizing and float32 VAE decode can overshoot the nominal image
    # range by a few ULPs.  TorchMetrics validates the range strictly before
    # computing LPIPS, so enforce the documented [0, 1] boundary here.
    reference = np.clip(reference, 0.0, 1.0)
    prediction = np.clip(prediction, 0.0, 1.0)
    reference_tensor = (
        torch.from_numpy(np.ascontiguousarray(reference.transpose(2, 0, 1)))
        .unsqueeze(0)
        .to(device=device, dtype=torch.float32)
    )
    prediction_tensor = (
        torch.from_numpy(np.ascontiguousarray(prediction.transpose(2, 0, 1)))
        .unsqueeze(0)
        .to(device=device, dtype=torch.float32)
    )
    with torch.inference_mode():
        value = metric(prediction_tensor, reference_tensor)
    return float(value.item())


def validate_sidecar(
    path: Path,
    *,
    target_sha256: str,
    identities: Mapping[str, str],
) -> dict[str, object]:
    payload = json.loads(path.read_text())
    expected = {
        "config_sha256": identities["config_sha256"],
        "evaluator_sha256": identities["evaluator_sha256"],
        "paired_development_module_sha256": identities[
            "paired_development_module_sha256"
        ],
        "representation_module_sha256": identities["representation_module_sha256"],
        "autoencoder_sha256": identities["autoencoder_sha256"],
        "target_sha256": target_sha256,
    }
    if payload.get("status") != "complete" or any(
        payload.get(key) != value for key, value in expected.items()
    ):
        raise RuntimeError(f"existing VAE sidecar identity differs: {path}")
    return payload


def evaluate_target(
    target: Mapping[str, str],
    *,
    config: Mapping[str, object],
    identities: Mapping[str, str],
    autoencoder,
    lpips_metric,
    device,
    sidecar_path: Path | None,
) -> dict[str, object]:
    import torch

    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats(device)
    target_path = workspace_path(target["target_path"], label="target image")
    if sha256(target_path) != target["target_sha256"]:
        raise RuntimeError(f"target identity changed: {target_path}")
    reference = load_rgb(target_path)
    if reference.shape[:2] != (int(target["height"]), int(target["width"])):
        raise RuntimeError(f"target dimensions changed: {target_path}")
    tiling = config["tiling"]
    metrics = config["metrics"]
    assert isinstance(tiling, dict) and isinstance(metrics, dict)
    processor = AutoencoderTileProcessor(
        autoencoder, device, int(tiling["alignment_multiple"])
    )
    reconstruction, tile_metadata = reconstruct_tiled(
        reference,
        processor,
        tile_size=int(tiling["tile_size"]),
        overlap=int(tiling["overlap"]),
    )
    reconstruction = quantize(reconstruction)
    crop_reference = centered_crop(reference, int(metrics["center_crop_size"]))
    crop_reconstruction = quantize(processor(crop_reference))
    lpips_maximum = int(metrics["lpips_resize_maximum_dimension"])
    full_lpips = lpips_value(
        lpips_metric,
        resize_for_lpips(reference, lpips_maximum),
        resize_for_lpips(reconstruction, lpips_maximum),
        device,
    )
    crop_lpips = lpips_value(lpips_metric, crop_reference, crop_reconstruction, device)
    frequency = high_frequency_diagnostic(
        reference,
        reconstruction,
        gaussian_sigma=float(metrics["high_frequency_gaussian_sigma"]),
    )
    result = {
        "status": "complete",
        **identities,
        "target_sha256": target["target_sha256"],
        "target_path": target["target_path"],
        "width": int(target["width"]),
        "height": int(target["height"]),
        "full_psnr": peak_signal_noise_ratio(reference, reconstruction),
        "full_ssim": rgb_structural_similarity(reference, reconstruction),
        "full_lpips_resized": full_lpips,
        "center_crop_width": int(crop_reference.shape[1]),
        "center_crop_height": int(crop_reference.shape[0]),
        "center_crop_psnr": peak_signal_noise_ratio(
            crop_reference, crop_reconstruction
        ),
        "center_crop_ssim": rgb_structural_similarity(
            crop_reference, crop_reconstruction
        ),
        "center_crop_lpips": crop_lpips,
        "mean_luma_bias": float(
            np.mean(
                0.2126 * reconstruction[..., 0]
                + 0.7152 * reconstruction[..., 1]
                + 0.0722 * reconstruction[..., 2]
                - 0.2126 * reference[..., 0]
                - 0.7152 * reference[..., 1]
                - 0.0722 * reference[..., 2]
            )
        ),
        "seam_error_ratio": seam_error_ratio(
            reference,
            reconstruction,
            tile_metadata,
            band_width=int(tiling["seam_band_width"]),
        ),
        "tile_count": int(tile_metadata["tile_count"]),
        **frequency,
        "runtime_seconds": time.monotonic() - started,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
    }
    for key, value in result.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise RuntimeError(f"non-finite VAE metric {key}: {target_path}")
    if sidecar_path is not None:
        content = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
        write_once_or_verify(sidecar_path, content)
    return result


def unique_targets(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    targets: dict[str, dict[str, str]] = {}
    for row in rows:
        candidate = {
            key: row[key] for key in ("target_sha256", "target_path", "width", "height")
        }
        existing = targets.get(row["target_sha256"])
        if existing is not None and (
            existing["width"] != candidate["width"]
            or existing["height"] != candidate["height"]
        ):
            raise RuntimeError("duplicate target hash has inconsistent dimensions")
        if existing is None or candidate["target_path"] < existing["target_path"]:
            targets[row["target_sha256"]] = candidate
    return [targets[key] for key in sorted(targets)]


def select_runtime_canary_targets(
    targets: list[dict[str, str]],
    *,
    count: int,
    tile_size: int,
    overlap: int,
) -> list[dict[str, str]]:
    """Choose the most expensive distinct targets for an end-to-end canary."""

    if count <= 0 or len(targets) < count:
        raise ValueError("runtime canary target count is invalid")

    def ranking(target: Mapping[str, str]) -> tuple[int, int, str]:
        height = int(target["height"])
        width = int(target["width"])
        tile_count = len(tile_origins(height, tile_size, overlap)) * len(
            tile_origins(width, tile_size, overlap)
        )
        return (-tile_count, -(height * width), target["target_sha256"])

    return sorted(targets, key=ranking)[:count]


def decision_from_summary(
    summary: Mapping[str, object], decision: Mapping[str, object]
) -> dict[str, object]:
    failures: list[str] = []
    datasets = summary["datasets"]
    assert isinstance(datasets, dict)
    for dataset, values in datasets.items():
        assert isinstance(values, dict)
        scene = values["scene_balanced"]
        assert isinstance(scene, dict)
        checks = (
            (
                "full_psnr",
                float(scene["full_psnr"]),
                ">=",
                float(decision["minimum_full_psnr_db"]),
            ),
            (
                "center_crop_psnr",
                float(scene["center_crop_psnr"]),
                ">=",
                float(decision["minimum_center_crop_psnr_db"]),
            ),
            (
                "full_lpips_resized",
                float(scene["full_lpips_resized"]),
                "<=",
                float(decision["maximum_full_lpips_resized"]),
            ),
            (
                "high_frequency_correlation",
                float(scene["high_frequency_correlation"]),
                ">=",
                float(decision["minimum_high_frequency_correlation"]),
            ),
            (
                "seam_error_ratio",
                float(scene["seam_error_ratio"]),
                "<=",
                float(decision["maximum_seam_error_ratio"]),
            ),
        )
        for metric, value, operator, threshold in checks:
            passed = value >= threshold if operator == ">=" else value <= threshold
            if not passed:
                failures.append(
                    f"{dataset}:{metric}={value:.8f} required {operator}{threshold:.8f}"
                )
    return {
        "pixel_frequency_bypass_required": bool(failures),
        "failures": failures,
        "policy": decision["policy"],
    }


def build_summary(
    *,
    config_path: Path,
    config: Mapping[str, object],
    rows: list[dict[str, str]],
    identities: Mapping[str, str],
    output_dir: Path,
    device_inventory: list[dict[str, object]],
) -> dict[str, object]:
    sidecar_dir = output_dir / "per_target"
    by_sha = {
        target["target_sha256"]: validate_sidecar(
            sidecar_dir / f"{target['target_sha256']}.json",
            target_sha256=target["target_sha256"],
            identities=identities,
        )
        for target in unique_targets(rows)
    }
    result_rows = []
    excluded_sidecar_fields = {
        "status",
        "config_sha256",
        "evaluator_sha256",
        "representation_module_sha256",
        "paired_development_module_sha256",
        "autoencoder_sha256",
        "target_sha256",
        "target_path",
    }
    for row in rows:
        metrics = by_sha[row["target_sha256"]]
        result_rows.append(
            {
                "development_id": row["development_id"],
                "dataset": row["dataset"],
                "scene_id": row["scene_id"],
                "target_sha256": row["target_sha256"],
                **{
                    key: value
                    for key, value in metrics.items()
                    if key not in excluded_sidecar_fields
                },
            }
        )
    fields = list(result_rows[0])
    csv_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(csv_buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(result_rows)
    csv_content = csv_buffer.getvalue().encode()
    metric_fields = (
        "full_psnr",
        "full_ssim",
        "full_lpips_resized",
        "center_crop_psnr",
        "center_crop_ssim",
        "center_crop_lpips",
        "high_frequency_energy_ratio",
        "high_frequency_error_rms",
        "high_frequency_correlation",
        "seam_error_ratio",
        "mean_luma_bias",
    )
    reporting = config["reporting"]
    decision = config["decision"]
    assert isinstance(reporting, dict) and isinstance(decision, dict)
    metric_summary = clustered_metric_summary(
        result_rows,
        fields=metric_fields,
        bootstrap_seed=int(reporting["bootstrap_seed"]),
        bootstrap_resamples=int(reporting["bootstrap_resamples"]),
        confidence=float(reporting["bootstrap_confidence"]),
    )
    automatic_decision = decision_from_summary(metric_summary, decision)
    output = config["output"]
    assert isinstance(output, dict)
    rows_path = workspace_path(str(output["rows_csv"]), label="result rows")
    write_once_or_verify(rows_path, csv_content)
    payload = {
        "schema_version": 1,
        "status": (
            "completed_pixel_frequency_bypass_required"
            if automatic_decision["pixel_frequency_bypass_required"]
            else "completed_latent_only_ceiling_clear"
        ),
        "official_foundir_inference_performed": False,
        "official_foundir_images_opened": False,
        "config": str(config_path.relative_to(ROOT)),
        **identities,
        "git_head": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "records": len(result_rows),
        "unique_targets": len(by_sha),
        "result_rows": str(rows_path.relative_to(ROOT)),
        "result_rows_sha256": hashlib.sha256(csv_content).hexdigest(),
        "device_inventory": device_inventory,
        "tiling": config["tiling"],
        "metrics": config["metrics"],
        "thresholds": decision,
        "automatic_decision": automatic_decision,
        "summary": metric_summary,
    }
    summary_path = workspace_path(str(output["summary_json"]), label="summary")
    content = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    write_once_or_verify(summary_path, content)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--contract-only", action="store_true")
    parser.add_argument("--runtime-canary", action="store_true")
    parser.add_argument("--skip-autoencoder-hash", action="store_true")
    args = parser.parse_args()
    if args.contract_only and args.runtime_canary:
        parser.error("--contract-only and --runtime-canary are mutually exclusive")
    config_path = args.config.resolve()
    try:
        config_path.relative_to(ROOT)
    except ValueError as exc:
        raise RuntimeError("config must be inside the workspace") from exc
    config, rows = contract_check(
        config_path, skip_autoencoder_hash=args.skip_autoencoder_hash
    )
    if args.contract_only:
        print(
            "LOWLIGHT_VAE_CEILING_CONTRACT_PASS "
            f"records={len(rows)} unique_targets={len(unique_targets(rows))}"
        )
        return

    import torch
    import torch.distributed as distributed
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the VAE ceiling")
    distributed.init_process_group(backend="nccl")
    try:
        rank = distributed.get_rank()
        world_size = distributed.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
        runtime = config["runtime"]
        model = config["model"]
        output = config["output"]
        metrics = config["metrics"]
        tiling = config["tiling"]
        assert isinstance(runtime, dict)
        assert isinstance(model, dict)
        assert isinstance(output, dict)
        assert isinstance(metrics, dict)
        assert isinstance(tiling, dict)
        if world_size != int(runtime["world_size"]):
            raise RuntimeError(f"VAE ceiling requires world size 2, got {world_size}")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        properties = torch.cuda.get_device_properties(device)
        if properties.total_memory < int(runtime["minimum_gpu_memory_mib"]) * 1024**2:
            raise RuntimeError("selected GPU has insufficient memory")
        if not any(name in properties.name for name in runtime["allowed_gpu_families"]):
            raise RuntimeError(
                f"unsupported GPU for frozen diagnostic: {properties.name}"
            )
        torch.manual_seed(2026082122)
        torch.cuda.manual_seed_all(2026082122)
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)

        identities = {
            "config_sha256": sha256(config_path),
            "evaluator_sha256": sha256(Path(__file__).resolve()),
            "representation_module_sha256": sha256(
                ROOT / "lowlight_restoration/representation_ceiling.py"
            ),
            "paired_development_module_sha256": sha256(
                ROOT / "lowlight_restoration/paired_development.py"
            ),
            "autoencoder_sha256": str(model["autoencoder_sha256"]),
        }
        targets = unique_targets(rows)
        if args.runtime_canary:
            canaries = select_runtime_canary_targets(
                targets,
                count=world_size,
                tile_size=int(tiling["tile_size"]),
                overlap=int(tiling["overlap"]),
            )
            target = canaries[rank]
            os.environ["AE_MODEL_PATH"] = str(
                workspace_path(str(model["autoencoder_weights"]), label="autoencoder")
            )
            from flux2.util import load_ae

            autoencoder = load_ae(str(model["name"]), device=device)
            autoencoder.eval().requires_grad_(False)
            lpips_metric = (
                LearnedPerceptualImagePatchSimilarity(
                    net_type=str(metrics["lpips_backbone"]), normalize=True
                )
                .to(device)
                .eval()
            )
            result = evaluate_target(
                target,
                config=config,
                identities=identities,
                autoencoder=autoencoder,
                lpips_metric=lpips_metric,
                device=device,
                sidecar_path=None,
            )
            canary_record = {
                "rank": rank,
                "target_sha256": target["target_sha256"],
                "tile_count": result["tile_count"],
                "full_psnr": result["full_psnr"],
                "full_lpips_resized": result["full_lpips_resized"],
                "peak_cuda_allocated_bytes": result["peak_cuda_allocated_bytes"],
            }
            gathered: list[dict[str, object] | None] = [None] * world_size
            distributed.all_gather_object(gathered, canary_record)
            if rank == 0:
                completed = [item for item in gathered if item is not None]
                if (
                    len(completed) != world_size
                    or len({str(item["target_sha256"]) for item in completed})
                    != world_size
                ):
                    raise RuntimeError("distributed runtime canary was incomplete")
                print(
                    "LOWLIGHT_VAE_CEILING_RUNTIME_CANARY_PASS "
                    f"targets={len(completed)} "
                    f"maximum_tiles={max(int(item['tile_count']) for item in completed)}",
                    flush=True,
                )
            distributed.barrier()
            return

        output_dir = workspace_path(str(output["directory"]), label="output directory")
        sidecar_dir = output_dir / "per_target"
        sidecar_dir.mkdir(parents=True, exist_ok=True)
        assigned = targets[rank::world_size]
        pending = []
        for target in assigned:
            sidecar = sidecar_dir / f"{target['target_sha256']}.json"
            if sidecar.exists():
                validate_sidecar(
                    sidecar,
                    target_sha256=target["target_sha256"],
                    identities=identities,
                )
            else:
                pending.append(target)

        if pending:
            os.environ["AE_MODEL_PATH"] = str(
                workspace_path(str(model["autoencoder_weights"]), label="autoencoder")
            )
            from flux2.util import load_ae

            autoencoder = load_ae(str(model["name"]), device=device)
            autoencoder.eval().requires_grad_(False)
            lpips_metric = (
                LearnedPerceptualImagePatchSimilarity(
                    net_type=str(metrics["lpips_backbone"]), normalize=True
                )
                .to(device)
                .eval()
            )
            for position, target in enumerate(pending, start=1):
                result = evaluate_target(
                    target,
                    config=config,
                    identities=identities,
                    autoencoder=autoencoder,
                    lpips_metric=lpips_metric,
                    device=device,
                    sidecar_path=sidecar_dir / f"{target['target_sha256']}.json",
                )
                print(
                    f"VAE_CEILING rank={rank} target={position}/{len(pending)} "
                    f"sha={target['target_sha256'][:12]} psnr={result['full_psnr']:.4f} "
                    f"lpips={result['full_lpips_resized']:.6f} tiles={result['tile_count']}",
                    flush=True,
                )
        device_record = {
            "rank": rank,
            "name": properties.name,
            "total_memory_bytes": properties.total_memory,
            "assigned_targets": len(assigned),
        }
        inventory: list[dict[str, object] | None] = [None] * world_size
        distributed.all_gather_object(inventory, device_record)
        distributed.barrier()
        if rank == 0:
            payload = build_summary(
                config_path=config_path,
                config=config,
                rows=rows,
                identities=identities,
                output_dir=output_dir,
                device_inventory=[item for item in inventory if item is not None],
            )
            scene = payload["summary"]["point_estimates"]["scene_balanced"]
            print(
                "LOWLIGHT_VAE_CEILING_PASS "
                f"status={payload['status']} full_psnr={scene['full_psnr']:.4f} "
                f"full_lpips={scene['full_lpips_resized']:.6f} "
                f"hf_corr={scene['high_frequency_correlation']:.6f}",
                flush=True,
            )
        distributed.barrier()
    finally:
        if distributed.is_initialized():
            distributed.destroy_process_group()


if __name__ == "__main__":
    main()
