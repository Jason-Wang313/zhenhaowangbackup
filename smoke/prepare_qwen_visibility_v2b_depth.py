#!/usr/bin/env python3
"""Prepare hash-bound relative-depth maps for the fresh v2b haze holdout."""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.io import loadmat, savemat
from scipy.stats import spearmanr
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.multidegradation_data import sha256, stable_digest, write_new_or_verify


MODEL_REPOSITORY = "depth-anything/Depth-Anything-V2-Small-hf"
MODEL_REVISION = "5426e4f0f36572d16453bbda7a8389317b1bef99"
MODEL_ROOT = ROOT / "models/depth_anything_v2_small_hf_5426e4f0"
MODEL_HASHES = {
    "README.md": "319566441daaac0c5ed83e75a8fd19bae0863f3275e23f147c06fe0906edf6fb",
    "config.json": "c56698d3643dde1f83ea2212759e6b31a22b8f827246a36dd007ee8a22b3ff75",
    "model.safetensors": "3152477ce0d8d6978d76b995120de97cb5b928701fd0f817769f59e249a16b70",
    "preprocessor_config.json": "d41175c0d889477ca8fc67191e540faef14baf6275157b3fdecf78469e6bbf84",
}
OLD_SOURCE_MANIFEST = ROOT / "manifests/low_light_source_pool_2000.json"
OLD_SOURCE_MANIFEST_SHA256 = "031a6e43cf46e8bf2363be7ca6c67c533ffcf7a44241a5f605a542daac1729d5"
EXTRA_ROOT = ROOT / "data/clean_pool/lsdir"
DEPTH_ROOT = ROOT / "data/qwen_multidegradation_visibility_v2b_depth"
REPORT = ROOT / "reports/qwen_multidegradation_visibility_v2b_depth.json"
SEED = 2026082831
DEPTH_SOURCE_COUNT = 20
CALIBRATION_SOURCE_COUNT = 5


def verify_model() -> None:
    for name, expected in MODEL_HASHES.items():
        path = MODEL_ROOT / name
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"depth model file failed its frozen hash: {path}")


def old_sources() -> list[dict[str, object]]:
    if sha256(OLD_SOURCE_MANIFEST) != OLD_SOURCE_MANIFEST_SHA256:
        raise RuntimeError("old source manifest changed")
    payload = json.loads(OLD_SOURCE_MANIFEST.read_text())
    records = payload.get("records")
    if not isinstance(records, list) or len(records) != 2000:
        raise RuntimeError("old source manifest cardinality changed")
    return records


def extra_sources(records: list[dict[str, object]]) -> list[Path]:
    used = {(ROOT / str(record["path"])).resolve() for record in records}
    candidates = [
        path.resolve()
        for path in sorted(EXTRA_ROOT.iterdir())
        if path.is_file() and path.resolve() not in used
    ]
    if len(candidates) != 200:
        raise RuntimeError(f"expected exactly 200 unused LSDIR sources, got {len(candidates)}")
    return candidates


def predict_relative_depth(
    image: Image.Image,
    *,
    processor: AutoImageProcessor,
    model: AutoModelForDepthEstimation,
) -> np.ndarray:
    inputs = processor(images=image, return_tensors="pt")
    with torch.inference_mode():
        prediction = model(**inputs).predicted_depth
    result = prediction[0].float().cpu().numpy()
    if result.ndim != 2 or not np.isfinite(result).all() or float(np.ptp(result)) <= 0:
        raise RuntimeError("depth estimator produced an invalid relative-depth field")
    return result


def calibration_orientation(
    records: list[dict[str, object]],
    *,
    processor: AutoImageProcessor,
    model: AutoModelForDepthEstimation,
) -> tuple[str, list[dict[str, object]]]:
    mio = sorted(
        (record for record in records if record["dataset"] == "mio100"),
        key=lambda record: stable_digest(SEED, "depth_calibration", record["source_id"]),
    )[:CALIBRATION_SOURCE_COUNT]
    if len(mio) != CALIBRATION_SOURCE_COUNT:
        raise RuntimeError("depth orientation calibration lacks MiO sources")
    rows: list[dict[str, object]] = []
    correlations: list[float] = []
    for record in mio:
        path = ROOT / str(record["path"])
        with Image.open(path) as image:
            image.load()
            prediction = predict_relative_depth(
                image.convert("RGB"), processor=processor, model=model
            )
        stem = path.stem
        official_path = ROOT / "data/agenticir/haze" / stem / "predict_depth.mat"
        official = loadmat(official_path)["data_obj"].astype(np.float32)
        resized = cv2.resize(
            prediction,
            (official.shape[1], official.shape[0]),
            interpolation=cv2.INTER_CUBIC,
        )
        correlation = float(spearmanr(resized.reshape(-1), official.reshape(-1)).statistic)
        if not np.isfinite(correlation):
            raise RuntimeError("depth orientation calibration correlation is non-finite")
        correlations.append(correlation)
        rows.append(
            {
                "source_id": record["source_id"],
                "source_sha256": record["sha256"],
                "official_depth_path": official_path.relative_to(ROOT).as_posix(),
                "official_depth_sha256": sha256(official_path),
                "raw_spearman_correlation": correlation,
            }
        )
    median = float(np.median(correlations))
    if abs(median) < 0.10:
        raise RuntimeError(
            f"depth orientation is not identifiable from MiO calibration: median={median}"
        )
    orientation = "identity" if median > 0 else "max_minus_prediction"
    for row in rows:
        row["selected_orientation"] = orientation
    return orientation, rows


def oriented_normalized(prediction: np.ndarray, orientation: str) -> np.ndarray:
    value = prediction.astype(np.float32)
    if orientation == "max_minus_prediction":
        value = float(value.max()) - value
    elif orientation != "identity":
        raise ValueError(orientation)
    value -= float(value.min())
    maximum = float(value.max())
    if maximum <= 0:
        raise RuntimeError("oriented depth is constant")
    return (value / maximum).astype(np.float32)


def deterministic_mat_payload(depth: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    savemat(buffer, {"data_obj": depth}, do_compression=False, oned_as="row")
    payload = bytearray(buffer.getvalue())
    header = b"MATLAB 5.0 MAT-file, qwen visibility v2b deterministic relative depth"
    payload[:116] = header.ljust(116, b" ")
    return bytes(payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()
    if args.threads <= 0:
        raise SystemExit("threads must be positive")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    verify_model()
    records = old_sources()
    extras = extra_sources(records)
    selected = sorted(
        extras,
        key=lambda path: stable_digest(SEED, "v2b_depth_source", path.name),
    )[:DEPTH_SOURCE_COUNT]
    processor = AutoImageProcessor.from_pretrained(MODEL_ROOT, local_files_only=True)
    model = AutoModelForDepthEstimation.from_pretrained(
        MODEL_ROOT, local_files_only=True
    ).eval()
    orientation, calibration = calibration_orientation(
        records, processor=processor, model=model
    )
    outputs: list[dict[str, object]] = []
    for index, path in enumerate(selected, start=1):
        with Image.open(path) as image:
            image.load()
            rgb = image.convert("RGB")
            prediction = predict_relative_depth(
                rgb, processor=processor, model=model
            )
            width, height = rgb.size
        depth = oriented_normalized(prediction, orientation)
        quarter_width = max(1, int(round(width / 4)))
        quarter_height = max(1, int(round(height / 4)))
        depth = cv2.resize(
            depth,
            (quarter_width, quarter_height),
            interpolation=cv2.INTER_CUBIC,
        ).clip(0.0, 1.0).astype(np.float32)
        source_id = f"lsdir_{path.stem.lower()}"
        output = DEPTH_ROOT / source_id / "predict_depth.mat"
        digest = write_new_or_verify(output, deterministic_mat_payload(depth))
        outputs.append(
            {
                "source_id": source_id,
                "source_path": path.relative_to(ROOT).as_posix(),
                "source_sha256": sha256(path),
                "depth_path": output.relative_to(ROOT).as_posix(),
                "depth_sha256": digest,
                "depth_shape": list(depth.shape),
                "minimum": float(depth.min()),
                "maximum": float(depth.max()),
            }
        )
        print(f"QWEN_VISIBILITY_V2B_DEPTH {index}/{len(selected)}", flush=True)
    report = {
        "schema_version": 1,
        "status": "pass",
        "model_repository": MODEL_REPOSITORY,
        "model_revision": MODEL_REVISION,
        "model_files": MODEL_HASHES,
        "license": "apache-2.0",
        "old_source_manifest": OLD_SOURCE_MANIFEST.relative_to(ROOT).as_posix(),
        "old_source_manifest_sha256": OLD_SOURCE_MANIFEST_SHA256,
        "selection_seed": SEED,
        "unused_source_candidates": len(extras),
        "depth_source_count": len(outputs),
        "orientation": orientation,
        "orientation_calibration": calibration,
        "depth_transform": "relative_depth_orientation_matched_to_official_mio_then_minmax_normalized",
        "outputs": outputs,
    }
    digest = write_new_or_verify(
        REPORT,
        (json.dumps(report, indent=2, sort_keys=True) + "\n").encode(),
    )
    print(
        "QWEN_VISIBILITY_V2B_DEPTH_PASS "
        + json.dumps(
            {
                "depth_sources": len(outputs),
                "orientation": orientation,
                "report_sha256": digest,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
