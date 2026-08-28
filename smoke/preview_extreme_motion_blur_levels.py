#!/usr/bin/env python3
"""Render a human-inspection preview for a more separated motion-blur scale."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import cv2
import numpy as np

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT))

from lowlight_restoration.multidegradation_data import CROP_SIZE, ROOT, _apply_blur


# Deliberately geometric growth in trajectory length.  These values are only a
# preview candidate; production BLUR_PARAMETERS remain frozen until approval.
CANDIDATE_PARAMETERS = {
    2: (22, 18.0),
    3: (30, 24.0),
    4: (90, 72.0),
    5: (200, 160.0),
}
SOURCE_PATHS = (
    Path("data/clean/002.png"),
    Path("data/clean/037.png"),
    Path("data/clean/073.png"),
)
ANGLE_DEGREES = 22.5
OUTPUT_ROOT = ROOT / "reports/blur_extreme_preview_v6"


def center_crop(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    if height < CROP_SIZE or width < CROP_SIZE:
        raise ValueError(f"source is smaller than {CROP_SIZE}: {width}x{height}")
    x = (width - CROP_SIZE) // 2
    y = (height - CROP_SIZE) // 2
    return image[y : y + CROP_SIZE, x : x + CROP_SIZE].copy()


def laplacian_variance(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def rmse(left: np.ndarray, right: np.ndarray) -> float:
    delta = left.astype(np.float32) - right.astype(np.float32)
    return float(np.sqrt(np.mean(delta * delta)))


def labeled_thumbnail(image: np.ndarray, label: str) -> np.ndarray:
    tile = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    cv2.rectangle(tile, (0, 0), (224, 30), (0, 0, 0), thickness=-1)
    cv2.putText(
        tile,
        label,
        (8, 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return tile


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {
        "status": "preview_only_not_production",
        "blur_type": "motion",
        "angle_degrees": ANGLE_DEGREES,
        "candidate_parameters": {
            f"L{level}": {"radius": radius, "sigma": sigma, "kernel_width": 2 * radius + 1}
            for level, (radius, sigma) in CANDIDATE_PARAMETERS.items()
        },
        "sources": [],
    }
    contact_rows: list[np.ndarray] = []
    for source_path in SOURCE_PATHS:
        absolute_source = ROOT / source_path
        source = cv2.imread(str(absolute_source), cv2.IMREAD_COLOR)
        if source is None:
            raise FileNotFoundError(absolute_source)
        clean = center_crop(source)
        source_output = OUTPUT_ROOT / source_path.stem
        source_output.mkdir(parents=True, exist_ok=True)
        images = {1: clean}
        for level, (radius, sigma) in CANDIDATE_PARAMETERS.items():
            images[level] = _apply_blur(
                clean,
                {
                    "radius": radius,
                    "sigma": sigma,
                    "angle_degrees": ANGLE_DEGREES,
                },
            )
        metrics = []
        previous = None
        thumbnails = []
        for level in range(1, 6):
            image = images[level]
            destination = source_output / f"L{level}.png"
            if not cv2.imwrite(str(destination), image):
                raise RuntimeError(f"failed to write {destination}")
            parameters = CANDIDATE_PARAMETERS.get(level)
            label = "L1 clean" if parameters is None else f"L{level} r={parameters[0]} s={parameters[1]:g}"
            thumbnails.append(labeled_thumbnail(image, label))
            metrics.append(
                {
                    "level": level,
                    "radius": parameters[0] if parameters else 0,
                    "sigma": parameters[1] if parameters else 0.0,
                    "laplacian_variance": laplacian_variance(image),
                    "rmse_from_clean": rmse(image, clean),
                    "rmse_from_previous_level": 0.0 if previous is None else rmse(image, previous),
                    "path": str(destination.relative_to(ROOT)),
                }
            )
            previous = image
        contact_rows.append(np.concatenate(thumbnails, axis=1))
        report["sources"].append(
            {
                "source": str(source_path),
                "crop": "center_448x448",
                "metrics": metrics,
            }
        )
    contact_sheet = np.concatenate(contact_rows, axis=0)
    contact_path = OUTPUT_ROOT / "contact_sheet.png"
    if not cv2.imwrite(str(contact_path), contact_sheet):
        raise RuntimeError(f"failed to write {contact_path}")
    report["contact_sheet"] = str(contact_path.relative_to(ROOT))
    report_path = OUTPUT_ROOT / "preview_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"EXTREME_BLUR_PREVIEW_PASS contact_sheet={contact_path}")
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
