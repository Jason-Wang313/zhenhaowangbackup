#!/usr/bin/env python3
"""Acquire a pinned 25-image non-FoundIR real degradation sanity pool."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import sys
import urllib.request
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.multidegradation_data import sha256, write_new_or_verify


POOL = ROOT / "data/qwen_multidegradation_real_sanity_v1"
TIMEOUT = 60

SOURCES = {
    "blur": {
        "dataset": "RealBlur-J via seiya12/instruct-pix2pix-dataset_RealblurJ",
        "repository": "seiya12/instruct-pix2pix-dataset_RealblurJ",
        "revision": "b66421985334b570984036e431006bd13131b63e",
        "license": "mirror_card_unspecified_internal_research_only",
        "kind": "first_rows",
        "field": "before",
    },
    "noise": {
        "dataset": "noisy_images grain/noise collection",
        "repository": "isabelxiao1/noisy_images",
        "revision": "426626810830b7b20640d270c408c6fdbf84c5cd",
        "license": "card_unspecified_internal_qualitative_only",
        "kind": "first_rows",
        "field": "image",
    },
    "rain": {
        "dataset": "RealRain-1k-H",
        "repository": "dronefreak/RealRain-1k",
        "revision": "cd0b5bb4fb78c50999084958f67d3b9c12e76ea4",
        "license": "mit",
        "kind": "paths",
        "paths": [
            "RealRain-1k-H/test/input/1002.png",
            "RealRain-1k-H/test/input/1003.png",
            "RealRain-1k-H/test/input/1004.png",
            "RealRain-1k-H/test/input/1006.png",
            "RealRain-1k-H/test/input/1009.png",
        ],
    },
    "haze": {
        "dataset": "RESIDE RTTS",
        "repository": "fuhao7i/RTTS",
        "revision": "6f5aaf4031975a6016157f144e5b373a8ba6f392",
        "license": "apache-2.0_mirror_metadata",
        "kind": "paths",
        "paths": [
            "VOCdevkit/VOC2007/JPEGImages/AM_Bing_211.png",
            "VOCdevkit/VOC2007/JPEGImages/AM_Bing_217.png",
            "VOCdevkit/VOC2007/JPEGImages/AM_Bing_222.png",
            "VOCdevkit/VOC2007/JPEGImages/AM_Bing_229.png",
            "VOCdevkit/VOC2007/JPEGImages/AM_Bing_232.png",
        ],
    },
    "low_light": {
        "dataset": "DICM author testing images",
        "repository": "local_verified_official_testing_archive",
        "revision": "archive_sha256_0033eb74d4c964755f02d4b46e6db3eb457637c116415571fa2d0212577b3f29",
        "license": "author_archive_terms_unspecified_internal_research_only",
        "kind": "local",
        "paths": [f"{index:02d}.JPG" for index in range(1, 6)],
    },
}


def request_bytes(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "qwen-real-sanity-v1"})
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
        return response.read()


def request_json(url: str) -> dict[str, object]:
    payload = json.loads(request_bytes(url))
    if not isinstance(payload, dict):
        raise RuntimeError(f"JSON endpoint did not return an object: {url}")
    return payload


def validate_image(payload: bytes) -> tuple[int, int, str]:
    with Image.open(io.BytesIO(payload)) as image:
        image.load()
        if image.width < 224 or image.height < 224:
            raise RuntimeError(f"real sanity image is too small: {image.size}")
        return image.width, image.height, image.format or "unknown"


def acquire_remote_type(name: str, spec: dict[str, object]) -> list[dict[str, object]]:
    sources: list[tuple[str, str]] = []
    if spec["kind"] == "first_rows":
        endpoint = (
            "https://datasets-server.huggingface.co/first-rows?dataset="
            f"{spec['repository']}&config=default&split=train"
        )
        response = request_json(endpoint)
        rows = response.get("rows")
        if not isinstance(rows, list) or len(rows) < 5:
            raise RuntimeError(f"first-rows endpoint lacks five {name} records")
        for item in rows[:5]:
            row = item.get("row") if isinstance(item, dict) else None
            cell = row.get(spec["field"]) if isinstance(row, dict) else None
            url = cell.get("src") if isinstance(cell, dict) else None
            if not isinstance(url, str) or str(spec["revision"]) not in url:
                raise RuntimeError(f"{name} first-row revision is not pinned")
            sources.append((f"row_{int(item['row_idx']):04d}", url))
    elif spec["kind"] == "paths":
        for index, path in enumerate(spec["paths"]):
            url = (
                f"https://huggingface.co/datasets/{spec['repository']}/resolve/"
                f"{spec['revision']}/{path}"
            )
            sources.append((f"path_{index:04d}", url))
    else:
        raise ValueError(f"unexpected remote kind: {spec['kind']}")

    records: list[dict[str, object]] = []
    for index, (source_key, url) in enumerate(sources):
        payload = request_bytes(url)
        width, height, image_format = validate_image(payload)
        extension = ".png" if image_format.upper() == "PNG" else ".jpg"
        output = POOL / name / f"{name}_{index:02d}{extension}"
        digest = write_new_or_verify(output, payload)
        records.append(
            {
                "record_id": f"real_{name}_{index:02d}",
                "expected_primary_type": name,
                "severity_ground_truth": None,
                "annotation_kind": "dataset_category_qualitative_not_exhaustive",
                "dataset": spec["dataset"],
                "repository": spec["repository"],
                "revision": spec["revision"],
                "license": spec["license"],
                "source_key": source_key,
                "source_url": url,
                "path": output.relative_to(ROOT).as_posix(),
                "sha256": digest,
                "width": width,
                "height": height,
                "format": image_format,
            }
        )
    return records


def acquire_local_lowlight(spec: dict[str, object]) -> list[dict[str, object]]:
    source_root = (
        ROOT
        / "data/real_lowlight_corrective_holdout/official_test_0033eb74d4c/Test/DICM"
    )
    records: list[dict[str, object]] = []
    for index, name in enumerate(spec["paths"]):
        path = source_root / name
        payload = path.read_bytes()
        width, height, image_format = validate_image(payload)
        records.append(
            {
                "record_id": f"real_low_light_{index:02d}",
                "expected_primary_type": "low_light",
                "severity_ground_truth": None,
                "annotation_kind": "dataset_category_qualitative_not_exhaustive",
                "dataset": spec["dataset"],
                "repository": spec["repository"],
                "revision": spec["revision"],
                "license": spec["license"],
                "source_key": name,
                "source_url": None,
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "width": width,
                "height": height,
                "format": image_format,
            }
        )
    return records


def validate_manifest(records: list[dict[str, object]]) -> dict[str, object]:
    if len(records) != 25 or len({record["record_id"] for record in records}) != 25:
        raise RuntimeError("real sanity pool must contain 25 unique records")
    counts: dict[str, int] = {}
    hashes: set[str] = set()
    for record in records:
        name = str(record["expected_primary_type"])
        counts[name] = counts.get(name, 0) + 1
        path = ROOT / str(record["path"])
        if sha256(path) != record["sha256"]:
            raise RuntimeError(f"real sanity hash mismatch: {path}")
        if record["sha256"] in hashes:
            raise RuntimeError("real sanity pool contains exact duplicate pixels")
        hashes.add(str(record["sha256"]))
    if counts != {name: 5 for name in ("blur", "noise", "rain", "haze", "low_light")}:
        raise RuntimeError(f"real sanity type counts changed: {counts}")
    return {
        "records": 25,
        "type_counts": counts,
        "exact_unique": 25,
        "foundir_used": False,
        "severity_annotation": "none_manual_qualitative_only",
    }


def main() -> None:
    records: list[dict[str, object]] = []
    for name, spec in SOURCES.items():
        if name == "low_light":
            records.extend(acquire_local_lowlight(spec))
        else:
            records.extend(acquire_remote_type(name, spec))
    validation = validate_manifest(records)
    manifest = ROOT / "manifests/qwen_multidegradation_real_sanity_v1.json"
    payload = {
        "schema_version": 1,
        "status": "frozen_before_qwen_inference",
        "selection_policy": "first_five_pinned_records_per_real_dataset_category",
        "manual_rubric": {
            "acceptable": "adjacent L3/L4 ambiguity and legitimate additional visible degradations",
            "failure": [
                "obvious_primary_type_omission",
                "obvious_unrelated_type_false_positive",
                "severity_contradicts_visible_strength",
            ],
        },
        "validation": validation,
        "records": records,
    }
    manifest_sha = write_new_or_verify(
        manifest, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    )
    print(
        f"QWEN_MULTIDEGRADATION_REAL_SANITY_ACQUISITION_PASS records=25 sha256={manifest_sha}",
        flush=True,
    )


if __name__ == "__main__":
    main()
