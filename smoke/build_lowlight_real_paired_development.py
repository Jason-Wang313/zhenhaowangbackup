#!/usr/bin/env python3
"""Build a reusable non-FoundIR paired-real Lowlight development suite."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.paired_development import (  # noqa: E402
    DevelopmentSource,
    select_paired_development_rows,
)


ADAPTATION_CSV = ROOT / "manifests/qwen_lowlight_real_adaptation.csv"
ADAPTATION_JSON = ROOT / "manifests/qwen_lowlight_real_adaptation.json"
REAL_VALIDATION_CSV = ROOT / "manifests/qwen_lowlight_real_validation.csv"
REAL_VALIDATION_JSON = ROOT / "manifests/qwen_lowlight_real_validation.json"
FOUNDIR_CSV = ROOT / "manifests/foundir_lowlight_official.csv"
OUTPUT_CSV = ROOT / "manifests/lowlight_real_paired_development_v1.csv"
OUTPUT_JSON = ROOT / "manifests/lowlight_real_paired_development_v1.json"
EXPECTED_HASHES = {
    ADAPTATION_CSV: "91f95644f852f628891513f116c0a3607b8de980bbe5252c360ff41d9bcc9ab7",
    ADAPTATION_JSON: "2b9f35ffdc2543a4a99bf95fbb4eb61938a022c54cba4569a14863c5b536f06f",
    REAL_VALIDATION_CSV: "52f6fc9c5cdc0347559b37a30f7f5a607ac9a3d5576c6b9e8dc22424c478106b",
    REAL_VALIDATION_JSON: "3b9be6683208580f51ddb93779436270b28e97ab4b27a2df720bfc6238cdd1fb",
    FOUNDIR_CSV: "5ea587a5da65902687c5f6d78b4c1ed4b80e2a3e0ecdaab5147979e003603a0f",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"source manifest is empty: {path}")
    return rows


def write_once_or_verify(path: Path, content: bytes) -> None:
    if path.exists():
        if path.read_bytes() != content:
            raise RuntimeError(f"existing immutable output differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(content)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-csv", type=Path, default=OUTPUT_CSV)
    parser.add_argument("--output-json", type=Path, default=OUTPUT_JSON)
    args = parser.parse_args()
    for path, expected in EXPECTED_HASHES.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"paired-development source identity changed: {path}")

    adaptation_rows = tuple(read_csv(ADAPTATION_CSV))
    validation_rows = tuple(read_csv(REAL_VALIDATION_CSV))
    rows = select_paired_development_rows(
        (
            DevelopmentSource(
                source_name="lolv2_real_train_validation",
                rows=adaptation_rows,
                required_dataset="LOL-v2-real-captured-train",
                required_split="validation",
                required_subset=None,
                target_kind="paired_normal_reference",
            ),
            DevelopmentSource(
                source_name="sice_reusable_validation",
                rows=validation_rows,
                required_dataset="SICE",
                required_split="validation",
                required_subset="sice_balanced_proxy",
                target_kind="selected_fusion_reference",
            ),
        )
    )
    foundir_rows = read_csv(FOUNDIR_CSV)
    foundir_hashes = {
        value
        for row in foundir_rows
        for value in (row["gt_sha256"], row["lq_sha256"])
    }
    output_rows: list[dict[str, object]] = []
    for row in rows:
        target = (ROOT / row["target_path"]).resolve()
        input_path = (ROOT / row["input_path"]).resolve()
        for path in (target, input_path):
            try:
                path.relative_to(ROOT)
            except ValueError as exc:
                raise RuntimeError(f"paired path escapes workspace: {path}") from exc
        if sha256(target) != row["target_sha256"] or sha256(input_path) != row["input_sha256"]:
            raise RuntimeError(f"paired image identity changed: {row['development_id']}")
        if row["target_sha256"] in foundir_hashes or row["input_sha256"] in foundir_hashes:
            raise RuntimeError(f"exact FoundIR overlap: {row['development_id']}")
        with Image.open(target) as target_image, Image.open(input_path) as input_image:
            target_rgb = target_image.convert("RGB")
            input_rgb = input_image.convert("RGB")
            if target_rgb.size != input_rgb.size:
                raise RuntimeError(f"unaligned paired dimensions: {row['development_id']}")
            width, height = target_rgb.size
        output_rows.append(
            {
                **row,
                "width": width,
                "height": height,
                "mode": "RGB",
                "exact_foundir_overlap": "false",
            }
        )

    fields = list(output_rows[0])
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(output_rows)
    csv_content = buffer.getvalue().encode("utf-8")
    dataset_counts = Counter(str(row["dataset"]) for row in output_rows)
    scene_counts = {
        dataset: len(
            {str(row["scene_id"]) for row in output_rows if row["dataset"] == dataset}
        )
        for dataset in sorted(dataset_counts)
    }
    payload = {
        "schema_version": 1,
        "status": "frozen_reusable_paired_real_development_v1",
        "purpose": "restoration_ceiling_and_causal_development_not_fresh_test",
        "official_foundir_inference_performed": False,
        "official_foundir_images_opened": False,
        "foundir_manifest_hashes_used_for_exact_overlap_only": True,
        "exact_foundir_overlap": 0,
        "perceptual_foundir_overlap_screen": "not_run_in_this_builder",
        "consumed_real_test_records_used": False,
        "absolute_l1_l5_ground_truth_claimed": False,
        "project_levels_are_retained_proxy_metadata_only": True,
        "selection": {
            "lolv2": "real_adaptation_validation_split_non_L1_only",
            "sice": "reusable_real_validation_split_non_L1_only",
            "reuse_policy": "development_only_with_scene_clustered_reporting",
        },
        "counts": {
            "records": len(output_rows),
            "scenes": len({str(row["scene_id"]) for row in output_rows}),
            "records_by_dataset": dict(sorted(dataset_counts.items())),
            "scenes_by_dataset": scene_counts,
            "proxy_levels": dict(
                sorted(Counter(str(row["proxy_project_level"]) for row in output_rows).items())
            ),
        },
        "parents": {
            str(path.relative_to(ROOT)): digest
            for path, digest in EXPECTED_HASHES.items()
        },
        "output_csv": str(args.output_csv.resolve().relative_to(ROOT)),
        "output_csv_sha256": hashlib.sha256(csv_content).hexdigest(),
    }
    json_content = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    write_once_or_verify(args.output_csv.resolve(), csv_content)
    write_once_or_verify(args.output_json.resolve(), json_content)
    print(
        "LOWLIGHT_REAL_PAIRED_DEVELOPMENT_PASS "
        f"records={len(output_rows)} scenes={payload['counts']['scenes']} "
        f"datasets={dict(dataset_counts)}"
    )


if __name__ == "__main__":
    main()
