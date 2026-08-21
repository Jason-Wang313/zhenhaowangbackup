#!/usr/bin/env python3
"""Freeze objective same-scene exposure ordering without absolute L1--L5 labels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.exposure_ordering import (  # noqa: E402
    build_exposure_stacks,
    exposure_pairs,
)


CANDIDATE_CSV = ROOT / "manifests/qwen_lowlight_corrective_holdout_candidates.csv"
CANDIDATE_JSON = ROOT / "manifests/qwen_lowlight_corrective_holdout_candidates.json"
OUTPUT_CSV = ROOT / "manifests/qwen_lowlight_real_exposure_ordering.csv"
OUTPUT_JSON = ROOT / "manifests/qwen_lowlight_real_exposure_ordering.json"
EXPECTED_CANDIDATE_CSV_SHA256 = (
    "81b3b6080f92124813f207a3d950b9dc922a006a7582c3276142f7c24611b265"
)
EXPECTED_CANDIDATE_JSON_SHA256 = (
    "c95b5069ad7756265ed0a3c71411a6cabbc0e877e8a2460122f8e93b293a0537"
)
SOURCE_ARCHIVE_SHA256 = (
    "0033eb74d4c964755f02d4b46e6db3eb457637c116415571fa2d0212577b3f29"
)
EXCLUDED_COMPONENTS = {
    # This component mixes exposures with files explicitly named test_out,
    # zero, and test_0.2; it is not a defensible raw exposure-only sequence.
    "official_test_scene_3c7dcd9ac2bac291": "contains_fused_or_generated_outputs",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_once_or_verify(path: Path, content: bytes) -> None:
    if path.exists():
        if path.read_bytes() != content:
            raise RuntimeError(f"existing immutable output differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(content)


def build_payload(
    candidate_csv: Path = CANDIDATE_CSV,
    candidate_json: Path = CANDIDATE_JSON,
) -> tuple[bytes, bytes]:
    if sha256(candidate_csv) != EXPECTED_CANDIDATE_CSV_SHA256:
        raise RuntimeError("screened candidate CSV identity changed")
    if sha256(candidate_json) != EXPECTED_CANDIDATE_JSON_SHA256:
        raise RuntimeError("screened candidate JSON identity changed")
    with candidate_csv.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    stacks = build_exposure_stacks(
        rows,
        required_subset="MEF",
        excluded_components=EXCLUDED_COMPONENTS,
    )

    output_rows: list[dict[str, object]] = []
    for stack_id, images in sorted(stacks.items()):
        for image in images:
            path = (ROOT / image.path).resolve()
            try:
                path.relative_to(ROOT)
            except ValueError as exc:
                raise RuntimeError(f"selected path escapes workspace: {image.path}") from exc
            if not path.is_file() or path.stat().st_size != image.byte_size:
                raise RuntimeError(f"selected exposure file is missing or changed: {image.path}")
            if sha256(path) != image.sha256:
                raise RuntimeError(f"selected exposure SHA-256 changed: {image.path}")
            output_rows.append(
                {
                    "stack_id": stack_id,
                    "image_id": f"{stack_id}_{image.order_dark_to_bright:02d}",
                    "path": image.path,
                    "sha256": image.sha256,
                    "subset": image.subset,
                    "filename": image.filename,
                    "bytes": image.byte_size,
                    "width": image.width,
                    "height": image.height,
                    "mode": image.mode,
                    "mean_luma": f"{image.mean_luma:.15g}",
                    "order_dark_to_bright": image.order_dark_to_bright,
                    "darkness_rank": image.darkness_rank,
                    "ev_darker_than_brightest": f"{image.ev_darker_than_brightest:.15g}",
                }
            )

    fields = list(output_rows[0])
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(output_rows)
    csv_content = buffer.getvalue().encode("utf-8")
    ev_gaps = (0.0, 0.5, 1.0, 2.0)
    pair_counts = {
        f"minimum_ev_{gap:g}": sum(
            len(exposure_pairs(images, minimum_ev_gap=gap))
            for images in stacks.values()
        )
        for gap in ev_gaps
    }
    payload = {
        "schema_version": 1,
        "status": "frozen_real_exposure_ordering_development_suite",
        "purpose": "objective_real_domain_ordinal_development_not_absolute_severity_ground_truth",
        "annotation_kind": "same_scene_relative_exposure_order_derived_before_model_inference",
        "absolute_project_l1_l5_labels_available": False,
        "official_foundir_used": False,
        "model_inference_performed_during_selection": False,
        "human_annotation_required": False,
        "reuse_policy": (
            "reusable_development_suite_for_causal_diagnosis_not_a_pristine_final_test"
        ),
        "source": {
            "identity": "official_Retinex-Net_Testing_Images_archive",
            "archive_sha256": SOURCE_ARCHIVE_SHA256,
            "candidate_csv": str(candidate_csv.relative_to(ROOT)),
            "candidate_csv_sha256": EXPECTED_CANDIDATE_CSV_SHA256,
            "candidate_json": str(candidate_json.relative_to(ROOT)),
            "candidate_json_sha256": EXPECTED_CANDIDATE_JSON_SHA256,
            "protected_exact_or_phash_leq4_overlaps": 0,
        },
        "selection": {
            "required_subset": "MEF",
            "minimum_exact_unique_images_per_stack": 2,
            "exact_duplicate_policy": "prefer_MEF_then_lexicographic_path",
            "ordering": "ascending_within_stack_mean_luma_dark_to_bright",
            "excluded_components": EXCLUDED_COMPONENTS,
            "reasoning": (
                "MEF contains author-provided same-scene exposure sequences; "
                "unrelated singleton images cannot support objective severity comparisons."
            ),
        },
        "counts": {
            "stacks": len(stacks),
            "images": len(output_rows),
            "pairwise_comparisons": pair_counts,
            "images_per_stack": {
                stack_id: len(images) for stack_id, images in sorted(stacks.items())
            },
        },
        "evaluation_contract": {
            "primary_soft_signal": (
                "expected_L1_L5_level_from_five_canonical_teacher_forced_target_NLLs"
            ),
            "primary_pair_gap_ev": 1.0,
            "minimum_primary_continuous_concordance": 0.75,
            "minimum_primary_hard_non_inversion_rate": 0.80,
            "minimum_median_stack_spearman": 0.60,
            "minimum_positive_stack_spearman_fraction": 0.80,
            "minimum_pooled_expected_level_span": 1.0,
            "interpretation": (
                "Passing establishes relative exposure sensitivity only. It does not "
                "establish absolute real-image L1-L5 accuracy or downstream restoration gain."
            ),
        },
        "output_csv": str(OUTPUT_CSV.relative_to(ROOT)),
        "output_csv_sha256": hashlib.sha256(csv_content).hexdigest(),
    }
    json_content = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    return csv_content, json_content


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-csv", type=Path, default=CANDIDATE_CSV)
    parser.add_argument("--candidate-json", type=Path, default=CANDIDATE_JSON)
    parser.add_argument("--output-csv", type=Path, default=OUTPUT_CSV)
    parser.add_argument("--output-json", type=Path, default=OUTPUT_JSON)
    args = parser.parse_args()
    csv_content, json_content = build_payload(
        args.candidate_csv.resolve(), args.candidate_json.resolve()
    )
    write_once_or_verify(args.output_csv.resolve(), csv_content)
    write_once_or_verify(args.output_json.resolve(), json_content)
    payload = json.loads(json_content)
    print(
        "QWEN_REAL_EXPOSURE_ORDERING_MANIFEST_PASS "
        f"stacks={payload['counts']['stacks']} images={payload['counts']['images']} "
        f"pairs_ev1={payload['counts']['pairwise_comparisons']['minimum_ev_1']}"
    )


if __name__ == "__main__":
    main()
