#!/usr/bin/env python3
"""Build immutable proxy-labeled SICE/LOL-v2 real Lowlight manifests."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image, ImageStat, UnidentifiedImageError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.qwen_real_validation import (  # noqa: E402
    PROJECT_LUMA_RATIO_ANCHORS,
    proxy_anchor_error_ev,
    proxy_level_from_luma_ratio,
    select_balanced_proxy_candidates,
)


SICE_REVISION = "1a65a70ad5ae9feef5f8402dfa42347652ce1f2c"
SICE_REPOSITORY = "okhater/SICE"
SICE_ORIGINAL_REPOSITORY = "csjcai/SICE"
SICE_MAXIMUM_PER_LEVEL = 25
SICE_MINIMUM_PER_LEVEL = 18
SICE_MAXIMUM_ANCHOR_ERROR_EV = 0.75
SELECTION_SEED = 2026081906
LOLV2_OFFICIAL_URL = (
    "https://drive.google.com/file/d/1dzuLCk9_gE2bFF222n3-7GVUlSVHpMYC/view"
)
LOLV2_EXPECTED_IDS = {f"{value:05d}" for value in range(690, 790)}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(payload: object) -> str:
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def workspace_path(path: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise SystemExit(f"PATH_OUTSIDE_WORKSPACE {resolved}") from exc
    return resolved


def relative(path: Path) -> str:
    return str(workspace_path(path).relative_to(ROOT.resolve()))


def inspect_image(path: Path) -> dict[str, object]:
    try:
        with Image.open(path) as handle:
            rgb = handle.convert("RGB")
            width, height = rgb.size
            means = ImageStat.Stat(rgb).mean
    except (OSError, ValueError, UnidentifiedImageError) as exc:
        raise RuntimeError(f"unreadable real-domain image: {path}") from exc
    if width <= 0 or height <= 0 or len(means) != 3:
        raise RuntimeError(f"bad real-domain image geometry: {path}")
    luma = 0.2126 * means[0] + 0.7152 * means[1] + 0.0722 * means[2]
    if not math.isfinite(luma) or luma <= 0.0:
        raise RuntimeError(f"bad real-domain image luma: {path}")
    return {
        "path": relative(path),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        "width": width,
        "height": height,
        "mode": "RGB",
        "mean_luma": luma,
    }


def inspect_paths(paths: list[Path]) -> dict[Path, dict[str, object]]:
    if len(paths) != len(set(paths)):
        raise RuntimeError("duplicate path in real-domain source inventory")
    with ThreadPoolExecutor(max_workers=8) as executor:
        values = list(executor.map(inspect_image, paths))
    return dict(zip(paths, values, strict=True))


def index_lolv2_pair_dirs(
    input_dir: Path, reference_dir: Path
) -> dict[str, tuple[Path, Path]]:
    def index(directory: Path, *, prefix: str) -> dict[str, Path]:
        if not directory.is_dir():
            raise RuntimeError(f"missing LOL-v2 directory: {directory}")
        images = [
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ]
        result: dict[str, Path] = {}
        for path in images:
            match = re.fullmatch(rf"{re.escape(prefix)}(\d{{5}})", path.stem)
            if match is None or match.group(1) in result:
                raise RuntimeError(f"bad or duplicate LOL-v2 image name: {path}")
            result[match.group(1)] = path
        if len(result) != len(images):
            raise RuntimeError(f"duplicate LOL-v2 image stem: {directory}")
        return result

    inputs = index(input_dir, prefix="low")
    references = index(reference_dir, prefix="normal")
    if set(inputs) != LOLV2_EXPECTED_IDS or set(references) != LOLV2_EXPECTED_IDS:
        raise RuntimeError(
            "LOL-v2 held-out IDs changed "
            f"input_missing={sorted(LOLV2_EXPECTED_IDS - set(inputs))[:5]} "
            f"reference_missing={sorted(LOLV2_EXPECTED_IDS - set(references))[:5]}"
        )
    return {key: (inputs[key], references[key]) for key in sorted(inputs)}


def index_sice_test(root: Path) -> dict[str, tuple[Path, list[Path]]]:
    if not root.is_dir():
        raise RuntimeError(f"missing SICE test root: {root}")
    result: dict[str, tuple[Path, list[Path]]] = {}
    for directory in sorted(root.iterdir()):
        if not directory.is_dir() or re.fullmatch(r"sample_\d+", directory.name) is None:
            raise RuntimeError(f"unexpected SICE test entry: {directory}")
        files = [
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ]
        label = [path for path in files if path.name == "label.jpg"]
        lows = sorted(
            (path for path in files if re.fullmatch(r"low\d+\.jpg", path.name)),
            key=lambda path: int(path.stem[3:]),
        )
        if len(label) != 1 or len(lows) < 3 or len(files) != 1 + len(lows):
            raise RuntimeError(f"bad SICE source composition: {directory}")
        result[directory.name] = (label[0], lows)
    if len(result) != 47:
        raise RuntimeError(f"SICE held-out source count changed: {len(result)}")
    return result


def base_record(
    *,
    pair_id: str,
    source_id: str,
    reference: dict[str, object],
    image: dict[str, object],
    level: int,
    split: str,
    dataset: str,
    subset: str,
    source_file: str,
    source_revision: str,
    annotation_method: str,
    ratio: float,
    anchor_error: float,
) -> dict[str, object]:
    return {
        "pair_id": pair_id,
        "source_id": source_id,
        "clean_path": reference["path"],
        "degraded_path": image["path"],
        "degradation_type": "low_light",
        "project_level": f"L{level}",
        "seed": SELECTION_SEED,
        "split": split,
        "clean_sha256": reference["sha256"],
        "degraded_sha256": image["sha256"],
        "dataset": dataset,
        "subset": subset,
        "annotation_kind": "reference_based_proxy_not_human_ground_truth",
        "annotation_method": annotation_method,
        "reference_path": reference["path"],
        "reference_sha256": reference["sha256"],
        "source_file": source_file,
        "source_revision": source_revision,
        "reference_mean_luma": float(reference["mean_luma"]),
        "input_mean_luma": float(image["mean_luma"]),
        "input_reference_luma_ratio": ratio,
        "proxy_anchor": 1.0 if level == 1 else PROJECT_LUMA_RATIO_ANCHORS[level],
        "proxy_anchor_error_ev": anchor_error,
    }


def build_sice_records(
    sources: dict[str, tuple[Path, list[Path]]],
    inspections: dict[Path, dict[str, object]],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    candidates: list[dict[str, object]] = []
    for source, (label_path, low_paths) in sorted(sources.items()):
        reference = inspections[label_path]
        for low_path in low_paths:
            image = inspections[low_path]
            if (reference["width"], reference["height"]) != (
                image["width"],
                image["height"],
            ):
                raise RuntimeError(f"unaligned SICE exposure: {source}/{low_path.name}")
            ratio = float(image["mean_luma"]) / float(reference["mean_luma"])
            candidates.append(
                {
                    "candidate_id": f"{source}/{low_path.name}",
                    "source": source,
                    "path": low_path,
                    "reference_path": label_path,
                    "luma_ratio": ratio,
                }
            )
    selected, selection = select_balanced_proxy_candidates(
        candidates,
        maximum_per_level=SICE_MAXIMUM_PER_LEVEL,
        minimum_per_level=SICE_MINIMUM_PER_LEVEL,
        maximum_anchor_error_ev=SICE_MAXIMUM_ANCHOR_ERROR_EV,
        seed=SELECTION_SEED,
    )
    quota = int(selection["selected_per_level"])
    clean_candidates = []
    for source, (label_path, _) in sorted(sources.items()):
        reference = inspections[label_path]
        clean_candidates.append(
            (
                hashlib.sha256(
                    f"{SELECTION_SEED}:L1:{source}".encode("utf-8")
                ).hexdigest(),
                source,
                label_path,
                reference,
            )
        )
    clean_candidates.sort()
    if len(clean_candidates) < quota:
        raise RuntimeError("not enough SICE proxy-clean references")

    records: list[dict[str, object]] = []
    for _, source, label_path, reference in clean_candidates[:quota]:
        records.append(
            base_record(
                pair_id=f"sice_{source}_L1",
                source_id=f"sice_{source}",
                reference=reference,
                image=reference,
                level=1,
                split="validation",
                dataset="SICE",
                subset="sice_balanced_proxy",
                source_file=relative(label_path),
                source_revision=SICE_REVISION,
                annotation_method="SICE_selected_fusion_reference_proxy_clean",
                ratio=1.0,
                anchor_error=0.0,
            )
        )
    for candidate in selected:
        source = str(candidate["source"])
        path = candidate["path"]
        reference_path = candidate["reference_path"]
        assert isinstance(path, Path) and isinstance(reference_path, Path)
        level = int(candidate["proxy_level"])
        records.append(
            base_record(
                pair_id=f"sice_{source}_{path.stem}_L{level}",
                source_id=f"sice_{source}",
                reference=inspections[reference_path],
                image=inspections[path],
                level=level,
                split="validation",
                dataset="SICE",
                subset="sice_balanced_proxy",
                source_file=relative(path),
                source_revision=SICE_REVISION,
                annotation_method=(
                    "nearest_project_luma_ratio_anchor_after_SICE_reference"
                ),
                ratio=float(candidate["luma_ratio"]),
                anchor_error=float(candidate["proxy_anchor_error_ev"]),
            )
        )
    records.sort(key=lambda row: (str(row["project_level"]), str(row["pair_id"])))
    counts = Counter(row["project_level"] for row in records)
    expected = {f"L{level}": quota for level in range(1, 6)}
    if dict(counts) != expected:
        raise RuntimeError(f"unbalanced SICE proxy selection: {dict(counts)}")
    selection["clean_selection_count"] = quota
    selection["record_count_by_level"] = expected
    selection["annotation_is_proxy"] = True
    selection["model_predictions_consulted_during_selection"] = False
    return records, selection


def build_lolv2_records(
    pairs: dict[str, tuple[Path, Path]],
    inspections: dict[Path, dict[str, object]],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    records: list[dict[str, object]] = []
    reference_groups: dict[str, list[tuple[str, Path]]] = {}
    for image_id, (_, reference_path) in sorted(pairs.items()):
        reference_sha = str(inspections[reference_path]["sha256"])
        reference_groups.setdefault(reference_sha, []).append((image_id, reference_path))
    for reference_sha, group in sorted(reference_groups.items()):
        image_id, reference_path = group[0]
        reference = inspections[reference_path]
        if reference["sha256"] != reference_sha:
            raise RuntimeError("LOL-v2 reference grouping changed")
        source = f"lolv2_real_reference_{image_id}"
        records.append(
            base_record(
                pair_id=f"{source}_L1",
                source_id=source,
                reference=reference,
                image=reference,
                level=1,
                split="test",
                dataset="LOL-v2-real",
                subset="lolv2_extreme_stress_proxy",
                source_file=relative(reference_path),
                source_revision="official_archive_1dzuLCk9_gE2bFF222n3-7GVUlSVHpMYC",
                annotation_method="LOL_v2_unique_normal_capture_proxy_clean",
                ratio=1.0,
                anchor_error=0.0,
            )
        )
    for image_id, (input_path, reference_path) in sorted(pairs.items()):
        image = inspections[input_path]
        reference = inspections[reference_path]
        if (image["width"], image["height"]) != (
            reference["width"],
            reference["height"],
        ):
            raise RuntimeError(f"unaligned LOL-v2 held-out pair: {image_id}")
        source = f"lolv2_real_test_{image_id}"
        ratio = float(image["mean_luma"]) / float(reference["mean_luma"])
        level = proxy_level_from_luma_ratio(ratio)
        records.append(
            base_record(
                pair_id=f"{source}_input_L{level}",
                source_id=source,
                reference=reference,
                image=image,
                level=level,
                split="test",
                dataset="LOL-v2-real",
                subset="lolv2_extreme_stress_proxy",
                source_file=relative(input_path),
                source_revision="official_archive_1dzuLCk9_gE2bFF222n3-7GVUlSVHpMYC",
                annotation_method="nearest_project_luma_ratio_anchor_after_LOLv2_GT",
                ratio=ratio,
                anchor_error=proxy_anchor_error_ev(ratio, level),
            )
        )
    records.sort(key=lambda row: (str(row["project_level"]), str(row["pair_id"])))
    expected_records = len(pairs) + len(reference_groups)
    if len(records) != expected_records:
        raise RuntimeError("LOL-v2 stress set cardinality changed")
    duplicate_groups = {
        digest: [image_id for image_id, _ in group]
        for digest, group in sorted(reference_groups.items())
        if len(group) > 1
    }
    evidence: dict[str, object] = {
        "held_out_pairs": len(pairs),
        "normal_reference_files": len(pairs),
        "unique_normal_reference_images": len(reference_groups),
        "repeated_normal_reference_files": len(pairs) - len(reference_groups),
        "duplicate_normal_reference_groups": len(duplicate_groups),
        "duplicate_normal_reference_group_sha256": canonical_sha256(duplicate_groups),
        "clean_proxy_records_deduplicated_by_exact_reference_sha256": True,
        "low_input_records_retained": len(pairs),
    }
    return records, evidence


def csv_bytes(records: list[dict[str, object]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=list(records[0]),
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(records)
    return buffer.getvalue().encode("utf-8")


def write_new_or_verify(path: Path, content: bytes) -> None:
    if path.exists():
        if not path.is_file() or path.read_bytes() != content:
            raise RuntimeError(f"existing immutable manifest differs: {path}")
        print(f"VERIFIED_EXISTING_OUTPUT {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(content)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lolv2-archive", type=Path, required=True)
    parser.add_argument("--lolv2-input-dir", type=Path, required=True)
    parser.add_argument("--lolv2-reference-dir", type=Path, required=True)
    parser.add_argument("--sice-snapshot-root", type=Path, required=True)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=ROOT / "manifests/qwen_lowlight_real_validation.csv",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=ROOT / "manifests/qwen_lowlight_real_validation.json",
    )
    args = parser.parse_args()

    archive = workspace_path(args.lolv2_archive)
    lolv2_input = workspace_path(args.lolv2_input_dir)
    lolv2_reference = workspace_path(args.lolv2_reference_dir)
    sice_snapshot = workspace_path(args.sice_snapshot_root)
    output_csv = workspace_path(args.output_csv)
    output_json = workspace_path(args.output_json)
    if not archive.is_file():
        raise RuntimeError(f"missing official LOL-v2 archive: {archive}")

    lolv2_pairs = index_lolv2_pair_dirs(lolv2_input, lolv2_reference)
    sice_sources = index_sice_test(sice_snapshot / "test" / "samples")
    all_paths = sorted(
        {
            path
            for pair in lolv2_pairs.values()
            for path in pair
        }
        | {
            path
            for label, lows in sice_sources.values()
            for path in [label, *lows]
        }
    )
    inspections = inspect_paths(all_paths)
    by_sha: dict[str, list[Path]] = {}
    for path, item in inspections.items():
        by_sha.setdefault(str(item["sha256"]), []).append(path)
    duplicate_groups = {
        digest: paths for digest, paths in by_sha.items() if len(paths) > 1
    }
    allowed_duplicate_paths = {reference for _, reference in lolv2_pairs.values()}
    unexpected_duplicates = {
        digest: paths
        for digest, paths in duplicate_groups.items()
        if any(path not in allowed_duplicate_paths for path in paths)
    }
    if unexpected_duplicates:
        raise RuntimeError("real-domain source files contain unexpected exact duplicates")

    sice_records, sice_selection = build_sice_records(sice_sources, inspections)
    lolv2_records, lolv2_reference_evidence = build_lolv2_records(
        lolv2_pairs, inspections
    )
    records = sice_records + lolv2_records
    if len({str(row["pair_id"]) for row in records}) != len(records):
        raise RuntimeError("duplicate pair ID in real-domain manifest")
    content = csv_bytes(records)
    manifest_sha = hashlib.sha256(content).hexdigest()
    inventory = sorted(inspections.values(), key=lambda item: str(item["path"]))
    subset_counts = Counter(str(row["subset"]) for row in records)
    subset_levels: dict[str, dict[str, int]] = {}
    for subset in sorted(subset_counts):
        subset_levels[subset] = dict(
            sorted(
                Counter(
                    str(row["project_level"])
                    for row in records
                    if row["subset"] == subset
                ).items()
            )
        )
    metadata = {
        "schema_version": 1,
        "status": "frozen_before_qwen_real_domain_inference",
        "scope": "non_FoundIR_real_Lowlight_VLM_validation",
        "annotation_kind": "reference_based_proxy_not_human_ground_truth",
        "annotation_warning": (
            "Project L1--L5 values are proxy labels derived from paired references; "
            "they are not official SICE or LOL-v2 severity ground truth."
        ),
        "model_predictions_consulted_during_selection": False,
        "official_foundir_used": False,
        "records": len(records),
        "manifest_csv": relative(output_csv),
        "manifest_sha256": manifest_sha,
        "record_bundle_sha256": canonical_sha256(records),
        "subset_counts": dict(sorted(subset_counts.items())),
        "subset_level_counts": subset_levels,
        "sice_selection": sice_selection,
        "sources": {
            "sice": {
                "mirror_repository": SICE_REPOSITORY,
                "mirror_revision": SICE_REVISION,
                "original_repository": SICE_ORIGINAL_REPOSITORY,
                "snapshot_root": relative(sice_snapshot),
                "held_out_sources": len(sice_sources),
                "held_out_images": sum(1 + len(lows) for _, lows in sice_sources.values()),
                "license_note": (
                    "The mirror declares Apache-2.0; the original authors' GitHub "
                    "repository does not expose a license file. Internal research use only."
                ),
            },
            "lolv2": {
                "official_url": LOLV2_OFFICIAL_URL,
                "archive": relative(archive),
                "archive_bytes": archive.stat().st_size,
                "archive_sha256": sha256(archive),
                "held_out_pairs": len(lolv2_pairs),
                "reference_duplication": lolv2_reference_evidence,
                "license_note": (
                    "The authors distribute the dataset publicly but their project "
                    "repository does not expose an explicit dataset license. Internal "
                    "research use only; cite the source papers."
                ),
            },
        },
        "source_file_count": len(inventory),
        "source_file_inventory_sha256": canonical_sha256(inventory),
        "source_files": inventory,
    }
    metadata_content = (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode()
    write_new_or_verify(output_csv, content)
    write_new_or_verify(output_json, metadata_content)
    print(
        "QWEN_REAL_VALIDATION_MANIFEST_PASS",
        f"records={len(records)}",
        f"manifest_sha256={manifest_sha}",
        f"subsets={dict(sorted(subset_counts.items()))}",
        f"levels={subset_levels}",
    )


if __name__ == "__main__":
    main()
