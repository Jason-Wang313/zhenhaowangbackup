#!/usr/bin/env python3
"""Screen the official Retinex-Net testing images before human annotation.

This tool never runs a model and never assigns severity labels.  It verifies
the downloaded archive/extraction, builds the complete protected-image union,
screens exact and 64-bit DCT-pHash overlap, groups internal duplicates and
obvious exposure series, and writes a fail-closed human-annotation package.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import stat
import sys
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from typing import Iterable

from PIL import Image, ImageStat, UnidentifiedImageError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from smoke.check_low_light_source_pool import hamming, perceptual_hash  # noqa: E402

ARCHIVE = ROOT / "data/real_lowlight_corrective_holdout/sources/Test.zip"
ARCHIVE_BYTES = 94_051_972
ARCHIVE_SHA256 = "0033eb74d4c964755f02d4b46e6db3eb457637c116415571fa2d0212577b3f29"
EXTRACTED = (
    ROOT / "data/real_lowlight_corrective_holdout/official_test_0033eb74d4c"
)
EXPECTED_SUBSET_COUNTS = {
    "DICM": 44,
    "Fusion": 18,
    "LIME": 10,
    "MEF": 79,
    "NPE": 75,
    "VV": 24,
    "low": 85,
}
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png"}
PERCEPTUAL_DISTANCE = 4
EXPECTED_PROTECTED_PATHS = 13_793
SUBSET_DISPLAY_PRIORITY = {
    "low": 0,
    "DICM": 1,
    "LIME": 2,
    "VV": 3,
    "MEF": 4,
    "Fusion": 5,
    "NPE": 6,
}
PROTECTED_SOURCES = {
    "real_adaptation": ROOT / "manifests/qwen_lowlight_real_adaptation.json",
    "prior_real_evaluation": ROOT / "manifests/qwen_lowlight_real_validation.json",
    "foundir": ROOT / "manifests/foundir_lowlight_official.json",
    "synthetic_pilot": ROOT / "manifests/low_light_pilot_10000.csv",
    "synthetic_sources": ROOT / "manifests/low_light_source_pool_2000.json",
}
PRIOR_REAL_EVALUATION_CSV = ROOT / "manifests/qwen_lowlight_real_validation.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise RuntimeError(f"path escapes workspace: {path}") from exc


def write_or_verify(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.read_bytes() != content:
            raise RuntimeError(f"refusing to overwrite differing artifact: {path}")
        return
    path.write_bytes(content)


def validate_archive_member(member: zipfile.ZipInfo) -> PurePosixPath:
    path = PurePosixPath(member.filename)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise RuntimeError(f"unsafe archive member: {member.filename!r}")
    mode = member.external_attr >> 16
    if stat.S_ISLNK(mode):
        raise RuntimeError(f"archive symlink is forbidden: {member.filename!r}")
    return path


def archive_image_members(path: Path) -> list[str]:
    if path.stat().st_size != ARCHIVE_BYTES or sha256(path) != ARCHIVE_SHA256:
        raise RuntimeError("official testing-image archive identity changed")
    names: list[str] = []
    with zipfile.ZipFile(path) as archive:
        bad_crc = archive.testzip()
        if bad_crc is not None:
            raise RuntimeError(f"bad ZIP member CRC: {bad_crc}")
        for member in archive.infolist():
            item = validate_archive_member(member)
            if member.is_dir() or item.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            if len(item.parts) != 3 or item.parts[0] != "Test":
                raise RuntimeError(f"unexpected image member layout: {item}")
            names.append(item.as_posix())
    if len(names) != len(set(names)):
        raise RuntimeError("archive has duplicate image member names")
    return sorted(names)


def inspect_image(path: Path) -> dict[str, object]:
    try:
        with Image.open(path) as handle:
            handle.load()
            original_mode = handle.mode
            rgb = handle.convert("RGB")
            width, height = rgb.size
            means = ImageStat.Stat(rgb).mean
            visual_hash = perceptual_hash(rgb)
    except (OSError, ValueError, UnidentifiedImageError) as exc:
        raise RuntimeError(f"unreadable image: {path}") from exc
    luma = 0.2126 * means[0] + 0.7152 * means[1] + 0.0722 * means[2]
    if width <= 0 or height <= 0 or not math.isfinite(luma):
        raise RuntimeError(f"invalid image geometry/luma: {path}")
    return {
        "path": relative(path),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        "width": width,
        "height": height,
        "mode": original_mode,
        "mean_luma": luma,
        "perceptual_hash": f"{visual_hash:016x}",
    }


def _add_protected(
    merged: dict[str, dict[str, object]],
    *,
    path: str,
    digest: str,
    source: str,
) -> None:
    item = merged.setdefault(path, {"path": path, "sha256": digest, "sources": []})
    if item["sha256"] != digest:
        raise RuntimeError(f"conflicting protected SHA-256 for {path}")
    sources = item["sources"]
    assert isinstance(sources, list)
    if source not in sources:
        sources.append(source)


def load_protected_inventory() -> list[dict[str, object]]:
    merged: dict[str, dict[str, object]] = {}
    adaptation = json.loads(PROTECTED_SOURCES["real_adaptation"].read_text())
    for item in adaptation["source_files"]:
        _add_protected(
            merged,
            path=str(item["path"]),
            digest=str(item["sha256"]),
            source="real_adaptation",
        )

    # Protect precisely the prior real records that were actually evaluated.
    # The acquisition inventory also contains 108 unselected SICE exposures;
    # including those would change the frozen 13,793-file protection contract.
    with PRIOR_REAL_EVALUATION_CSV.open(newline="") as handle:
        for record in csv.DictReader(handle):
            if record["source_file"] != record["degraded_path"]:
                raise RuntimeError("prior-real source_file/degraded_path changed")
            for prefix in ("clean", "degraded", "reference"):
                _add_protected(
                    merged,
                    path=record[f"{prefix}_path"],
                    digest=record[f"{prefix}_sha256"],
                    source="prior_real_evaluation",
                )

    foundir = json.loads(PROTECTED_SOURCES["foundir"].read_text())
    for record in foundir["records"]:
        for prefix in ("lq", "gt"):
            _add_protected(
                merged,
                path=str(record[f"{prefix}_path"]),
                digest=str(record[f"{prefix}_sha256"]),
                source="foundir",
            )

    with PROTECTED_SOURCES["synthetic_pilot"].open(newline="") as handle:
        for record in csv.DictReader(handle):
            _add_protected(
                merged,
                path=record["degraded_path"],
                digest=record["degraded_sha256"],
                source="synthetic_pilot",
            )

    sources = json.loads(PROTECTED_SOURCES["synthetic_sources"].read_text())
    for record in sources["records"]:
        _add_protected(
            merged,
            path=str(record["path"]),
            digest=str(record["sha256"]),
            source="synthetic_sources",
        )

    items = sorted(merged.values(), key=lambda item: str(item["path"]))
    if len(items) != EXPECTED_PROTECTED_PATHS:
        raise RuntimeError(
            f"protected union changed: {len(items)} != {EXPECTED_PROTECTED_PATHS}"
        )
    return items


def inspect_protected(item: dict[str, object]) -> dict[str, object]:
    path = ROOT / str(item["path"])
    if not path.is_file():
        raise RuntimeError(f"missing protected image: {path}")
    inspected = inspect_image(path)
    if inspected["sha256"] != item["sha256"]:
        raise RuntimeError(f"protected image SHA-256 changed: {path}")
    return {**item, "perceptual_hash": inspected["perceptual_hash"]}


def inspect_candidates(extracted: Path, names: list[str]) -> list[dict[str, object]]:
    expected_paths = {(extracted / name).resolve() for name in names}
    actual_paths = {
        path.resolve()
        for path in extracted.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }
    if expected_paths != actual_paths:
        missing = sorted(str(path) for path in expected_paths - actual_paths)
        extra = sorted(str(path) for path in actual_paths - expected_paths)
        raise RuntimeError(f"extraction mismatch: missing={missing[:3]} extra={extra[:3]}")
    with ThreadPoolExecutor(max_workers=4) as executor:
        inspected = list(executor.map(inspect_image, sorted(actual_paths)))
    for item in inspected:
        parts = Path(str(item["path"])).parts
        item["subset"] = parts[-2]
        item["filename"] = parts[-1]
    counts = Counter(str(item["subset"]) for item in inspected)
    if dict(sorted(counts.items())) != EXPECTED_SUBSET_COUNTS:
        raise RuntimeError(f"testing-image subset counts changed: {dict(counts)}")
    return sorted(inspected, key=lambda item: str(item["path"]))


class DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, index: int) -> int:
        while self.parent[index] != index:
            self.parent[index] = self.parent[self.parent[index]]
            index = self.parent[index]
        return index

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


def exposure_group_key(item: dict[str, object]) -> str | None:
    """Return only conservative filename-based exposure/encoding group keys."""
    subset = str(item["subset"])
    stem = Path(str(item["filename"])).stem
    lowered = stem.lower()
    if subset in {"low", "NPE"}:
        match = re.fullmatch(r"(cloudy \(\d+\))(?:_(?:bmp|jpeg|jpg))?", lowered)
        if match:
            return f"cloudy-encoding:{match.group(1)}"
    if subset == "Fusion" and lowered in {"low20", "high20"}:
        return "fusion-exposure:20"
    if subset != "MEF":
        return None
    patterns = (
        r"(belg)\d+",
        r"(cave)\d+",
        r"(chinese_garden)\d+",
        r"(dsc0364)\d",
        r"(dsc_016)\d",
        r"(imgp636)\d",
        r"(kluki)\d+",
        r"(lampicka)\d+",
        r"(memorial)\d+",
        r"(office_)\d+",
    )
    for pattern in patterns:
        match = re.fullmatch(pattern, lowered)
        if match:
            return f"mef-exposure:{match.group(1)}"
    match = re.fullmatch(r"(.+?)_(normal|under)", lowered)
    if match:
        return f"mef-exposure:{match.group(1)}"
    return None


def build_components(candidates: list[dict[str, object]]) -> list[dict[str, object]]:
    disjoint = DisjointSet(len(candidates))
    exact: dict[str, int] = {}
    exposure: dict[str, int] = {}
    for index, item in enumerate(candidates):
        digest = str(item["sha256"])
        if digest in exact:
            disjoint.union(index, exact[digest])
        else:
            exact[digest] = index
        group = exposure_group_key(item)
        if group is not None:
            if group in exposure:
                disjoint.union(index, exposure[group])
            else:
                exposure[group] = index
    hashes = [int(str(item["perceptual_hash"]), 16) for item in candidates]
    for right in range(len(candidates)):
        for left in range(right):
            if hamming(hashes[left], hashes[right]) <= PERCEPTUAL_DISTANCE:
                disjoint.union(left, right)

    grouped: dict[int, list[int]] = defaultdict(list)
    for index in range(len(candidates)):
        grouped[disjoint.find(index)].append(index)
    components: list[dict[str, object]] = []
    for indices in grouped.values():
        members = [candidates[index] for index in indices]
        paths = sorted(str(item["path"]) for item in members)
        evidence = {
            "paths": paths,
            "sha256": sorted({str(item["sha256"]) for item in members}),
            "perceptual_hashes": sorted(
                {str(item["perceptual_hash"]) for item in members}
            ),
            "exposure_group_keys": sorted(
                {key for item in members if (key := exposure_group_key(item))}
            ),
        }
        digest = canonical_sha256(evidence)
        display = min(
            members,
            key=lambda item: (
                SUBSET_DISPLAY_PRIORITY[str(item["subset"])], str(item["path"])
            ),
        )
        components.append(
            {
                "component_id": f"official_test_scene_{digest[:16]}",
                "component_sha256": digest,
                "display_path": display["path"],
                "members": paths,
                "exact_unique_images": len(evidence["sha256"]),
                "subsets": sorted({str(item["subset"]) for item in members}),
                "exposure_group_keys": evidence["exposure_group_keys"],
            }
        )
    return sorted(components, key=lambda item: str(item["component_id"]))


def screen_overlaps(
    candidates: list[dict[str, object]], protected: list[dict[str, object]]
) -> None:
    protected_by_sha: dict[str, list[dict[str, object]]] = defaultdict(list)
    for item in protected:
        protected_by_sha[str(item["sha256"])].append(item)
    protected_hashes = [int(str(item["perceptual_hash"]), 16) for item in protected]
    for candidate in candidates:
        candidate_hash = int(str(candidate["perceptual_hash"]), 16)
        exact_matches = protected_by_sha.get(str(candidate["sha256"]), [])
        perceptual_matches = []
        minimum = 64
        for index, protected_hash in enumerate(protected_hashes):
            distance = hamming(candidate_hash, protected_hash)
            minimum = min(minimum, distance)
            if distance <= PERCEPTUAL_DISTANCE:
                item = protected[index]
                perceptual_matches.append(
                    {
                        "path": item["path"],
                        "sources": item["sources"],
                        "hamming_distance": distance,
                    }
                )
        candidate["protected_exact_matches"] = [
            {"path": item["path"], "sources": item["sources"]}
            for item in exact_matches
        ]
        candidate["protected_perceptual_matches"] = sorted(
            perceptual_matches,
            key=lambda item: (int(item["hamming_distance"]), str(item["path"])),
        )
        candidate["minimum_protected_phash_distance"] = minimum


def build_annotation_rows(
    candidates: list[dict[str, object]], components: list[dict[str, object]]
) -> tuple[list[dict[str, object]], set[str]]:
    by_path = {str(item["path"]): item for item in candidates}
    rejected_components: set[str] = set()
    rows: list[dict[str, object]] = []
    for component in components:
        members = [by_path[str(path)] for path in component["members"]]
        protected_paths = sorted(
            {
                str(match["path"])
                for item in members
                for match in item["protected_perceptual_matches"]
            }
        )
        if protected_paths:
            rejected_components.add(str(component["component_id"]))
            component["screening_status"] = "rejected_protected_scene_overlap"
            component["protected_match_paths"] = protected_paths
            continue
        component["screening_status"] = "pending_human_annotation"
        component["protected_match_paths"] = []
        rows.append(
            {
                "component_id": component["component_id"],
                "display_path": component["display_path"],
                "candidate_count": len(members),
                "candidate_paths_json": json.dumps(
                    component["members"], ensure_ascii=False, separators=(",", ":")
                ),
                "subsets_json": json.dumps(
                    component["subsets"], ensure_ascii=False, separators=(",", ":")
                ),
                "mean_lumas_json": json.dumps(
                    {
                        str(item["path"]): round(float(item["mean_luma"]), 6)
                        for item in members
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "selected_path": "",
                "valid_natural_lowlight_input": "",
                "project_level": "",
                "rater_id": "",
                "confidence": "",
                "notes": "",
            }
        )
    return rows, rejected_components


def csv_bytes(rows: Iterable[dict[str, object]], fieldnames: list[str]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--extracted", type=Path, default=EXTRACTED)
    parser.add_argument(
        "--manifest-json",
        type=Path,
        default=ROOT / "manifests/qwen_lowlight_corrective_holdout_candidates.json",
    )
    parser.add_argument(
        "--manifest-csv",
        type=Path,
        default=ROOT / "manifests/qwen_lowlight_corrective_holdout_candidates.csv",
    )
    parser.add_argument(
        "--annotation-csv",
        type=Path,
        default=ROOT / "manifests/qwen_lowlight_corrective_holdout_annotation.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    names = archive_image_members(args.archive)
    candidates = inspect_candidates(args.extracted, names)
    protected = load_protected_inventory()
    with ThreadPoolExecutor(max_workers=4) as executor:
        protected = list(executor.map(inspect_protected, protected))
    screen_overlaps(candidates, protected)
    components = build_components(candidates)
    annotation_rows, rejected_components = build_annotation_rows(candidates, components)
    component_by_path = {
        str(path): component
        for component in components
        for path in component["members"]
    }
    for item in candidates:
        component = component_by_path[str(item["path"])]
        item["component_id"] = component["component_id"]
        item["screening_status"] = component["screening_status"]

    manifest_sources = {
        name: {
            "path": relative(path),
            "sha256": sha256(path),
        }
        for name, path in PROTECTED_SOURCES.items()
    }
    manifest_sources["prior_real_evaluation_csv"] = {
        "path": relative(PRIOR_REAL_EVALUATION_CSV),
        "sha256": sha256(PRIOR_REAL_EVALUATION_CSV),
    }
    protected_bundle = [
        {
            "path": item["path"],
            "sha256": item["sha256"],
            "perceptual_hash": item["perceptual_hash"],
            "sources": sorted(item["sources"]),
        }
        for item in protected
    ]
    payload = {
        "schema_version": 1,
        "status": "screened_unlabeled_no_model_inference",
        "scope": "Lowlight VLM corrective holdout candidate acquisition",
        "source": {
            "official_page": "https://daooshee.github.io/BMVC2018website/",
            "official_page_label": "Testing Images",
            "archive_path": relative(args.archive),
            "archive_bytes": args.archive.stat().st_size,
            "archive_sha256": sha256(args.archive),
            "extracted_root": relative(args.extracted),
        },
        "overlap_policy": {
            "exact": "SHA-256",
            "perceptual": "64-bit DCT pHash",
            "maximum_hamming_distance": PERCEPTUAL_DISTANCE,
            "component_wide_rejection": True,
        },
        "protected_inventory": {
            "unique_paths": len(protected),
            "bundle_sha256": canonical_sha256(protected_bundle),
            "source_manifests": manifest_sources,
            "source_membership_counts": dict(
                sorted(
                    Counter(
                        source for item in protected for source in item["sources"]
                    ).items()
                )
            ),
        },
        "candidate_counts": {
            "archive_images": len(candidates),
            "subsets": EXPECTED_SUBSET_COUNTS,
            "exact_unique_images": len({item["sha256"] for item in candidates}),
            "scene_components": len(components),
            "protected_overlapping_images": sum(
                bool(item["protected_perceptual_matches"]) for item in candidates
            ),
            "rejected_scene_components": len(rejected_components),
            "pending_human_annotation_components": len(annotation_rows),
        },
        "annotation_contract": {
            "annotation_kind": "pending_human_labels_not_model_or_proxy_labels",
            "one_selected_image_maximum_per_scene_component": True,
            "model_inference_performed": False,
            "official_foundir_inference_performed": False,
            "consumed_real_test_inference_performed": False,
        },
        "components": components,
        "candidates": candidates,
    }
    json_content = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    candidate_fields = [
        "path",
        "subset",
        "filename",
        "sha256",
        "perceptual_hash",
        "bytes",
        "width",
        "height",
        "mode",
        "mean_luma",
        "minimum_protected_phash_distance",
        "component_id",
        "screening_status",
        "protected_exact_match_count",
        "protected_perceptual_match_count",
    ]
    candidate_rows = []
    for item in candidates:
        candidate_rows.append(
            {
                **{field: item.get(field, "") for field in candidate_fields},
                "protected_exact_match_count": len(item["protected_exact_matches"]),
                "protected_perceptual_match_count": len(
                    item["protected_perceptual_matches"]
                ),
            }
        )
    annotation_fields = [
        "component_id",
        "display_path",
        "candidate_count",
        "candidate_paths_json",
        "subsets_json",
        "mean_lumas_json",
        "selected_path",
        "valid_natural_lowlight_input",
        "project_level",
        "rater_id",
        "confidence",
        "notes",
    ]
    write_or_verify(args.manifest_json, json_content)
    write_or_verify(args.manifest_csv, csv_bytes(candidate_rows, candidate_fields))
    write_or_verify(args.annotation_csv, csv_bytes(annotation_rows, annotation_fields))
    print(
        "CORRECTIVE_HOLDOUT_SCREEN_PASS "
        f"candidates={len(candidates)} components={len(components)} "
        f"protected_overlap_images={payload['candidate_counts']['protected_overlapping_images']} "
        f"annotation_components={len(annotation_rows)}"
    )


if __name__ == "__main__":
    main()
