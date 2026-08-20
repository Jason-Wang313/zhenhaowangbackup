#!/usr/bin/env python3
"""Build the immutable source-disjoint LOL-v2 real-adaptation manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageStat, UnidentifiedImageError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.qwen_real_validation import (  # noqa: E402
    PROJECT_LUMA_RATIO_ANCHORS,
    proxy_anchor_error_ev,
    proxy_level_from_luma_ratio,
)
from lowlight_restoration.qwen_supervised import canonical_target  # noqa: E402
from smoke.check_low_light_source_pool import hamming  # noqa: E402
from smoke.check_low_light_source_pool import perceptual_hash

ARCHIVE_SHA256 = "9820d8b112438d94d1f5d4d25817eee618a8cf4bc63a65cd6b19f5a98c3faefa"
ARCHIVE_BYTES = 1_046_491_030
OBSERVED_METADATA_SHA256 = (
    "3b9be6683208580f51ddb93779436270b28e97ab4b27a2df720bfc6238cdd1fb"
)
EXPECTED_IDS = {f"{value:05d}" for value in range(1, 690)}
SPLIT_SEED = 2026082001
PERCEPTUAL_DISTANCE = 4
SPLIT_ORDER = ("train", "validation", "test")
SPLIT_TARGETS = {
    "train": {1: 190, 2: 14, 3: 18, 4: 52, 5: 399},
    "validation": {1: 41, 2: 3, 3: 3, 4: 11, 5: 85},
    "test": {1: 41, 2: 3, 3: 4, 4: 11, 5: 86},
}
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
        raise RuntimeError(f"path escapes workspace: {path}") from exc
    return resolved


def relative(path: Path) -> str:
    return str(workspace_path(path).relative_to(ROOT.resolve()))


def inspect_image(path: Path, *, include_perceptual_hash: bool) -> dict[str, object]:
    try:
        with Image.open(path) as handle:
            handle.load()
            original_mode = handle.mode
            rgb = handle.convert("RGB")
            width, height = rgb.size
            means = ImageStat.Stat(rgb).mean
            visual_hash = perceptual_hash(rgb) if include_perceptual_hash else None
    except (OSError, ValueError, UnidentifiedImageError) as exc:
        raise RuntimeError(f"unreadable LOL-v2 training image: {path}") from exc
    luma = 0.2126 * means[0] + 0.7152 * means[1] + 0.0722 * means[2]
    if (width, height) != (600, 400) or not math.isfinite(luma) or luma <= 0.0:
        raise RuntimeError(f"unexpected LOL-v2 training image geometry/luma: {path}")
    return {
        "path": relative(path),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        "width": width,
        "height": height,
        "original_mode": original_mode,
        "mean_luma": luma,
        "perceptual_hash": (None if visual_hash is None else f"{visual_hash:016x}"),
    }


def index_images(
    directory: Path, *, prefix: str, expected_ids: set[str]
) -> dict[str, Path]:
    if not directory.is_dir():
        raise RuntimeError(f"missing LOL-v2 directory: {directory}")
    result: dict[str, Path] = {}
    files = [
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    for path in files:
        match = re.fullmatch(rf"{re.escape(prefix)}(\d{{5}})", path.stem)
        if match is None or match.group(1) in result:
            raise RuntimeError(f"bad or duplicate LOL-v2 filename: {path}")
        result[match.group(1)] = path
    if set(result) != expected_ids or len(result) != len(files):
        raise RuntimeError(
            f"LOL-v2 ID set changed for {directory}: "
            f"missing={sorted(expected_ids - set(result))[:5]} "
            f"extra={sorted(set(result) - expected_ids)[:5]}"
        )
    return result


def inspect_training_pairs(
    low: dict[str, Path], normal: dict[str, Path]
) -> dict[Path, dict[str, object]]:
    paths = [*low.values(), *normal.values()]
    if len(paths) != 1_378 or len(paths) != len(set(paths)):
        raise RuntimeError("LOL-v2 training pair paths are incomplete or duplicated")

    def inspect(path: Path) -> tuple[Path, dict[str, object]]:
        return path, inspect_image(
            path, include_perceptual_hash=path in normal.values()
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        return dict(executor.map(inspect, paths))


def scene_components(
    normal: dict[str, Path], inspections: dict[Path, dict[str, object]]
) -> tuple[list[dict[str, object]], dict[str, object]]:
    by_exact: dict[str, list[str]] = defaultdict(list)
    for image_id, path in sorted(normal.items()):
        by_exact[str(inspections[path]["sha256"])].append(image_id)
    nodes: list[dict[str, object]] = []
    for digest, ids in sorted(by_exact.items()):
        hashes = {
            str(inspections[normal[image_id]]["perceptual_hash"]) for image_id in ids
        }
        if len(hashes) != 1:
            raise RuntimeError(
                "exact Normal duplicates have different perceptual hashes"
            )
        nodes.append(
            {
                "exact_sha256": digest,
                "ids": sorted(ids),
                "perceptual_hash": next(iter(hashes)),
            }
        )

    parent = list(range(len(nodes)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    edges: list[dict[str, object]] = []
    for right in range(len(nodes)):
        right_hash = int(str(nodes[right]["perceptual_hash"]), 16)
        for left in range(right):
            distance = hamming(int(str(nodes[left]["perceptual_hash"]), 16), right_hash)
            if distance <= PERCEPTUAL_DISTANCE:
                union(left, right)
                edges.append(
                    {
                        "left_exact_sha256": nodes[left]["exact_sha256"],
                        "right_exact_sha256": nodes[right]["exact_sha256"],
                        "hamming_distance": distance,
                    }
                )

    grouped: dict[int, list[int]] = defaultdict(list)
    for index in range(len(nodes)):
        grouped[find(index)].append(index)
    components: list[dict[str, object]] = []
    for indices in grouped.values():
        ids = sorted(image_id for index in indices for image_id in nodes[index]["ids"])
        numeric = [int(image_id) for image_id in ids]
        if max(numeric) - min(numeric) + 1 > len(numeric) + 3:
            raise RuntimeError(
                "perceptual scene component spans nonlocal LOL-v2 IDs: " f"{ids}"
            )
        component_evidence = {
            "normal_ids": ids,
            "exact_sha256": sorted(
                str(nodes[index]["exact_sha256"]) for index in indices
            ),
            "perceptual_hashes": sorted(
                str(nodes[index]["perceptual_hash"]) for index in indices
            ),
        }
        component_digest = canonical_sha256(component_evidence)
        components.append(
            {
                **component_evidence,
                "component_sha256": component_digest,
                "source_id": (
                    f"lolv2_real_train_scene_{ids[0]}_{component_digest[:12]}"
                ),
            }
        )
    components.sort(key=lambda item: str(item["source_id"]))
    ids = [image_id for item in components for image_id in item["normal_ids"]]
    if len(components) != 272 or sorted(ids) != sorted(EXPECTED_IDS):
        raise RuntimeError("LOL-v2 scene-component inventory changed")
    sizes = Counter(len(item["normal_ids"]) for item in components)
    evidence = {
        "method": "exact_SHA256_union_64bit_DCT_pHash_hamming_le_4",
        "perceptual_hash_bits": 64,
        "maximum_hamming_distance": PERCEPTUAL_DISTANCE,
        "exact_unique_normal_images": len(nodes),
        "exact_duplicate_normal_files": len(normal) - len(nodes),
        "exact_duplicate_groups": sum(len(node["ids"]) > 1 for node in nodes),
        "perceptual_edges": len(edges),
        "perceptual_edge_bundle_sha256": canonical_sha256(edges),
        "scene_components": len(components),
        "scene_component_size_distribution": {
            str(size): count for size, count in sorted(sizes.items())
        },
        "maximum_scene_component_size": max(sizes),
        "all_components_local_in_official_numeric_sequence": True,
    }
    return components, evidence


def assign_proxy_levels(
    low: dict[str, Path],
    normal: dict[str, Path],
    inspections: dict[Path, dict[str, object]],
) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for image_id in sorted(low):
        low_item = inspections[low[image_id]]
        normal_item = inspections[normal[image_id]]
        ratio = float(low_item["mean_luma"]) / float(normal_item["mean_luma"])
        level = proxy_level_from_luma_ratio(ratio)
        result[image_id] = {
            "level": level,
            "ratio": ratio,
            "anchor": PROJECT_LUMA_RATIO_ANCHORS[level],
            "anchor_error_ev": proxy_anchor_error_ev(ratio, level),
        }
    if Counter(int(item["level"]) for item in result.values()) != Counter(
        {2: 20, 3: 25, 4: 74, 5: 570}
    ):
        raise RuntimeError("LOL-v2 training proxy-level distribution changed")
    return result


def split_components(
    components: list[dict[str, object]], levels: dict[str, dict[str, object]]
) -> tuple[dict[str, str], dict[str, object]]:
    groups: list[dict[str, object]] = []
    for component in components:
        ids = list(component["normal_ids"])
        counts = Counter(int(levels[image_id]["level"]) for image_id in ids)
        vector = [1, *(counts[level] for level in range(2, 6))]
        groups.append({**component, "vector": vector})
    total = [sum(int(group["vector"][index]) for group in groups) for index in range(5)]
    if total != [272, 20, 25, 74, 570]:
        raise RuntimeError("scene-component severity totals changed")
    caps = [SPLIT_TARGETS[split][1] for split in SPLIT_ORDER]
    targets = [
        [SPLIT_TARGETS[split][level] for level in range(1, 6)] for split in SPLIT_ORDER
    ]
    groups.sort(
        key=lambda group: (
            -sum(int(group["vector"][index]) / total[index] for index in range(1, 5)),
            -sum(int(value) for value in group["vector"]),
            hashlib.sha256(f"{SPLIT_SEED}:{group['source_id']}".encode()).hexdigest(),
        )
    )
    assigned: list[list[dict[str, object]]] = [[] for _ in SPLIT_ORDER]
    current = [[0] * 5 for _ in SPLIT_ORDER]
    for group in groups:
        choices: list[tuple[float, int]] = []
        vector = [int(value) for value in group["vector"]]
        for split_index in range(len(SPLIT_ORDER)):
            if current[split_index][0] >= caps[split_index]:
                continue
            realized = [
                current[split_index][dimension] + vector[dimension]
                for dimension in range(5)
            ]
            score = 0.0
            for dimension in range(5):
                denominator = max(targets[split_index][dimension], 1)
                difference = (
                    realized[dimension] - targets[split_index][dimension]
                ) / denominator
                score += difference * difference * (10.0 if difference > 0 else 1.0)
            score += (realized[0] / caps[split_index]) ** 2 * 0.01
            choices.append((score, split_index))
        if not choices:
            raise RuntimeError("no capacity remains during scene split")
        split_index = min(choices)[1]
        assigned[split_index].append(group)
        current[split_index] = [
            current[split_index][dimension] + vector[dimension]
            for dimension in range(5)
        ]

    def objective(values: list[list[int]]) -> float:
        return sum(
            (
                (values[split_index][dimension] - targets[split_index][dimension])
                / max(targets[split_index][dimension], 1)
            )
            ** 2
            for split_index in range(len(SPLIT_ORDER))
            for dimension in range(1, 5)
        )

    rng = random.Random(SPLIT_SEED)
    score = objective(current)
    accepted_swaps = 0
    iterations = 0
    for iterations in range(1, 500_001):
        if current == targets:
            break
        left, right = rng.sample(range(len(SPLIT_ORDER)), 2)
        left_index = rng.randrange(len(assigned[left]))
        right_index = rng.randrange(len(assigned[right]))
        left_group = assigned[left][left_index]
        right_group = assigned[right][right_index]
        left_vector = [int(value) for value in left_group["vector"]]
        right_vector = [int(value) for value in right_group["vector"]]
        candidate = [list(values) for values in current]
        candidate[left] = [
            current[left][dimension] - left_vector[dimension] + right_vector[dimension]
            for dimension in range(5)
        ]
        candidate[right] = [
            current[right][dimension] - right_vector[dimension] + left_vector[dimension]
            for dimension in range(5)
        ]
        candidate_score = objective(candidate)
        if candidate_score < score - 1e-15:
            assigned[left][left_index], assigned[right][right_index] = (
                right_group,
                left_group,
            )
            current = candidate
            score = candidate_score
            accepted_swaps += 1
    if current != targets:
        raise RuntimeError(
            "deterministic scene split could not realize the frozen targets: "
            f"{current}"
        )
    source_to_split: dict[str, str] = {}
    for split, values in zip(SPLIT_ORDER, assigned, strict=True):
        for group in values:
            source = str(group["source_id"])
            if source in source_to_split:
                raise RuntimeError(f"scene assigned twice: {source}")
            source_to_split[source] = split
    evidence = {
        "seed": SPLIT_SEED,
        "method": "deterministic_greedy_then_strictly_improving_cross_split_swaps",
        "optimizer_iterations": iterations,
        "accepted_swaps": accepted_swaps,
        "objective": score,
        "exact_targets_realized": True,
        "targets": {
            split: {f"L{level}": SPLIT_TARGETS[split][level] for level in range(1, 6)}
            for split in SPLIT_ORDER
        },
        "source_assignment_sha256": canonical_sha256(source_to_split),
    }
    return source_to_split, evidence


def verify_observed_separation(
    *,
    training_inspections: dict[Path, dict[str, object]],
    normal: dict[str, Path],
    observed_metadata: Path,
    test_normal_dir: Path,
) -> dict[str, object]:
    if sha256(observed_metadata) != OBSERVED_METADATA_SHA256:
        raise RuntimeError("previously evaluated real-source metadata changed")
    observed = json.loads(observed_metadata.read_text())
    source_files = observed.get("source_files")
    if not isinstance(source_files, list):
        raise RuntimeError("observed real-source inventory is missing")
    observed_hashes = {
        str(item["sha256"])
        for item in source_files
        if isinstance(item, dict) and isinstance(item.get("sha256"), str)
    }
    training_hashes = {str(item["sha256"]) for item in training_inspections.values()}
    overlap = training_hashes & observed_hashes
    if overlap:
        raise RuntimeError("LOL-v2 adaptation data overlaps an evaluated real image")

    test_normal = index_images(
        test_normal_dir,
        prefix="normal",
        expected_ids={f"{value:05d}" for value in range(690, 790)},
    )
    unique_test_hashes: dict[str, int] = {}
    for path in test_normal.values():
        digest = sha256(path)
        if digest not in unique_test_hashes:
            item = inspect_image(path, include_perceptual_hash=True)
            unique_test_hashes[digest] = int(str(item["perceptual_hash"]), 16)
    train_nodes: dict[str, int] = {}
    for path in normal.values():
        item = training_inspections[path]
        train_nodes.setdefault(
            str(item["sha256"]), int(str(item["perceptual_hash"]), 16)
        )
    near_edges = [
        (train_sha, test_sha, hamming(train_hash, test_hash))
        for train_sha, train_hash in train_nodes.items()
        for test_sha, test_hash in unique_test_hashes.items()
        if hamming(train_hash, test_hash) <= PERCEPTUAL_DISTANCE
    ]
    if near_edges:
        raise RuntimeError("LOL-v2 Train/Test Normal captures overlap perceptually")
    return {
        "observed_metadata": relative(observed_metadata),
        "observed_metadata_sha256": OBSERVED_METADATA_SHA256,
        "observed_source_files": len(source_files),
        "observed_unique_sha256": len(observed_hashes),
        "training_source_files": len(training_inspections),
        "exact_sha256_overlap": 0,
        "train_unique_normal_sha256": len(train_nodes),
        "test_unique_normal_sha256": len(unique_test_hashes),
        "train_test_normal_perceptual_edges_hamming_le_4": 0,
        "previously_evaluated_records_used_for_training": False,
    }


def record_bundle_sha256(records: Iterable[dict[str, object]]) -> str:
    payload = [
        {
            "degraded_sha256": record["degraded_sha256"],
            "level": int(str(record["project_level"])[1:]),
            "pair_id": record["pair_id"],
            "source_id": record["source_id"],
            "target": canonical_target(int(str(record["project_level"])[1:])),
        }
        for record in records
    ]
    return canonical_sha256(payload)


def build_records(
    *,
    components: list[dict[str, object]],
    source_to_split: dict[str, str],
    levels: dict[str, dict[str, object]],
    low: dict[str, Path],
    normal: dict[str, Path],
    inspections: dict[Path, dict[str, object]],
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for component in components:
        source_id = str(component["source_id"])
        split = source_to_split[source_id]
        ids = list(component["normal_ids"])
        canonical_id = ids[0]
        canonical_path = normal[canonical_id]
        canonical_item = inspections[canonical_path]
        records.append(
            {
                "pair_id": f"{source_id}_normal_L1",
                "source_id": source_id,
                "clean_path": canonical_item["path"],
                "degraded_path": canonical_item["path"],
                "degradation_type": "low_light",
                "project_level": "L1",
                "seed": SPLIT_SEED,
                "split": split,
                "clean_sha256": canonical_item["sha256"],
                "degraded_sha256": canonical_item["sha256"],
                "dataset": "LOL-v2-real-captured-train",
                "annotation_kind": "reference_based_proxy_not_human_ground_truth",
                "annotation_method": "perceptual_scene_component_canonical_Normal_proxy_clean",
                "official_pair_id": canonical_id,
                "scene_component_sha256": component["component_sha256"],
                "reference_path": canonical_item["path"],
                "reference_sha256": canonical_item["sha256"],
                "input_reference_luma_ratio": 1.0,
                "proxy_anchor": 1.0,
                "proxy_anchor_error_ev": 0.0,
            }
        )
        for image_id in ids:
            low_item = inspections[low[image_id]]
            normal_item = inspections[normal[image_id]]
            proxy = levels[image_id]
            level = int(proxy["level"])
            records.append(
                {
                    "pair_id": f"lolv2_real_train_{image_id}_input_L{level}",
                    "source_id": source_id,
                    "clean_path": normal_item["path"],
                    "degraded_path": low_item["path"],
                    "degradation_type": "low_light",
                    "project_level": f"L{level}",
                    "seed": SPLIT_SEED,
                    "split": split,
                    "clean_sha256": normal_item["sha256"],
                    "degraded_sha256": low_item["sha256"],
                    "dataset": "LOL-v2-real-captured-train",
                    "annotation_kind": "reference_based_proxy_not_human_ground_truth",
                    "annotation_method": (
                        "nearest_frozen_project_luma_ratio_anchor_after_" "LOLv2_Normal"
                    ),
                    "official_pair_id": image_id,
                    "scene_component_sha256": component["component_sha256"],
                    "reference_path": normal_item["path"],
                    "reference_sha256": normal_item["sha256"],
                    "input_reference_luma_ratio": proxy["ratio"],
                    "proxy_anchor": proxy["anchor"],
                    "proxy_anchor_error_ev": proxy["anchor_error_ev"],
                }
            )
    split_rank = {name: index for index, name in enumerate(SPLIT_ORDER)}
    records.sort(
        key=lambda record: (
            split_rank[str(record["split"])],
            str(record["source_id"]),
            str(record["project_level"]),
            str(record["pair_id"]),
        )
    )
    if (
        len(records) != 961
        or len({str(record["pair_id"]) for record in records}) != 961
    ):
        raise RuntimeError("real-adaptation record cardinality changed")
    return records


def csv_bytes(records: list[dict[str, object]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=list(records[0]),
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(records)
    return buffer.getvalue().encode()


def write_new_or_verify(path: Path, content: bytes) -> None:
    if path.exists():
        if not path.is_file() or path.read_bytes() != content:
            raise RuntimeError(
                f"existing immutable adaptation manifest differs: {path}"
            )
        print(f"VERIFIED_EXISTING_OUTPUT {relative(path)}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(content)
    print(f"WROTE_NEW_OUTPUT {relative(path)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--real-root", type=Path, required=True)
    parser.add_argument("--observed-metadata", type=Path, required=True)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=ROOT / "manifests/qwen_lowlight_real_adaptation.csv",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=ROOT / "manifests/qwen_lowlight_real_adaptation.json",
    )
    args = parser.parse_args()

    archive = workspace_path(args.archive)
    real_root = workspace_path(args.real_root)
    observed_metadata = workspace_path(args.observed_metadata)
    output_csv = workspace_path(args.output_csv)
    output_json = workspace_path(args.output_json)
    if (
        not archive.is_file()
        or archive.stat().st_size != ARCHIVE_BYTES
        or sha256(archive) != ARCHIVE_SHA256
    ):
        raise RuntimeError("verified official LOL-v2 archive changed")
    low = index_images(
        real_root / "Train" / "Low", prefix="low", expected_ids=EXPECTED_IDS
    )
    normal = index_images(
        real_root / "Train" / "Normal", prefix="normal", expected_ids=EXPECTED_IDS
    )
    inspections = inspect_training_pairs(low, normal)
    if len({str(item["sha256"]) for item in inspections.values()}) != 1_288:
        raise RuntimeError("LOL-v2 training exact-duplicate inventory changed")
    components, component_evidence = scene_components(normal, inspections)
    levels = assign_proxy_levels(low, normal, inspections)
    source_to_split, split_evidence = split_components(components, levels)
    separation = verify_observed_separation(
        training_inspections=inspections,
        normal=normal,
        observed_metadata=observed_metadata,
        test_normal_dir=real_root / "Test" / "Normal",
    )
    records = build_records(
        components=components,
        source_to_split=source_to_split,
        levels=levels,
        low=low,
        normal=normal,
        inspections=inspections,
    )
    csv_content = csv_bytes(records)
    manifest_sha = hashlib.sha256(csv_content).hexdigest()
    split_contracts: dict[str, dict[str, object]] = {}
    for split in SPLIT_ORDER:
        selected = [record for record in records if record["split"] == split]
        per_level = Counter(str(record["project_level"]) for record in selected)
        split_contracts[split] = {
            "records": len(selected),
            "sources": len({str(record["source_id"]) for record in selected}),
            "records_per_level": {
                str(level): per_level[f"L{level}"] for level in range(1, 6)
            },
            "bundle_sha256": record_bundle_sha256(selected),
        }
    expected_contracts = {
        split: {str(level): SPLIT_TARGETS[split][level] for level in range(1, 6)}
        for split in SPLIT_ORDER
    }
    if any(
        split_contracts[split]["records_per_level"] != expected_contracts[split]
        for split in SPLIT_ORDER
    ):
        raise RuntimeError("real-adaptation split records differ from frozen targets")
    inventory = sorted(inspections.values(), key=lambda item: str(item["path"]))
    ratios = [float(item["ratio"]) for item in levels.values()]
    errors = [float(item["anchor_error_ev"]) for item in levels.values()]
    metadata = {
        "schema_version": 1,
        "status": "frozen_before_qwen_real_adaptation_inference",
        "scope": "LOL-v2_Real_captured_Train_source_disjoint_real_adaptation",
        "annotation_kind": "reference_based_proxy_not_human_ground_truth",
        "annotation_warning": (
            "Project L1--L5 values are proxy labels derived from paired Normal "
            "references; they are not official LOL-v2 severity ground truth."
        ),
        "model_predictions_consulted_during_split": False,
        "previously_evaluated_real_records_used": False,
        "official_foundir_used": False,
        "manifest_csv": relative(output_csv),
        "manifest_sha256": manifest_sha,
        "record_bundle_sha256": record_bundle_sha256(records),
        "records": len(records),
        "scene_sources": len(components),
        "split_contracts": split_contracts,
        "holdout_protection": {
            "split": "test",
            "used_for_training": False,
            "used_for_checkpoint_selection": False,
            "evaluate_once_after_checkpoint_selection": True,
            "predictions_consulted_before_manifest_freeze": False,
        },
        "proxy_levels": {
            "anchors": {
                str(level): value for level, value in PROJECT_LUMA_RATIO_ANCHORS.items()
            },
            "method": "nearest_anchor_in_absolute_log2_EV_distance_lower_level_tiebreak",
            "low_input_counts": {
                str(level): Counter(int(item["level"]) for item in levels.values())[
                    level
                ]
                for level in range(2, 6)
            },
            "minimum_input_reference_luma_ratio": min(ratios),
            "maximum_input_reference_luma_ratio": max(ratios),
            "minimum_anchor_error_ev": min(errors),
            "maximum_anchor_error_ev": max(errors),
            "L5_is_saturating_extreme_class_below_anchor": True,
        },
        "scene_grouping": component_evidence,
        "split_assignment": split_evidence,
        "separation_from_previously_evaluated_real_data": separation,
        "source": {
            "archive": relative(archive),
            "archive_bytes": ARCHIVE_BYTES,
            "archive_sha256": ARCHIVE_SHA256,
            "real_root": relative(real_root),
            "official_pair_ids": ["00001", "00689"],
            "pairs": 689,
            "low_files": 689,
            "normal_files": 689,
            "license_note": (
                "The authors distribute LOL-v2 publicly but expose no explicit "
                "dataset license in the project repository; internal research use only."
            ),
        },
        "source_file_count": len(inventory),
        "source_file_inventory_sha256": canonical_sha256(inventory),
        "source_files": inventory,
        "component_inventory_sha256": canonical_sha256(components),
        "components": components,
    }
    json_content = (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode()
    write_new_or_verify(output_csv, csv_content)
    write_new_or_verify(output_json, json_content)
    print(
        "QWEN_REAL_ADAPTATION_MANIFEST_PASS "
        + json.dumps(
            {
                "manifest_sha256": manifest_sha,
                "metadata_sha256": hashlib.sha256(json_content).hexdigest(),
                "records": len(records),
                "scene_sources": len(components),
                "splits": split_contracts,
                "previously_evaluated_exact_overlap": 0,
                "train_test_perceptual_overlap": 0,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
