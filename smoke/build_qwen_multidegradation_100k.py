#!/usr/bin/env python3
"""Build the immutable balanced 100k-train Qwen degradation manifest."""

from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.multidegradation_data import (
    AGENTICIR_REVISION,
    BLUR_PARAMETERS_EXTREME_V6,
    BLUR_PARAMETERS_MILD_V7,
    NOISE_SIGMA_RANGES,
    NOISE_SIGMA_RANGES_MILD_V7,
    SPLITS,
    _balanced_levels,
    _operator_parameters,
    assign_source_splits,
    encode_jsonl,
    load_source_inventory,
    sha256,
    stable_digest,
    stable_seed,
    write_new_or_verify,
)
from lowlight_restoration.qwen_multidegradation import (
    DEGRADATIONS,
    canonical_target,
    make_label,
)


V6_STATUS = "human_approved_extreme_blur_v6_100k_contract"
V7_STATUS = "advisor_directed_mild_blur_noise_v7_100k_contract"


def parameter_profile(
    contract: Mapping[str, object],
) -> tuple[
    Mapping[int, tuple[int, float]],
    Mapping[int, tuple[float, float]],
    str,
    str | None,
]:
    status = contract.get("status")
    if status == V6_STATUS:
        return (
            BLUR_PARAMETERS_EXTREME_V6,
            NOISE_SIGMA_RANGES,
            "extreme_v6_human_approved_2026_08_28",
            None,
        )
    if status == V7_STATUS:
        return (
            BLUR_PARAMETERS_MILD_V7,
            NOISE_SIGMA_RANGES_MILD_V7,
            "mild_v7_advisor_directed_2026_09_05",
            "mild_v7_advisor_directed_2026_09_05",
        )
    raise ValueError("100k data contract status is not approved")


def workspace_path(value: object) -> Path:
    path = (ROOT / str(value)).resolve()
    path.relative_to(ROOT.resolve())
    return path


def load_contract(path: Path) -> tuple[dict[str, object], str]:
    raw = path.read_bytes()
    contract = json.loads(raw)
    if contract.get("schema_version") != 1:
        raise ValueError("100k data contract is not approved")
    blur_parameters, noise_ranges, _, _ = parameter_profile(contract)
    if sha256(workspace_path(contract["source_manifest"])) != contract.get(
        "source_manifest_sha256"
    ):
        raise RuntimeError("100k source manifest changed")
    blur = contract.get("blur")
    if not isinstance(blur, dict):
        raise ValueError("100k blur contract is missing")
    if contract.get("status") == V6_STATUS:
        for path_key, hash_key in (
            ("preview", "preview_sha256"),
            ("preview_report", "preview_report_sha256"),
        ):
            if sha256(workspace_path(blur[path_key])) != blur.get(hash_key):
                raise RuntimeError(f"approved blur evidence changed: {path_key}")
    expected = {
        str(level): {
            "radius": radius,
            "sigma": sigma,
            "kernel_width": 2 * radius + 1,
        }
        for level, (radius, sigma) in blur_parameters.items()
    }
    if blur.get("parameters") != expected:
        raise RuntimeError("100k blur parameters differ from approved preview")
    noise = contract.get("noise")
    if contract.get("status") == V7_STATUS:
        expected_noise = {
            str(level): {"sigma_0_255_min": low, "sigma_0_255_max": high}
            for level, (low, high) in noise_ranges.items()
        }
        if not isinstance(noise, dict) or noise.get("parameters") != expected_noise:
            raise RuntimeError("100k noise parameters differ from advisor-directed v7")
    return contract, __import__("hashlib").sha256(raw).hexdigest()


def build_record(
    *,
    split: str,
    size: int,
    combination_index: int,
    index: int,
    combo: tuple[str, ...],
    ordered_sources: Sequence[Mapping[str, object]],
    contract: Mapping[str, object],
    contract_sha: str,
) -> dict[str, object]:
    blur_parameters, noise_ranges, blur_profile, noise_profile = parameter_profile(
        contract
    )
    seed = int(contract["seed"])
    source = ordered_sources[index % len(ordered_sources)]
    record_seed = stable_seed(
        seed, "100k", split, size, combination_index, index
    )
    rng = random.Random(record_seed)
    crop_size = int(contract["crop_size"])
    crop_x = rng.randint(0, int(source["width"]) - crop_size)
    crop_y = rng.randint(0, int(source["height"]) - crop_size)
    levels = _balanced_levels(combo, index)
    physical_order = tuple(map(str, contract["physical_application_order"]))
    application_order = tuple(name for name in physical_order if name in combo)
    if set(application_order) != set(combo):
        raise RuntimeError("physical application order does not cover a combination")
    operators: dict[str, object] = {}
    for operator_index, name in enumerate(application_order):
        operator_seed = stable_seed(record_seed, operator_index, name)
        parameters = _operator_parameters(
            name,
            levels[name],
            seed=operator_seed,
            source=source,
            blur_parameters=blur_parameters,
            noise_sigma_ranges=noise_ranges,
        )
        parameters["seed"] = operator_seed
        operators[name] = parameters
    label = make_label(combo, levels)
    record_id = (
        f"{split[:3]}_100k_c{size}_{combination_index:02d}_{index:05d}"
    )
    record = {
        "record_id": record_id,
        "split": split,
        "source_id": source["source_id"],
        "source_dataset": source["dataset"],
        "source_path": source["path"],
        "source_sha256": source["sha256"],
        "crop": {
            "x": crop_x,
            "y": crop_y,
            "width": crop_size,
            "height": crop_size,
        },
        "composition_size": size,
        "degradations": list(label.degradations),
        "levels": label.level_map,
        "application_order": list(application_order),
        "operators": operators,
        "seed": record_seed,
        "target": canonical_target(label),
        "output_path": (
            f"{contract['data_root']}/{split}/{record_id}.png"
        ),
        "agenticir_revision": AGENTICIR_REVISION,
        "data_contract_sha256": contract_sha,
        "blur_profile": blur_profile,
    }
    if noise_profile is not None:
        record["noise_profile"] = noise_profile
    return record


def build_records(
    inventory: Sequence[Mapping[str, object]],
    contract: Mapping[str, object],
    contract_sha: str,
) -> list[dict[str, object]]:
    assignments = assign_source_splits(inventory, seed=int(contract["seed"]))
    records: list[dict[str, object]] = []
    counts = contract["split_composition_counts"]
    if not isinstance(counts, Mapping):
        raise ValueError("100k split counts are missing")
    for split in SPLITS:
        sources = assignments[split]
        haze_sources = tuple(
            source for source in sources if source["dataset"] == "mio100"
        )
        split_counts = counts[split]
        if not isinstance(split_counts, Mapping):
            raise ValueError(f"100k split counts are malformed: {split}")
        for size in range(4):
            combos = [()] if size == 0 else list(
                itertools.combinations(DEGRADATIONS, size)
            )
            quota = int(split_counts[str(size)])
            if quota % len(combos):
                raise ValueError("100k composition quota is not combination-balanced")
            per_combo = quota // len(combos)
            if size and per_combo % 4:
                raise ValueError("100k combination quota is not level-balanced")
            for combination_index, combo in enumerate(combos):
                pool = haze_sources if "haze" in combo else sources
                ordered_pool = sorted(
                    pool,
                    key=lambda source: stable_digest(
                        int(contract["seed"]),
                        "100k",
                        split,
                        combo,
                        source["source_id"],
                    ),
                )
                for index in range(per_combo):
                    records.append(
                        build_record(
                            split=split,
                            size=size,
                            combination_index=combination_index,
                            index=index,
                            combo=combo,
                            ordered_sources=ordered_pool,
                            contract=contract,
                            contract_sha=contract_sha,
                        )
                    )
    return records


def bind_mio_depth_once(
    inventory: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Attach each MiO depth hash once instead of rehashing it per record."""

    bound: list[dict[str, object]] = []
    for source_input in inventory:
        source = dict(source_input)
        if source["dataset"] == "mio100":
            stem = Path(str(source["path"])).stem
            depth_path = Path("data/agenticir/haze") / stem / "predict_depth.mat"
            absolute = workspace_path(depth_path)
            source["depth_path"] = depth_path.as_posix()
            source["depth_sha256"] = sha256(absolute)
        bound.append(source)
    return bound


def validate_records(
    records: Sequence[Mapping[str, object]],
    contract: Mapping[str, object],
    contract_sha: str,
) -> dict[str, object]:
    blur_parameters, noise_ranges, blur_profile, noise_profile = parameter_profile(
        contract
    )
    expected_total = sum(
        int(value)
        for split_counts in contract["split_composition_counts"].values()
        for value in split_counts.values()
    )
    if len(records) != expected_total or expected_total != 120000:
        raise RuntimeError(f"100k manifest cardinality changed: {len(records)}")
    ids = [str(record["record_id"]) for record in records]
    if len(ids) != len(set(ids)):
        raise RuntimeError("100k manifest has duplicate record IDs")
    roles: dict[str, str] = {}
    composition = Counter()
    combinations = Counter()
    levels = Counter()
    blur_counts = Counter()
    physical_order = tuple(map(str, contract["physical_application_order"]))
    for record in records:
        split = str(record["split"])
        size = int(record["composition_size"])
        names = tuple(record["degradations"])
        label = make_label(names, record["levels"])
        if (
            len(names) != size
            or record["target"] != canonical_target(label)
            or record.get("data_contract_sha256") != contract_sha
        ):
            raise RuntimeError(f"100k label contract changed: {record['record_id']}")
        expected_order = tuple(name for name in physical_order if name in names)
        if tuple(record["application_order"]) != expected_order:
            raise RuntimeError("100k physical order changed")
        source_id = str(record["source_id"])
        previous = roles.setdefault(source_id, split)
        if previous != split:
            raise RuntimeError(f"100k source leakage: {source_id}")
        if "haze" in names and record["source_dataset"] != "mio100":
            raise RuntimeError("100k haze lacks an objective aligned depth map")
        if "blur" in names:
            level = int(record["levels"]["blur"])
            radius, sigma = blur_parameters[level]
            actual = record["operators"]["blur"]
            if (actual["radius"], actual["sigma"]) != (radius, sigma):
                raise RuntimeError("100k blur parameters changed")
            blur_counts[level] += 1
        if "noise" in names:
            level = int(record["levels"]["noise"])
            low, high = noise_ranges[level]
            sigma = float(record["operators"]["noise"]["sigma_0_255"])
            if not low <= sigma <= high:
                raise RuntimeError("100k noise parameters changed")
        composition[(split, size)] += 1
        combinations[(split, names)] += 1
        for name, level in label.levels:
            levels[(split, names, name, level)] += 1
    for split, expected in contract["split_composition_counts"].items():
        for size_text, count in expected.items():
            if composition[(split, int(size_text))] != int(count):
                raise RuntimeError("100k composition balance changed")
    for (split, names), count in combinations.items():
        if not names:
            continue
        for name in names:
            observed = [levels[(split, names, name, level)] for level in range(2, 6)]
            if observed != [count // 4] * 4:
                raise RuntimeError(f"100k level balance changed: {split} {names}")
    return {
        "records": len(records),
        "split_records": dict(Counter(str(record["split"]) for record in records)),
        "split_composition_counts": {
            split: {
                str(size): composition[(split, size)] for size in range(4)
            }
            for split in SPLITS
        },
        "unique_sources_used": len(roles),
        "source_disjoint": True,
        "combination_and_level_balanced": True,
        "blur_level_counts": {str(level): blur_counts[level] for level in range(2, 6)},
        "blur_profile": blur_profile,
        "noise_profile": noise_profile or "legacy_agenticir_ranges",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/qwen_multidegradation_100k_data_v1.json",
    )
    args = parser.parse_args()
    config_path = args.config.resolve()
    contract, contract_sha = load_contract(config_path)
    inventory = bind_mio_depth_once(
        load_source_inventory(workspace_path(contract["source_manifest"]))
    )
    records = build_records(inventory, contract, contract_sha)
    validation = validate_records(records, contract, contract_sha)
    manifest_path = workspace_path(contract["manifest"])
    manifest_sha = write_new_or_verify(manifest_path, encode_jsonl(records))
    metadata = {
        "schema_version": 1,
        "status": "frozen_100k_train_manifest_images_not_implied_generated",
        "data_contract": config_path.relative_to(ROOT).as_posix(),
        "data_contract_sha256": contract_sha,
        "manifest": contract["manifest"],
        "manifest_sha256": manifest_sha,
        "validation": validation,
        "protected_test_inference_performed": False,
        "foundir_used": False,
    }
    metadata_path = workspace_path(contract["metadata"])
    metadata_sha = write_new_or_verify(
        metadata_path,
        (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode(),
    )
    print(
        "QWEN_100K_MANIFEST_PASS "
        + json.dumps(
            {
                "manifest_sha256": manifest_sha,
                "metadata_sha256": metadata_sha,
                **validation,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
