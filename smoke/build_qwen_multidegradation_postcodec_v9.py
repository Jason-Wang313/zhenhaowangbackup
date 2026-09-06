#!/usr/bin/env python3
"""Build a bounded synthetic-only post-codec continuation/replay manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Callable, Hashable, Mapping, Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.multidegradation_data import (
    read_jsonl,
    sha256,
    stable_digest,
    write_new_or_verify,
)


JPEG_QUALITIES = (75, 85, 94, 99)
TRAIN_ORIGINAL_QUOTAS = {0: 1040, 1: 3120, 2: 3120, 3: 3120}
TRAIN_POST_POSITIVE_QUOTAS = {1: 1600, 2: 1600, 3: 1600}
TRAIN_POST_NEGATIVE_QUOTAS = {0: 1200, 1: 1200, 2: 1200, 3: 1200}
VALIDATION_POST_POSITIVE_QUOTAS = {1: 288, 2: 456, 3: 456}
VALIDATION_POST_NEGATIVE_QUOTAS = {0: 300, 1: 300, 2: 300, 3: 300}


def canonical_bytes(value: object) -> bytes:
    return json.dumps(value, separators=(",", ":"), sort_keys=True).encode()


def manifest_record_sha256(record: Mapping[str, object]) -> str:
    return hashlib.sha256(canonical_bytes(record)).hexdigest()


def deterministic_balanced_take(
    records: Sequence[dict[str, object]],
    quota: int,
    *,
    seed: int,
    salt: str,
    key: Callable[[dict[str, object]], Hashable],
) -> list[dict[str, object]]:
    groups: dict[Hashable, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        groups[key(record)].append(record)
    if quota <= 0 or not groups:
        raise ValueError(f"empty or invalid balanced selection: {salt}")
    for group_key, values in groups.items():
        values.sort(
            key=lambda record: stable_digest(
                seed, salt, repr(group_key), str(record["record_id"])
            )
        )
    ordered_keys = sorted(groups, key=repr)
    offsets = {group_key: 0 for group_key in ordered_keys}
    selected: list[dict[str, object]] = []
    while len(selected) < quota:
        progressed = False
        for group_key in ordered_keys:
            offset = offsets[group_key]
            values = groups[group_key]
            if offset < len(values):
                selected.append(values[offset])
                offsets[group_key] += 1
                progressed = True
                if len(selected) == quota:
                    break
        if not progressed:
            raise RuntimeError(f"insufficient candidates for {salt}: {quota}")
    return selected


def noise_stratum(record: dict[str, object]) -> tuple[object, ...]:
    noise = record["operators"]["noise"]
    return (
        tuple(record["degradations"]),
        noise["family"],
        int(record["levels"]["noise"]),
    )


def negative_stratum(record: dict[str, object]) -> tuple[str, ...]:
    return tuple(record["degradations"])


def original_stratum(record: dict[str, object]) -> tuple[object, ...]:
    return (
        tuple(record["degradations"]),
        tuple(sorted((str(k), int(v)) for k, v in record["levels"].items())),
        record.get("non_target_nuisance", {}).get("family", "none"),
    )


def clone_record(
    record: dict[str, object],
    *,
    domain: str,
    quality: int | None,
) -> dict[str, object]:
    cloned = dict(record)
    cloned["base_record_id"] = record["record_id"]
    cloned["base_manifest_record_sha256"] = manifest_record_sha256(record)
    cloned["evaluation_domain"] = domain
    if quality is None:
        cloned["record_id"] = f"v9_original_{record['record_id']}"
        cloned["input_transform"] = {"family": "none"}
    else:
        cloned["record_id"] = f"v9_postjpeg_q{quality}_{record['record_id']}"
        cloned["input_transform"] = {
            "family": "jpeg",
            "quality": quality,
            "subsampling": 2,
            "order": "after_target_degradations",
        }
    return cloned


def select_postcodec(
    records: Sequence[dict[str, object]],
    *,
    positive_quotas: Mapping[int, int],
    negative_quotas: Mapping[int, int],
    seed: int,
    salt: str,
) -> list[dict[str, object]]:
    eligible = [
        record
        for record in records
        if record.get("non_target_nuisance", {}).get("family") == "none"
    ]
    selected: list[dict[str, object]] = []
    for composition_size, quota in positive_quotas.items():
        pool = [
            record
            for record in eligible
            if int(record["composition_size"]) == composition_size
            and "noise" in record["degradations"]
        ]
        selected.extend(
            deterministic_balanced_take(
                pool,
                quota,
                seed=seed,
                salt=f"{salt}_noise_c{composition_size}",
                key=noise_stratum,
            )
        )
    for composition_size, quota in negative_quotas.items():
        pool = [
            record
            for record in eligible
            if int(record["composition_size"]) == composition_size
            and "noise" not in record["degradations"]
        ]
        selected.extend(
            deterministic_balanced_take(
                pool,
                quota,
                seed=seed,
                salt=f"{salt}_negative_c{composition_size}",
                key=negative_stratum,
            )
        )
    selected.sort(key=lambda record: stable_digest(seed, salt, record["record_id"]))
    return [
        clone_record(
            record,
            domain="post_target_jpeg",
            quality=JPEG_QUALITIES[index % len(JPEG_QUALITIES)],
        )
        for index, record in enumerate(selected)
    ]


def build(config: Mapping[str, object]) -> tuple[list[dict[str, object]], dict[str, object]]:
    seed = int(config["seed"])
    base_manifest = ROOT / str(config["base_manifest"])
    if sha256(base_manifest) != config["base_manifest_sha256"]:
        raise RuntimeError("physical-v8 base manifest changed")
    records = read_jsonl(base_manifest)
    if len(records) != 120000:
        raise RuntimeError("physical-v8 base manifest cardinality changed")
    by_split = {
        split: [record for record in records if record["split"] == split]
        for split in ("train", "validation")
    }
    original_train: list[dict[str, object]] = []
    for composition_size, quota in TRAIN_ORIGINAL_QUOTAS.items():
        pool = [
            record
            for record in by_split["train"]
            if int(record["composition_size"]) == composition_size
        ]
        original_train.extend(
            clone_record(record, domain="v8_original", quality=None)
            for record in deterministic_balanced_take(
                pool,
                quota,
                seed=seed,
                salt=f"train_original_c{composition_size}",
                key=original_stratum,
            )
        )
    post_train = select_postcodec(
        by_split["train"],
        positive_quotas=TRAIN_POST_POSITIVE_QUOTAS,
        negative_quotas=TRAIN_POST_NEGATIVE_QUOTAS,
        seed=seed,
        salt="train_postcodec",
    )
    original_validation = [
        clone_record(record, domain="v8_original", quality=None)
        for record in by_split["validation"]
    ]
    post_validation = select_postcodec(
        by_split["validation"],
        positive_quotas=VALIDATION_POST_POSITIVE_QUOTAS,
        negative_quotas=VALIDATION_POST_NEGATIVE_QUOTAS,
        seed=seed,
        salt="validation_postcodec",
    )
    train = sorted(
        original_train + post_train,
        key=lambda record: stable_digest(seed, "train_final", record["record_id"]),
    )
    validation = sorted(
        original_validation + post_validation,
        key=lambda record: stable_digest(seed, "validation_final", record["record_id"]),
    )
    combined = train + validation
    if len(train) != 20000 or len(validation) != 12400:
        raise RuntimeError("v9 replay cardinality changed")
    if any("foundir" in str(record["output_path"]).lower() for record in combined):
        raise RuntimeError("FoundIR image entered the Qwen replay manifest")
    if any(record["split"] == "test" for record in combined):
        raise RuntimeError("protected synthetic test entered the Qwen replay manifest")
    counts = Counter(
        (record["split"], record["evaluation_domain"], "noise" in record["degradations"])
        for record in combined
    )
    metadata = {
        "schema_version": 1,
        "status": "synthetic_only_postcodec_v9_replay_frozen",
        "base_manifest": config["base_manifest"],
        "base_manifest_sha256": config["base_manifest_sha256"],
        "records": len(combined),
        "split_counts": dict(Counter(record["split"] for record in combined)),
        "domain_presence_counts": {
            f"{split}:{domain}:{'noise_present' if present else 'noise_absent'}": count
            for (split, domain, present), count in sorted(counts.items())
        },
        "jpeg_qualities": list(JPEG_QUALITIES),
        "jpeg_order": "after_target_degradations",
        "foundir_images_used_for_qwen_training": False,
        "protected_synthetic_test_accessed": False,
    }
    return combined, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/qwen_multidegradation_postcodec_v9_data.json",
    )
    args = parser.parse_args()
    raw = args.config.read_bytes()
    config = json.loads(raw)
    if config.get("status") != "synthetic_only_postcodec_v9_data_contract":
        raise ValueError("unexpected v9 data contract")
    records, metadata = build(config)
    manifest = ROOT / str(config["manifest"])
    payload = b"".join(canonical_bytes(record) + b"\n" for record in records)
    write_new_or_verify(manifest, payload)
    metadata.update(
        {
            "data_contract": args.config.relative_to(ROOT).as_posix(),
            "data_contract_sha256": hashlib.sha256(raw).hexdigest(),
            "manifest": config["manifest"],
            "manifest_sha256": hashlib.sha256(payload).hexdigest(),
        }
    )
    output = ROOT / str(config["metadata"])
    write_new_or_verify(
        output, (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode()
    )
    print("QWEN_POSTCODEC_V9_MANIFEST_PASS " + json.dumps(metadata, sort_keys=True))


if __name__ == "__main__":
    main()
