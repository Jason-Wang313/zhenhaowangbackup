#!/usr/bin/env python3
"""Freeze the fresh 1,200-record, source-disjoint v2b validation manifest."""

from __future__ import annotations

import argparse
import concurrent.futures
import itertools
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import cv2
import numpy as np

cv2.setNumThreads(1)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.multidegradation_data import (
    encode_jsonl,
    sha256,
    stable_digest,
    write_new_or_verify,
)
from lowlight_restoration.multidegradation_visibility_v2 import (
    build_candidate_record,
    load_visibility_anchors,
    ordered_source_pool,
    physical_order,
    select_visible_candidate,
)
from lowlight_restoration.qwen_multidegradation import (
    DEGRADATIONS,
    canonical_target,
    make_label,
)
from smoke.build_qwen_multidegradation_visibility_v2 import (
    _validate_visibility_evidence,
)

DEFAULT_CONFIG = (
    ROOT / "configs/qwen_multidegradation_visibility_v2b_validation.json"
)


def workspace_path(value: object) -> Path:
    path = (ROOT / str(value)).resolve()
    path.relative_to(ROOT.resolve())
    return path


def load_contract(path: Path) -> dict[str, object]:
    path = path.resolve()
    path.relative_to(ROOT.resolve())
    config = json.loads(path.read_text())
    if config.get("status") != (
        "frozen_source_disjoint_validation_before_model_correction_or_inference"
    ):
        raise RuntimeError("v2b validation contract status changed")
    for path_key, hash_key in (
        ("source_manifest", "source_manifest_sha256"),
        ("old_manifest", "old_manifest_sha256"),
        ("parent_visibility_contract", "parent_visibility_contract_sha256"),
        ("visibility_anchor", "visibility_anchor_sha256"),
    ):
        if sha256(workspace_path(config[path_key])) != config[hash_key]:
            raise RuntimeError(f"v2b frozen input changed: {path_key}")
    parent = json.loads(workspace_path(config["parent_visibility_contract"]).read_text())
    if config.get("visibility_policy") != parent.get("visibility_policy"):
        raise RuntimeError("v2b visibility policy differs from frozen v2 policy")
    if config.get("composition_counts") != {
        "0": 120,
        "1": 360,
        "2": 360,
        "3": 360,
    } or int(config.get("expected_records", -1)) != 1200:
        raise RuntimeError("v2b validation cardinality contract changed")
    for key in (
        "manifest",
        "metadata",
        "screening_root",
        "generation_evidence_root",
        "data_root",
    ):
        workspace_path(config[key])
    return config


def load_fresh_sources(config: Mapping[str, object]) -> list[dict[str, object]]:
    payload = json.loads(workspace_path(config["source_manifest"]).read_text())
    records = payload.get("records")
    if (
        payload.get("status") != "frozen_before_model_correction_or_inference"
        or payload.get("exact_source_overlap_count") != 0
        or payload.get("perceptual_source_overlap_count_at_hamming_le_4") != 0
        or not isinstance(records, list)
        or len(records) != int(config["expected_sources"])
        or sum("depth_path" in record for record in records)
        != int(config["expected_depth_sources"])
    ):
        raise RuntimeError("v2b fresh source manifest is malformed")
    for record in records:
        source_path = workspace_path(record["path"])
        if sha256(source_path) != record["sha256"]:
            raise RuntimeError("v2b source hash changed")
        if "depth_path" in record:
            if sha256(workspace_path(record["depth_path"])) != record["depth_sha256"]:
                raise RuntimeError("v2b source depth hash changed")
    return [dict(record) for record in records]


def renamed_record(record: Mapping[str, object], data_root: str) -> dict[str, object]:
    result = dict(record)
    result["record_id"] = f"v2b_{record['record_id']}"
    result["output_path"] = (
        f"{data_root}/validation/{result['record_id']}.png"
    )
    result["validation_generation"] = "source_disjoint_v2b"
    return result


def sidecar_path(group: Mapping[str, object], within_index: int) -> Path:
    record_id = (
        f"v2b_val_v2_c{int(group['size'])}_"
        f"{int(group['combination_index']):02d}_{within_index:04d}"
    )
    return workspace_path(
        Path(str(group["screening_root"])) / f"{record_id}.json"
    )


def verify_cached(
    payload: Mapping[str, object],
    *,
    group: Mapping[str, object],
    within_index: int,
) -> dict[str, object]:
    if payload.get("contract_sha256") != group["contract_sha256"]:
        raise RuntimeError("v2b screening sidecar contract changed")
    result = payload.get("result")
    if not isinstance(result, dict) or result.get("accepted") is not True:
        raise RuntimeError("v2b screening sidecar is incomplete")
    record = result.get("record")
    evidence = result.get("visibility")
    if not isinstance(record, dict) or not isinstance(evidence, list):
        raise RuntimeError("v2b screening sidecar is malformed")
    expected = build_candidate_record(
        split="validation",
        composition_size=int(group["size"]),
        combination_index=int(group["combination_index"]),
        within_combination_index=within_index,
        combo=tuple(group["combo"]),
        ordered_sources=group["sources"],
        attempt=int(record["screening_attempt"]),
        seed=int(group["seed"]),
        policy_order=group["policy"]["physical_application_order"],
        visibility_policy_name=str(group["policy"]["policy_name"]),
        data_root=str(group["data_root"]),
        composition_rescue=group["policy"].get("composition_rescue"),
    )
    expected = renamed_record(expected, str(group["data_root"]))
    if expected != record:
        raise RuntimeError("v2b cached candidate is not deterministic")
    _validate_visibility_evidence(record, evidence, policy=group["policy"])
    return dict(result)


def build_group(group: Mapping[str, object]) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for within_index in range(int(group["start"]), int(group["end"])):
        sidecar = sidecar_path(group, within_index)
        if sidecar.is_file():
            results.append(
                verify_cached(json.loads(sidecar.read_text()), group=group, within_index=within_index)
            )
            continue
        result = select_visible_candidate(
            split="validation",
            composition_size=int(group["size"]),
            combination_index=int(group["combination_index"]),
            within_combination_index=within_index,
            combo=tuple(group["combo"]),
            sources=group["sources"],
            thresholds=group["thresholds"],
            policy=group["policy"],
            seed=int(group["seed"]),
            data_root=str(group["data_root"]),
        )
        if result.get("accepted") is not True:
            raise RuntimeError(f"v2b visibility screening exhausted: {result['record_id']}")
        result = dict(result)
        result["record"] = renamed_record(result["record"], str(group["data_root"]))
        result["record_id"] = result["record"]["record_id"]
        payload = {
            "schema_version": 1,
            "contract_sha256": group["contract_sha256"],
            "result": result,
        }
        write_new_or_verify(
            sidecar,
            (json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n").encode(),
        )
        results.append(result)
    return results


def build_tasks(
    sources: Sequence[Mapping[str, object]],
    *,
    config: Mapping[str, object],
    thresholds: Mapping[tuple[str, int], float],
    contract_sha256: str,
) -> list[dict[str, object]]:
    tasks: list[dict[str, object]] = []
    for size in range(4):
        combos = [()] if size == 0 else list(itertools.combinations(DEGRADATIONS, size))
        quota = int(config["composition_counts"][str(size)])
        if quota % len(combos):
            raise RuntimeError("v2b combination quota is not balanced")
        per_combo = quota // len(combos)
        if size and per_combo % 4:
            raise RuntimeError("v2b level quota is not balanced")
        for combo_index, combo in enumerate(combos):
            pool = ordered_source_pool(
                sources,
                seed=int(config["seed"]),
                split="validation_v2b",
                combo=combo,
            )
            if not pool:
                raise RuntimeError(f"v2b source pool is empty: {combo}")
            for start in range(0, per_combo, 8):
                tasks.append(
                    {
                        "size": size,
                        "combination_index": combo_index,
                        "combo": combo,
                        "sources": pool,
                        "thresholds": thresholds,
                        "policy": config["visibility_policy"],
                        "seed": config["seed"],
                        "data_root": config["data_root"],
                        "screening_root": config["screening_root"],
                        "contract_sha256": contract_sha256,
                        "start": start,
                        "end": min(start + 8, per_combo),
                    }
                )
    if sum(int(task["end"]) - int(task["start"]) for task in tasks) != 1200:
        raise RuntimeError("v2b task count changed")
    return tasks


def validate_records(
    records: Sequence[Mapping[str, object]],
    *,
    config: Mapping[str, object],
    fresh_sources: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    if len(records) != 1200:
        raise RuntimeError("v2b validation does not contain 1,200 records")
    source_lookup = {str(row["source_id"]): row for row in fresh_sources}
    old_records = [
        json.loads(line)
        for line in workspace_path(config["old_manifest"]).read_text().splitlines()
        if line
    ]
    old_ids = {str(row["source_id"]) for row in old_records}
    old_paths = {str(row["source_path"]) for row in old_records}
    old_hashes = {str(row["source_sha256"]) for row in old_records}
    counts: Counter[tuple[int, tuple[str, ...]]] = Counter()
    levels: Counter[tuple[tuple[str, ...], str, int]] = Counter()
    ids: set[str] = set()
    used_sources: set[str] = set()
    attempts: list[int] = []
    active_components = 0
    for record in records:
        record_id = str(record["record_id"])
        if record_id in ids or not record_id.startswith("v2b_val_v2_"):
            raise RuntimeError("v2b record identity is duplicated or malformed")
        ids.add(record_id)
        if record.get("split") != "validation":
            raise RuntimeError("v2b contains a non-validation record")
        source_id = str(record["source_id"])
        source = source_lookup.get(source_id)
        if source is None:
            raise RuntimeError("v2b record uses a non-frozen source")
        if (
            record["source_path"] != source["path"]
            or record["source_sha256"] != source["sha256"]
            or source_id in old_ids
            or str(record["source_path"]) in old_paths
            or str(record["source_sha256"]) in old_hashes
        ):
            raise RuntimeError("v2b source-disjointness failed")
        used_sources.add(source_id)
        names = tuple(record["degradations"])
        size = int(record["composition_size"])
        label = make_label(names, record["levels"])
        if (
            names != label.degradations
            or len(names) != size
            or record["target"] != canonical_target(label)
            or tuple(record["application_order"])
            != physical_order(names, config["visibility_policy"]["physical_application_order"])
        ):
            raise RuntimeError("v2b label or physical order changed")
        if "haze" in names:
            operator = record["operators"]["haze"]
            if (
                source.get("depth_path") != operator.get("depth_path")
                or source.get("depth_sha256") != operator.get("depth_sha256")
            ):
                raise RuntimeError("v2b haze lacks its source-bound depth")
        evidence = record.get("visibility_evidence")
        if not isinstance(evidence, list):
            raise RuntimeError("v2b visibility evidence is missing")
        _validate_visibility_evidence(record, evidence, policy=config["visibility_policy"])
        counts[(size, names)] += 1
        for name, level in label.levels:
            levels[(names, name, level)] += 1
        attempts.append(int(record["screening_attempt"]) + 1)
        active_components += len(evidence)
    for size in range(4):
        combos = [()] if size == 0 else list(itertools.combinations(DEGRADATIONS, size))
        quota = int(config["composition_counts"][str(size)])
        expected = quota // len(combos)
        for combo in combos:
            if counts[(size, combo)] != expected:
                raise RuntimeError("v2b combination balance changed")
            for name in combo:
                if [levels[(combo, name, level)] for level in range(2, 6)] != [expected // 4] * 4:
                    raise RuntimeError("v2b severity balance changed")
    array = np.asarray(attempts, dtype=np.float64)
    return {
        "records": len(records),
        "unique_record_ids": len(ids),
        "unique_sources_used": len(used_sources),
        "source_disjoint_from_old_train_validation_and_protected_test": True,
        "combination_and_l2_l5_exactly_balanced": True,
        "visible_active_components": active_components,
        "mean_screening_attempts": float(array.mean()),
        "p95_screening_attempts": float(np.quantile(array, 0.95)),
        "maximum_screening_attempts": int(array.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--workers", type=int, default=min(48, os.cpu_count() or 1))
    args = parser.parse_args()
    if args.workers <= 0:
        raise SystemExit("workers must be positive")
    config_path = args.config.resolve()
    config = load_contract(config_path)
    contract_sha256 = sha256(config_path)
    sources = load_fresh_sources(config)
    policy = config["visibility_policy"]
    thresholds = load_visibility_anchors(
        workspace_path(config["visibility_anchor"]),
        metric=str(policy["primary_metric"]),
        quantile=str(policy["anchor_quantile"]),
    )
    multiplier = float(policy["anchor_multiplier"])
    thresholds = {key: value * multiplier for key, value in thresholds.items()}
    tasks = build_tasks(
        sources,
        config=config,
        thresholds=thresholds,
        contract_sha256=contract_sha256,
    )
    results: list[dict[str, object]] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        completed = 0
        for rows in executor.map(build_group, tasks):
            results.extend(rows)
            completed += len(rows)
            if completed % 120 == 0 or completed == 1200:
                print(f"QWEN_VISIBILITY_V2B_BUILD_PROGRESS {completed}/1200", flush=True)
    records: list[dict[str, object]] = []
    for result in results:
        record = dict(result["record"])
        record["visibility_evidence"] = result["visibility"]
        records.append(record)
    validation = validate_records(records, config=config, fresh_sources=sources)
    manifest_path = workspace_path(config["manifest"])
    metadata_path = workspace_path(config["metadata"])
    manifest_sha = write_new_or_verify(manifest_path, encode_jsonl(records))
    metadata = {
        "schema_version": 1,
        "status": "frozen_validation_manifest_before_model_correction_or_inference",
        "config": config_path.relative_to(ROOT).as_posix(),
        "config_sha256": contract_sha256,
        "manifest": manifest_path.relative_to(ROOT).as_posix(),
        "manifest_sha256": manifest_sha,
        "source_manifest": config["source_manifest"],
        "source_manifest_sha256": config["source_manifest_sha256"],
        "old_manifest": config["old_manifest"],
        "old_manifest_sha256": config["old_manifest_sha256"],
        "visibility_anchor": config["visibility_anchor"],
        "visibility_anchor_sha256": config["visibility_anchor_sha256"],
        "validation": validation,
        "model_checkpoint_accessed": False,
        "protected_test_inference_performed": False,
    }
    metadata_sha = write_new_or_verify(
        metadata_path,
        (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode(),
    )
    print(
        "QWEN_VISIBILITY_V2B_MANIFEST_PASS "
        + json.dumps(
            {
                "manifest_sha256": manifest_sha,
                "metadata_sha256": metadata_sha,
                "validation": validation,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
