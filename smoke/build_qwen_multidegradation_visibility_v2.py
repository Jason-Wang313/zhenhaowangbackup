#!/usr/bin/env python3
"""Build the complete balanced visibility-aware v2 manifest."""

from __future__ import annotations

import argparse
import concurrent.futures
import itertools
import json
import os
import sys
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.multidegradation_data import (
    SPLITS,
    SPLIT_COMPOSITION_COUNTS,
    assign_source_splits,
    encode_jsonl,
    load_source_inventory,
    sha256,
    validate_pilot_records,
    write_new_or_verify,
)
from lowlight_restoration.multidegradation_visibility_v2 import (
    build_candidate_record,
    load_visibility_anchors,
    ordered_source_pool,
    physical_order,
    select_visible_candidate,
)
from lowlight_restoration.qwen_multidegradation import DEGRADATIONS


def _workspace_path(value: object) -> Path:
    path = (ROOT / str(value)).resolve()
    path.relative_to(ROOT.resolve())
    return path


def load_production_contract(config_path: Path) -> dict[str, object]:
    """Load a fail-closed production contract and verify its frozen pilot."""

    config_path = config_path.resolve()
    config_path.relative_to(ROOT.resolve())
    config = json.loads(config_path.read_text())
    for path_key, hash_key in (
        ("source_manifest", "source_manifest_sha256"),
        ("visibility_anchor", "visibility_anchor_sha256"),
        ("causal_audit", "causal_audit_sha256"),
        ("passing_pilot", "passing_pilot_sha256"),
        ("passing_pilot_config", "passing_pilot_config_sha256"),
    ):
        path = _workspace_path(config[path_key])
        if sha256(path) != config[hash_key]:
            raise RuntimeError(f"v2 production input changed: {path_key}")
    pilot = json.loads(_workspace_path(config["passing_pilot"]).read_text())
    if (
        pilot.get("status") != "pass"
        or pilot.get("automatic_gate_pass") is not True
        or pilot.get("records") != 40
        or pilot.get("accepted_records") != 40
        or not all(bool(value) for value in pilot.get("checks", {}).values())
    ):
        raise RuntimeError("v2 production requires the complete frozen passing pilot")
    if pilot.get("config") != config["passing_pilot_config"]:
        raise RuntimeError("v2 passing pilot config path changed")
    if pilot.get("config_sha256") != config["passing_pilot_config_sha256"]:
        raise RuntimeError("v2 passing pilot config hash changed")
    causal_audit = json.loads(_workspace_path(config["causal_audit"]).read_text())
    if causal_audit.get("status") != "pass":
        raise RuntimeError("v2 production requires the passing causal audit")
    pilot_config = json.loads(
        _workspace_path(config["passing_pilot_config"]).read_text()
    )
    pilot_policy = dict(pilot_config["visibility_policy"])
    production_policy = dict(config["visibility_policy"])
    pilot_attempts = int(pilot_policy.pop("maximum_attempts_per_record"))
    production_attempts = int(
        production_policy.pop("maximum_attempts_per_record")
    )
    rescue = production_policy.pop("composition_rescue", None)
    if production_policy != pilot_policy or production_attempts < pilot_attempts:
        raise RuntimeError("v2 production visibility policy differs from passing pilot")
    if rescue is not None:
        if (
            not isinstance(rescue, dict)
            or int(rescue.get("start_attempt", -1)) < 256
            or int(rescue["start_attempt"]) >= production_attempts
        ):
            raise RuntimeError("v2 composition rescue attempt boundary is invalid")
        for key in (
            "haze_beta_scale_range",
            "haze_airlight_contrast_scale_range",
            "noise_sigma_scale_range",
        ):
            values = rescue.get(key)
            if (
                not isinstance(values, list)
                or len(values) != 2
                or not 0 < float(values[0]) <= float(values[1]) <= 1
            ):
                raise RuntimeError(f"v2 composition rescue range is invalid: {key}")
    compatible = config.get("compatible_screening_contract_sha256", [])
    if (
        not isinstance(compatible, list)
        or any(
            not isinstance(value, str) or len(value) != 64
            for value in compatible
        )
    ):
        raise RuntimeError("v2 compatible screening contracts are malformed")
    supersedes = config.get("supersedes_failed_production_contract")
    if supersedes is not None:
        if (
            not isinstance(supersedes, dict)
            or sha256(_workspace_path(supersedes.get("path")))
            != supersedes.get("sha256")
            or supersedes.get("sha256") not in compatible
        ):
            raise RuntimeError("v2 superseded production contract changed")
    for output_key in ("manifest", "metadata", "screening_root", "data_root"):
        _workspace_path(config[output_key])
    return config


def _validate_visibility_evidence(
    record: Mapping[str, object],
    evidence: Sequence[Mapping[str, object]],
    *,
    policy: Mapping[str, object],
) -> None:
    order = tuple(str(name) for name in record["application_order"])
    if len(evidence) != len(order):
        raise RuntimeError("v2 visibility evidence cardinality changed")
    full_hashes: set[str] = set()
    for position, (name, component) in enumerate(zip(order, evidence)):
        if (
            component.get("degradation") != name
            or int(component.get("level", -1)) != int(record["levels"][name])
            or int(component.get("position", -1)) != position
            or component.get("primary_metric") != policy["primary_metric"]
            or component.get("visibility_mode") != policy["visibility_mode"]
        ):
            raise RuntimeError("v2 visibility evidence identity changed")
        anchor = float(component["anchor_observed_value"])
        minimum_anchor = float(component["minimum_anchor_value"])
        retention = float(component["retention_ratio"])
        minimum_retention = float(component["minimum_retention_ratio"])
        if (
            component.get("pass") is not True
            or anchor < minimum_anchor
            or retention < minimum_retention
            or minimum_retention != float(policy["minimum_rmse_survival_ratio"])
        ):
            raise RuntimeError("v2 manifest contains an invisible active component")
        full_hash = str(component["full_sha256"])
        counterfactual_hash = str(component["counterfactual_sha256"])
        if len(full_hash) != 64 or len(counterfactual_hash) != 64:
            raise RuntimeError("v2 visibility evidence hash is malformed")
        full_hashes.add(full_hash)
    if len(full_hashes) > 1:
        raise RuntimeError("v2 component evidence disagrees on the final render hash")


def _sidecar_path(group: Mapping[str, object], within_index: int) -> Path:
    record_id = (
        f"{str(group['split'])[:3]}_v2_c{int(group['size'])}_"
        f"{int(group['combination_index']):02d}_{within_index:04d}"
    )
    return _workspace_path(
        Path(str(group["screening_root"]))
        / str(group["split"])
        / f"{record_id}.json"
    )


def _failure_path(group: Mapping[str, object], within_index: int) -> Path:
    record_id = (
        f"{str(group['split'])[:3]}_v2_c{int(group['size'])}_"
        f"{int(group['combination_index']):02d}_{within_index:04d}"
    )
    return _workspace_path(
        Path(str(group["screening_root"]))
        / "failures"
        / str(group["contract_sha256"])[:12]
        / str(group["split"])
        / f"{record_id}.json"
    )


def _verify_cached_result(
    payload: Mapping[str, object],
    *,
    group: Mapping[str, object],
    within_index: int,
) -> dict[str, object]:
    compatible_contracts = {
        str(group["contract_sha256"]),
        *map(str, group.get("compatible_contract_sha256", [])),
    }
    if payload.get("contract_sha256") not in compatible_contracts:
        raise RuntimeError("v2 screening sidecar belongs to a different contract")
    result = payload.get("result")
    if not isinstance(result, dict) or result.get("accepted") is not True:
        raise RuntimeError("v2 screening sidecar is incomplete")
    record = result.get("record")
    evidence = result.get("visibility")
    if not isinstance(record, dict) or not isinstance(evidence, list):
        raise RuntimeError("v2 screening sidecar is malformed")
    expected_record = build_candidate_record(
        split=str(group["split"]),
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
        composition_rescue=(
            group["policy"].get("composition_rescue")
            if isinstance(group["policy"].get("composition_rescue"), Mapping)
            else None
        ),
    )
    if record != expected_record:
        raise RuntimeError("v2 screening sidecar candidate is not deterministic")
    _validate_visibility_evidence(record, evidence, policy=group["policy"])
    return result


def _build_group(group: Mapping[str, object]) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for within_index in range(
        int(group["start_index"]), int(group["end_index"])
    ):
        sidecar_path = _sidecar_path(group, within_index)
        if sidecar_path.is_file():
            payload = json.loads(sidecar_path.read_text())
            results.append(
                _verify_cached_result(
                    payload,
                    group=group,
                    within_index=within_index,
                )
            )
            continue
        result = select_visible_candidate(
            split=str(group["split"]),
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
        if not bool(result["accepted"]):
            failure = {
                "schema_version": 1,
                "status": "failed_maximum_attempts_exhausted",
                "contract_sha256": group["contract_sha256"],
                "maximum_attempts_per_record": group["policy"][
                    "maximum_attempts_per_record"
                ],
                "result": result,
            }
            write_new_or_verify(
                _failure_path(group, within_index),
                (
                    json.dumps(failure, separators=(",", ":"), sort_keys=True)
                    + "\n"
                ).encode(),
            )
            raise RuntimeError(
                f"v2 production exhausted attempts: {result['record_id']}"
            )
        payload = {
            "schema_version": 1,
            "contract_sha256": group["contract_sha256"],
            "result": result,
        }
        write_new_or_verify(
            sidecar_path,
            (json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n").encode(),
        )
        results.append(result)
    return results


def build_group_tasks(
    assignments: Mapping[str, Sequence[Mapping[str, object]]],
    *,
    config: Mapping[str, object],
    thresholds: Mapping[tuple[str, int], float],
    contract_sha256: str,
) -> list[dict[str, object]]:
    fresh_chunk_size = 16
    cached_chunk_size = 128
    groups: list[dict[str, object]] = []
    for split in SPLITS:
        for size in range(4):
            combinations = [()] if size == 0 else list(
                itertools.combinations(DEGRADATIONS, size)
            )
            quota = SPLIT_COMPOSITION_COUNTS[split][size]
            if quota % len(combinations):
                raise RuntimeError("v2 production quota is not combination-balanced")
            per_combination = quota // len(combinations)
            for combination_index, combo in enumerate(combinations):
                sources = ordered_source_pool(
                    assignments[split],
                    seed=int(config["seed"]),
                    split=split,
                    combo=combo,
                )
                if not sources:
                    raise RuntimeError(f"v2 source pool is empty: {split} {combo}")
                sidecar_directory = (
                    _workspace_path(config["screening_root"]) / split
                )

                def has_sidecar(index: int) -> bool:
                    record_id = (
                        f"{split[:3]}_v2_c{size}_{combination_index:02d}_"
                        f"{index:04d}"
                    )
                    return (sidecar_directory / f"{record_id}.json").is_file()

                resume_mode = any(
                    has_sidecar(index)
                    for index in range(min(per_combination, fresh_chunk_size))
                )
                ranges: list[tuple[int, int]] = []
                start_index = 0
                while start_index < per_combination:
                    if not resume_mode:
                        end_index = min(
                            start_index + fresh_chunk_size, per_combination
                        )
                    elif not has_sidecar(start_index):
                        end_index = start_index + 1
                    else:
                        end_index = start_index + 1
                        while (
                            end_index < per_combination
                            and end_index - start_index < cached_chunk_size
                            and has_sidecar(end_index)
                        ):
                            end_index += 1
                    ranges.append((start_index, end_index))
                    start_index = end_index
                for start_index, end_index in ranges:
                    groups.append(
                        {
                            "split": split,
                            "size": size,
                            "combination_index": combination_index,
                            "combo": combo,
                            "sources": sources,
                            "thresholds": thresholds,
                            "policy": config["visibility_policy"],
                            "seed": config["seed"],
                            "data_root": config["data_root"],
                            "screening_root": config["screening_root"],
                            "contract_sha256": contract_sha256,
                            "compatible_contract_sha256": config.get(
                                "compatible_screening_contract_sha256", []
                            ),
                            "start_index": start_index,
                            "end_index": end_index,
                        }
                    )
    if sum(
        int(group["end_index"]) - int(group["start_index"])
        for group in groups
    ) != 12000:
        raise RuntimeError("v2 production task count changed")
    return groups


def validate_visibility_records(
    records: list[dict[str, object]],
    *,
    policy: Mapping[str, object],
) -> dict[str, object]:
    base = validate_pilot_records(records)
    attempts = []
    component_count = 0
    for record in records:
        expected_order = physical_order(
            tuple(record["degradations"]),
            policy["physical_application_order"],
        )
        if tuple(record["application_order"]) != expected_order:
            raise RuntimeError("v2 record violates physical application order")
        if record["visibility_policy"] != policy["policy_name"]:
            raise RuntimeError("v2 record visibility policy changed")
        attempts.append(int(record["screening_attempt"]) + 1)
        evidence = record.get("visibility_evidence", [])
        _validate_visibility_evidence(record, evidence, policy=policy)
        component_count += len(evidence)
    array = np.asarray(attempts, dtype=np.float64)
    return {
        **base,
        "visibility_policy": policy["policy_name"],
        "visible_active_components": component_count,
        "mean_screening_attempts": float(array.mean()),
        "p95_screening_attempts": float(np.quantile(array, 0.95)),
        "maximum_screening_attempts": int(array.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "configs/qwen_multidegradation_visibility_v2_production_correction1.json",
    )
    parser.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 1))
    args = parser.parse_args()
    if args.workers <= 0:
        raise SystemExit("workers must be positive")
    config_path = args.config.resolve()
    config = load_production_contract(config_path)
    contract_sha256 = sha256(config_path)
    policy = config["visibility_policy"]
    source_path = ROOT / config["source_manifest"]
    inventory = load_source_inventory(source_path)
    assignments = assign_source_splits(inventory, seed=int(config["seed"]))
    anchor_path = ROOT / config["visibility_anchor"]
    thresholds = load_visibility_anchors(
        anchor_path,
        metric=str(policy["primary_metric"]),
        quantile=str(policy["anchor_quantile"]),
    )
    multiplier = float(policy.get("anchor_multiplier", 1.0))
    thresholds = {key: value * multiplier for key, value in thresholds.items()}
    groups = build_group_tasks(
        assignments,
        config=config,
        thresholds=thresholds,
        contract_sha256=contract_sha256,
    )
    results: list[dict[str, object]] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        completed = 0
        for group_results in executor.map(_build_group, groups):
            results.extend(group_results)
            completed += len(group_results)
            print(
                f"QWEN_VISIBILITY_V2_BUILD_PROGRESS {completed}/12000",
                flush=True,
            )
    records: list[dict[str, object]] = []
    for result in results:
        record = dict(result["record"])
        record["visibility_evidence"] = result["visibility"]
        records.append(record)
    validation = validate_visibility_records(records, policy=policy)
    manifest_path = (ROOT / config["manifest"]).resolve()
    metadata_path = (ROOT / config["metadata"]).resolve()
    manifest_sha = write_new_or_verify(manifest_path, encode_jsonl(records))
    metadata = {
        "schema_version": 1,
        "status": "frozen_manifest_only_images_not_implied_generated",
        "config": config_path.relative_to(ROOT).as_posix(),
        "config_sha256": contract_sha256,
        "source_manifest": config["source_manifest"],
        "source_manifest_sha256": config["source_manifest_sha256"],
        "visibility_anchor": config["visibility_anchor"],
        "visibility_anchor_sha256": config["visibility_anchor_sha256"],
        "causal_audit": config["causal_audit"],
        "causal_audit_sha256": config["causal_audit_sha256"],
        "passing_pilot": config["passing_pilot"],
        "passing_pilot_sha256": config["passing_pilot_sha256"],
        "passing_pilot_config": config["passing_pilot_config"],
        "passing_pilot_config_sha256": config["passing_pilot_config_sha256"],
        "screening_root": config["screening_root"],
        "manifest": manifest_path.relative_to(ROOT).as_posix(),
        "manifest_sha256": manifest_sha,
        "validation": validation,
    }
    metadata_sha = write_new_or_verify(
        metadata_path,
        (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode(),
    )
    print(
        "QWEN_VISIBILITY_V2_MANIFEST_PASS "
        + json.dumps(
            {
                "records": len(records),
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
