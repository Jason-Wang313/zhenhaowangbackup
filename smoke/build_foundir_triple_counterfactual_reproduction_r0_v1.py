#!/usr/bin/env python3
"""Build and validate the CPU-only counterfactual reproduction mini-protocol."""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/foundir_triple_counterfactual_reproduction_r0_v1.json"
DEGRADATION_ORDER = ("blur", "noise", "rain", "haze", "low_light")
DEGRADATION_RANK = {name: index for index, name in enumerate(DEGRADATION_ORDER)}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def ordered_subsets(order: Sequence[str]) -> list[tuple[str, ...]]:
    return [
        tuple(order[index] for index in indices)
        for size in range(len(order) + 1)
        for indices in itertools.combinations(range(len(order)), size)
    ]


def stable_rank(seed: int, record_id: str) -> str:
    return hashlib.sha256(f"{seed}:{record_id}".encode()).hexdigest()


def combination_key(record: Mapping[str, Any]) -> str:
    names = tuple(map(str, record["degradations"]))
    if any(name not in DEGRADATION_RANK for name in names):
        raise ValueError(f"unknown degradation in combination: {names}")
    return "+".join(sorted(names, key=DEGRADATION_RANK.__getitem__))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def select_parents(config: Mapping[str, Any], records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    data = config["data"]
    seed = int(data["builder_seed"])
    parent_split = str(data["parent_split"])
    candidates = [record for record in records if record["split"] == parent_split]
    candidates.sort(key=lambda record: stable_rank(seed, str(record["record_id"])))

    selected: list[dict[str, Any]] = []
    used_sources: set[str] = set()
    quotas = {
        role: {int(size): int(count) for size, count in by_size.items()}
        for role, by_size in data["selection"].items()
    }
    counts: Counter[tuple[str, int]] = Counter()
    required_triples = set(map(str, data["required_heldout_triple_combinations"]))
    triple_counts: Counter[str] = Counter()

    def add(role: str, record: dict[str, Any]) -> bool:
        size = int(record["composition_size"])
        source = str(record["source_id"])
        if source in used_sources or counts[(role, size)] >= quotas[role].get(size, 0):
            return False
        copied = dict(record)
        copied["protocol_role"] = role
        selected.append(copied)
        used_sources.add(source)
        counts[(role, size)] += 1
        if role == "heldout_triple":
            triple_counts[combination_key(record)] += 1
        return True

    # Guarantee support for every declared triple combination before filling quotas.
    for combo in sorted(required_triples):
        match = next(
            (
                record
                for record in candidates
                if int(record["composition_size"]) == 3
                and combination_key(record) == combo
                and str(record["source_id"]) not in used_sources
            ),
            None,
        )
        if match is None or not add("heldout_triple", match):
            raise RuntimeError(f"cannot satisfy heldout triple combination: {combo}")

    for role in ("train", "development", "heldout_triple"):
        for record in candidates:
            size = int(record["composition_size"])
            if size not in quotas[role]:
                continue
            add(role, record)

    missing = {
        f"{role}:{size}": wanted - counts[(role, size)]
        for role, by_size in quotas.items()
        for size, wanted in by_size.items()
        if counts[(role, size)] != wanted
    }
    if missing:
        raise RuntimeError(f"selection quotas unsatisfied: {missing}")
    if set(triple_counts) != required_triples:
        raise RuntimeError("heldout triple combination coverage is incomplete")
    return selected


def build_lattice(parent: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    order = tuple(map(str, parent["application_order"]))
    subsets = ordered_subsets(order)
    nodes = []
    for subset in subsets:
        nodes.append(
            {
                "node_id": "clean" if not subset else "+".join(subset),
                "application_order": list(subset),
                "degradations": sorted(subset),
                "levels": {name: parent["levels"][name] for name in subset},
                "operators": {name: copy.deepcopy(parent["operators"][name]) for name in subset},
            }
        )

    pairs = []
    for input_subset in subsets:
        for remove_subset in ordered_subsets(input_subset):
            remove = set(remove_subset)
            target_subset = tuple(name for name in input_subset if name not in remove)
            pairs.append(
                {
                    "input_node": "clean" if not input_subset else "+".join(input_subset),
                    "remove": list(remove_subset),
                    "target_node": "clean" if not target_subset else "+".join(target_subset),
                }
            )

    role = str(parent["protocol_role"])
    protocol = config["protocol"]
    return {
        "schema_version": 1,
        "parent_record_id": parent["record_id"],
        "protocol_role": role,
        "gradient_eligible": role in protocol["gradient_eligible_roles"],
        "checkpoint_selection_eligible": role in protocol["checkpoint_selection_roles"],
        "source_id": parent["source_id"],
        "source_path": parent["source_path"],
        "source_sha256": parent["source_sha256"],
        "crop": copy.deepcopy(parent["crop"]),
        "non_target_nuisance": copy.deepcopy(parent.get("non_target_nuisance")),
        "composition_size": parent["composition_size"],
        "degradations": parent["degradations"],
        "levels": parent["levels"],
        "application_order": parent["application_order"],
        "operators": copy.deepcopy(parent["operators"]),
        "agenticir_revision": parent["agenticir_revision"],
        "nodes": nodes,
        "counterfactual_pairs": pairs,
    }


def validate_lattices(config: Mapping[str, Any], lattices: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    protocol = config["protocol"]
    role_sources: dict[str, set[str]] = defaultdict(set)
    role_sizes: Counter[tuple[str, int]] = Counter()
    triple_combos: Counter[str] = Counter()
    verified_sources: set[str] = set()

    for lattice in lattices:
        role = str(lattice["protocol_role"])
        size = int(lattice["composition_size"])
        source_id = str(lattice["source_id"])
        role_sources[role].add(source_id)
        role_sizes[(role, size)] += 1
        if role == "heldout_triple":
            triple_combos[combination_key(lattice)] += 1

        if lattice["gradient_eligible"] and size > int(protocol["maximum_gradient_composition_size"]):
            raise ValueError("triple leaked into gradient-eligible data")
        if lattice["checkpoint_selection_eligible"] and size > int(protocol["maximum_selection_composition_size"]):
            raise ValueError("triple leaked into checkpoint selection")
        if role == "heldout_triple" and size != int(protocol["heldout_composition_size"]):
            raise ValueError("heldout role contains a non-triple parent")
        if role == "heldout_triple" and (lattice["gradient_eligible"] or lattice["checkpoint_selection_eligible"]):
            raise ValueError("heldout triple is not fail-closed")

        order = tuple(map(str, lattice["application_order"]))
        if len(order) != size or len(order) != len(set(order)):
            raise ValueError("invalid parent application order")
        if set(order) != set(map(str, lattice["degradations"])):
            raise ValueError("parent degradation/order mismatch")
        if set(lattice["levels"]) != set(order) or set(lattice["operators"]) != set(order):
            raise ValueError("parent component metadata mismatch")

        expected_nodes = {"clean" if not subset else "+".join(subset): subset for subset in ordered_subsets(order)}
        actual_nodes = {str(node["node_id"]): node for node in lattice["nodes"]}
        if set(actual_nodes) != set(expected_nodes):
            raise ValueError("counterfactual lattice node coverage mismatch")
        for node_id, subset in expected_nodes.items():
            node = actual_nodes[node_id]
            if tuple(node["application_order"]) != subset:
                raise ValueError("counterfactual node reordered operators")
            for name in subset:
                if canonical_json(node["operators"][name]) != canonical_json(lattice["operators"][name]):
                    raise ValueError("counterfactual node changed operator parameters")
                if node["levels"][name] != lattice["levels"][name]:
                    raise ValueError("counterfactual node changed severity")

        expected_pairs = {
            (
                "clean" if not input_subset else "+".join(input_subset),
                tuple(remove_subset),
                "clean" if not target else "+".join(target),
            )
            for input_subset in ordered_subsets(order)
            for remove_subset in ordered_subsets(input_subset)
            for target in [tuple(name for name in input_subset if name not in set(remove_subset))]
        }
        actual_pairs = {
            (str(pair["input_node"]), tuple(pair["remove"]), str(pair["target_node"]))
            for pair in lattice["counterfactual_pairs"]
        }
        if actual_pairs != expected_pairs:
            raise ValueError("counterfactual pair algebra mismatch")

        if config["data"]["verify_source_sha256"] and source_id not in verified_sources:
            source_path = ROOT / str(lattice["source_path"])
            if sha256_file(source_path) != str(lattice["source_sha256"]):
                raise ValueError(f"source hash mismatch: {source_id}")
            verified_sources.add(source_id)

    roles = sorted(role_sources)
    for index, left in enumerate(roles):
        for right in roles[index + 1 :]:
            overlap = role_sources[left] & role_sources[right]
            if overlap:
                raise ValueError(f"source overlap between {left} and {right}: {sorted(overlap)[:3]}")

    declared = set(map(str, config["data"]["required_heldout_triple_combinations"]))
    if set(triple_combos) != declared:
        raise ValueError("declared heldout triple coverage mismatch")
    return {
        "status": "pass",
        "parents": len(lattices),
        "nodes": sum(len(lattice["nodes"]) for lattice in lattices),
        "counterfactual_pairs": sum(len(lattice["counterfactual_pairs"]) for lattice in lattices),
        "role_size_counts": {
            f"{role}:{size}": count for (role, size), count in sorted(role_sizes.items())
        },
        "unique_sources": len(verified_sources),
        "heldout_triple_combinations": dict(sorted(triple_combos.items())),
        "triple_gradient_leakage": 0,
        "triple_selection_leakage": 0,
        "official_foundir_test_accessed": False,
        "foundir_60_image_triple_accessed": False,
    }


def write_or_verify(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"refusing to overwrite non-identical artifact: {path}")
        return
    path.write_bytes(payload)


def build(config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    parent_path = ROOT / config["data"]["parent_manifest"]
    if sha256_file(parent_path) != config["data"]["parent_manifest_sha256"]:
        raise RuntimeError("parent manifest hash mismatch")
    records = load_jsonl(parent_path)
    parents = select_parents(config, records)
    lattices = [build_lattice(parent, config) for parent in parents]
    audit = validate_lattices(config, lattices)

    output_path = ROOT / config["data"]["output_manifest"]
    payload = b"".join((canonical_json(lattice) + "\n").encode() for lattice in lattices)
    metadata = {
        "schema_version": 1,
        "status": "foundir_triple_counterfactual_reproduction_r0_v1_cpu_ready",
        "config_path": str(config_path.relative_to(ROOT)),
        "config_sha256": sha256_file(config_path),
        "parent_manifest": config["data"]["parent_manifest"],
        "parent_manifest_sha256": config["data"]["parent_manifest_sha256"],
        "output_manifest": config["data"]["output_manifest"],
        "output_manifest_sha256": hashlib.sha256(payload).hexdigest(),
        "audit": audit,
        "gpu_training_launched": False,
        "gpu_hours": 0.0,
    }
    metadata_payload = (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode()
    write_or_verify(output_path, payload)
    write_or_verify(ROOT / config["data"]["output_metadata"], metadata_payload)
    return metadata


def check(config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    manifest_path = ROOT / config["data"]["output_manifest"]
    metadata_path = ROOT / config["data"]["output_metadata"]
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if sha256_file(manifest_path) != metadata["output_manifest_sha256"]:
        raise RuntimeError("output manifest hash mismatch")
    lattices = load_jsonl(manifest_path)
    audit = validate_lattices(config, lattices)
    if canonical_json(audit) != canonical_json(metadata["audit"]):
        raise RuntimeError("stored audit differs from recomputed audit")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve()
    result = check(config_path) if args.check else build(config_path)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
