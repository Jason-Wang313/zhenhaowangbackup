#!/usr/bin/env python3
"""Validate and materialize the FoundIR-Lowlight Gate 1 freeze.

This program reads only non-FoundIR development data manifests and hashes the
already-present protected manifest as an exclusion index. It never opens a
protected FoundIR image.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEV_CONFIG = ROOT / "configs/foundir_lowlight_development_contract_v1.json"
TEACHER_CONFIG = ROOT / "configs/foundir_lowlight_teacher_contract_v1.json"
PROPOSAL_CONFIG = ROOT / "configs/foundir_lowlight_proposal_lattice_v1.json"
LEDGER = ROOT / "manifests/foundir_lowlight_experiment_ledger_v1.json"
OUTPUT_CSV = ROOT / "manifests/foundir_lowlight_development_contract_v1.csv"
OUTPUT_JSON = ROOT / "reports/foundir_lowlight_gate1_result_v1.json"
OUTPUT_REPORT = ROOT / "reports/foundir_lowlight_gate1_result_2026-08-22.md"

FIELDS = [
    "family",
    "source_dataset",
    "subset",
    "role",
    "group_id",
    "record_id",
    "input_path",
    "target_path",
    "input_sha256",
    "target_sha256",
    "width",
    "height",
    "stored_color_order",
    "metric_color_order",
    "alignment_dx_pixels",
    "alignment_dy_pixels",
    "alignment_confidence",
    "proxy_project_level",
    "absolute_l1_l5_label_available",
    "protected_exact_overlap",
    "redistribution_policy",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def check_identity(path: str, expected: str) -> None:
    resolved = ROOT / path
    assert resolved.is_file(), path
    assert sha256(resolved) == expected, path


def normalize_bool(value: str) -> bool:
    return value.strip().lower() == "true"


def materialize_rows(config: dict) -> list[dict[str, str]]:
    paired_parent = config["parents"]["paired_real_csv"]
    sdsd_parent = config["parents"]["sdsd_csv"]
    paired = load_csv(ROOT / paired_parent["path"])
    sdsd = load_csv(ROOT / sdsd_parent["path"])

    sice_outer = set(config["families"]["sice"]["outer_scene_ids"])
    rows: list[dict[str, str]] = []
    for row in paired:
        if row["dataset"] == "lolv2_real_train_validation":
            family = "lolv2"
            role = "outer_development"
        elif row["dataset"] == "sice_reusable_validation":
            family = "sice"
            role = "outer_development" if row["scene_id"] in sice_outer else "training"
        else:
            raise AssertionError(row["dataset"])
        rows.append(
            {
                "family": family,
                "source_dataset": row["dataset"],
                "subset": "real_captured_train" if family == "lolv2" else "reusable_test_source",
                "role": role,
                "group_id": row["scene_id"],
                "record_id": row["development_id"],
                "input_path": row["input_path"],
                "target_path": row["target_path"],
                "input_sha256": row["input_sha256"],
                "target_sha256": row["target_sha256"],
                "width": row["width"],
                "height": row["height"],
                "stored_color_order": "RGB",
                "metric_color_order": "RGB",
                "alignment_dx_pixels": "",
                "alignment_dy_pixels": "",
                "alignment_confidence": "",
                "proxy_project_level": row["proxy_project_level"],
                "absolute_l1_l5_label_available": "false",
                "protected_exact_overlap": row["exact_foundir_overlap"].lower(),
                "redistribution_policy": "preserve_parent_terms_do_not_package_dataset_bytes",
            }
        )

    for row in sdsd:
        record_id = f"sdsd:{row['subset']}:{row['sequence_id']}:{int(row['frame_position']):06d}"
        rows.append(
            {
                "family": "sdsd",
                "source_dataset": "sdsd",
                "subset": row["subset"],
                "role": row["role"],
                "group_id": row["sequence_id"],
                "record_id": record_id,
                "input_path": row["lq_path"],
                "target_path": row["gt_path"],
                "input_sha256": row["lq_file_sha256"],
                "target_sha256": row["gt_file_sha256"],
                "width": row["width"],
                "height": row["height"],
                "stored_color_order": row["stored_color_order"],
                "metric_color_order": row["metric_color_order"],
                "alignment_dx_pixels": row["alignment_dx_pixels"],
                "alignment_dy_pixels": row["alignment_dy_pixels"],
                "alignment_confidence": row["alignment_phase_response"],
                "proxy_project_level": "",
                "absolute_l1_l5_label_available": row["absolute_l1_l5_label_available"].lower(),
                "protected_exact_overlap": row["protected_exact_file_overlap"].lower(),
                "redistribution_policy": "internal_research_only_do_not_redistribute_or_package",
            }
        )
    return sorted(rows, key=lambda x: (x["family"], x["role"], x["group_id"], x["record_id"]))


def validate_rows(rows: list[dict[str, str]], config: dict) -> dict:
    assert len(rows) == config["expected_combined"]["records"]
    assert len({row["record_id"] for row in rows}) == len(rows)
    assert all(not normalize_bool(row["protected_exact_overlap"]) for row in rows)
    assert all(not normalize_bool(row["absolute_l1_l5_label_available"]) for row in rows)

    by_family_role = Counter((row["family"], row["role"]) for row in rows)
    expected = {
        ("lolv2", "outer_development"): 102,
        ("sice", "training"): 54,
        ("sice", "outer_development"): 18,
        ("sdsd", "training"): 375,
        ("sdsd", "outer_development"): 125,
    }
    assert by_family_role == expected, (by_family_role, expected)

    group_roles: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        group_roles[(row["family"], row["group_id"])].add(row["role"])
        assert (ROOT / row["input_path"]).is_file(), row["input_path"]
        assert (ROOT / row["target_path"]).is_file(), row["target_path"]
    assert all(len(roles) == 1 for roles in group_roles.values())

    for key in ("input_sha256", "target_sha256"):
        roles_by_hash: dict[str, set[str]] = defaultdict(set)
        for row in rows:
            roles_by_hash[row[key]].add(row["role"])
        assert all(len(roles) == 1 for roles in roles_by_hash.values()), key

    sice_scenes = sorted({row["group_id"] for row in rows if row["family"] == "sice"})
    expected_outer = set(
        sorted(
            sice_scenes,
            key=lambda scene: (
                hashlib.sha256(f"foundir-g1-sice-outer-v1:{scene}".encode()).hexdigest(),
                scene,
            ),
        )[:10]
    )
    assert expected_outer == set(config["families"]["sice"]["outer_scene_ids"])

    counts = {
        f"{family}:{role}": count
        for (family, role), count in sorted(by_family_role.items())
    }
    groups = Counter((row["family"], row["role"], row["group_id"]) for row in rows)
    group_counts = Counter((family, role) for family, role, _ in groups)
    return {
        "records": len(rows),
        "groups": len(group_roles),
        "records_by_family_role": counts,
        "groups_by_family_role": {
            f"{family}:{role}": count
            for (family, role), count in sorted(group_counts.items())
        },
    }


def validate_contracts() -> tuple[dict, dict, dict, dict]:
    development = load_json(DEV_CONFIG)
    teacher = load_json(TEACHER_CONFIG)
    proposal = load_json(PROPOSAL_CONFIG)
    ledger = load_json(LEDGER)

    for entry in development["parents"].values():
        check_identity(entry["path"], entry["sha256"])
    for entry in teacher["checkpoints"].values():
        check_identity(entry["path"], entry["sha256"])
        assert (ROOT / entry["path"]).stat().st_size == entry["bytes"]
    check_identity(teacher["repository"]["license_path"], teacher["repository"]["license_sha256"])

    assert proposal["development_contract"] == str(DEV_CONFIG.relative_to(ROOT))
    assert proposal["metric_contract"] == development["parents"]["metric_contract"]["path"]
    assert proposal["teacher_contract"] == str(TEACHER_CONFIG.relative_to(ROOT))
    candidate_ids = [item["id"] for item in proposal["candidates"]]
    assert len(candidate_ids) == len(set(candidate_ids)) == 12
    assert {item["id"] for item in proposal["candidates"] if item["target_access"]} == {
        "O1_EXPOSURE",
        "O2_MONOTONE_TONE_WB",
        "O3_TONE_WB_SMOOTH_ILLUMINATION",
        "O4_GAMMA_FULL_MATRIX",
        "B_CONVEX_ORACLE",
        "B_FREQUENCY_ORACLE",
    }
    historical = next(item for item in proposal["candidates"] if item["id"] == "P0_HISTORICAL_FLUX")
    check_identity(historical["checkpoint"], historical["checkpoint_sha256"])

    entries = ledger["entries"]
    assert len({entry["id"] for entry in entries}) == len(entries)
    assert ledger["seed_policy"]["finalist_seeds"] == [2026082201, 2026082202, 2026082203]
    assert ledger["seed_policy"]["deploy_predeclared_seed"] == 2026082201
    assert next(entry for entry in entries if entry["id"] == "G7_PROTECTED_EVENT")["status"] == "blocked_pending_user_authorization"
    assert all(entry["protected_test_access"] is False for entry in entries)
    return development, teacher, proposal, ledger


def write_outputs(rows: list[dict[str, str]], summary: dict, contracts: tuple[dict, dict, dict, dict]) -> None:
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    development, teacher, proposal, ledger = contracts
    identities = {
        "development_contract_sha256": sha256(DEV_CONFIG),
        "teacher_contract_sha256": sha256(TEACHER_CONFIG),
        "proposal_contract_sha256": sha256(PROPOSAL_CONFIG),
        "experiment_ledger_sha256": sha256(LEDGER),
        "combined_manifest_sha256": sha256(OUTPUT_CSV),
    }
    result = {
        "schema_version": 1,
        "status": "pass",
        "gate": 1,
        "decision": "protocol_development_teacher_and_experiment_contracts_frozen",
        "protected_foundir_status": "NOT_ACCESSED_DURING_GATE_1",
        "summary": summary,
        "identities": identities,
        "metric_gate_pass": load_json(ROOT / development["parents"]["metric_result"]["path"])["status"] == "pass",
        "teacher_deployment_status": "conditional_not_yet_admitted",
        "teacher_overlap_status": teacher["training_provenance"]["development_overlap"],
        "declared_gate2_candidate_count": len(proposal["candidates"]),
        "declared_ledger_entry_count": len(ledger["entries"]),
    }
    with OUTPUT_JSON.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")

    report = f"""# FoundIR-Lowlight Gate 1 result

Date: 2026-08-22 UTC

## Outcome

`FOUNDIR_LOWLIGHT_GATE1_PASS`

The metric implementation, three-family reusable development suite, leakage
policy, released-teacher status, experiment ledger, and Gate 2 proposal lattice
are frozen. No protected FoundIR-Lowlight image was opened and no protected
inference was performed.

## Development suite

| Family / role | Records | Groups |
|---|---:|---:|
| LOL-v2 / outer development | 102 | 41 |
| SICE / training | 54 | 31 |
| SICE / outer development | 18 | 10 |
| SDSD / training | 375 | 125 |
| SDSD / outer development | 125 | 25 |
| **Total** | **674** | **232** |

All splits are scene- or sequence-disjoint. No input or target content hash
crosses training and outer roles. Outer strata are explicitly reusable
development sets, not new hidden tests. The metric contract evaluates saved,
reloaded uint8 RGB and conservatively takes the lower PSNR and higher LPIPS
across its two frozen paths.

## Teacher decision

The released generalist and documented generalist-to-Lowlight-specialist
cascade are admitted for internal non-FoundIR proposal measurement. Their
development-family training overlap is unverifiable, checkpoint-specific
redistribution terms are not separately stated, and runtime/memory have not yet
been measured. Therefore teacher-at-inference is not a deployable dependency at
Gate 1; Gate 2 must measure value and cost, and packaging still requires
clearance.

## Identities

- Combined manifest SHA-256: `{identities['combined_manifest_sha256']}`
- Development contract SHA-256: `{identities['development_contract_sha256']}`
- Teacher contract SHA-256: `{identities['teacher_contract_sha256']}`
- Proposal contract SHA-256: `{identities['proposal_contract_sha256']}`
- Experiment ledger SHA-256: `{identities['experiment_ledger_sha256']}`

The machine-readable result is `reports/foundir_lowlight_gate1_result_v1.json`.
"""
    OUTPUT_REPORT.write_text(report, encoding="utf-8")


def main() -> None:
    contracts = validate_contracts()
    rows = materialize_rows(contracts[0])
    summary = validate_rows(rows, contracts[0])
    write_outputs(rows, summary, contracts)
    print(
        "FOUNDIR_LOWLIGHT_GATE1_PASS "
        f"records={summary['records']} groups={summary['groups']} "
        f"manifest_sha256={sha256(OUTPUT_CSV)}"
    )


if __name__ == "__main__":
    main()
