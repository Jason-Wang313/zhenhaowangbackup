from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_manifest_contains_only_correctly_accepted_development_rows() -> None:
    with (ROOT / "manifests/foundir_noise_family_integrated_development_v1.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1842
    assert Counter(row["final_route"] for row in rows) == Counter({
        "noise": 674,
        "noise+low_light": 1168,
    })
    assert {row["split"] for row in rows} == {"validation"}
    assert {row["routing_status"] for row in rows} == {"correct"}
    assert all(row["router_truth"] == row["final_route"] for row in rows)
    assert all(
        json.loads(row["expert_order_json"])
        == (["noise"] if row["final_route"] == "noise" else ["noise", "low_light"])
        for row in rows
    )
    with (ROOT / "checkpoints/foundir_noise_family_fail_closed_consensus_v1/development_decisions.csv").open(newline="") as handle:
        decisions = list(csv.DictReader(handle))
    expected_ids = {
        row["record_id"] for row in decisions
        if row["status"] == "correct" and row["final_route"] in {"noise", "noise+low_light"}
    }
    assert {row["record_id"] for row in rows} == expected_ids


def test_config_is_frozen_inference_only_and_forbids_official_test() -> None:
    config = json.loads((ROOT / "configs/foundir_noise_family_integrated_development_v1.json").read_text())
    assert config["scope"]["split"] == "validation"
    assert config["scope"]["model_weight_updates"] is False
    assert config["scope"]["development_pixels_accessed"] is True
    assert config["scope"]["official_foundir_test_accessed"] is False
    assert config["orders"] == {
        "05Noise": ["noise"],
        "16Lowlight_Noise": ["noise", "low_light"],
    }
    assert config["data"]["records_by_category"] == {
        "05Noise": 674,
        "16Lowlight_Noise": 1168,
    }
    assert config["inference"]["num_steps"] == 50
    assert config["gate"]["absolute_thresholds"]["05Noise"] == {
        "psnr_minimum": 38.9,
        "lpips_maximum": 0.10,
    }
