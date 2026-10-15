from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "manifests/foundir_final_eleven_route_integration_v1.csv"
CONFIGS = [
    ROOT / "configs/foundir_final_eleven_route_integration_routed_v1.json",
    ROOT / "configs/foundir_final_eleven_route_integration_direct_v1.json",
]
EXPECTED_ORDERS = {
    "01Blur": ["blur"],
    "02Blur_Noise": ["blur_noise"],
    "05Noise": ["noise"],
    "08Haze": ["haze"],
    "09Lowlight_Haze": ["haze", "low_light"],
    "10Rain": ["rain"],
    "13Rain_Haze": ["rain", "haze"],
    "14Lowlight": ["low_light"],
    "15Lowlight_Blur": ["low_light", "blur"],
    "16Lowlight_Noise": ["noise", "low_light"],
    "18Lowlight_Blur_Noise": ["noise", "low_light", "blur"],
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_manifest_is_balanced_and_source_disjoint_within_category() -> None:
    with MANIFEST.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 176
    assert Counter(row["category"] for row in rows) == Counter({key: 16 for key in EXPECTED_ORDERS})
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["category"]].append(row)
        assert row["split"] == "validation"
        assert row["routing_status"] == "correct"
        assert row["router_truth"] == row["final_route"]
        assert min(int(row["width"]), int(row["height"])) >= 512
        assert json.loads(row["expert_order_json"]) == EXPECTED_ORDERS[row["category"]]
    assert all(len({row["source_group"] for row in values}) == 16 for values in grouped.values())


def test_configs_are_frozen_development_only_twins() -> None:
    routed, direct = [json.loads(path.read_text()) for path in CONFIGS]
    assert routed["integration_role"] == "routed"
    assert direct["integration_role"] == "direct"
    for config in (routed, direct):
        assert config["scope"]["model_weight_updates"] is False
        assert config["scope"]["development_pixels_accessed"] is True
        assert config["scope"]["official_foundir_test_accessed"] is False
        assert config["orders"] == EXPECTED_ORDERS
        assert config["data"]["manifest_sha256"] == sha256(MANIFEST)
        assert config["inference"]["stage_order"] == ["rain", "blur_noise", "noise", "haze", "low_light", "blur"]
    assert routed["orders"] == direct["orders"]
    assert routed["experts"] == direct["experts"]
    assert routed["inference"] == direct["inference"]


def test_blur_noise_is_project_checkpoint_not_released_foundir() -> None:
    config = json.loads(CONFIGS[0].read_text())
    expert = config["experts"]["blur_noise"]
    assert expert["checkpoint"] == "checkpoints/foundir_blur_noise_restormer_pilot_v1/step_001000.pt"
    assert expert["checkpoint_step"] == 1000
    assert "FoundIR" not in expert["checkpoint"]
