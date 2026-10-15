from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "manifests/foundir_final_eleven_all_in_one_development_rehearsal_v1.csv"
CONFIG = ROOT / "configs/foundir_final_eleven_all_in_one_development_rehearsal_v1.json"


def test_manifest_is_balanced_and_keeps_router_failures() -> None:
    rows = list(csv.DictReader(MANIFEST.open()))
    assert len(rows) == 1408
    assert set(Counter(row["category"] for row in rows).values()) == {128}
    assert Counter(row["routing_status"] for row in rows) == {"correct": 1247, "abstain": 160, "wrong": 1}
    for row in rows:
        order = json.loads(row["expert_order_json"])
        assert min(int(row["width"]), int(row["height"])) >= 512
        assert (row["routing_status"] == "abstain") == (order == [])


def test_config_is_frozen_development_only() -> None:
    config = json.loads(CONFIG.read_text())
    assert config["scope"]["model_weight_updates"] is False
    assert config["scope"]["official_foundir_test_accessed"] is False
    assert config["scope"]["not_an_independent_lockbox"] is True
    assert config["inference"]["abstention_output"] == "identity_input"
    assert config["data"]["manifest_sha256"] == hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
    assert config["gate"]["absolute_thresholds"]["01Blur"] == {"psnr_minimum": 25.9, "lpips_maximum": 0.18}
