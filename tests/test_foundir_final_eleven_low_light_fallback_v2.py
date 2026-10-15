from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from lowlight_restoration.final_eleven_route_router import safe_expert_plan

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "manifests/foundir_final_eleven_all_in_one_development_rehearsal_v2.csv"
CONFIG = ROOT / "configs/foundir_final_eleven_all_in_one_development_rehearsal_v2.json"


def test_dark_abstention_plan_is_bounded() -> None:
    assert safe_expert_plan(None, exposure=0.15) == ("low_light",)
    assert safe_expert_plan(None, exposure=0.150001) == ()
    assert safe_expert_plan("rain", exposure=0.01) == ("rain",)


def test_v2_manifest_changes_only_bounded_dark_abstentions() -> None:
    rows = list(csv.DictReader(MANIFEST.open()))
    assert len(rows) == 1408
    assert Counter(row["routing_status"] for row in rows) == {
        "correct": 1247,
        "abstain": 93,
        "fallback": 67,
        "wrong": 1,
    }
    fallback = [row for row in rows if row["routing_status"] == "fallback"]
    assert Counter(row["router_truth"] for row in fallback) == {
        "low_light": 18,
        "haze+low_light": 4,
        "blur+low_light": 10,
        "noise+low_light": 15,
        "blur+noise+low_light": 20,
    }
    assert all(float(row["fallback_exposure"]) <= 0.15 for row in fallback)
    assert all(json.loads(row["expert_order_json"]) == ["low_light"] for row in fallback)


def test_v2_config_is_development_only_and_hash_pinned() -> None:
    config = json.loads(CONFIG.read_text())
    assert config["scope"]["model_weight_updates"] is False
    assert config["scope"]["official_foundir_test_accessed"] is False
    assert config["scope"]["not_an_independent_lockbox"] is True
    assert config["fallback"] == {
        "when": "router_abstains",
        "maximum_exposure": 0.15,
        "maximum_side": 128,
        "expert_plan": ["low_light"],
        "model_weight_updates": False,
    }
    assert config["data"]["manifest_sha256"] == hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
