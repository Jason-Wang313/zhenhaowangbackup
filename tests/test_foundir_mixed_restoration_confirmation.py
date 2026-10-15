import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_confirmation_is_balanced_fresh_and_training_only():
    metadata = json.loads((ROOT / "manifests/foundir_mixed_restoration_confirmation_v1.json").read_text())
    with (ROOT / "manifests/foundir_mixed_restoration_confirmation_v1.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 640
    assert len({row["record_id"] for row in rows}) == 640
    assert len({row["source_group"] for row in rows}) == 640
    assert Counter(row["category"] for row in rows) == Counter({category: 128 for category in metadata["expert_orders"]})
    assert {row["split"] for row in rows} == {"train"}
    assert metadata["pilot_record_overlap"] == 0
    assert metadata["pilot_source_group_overlap"] == 0
    assert metadata["qwen_record_overlap"] == 0
    assert metadata["singleton_training_source_disjoint"] is False
    assert metadata["singleton_training_source_group_overlap"] == 640
    assert metadata["official_foundir_test_accessed"] is False


def test_confirmation_config_freezes_existing_orders_and_requires_bootstrap():
    config = json.loads((ROOT / "configs/foundir_mixed_restoration_confirmation_v1.json").read_text())
    assert config["data"]["records"] == 640
    assert config["data"]["records_per_category"] == 128
    assert config["scope"]["model_weight_updates"] is False
    assert config["scope"]["checkpoint_selection_reopened"] is False
    assert config["scope"]["pilot_source_disjoint"] is True
    assert config["scope"]["singleton_training_source_disjoint"] is False
    assert config["gate"]["source_group_bootstrap_required"] is True
    assert config["scope"]["development_pixels_accessed"] is False
    assert config["scope"]["official_foundir_test_accessed"] is False
