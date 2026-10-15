import csv
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_mixed_pilot_is_training_only_and_frozen():
    config = json.loads((ROOT / "configs/foundir_mixed_restoration_pilot_v1.json").read_text())
    assert config["scope"] == {
        "split": "train",
        "mixed_degradations": True,
        "model_weight_updates": False,
        "checkpoint_selection_reopened": False,
        "development_pixels_accessed": False,
        "official_foundir_test_accessed": False,
    }
    assert config["inference"]["stage_order"] == ["rain", "noise", "haze", "low_light", "blur"]
    assert config["inference"]["num_steps"] == 50


def test_mixed_pilot_manifest_is_balanced_and_globally_source_disjoint():
    with (ROOT / "manifests/foundir_mixed_restoration_pilot_v1.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 96
    assert len({row["record_id"] for row in rows}) == 96
    assert len({row["source_group"] for row in rows}) == 96
    assert set(row["split"] for row in rows) == {"train"}
    assert set(Counter(row["category"] for row in rows).values()) == {16}


def test_orders_have_one_fixed_topological_stage_order():
    config = json.loads((ROOT / "configs/foundir_mixed_restoration_pilot_v1.json").read_text())
    stage_positions = {stage: index for index, stage in enumerate(config["inference"]["stage_order"])}
    for order in config["orders"].values():
        assert order == sorted(order, key=stage_positions.__getitem__)
        assert len(order) in {2, 3}
