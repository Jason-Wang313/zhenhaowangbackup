from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from smoke.build_foundir_blur_only_sanity_manifest import frozen_row, select, stable
from smoke.train_foundir_blur_only_flux2_overfit_ddp import scheduled_index, validate_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/foundir_blur_only_flux2_sanity_v1.json"


def loaded_config() -> dict[str, object]:
    return json.loads(CONFIG.read_text())


def test_config_locks_advisor_scope_and_official_schedule() -> None:
    config = loaded_config()
    validate_config(config)
    assert config["scope"] == {
        "category": "01Blur",
        "qwen_degradation_conditioning": False,
        "mixed_degradations": False,
        "pixel_refiner": False,
        "official_foundir_test_accessed": False,
        "sealed_internal_accessed": False,
    }
    assert config["inference"]["num_steps"] == 50
    assert config["inference"]["guidance"] == 1.0


def test_config_rejects_goal_drift() -> None:
    config = loaded_config()
    config["scope"]["mixed_degradations"] = True
    with pytest.raises(RuntimeError, match="scope contract"):
        validate_config(config)


def test_distinct_source_group_selection_is_deterministic() -> None:
    rows = [
        {"record_id": f"r{i}", "split": "train", "source_group": f"g{i // 2}"}
        for i in range(12)
    ]
    first = select(rows, split="train", count=5, seed=7)
    second = select(list(reversed(rows)), split="train", count=5, seed=7)
    assert [row["record_id"] for row in first] == [row["record_id"] for row in second]
    assert len({row["source_group"] for row in first}) == 5


def test_frozen_crop_is_in_bounds_and_deterministic() -> None:
    row = {
        "record_id": "r", "split": "train", "source_group": "g", "category": "01Blur",
        "composition_size": "1", "lq_path": "lq", "gt_path": "gt", "lq_sha256": "a" * 64,
        "gt_sha256": "b" * 64, "width": "640", "height": "600",
    }
    a = frozen_row(row, role="tiny_overfit_train", crop_size=512, crop_seed=9)
    b = frozen_row(row, role="tiny_overfit_train", crop_size=512, crop_seed=9)
    assert a == b
    assert 0 <= a["crop_x"] <= 128
    assert 0 <= a["crop_y"] <= 88


def test_schedule_visits_every_record_before_repeat() -> None:
    order = [scheduled_index(position, 16, 123) for position in range(16)]
    assert sorted(order) == list(range(16))


def test_frozen_manifest_contains_only_train_and_future_validation() -> None:
    config = loaded_config()
    path = ROOT / config["data"]["frozen_manifest"]
    metadata_path = ROOT / config["data"]["frozen_manifest_metadata"]
    assert path.is_file() and metadata_path.is_file()
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["role"] for row in rows} == {"tiny_overfit_train", "future_blur_validation"}
    assert {row["split"] for row in rows} == {"train", "validation"}
    assert {row["category"] for row in rows} == {"01Blur"}
    assert {row["composition_size"] for row in rows} == {"1"}
    metadata = json.loads(metadata_path.read_text())
    assert metadata["protected_foundir_test_accessed"] is False
    assert metadata["sealed_internal_accessed"] is False
    assert metadata["frozen_manifest_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
