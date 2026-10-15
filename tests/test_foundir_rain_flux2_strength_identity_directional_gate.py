from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from smoke.train_foundir_blur_only_flux2_overfit_ddp import (
    strength_balanced_index,
    validate_config,
    validate_external_continuation,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/foundir_rain_flux2_strength_identity_directional_gate_v1.json"


def loaded() -> dict:
    config = json.loads(CONFIG.read_text())
    config["_config_path"] = str(CONFIG)
    return config


def test_exact_bounded_identity_directional_contract() -> None:
    config = loaded()
    validate_config(config)
    validate_external_continuation(config)
    assert config["initialization"]["start_step"] == 2000
    assert config["training"]["max_steps"] == 2250
    assert config["training"]["learning_rate"] == 2.5e-5
    assert config["inference"]["evaluate_steps"] == [2000, 2250]
    assert config["training"]["strength_identity_policy"] == {
        "sampling": "equal_three_bins_absolute_position",
        "input_psnr_bins": [28.0, 34.0],
        "heavy_target": "gt",
        "moderate_target": "gt",
        "near_clean_target": "lq_identity",
    }
    assert config["gate"]["automatic_continuation"] is False


def test_parent_artifacts_are_hash_bound() -> None:
    initialization = loaded()["initialization"]
    for path_key, hash_key in (
        ("parent_config", "parent_config_sha256"),
        ("parent_checkpoint", "parent_checkpoint_sha256"),
        ("parent_optimizer", "parent_optimizer_sha256"),
        ("parent_result", "parent_result_sha256"),
    ):
        assert hashlib.sha256((ROOT / initialization[path_key]).read_bytes()).hexdigest() == initialization[hash_key]


def test_manifest_reuses_training_and_freezes_balanced_disjoint_development() -> None:
    config = loaded()
    data = config["data"]
    with (ROOT / data["frozen_manifest"]).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    with (ROOT / data["parent_manifest"]).open(newline="") as handle:
        parent = list(csv.DictReader(handle))
    train = [row for row in rows if row["role"] == data["train_role"]]
    development = [row for row in rows if row["role"] == data["validation_role"]]
    parent_train = [row for row in parent if row["role"] == "pilot_train"]
    strip_role = lambda row: {key: value for key, value in row.items() if key != "role"}
    assert list(map(strip_role, train)) == list(map(strip_role, parent_train))
    assert len(train) == len({row["source_group"] for row in train}) == 1580
    assert len(development) == len({row["source_group"] for row in development}) == 48
    assert not ({row["source_group"] for row in train} & {row["source_group"] for row in development})
    metadata = json.loads((ROOT / data["frozen_manifest_metadata"]).read_text())
    assert metadata["development_strength_bin_counts"] == {"heavy": 16, "moderate": 16, "near_clean": 16}
    assert metadata["config_sha256"] == hashlib.sha256(CONFIG.read_bytes()).hexdigest()


def test_absolute_positions_sample_bins_equally() -> None:
    buckets = {"heavy": [10, 11], "moderate": [20, 21], "near_clean": [30, 31]}
    chosen = [strength_balanced_index(position, buckets, 17)[1] for position in range(96)]
    assert {name: chosen.count(name) for name in buckets} == {name: 32 for name in buckets}


@pytest.mark.parametrize("field,value", [("max_steps", 2251), ("learning_rate", 5e-5)])
def test_rejects_training_drift(field: str, value: float) -> None:
    config = loaded()
    config["training"][field] = value
    with pytest.raises(RuntimeError):
        validate_config(config)


def test_runner_is_eight_gpu_and_has_no_automatic_extension() -> None:
    runner = (ROOT / "smoke/run_foundir_rain_flux2_strength_identity_directional_gate_v1_8gpu.sh").read_text()
    assert "--nproc_per_node=8" in runner
    assert "250_steps" in runner and "near_clean_LQ_identity" in runner
    assert "automatic_continuation" in runner
    assert "--require-decision" in runner
