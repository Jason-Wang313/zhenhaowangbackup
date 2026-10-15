from __future__ import annotations

import json
from pathlib import Path

import pytest

from smoke.freeze_foundir_haze_flux2_source_diverse_pilot import select_rows
from smoke.train_foundir_blur_only_flux2_overfit_ddp import validate_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/foundir_haze_flux2_source_diverse_pilot_v1.json"


def loaded() -> dict:
    config = json.loads(CONFIG.read_text())
    config["_config_path"] = str(CONFIG)
    return config


def test_contract_is_exact_and_simple() -> None:
    config = loaded()
    validate_config(config)
    assert config["scope"]["category"] == "08Haze"
    assert config["data"]["tiny_overfit_records"] == 1580
    assert config["data"]["frozen_validation_records"] == 128
    assert config["training"]["max_steps"] == 2000
    assert config["training"]["global_batch_size"] == 32
    assert config["lora"]["initialization"] == "fresh_zero_lora"
    assert config["gate"]["automatic_full_development_launch"] is False
    for key in ("qwen_degradation_conditioning", "mixed_degradations", "pixel_refiner"):
        assert config["scope"][key] is False


def test_manifest_selection_is_source_disjoint_and_excludes_prior_gate() -> None:
    config = loaded()
    rows, available = select_rows(config)
    train = [row for row in rows if row["role"] == "pilot_train"]
    validation = [row for row in rows if row["role"] == "pilot_validation"]
    assert len(train) == len({row["source_group"] for row in train}) == 1580
    assert len(validation) == len({row["source_group"] for row in validation}) == 128
    assert not ({row["source_group"] for row in train} & {row["source_group"] for row in validation})
    assert available["pilot_train"] >= 1580
    assert available["pilot_validation"] >= 128
    excluded = set()
    import csv
    with (ROOT / config["data"]["excluded_manifest"]).open(newline="") as handle:
        excluded = {row["source_group"] for row in csv.DictReader(handle) if row["category"] == "08Haze"}
    assert not ({row["source_group"] for row in rows} & excluded)


@pytest.mark.parametrize("field,value", [
    ("tiny_overfit_records", 1579),
    ("frozen_validation_records", 127),
])
def test_contract_rejects_cardinality_drift(field: str, value: int) -> None:
    config = loaded()
    config["data"][field] = value
    with pytest.raises(RuntimeError, match="pilot contract changed"):
        validate_config(config)


def test_runner_is_eight_gpu_and_stops_after_pilot() -> None:
    runner = (ROOT / "smoke/run_foundir_haze_flux2_source_diverse_pilot_v1_8gpu.sh").read_text()
    assert "--nproc_per_node=8" in runner
    assert "no_Qwen_degradation no_mixed no_pixel_refiner" in runner
    assert "automatic_full_development_launch" in runner
    assert "--require-decision" in runner
