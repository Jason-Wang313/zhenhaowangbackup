from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from smoke.train_foundir_blur_only_flux2_overfit_ddp import (
    validate_config,
    validate_external_continuation,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/foundir_haze_flux2_source_diverse_continuation_v1.json"


def loaded() -> dict:
    config = json.loads(CONFIG.read_text())
    config["_config_path"] = str(CONFIG)
    return config


def test_exact_bounded_continuation_contract() -> None:
    config = loaded()
    validate_config(config)
    validate_external_continuation(config)
    assert config["initialization"]["start_step"] == 2000
    assert config["training"]["max_steps"] == 3000
    assert config["training"]["checkpoint_steps"] == [3000]
    assert config["inference"]["evaluate_steps"] == [3000]
    assert config["data"]["tiny_overfit_records"] == 1580
    assert config["data"]["frozen_validation_records"] == 128
    assert config["lora"]["initialization"] == "resume_exact_checkpoint_and_optimizer"


def test_parent_artifacts_are_hash_bound() -> None:
    config = loaded()
    initialization = config["initialization"]
    for path_key, hash_key in (
        ("parent_config", "parent_config_sha256"),
        ("parent_checkpoint", "parent_checkpoint_sha256"),
        ("parent_optimizer", "parent_optimizer_sha256"),
        ("parent_result", "parent_result_sha256"),
    ):
        path = ROOT / initialization[path_key]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == initialization[hash_key]


@pytest.mark.parametrize("field,value", [("start_step", 1999), ("start_step", 2001)])
def test_rejects_step_offset_drift(field: str, value: int) -> None:
    config = loaded()
    config["initialization"][field] = value
    with pytest.raises(RuntimeError, match="continuation contract changed"):
        validate_config(config)


def test_runner_normalizes_scientific_result_without_runtime_failure() -> None:
    runner = (ROOT / "smoke/run_foundir_haze_flux2_source_diverse_continuation_v1_8gpu.sh").read_text()
    assert "--nproc_per_node=8" in runner
    assert "trainer_status\" -ne 0" in runner
    assert "automatic_gate_pass" in runner
    assert "SCIENTIFIC_FAILURE_STOP" in runner
    assert "no_new_data no_new_loss" in runner
