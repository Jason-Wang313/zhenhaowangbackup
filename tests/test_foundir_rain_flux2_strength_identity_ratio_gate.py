from __future__ import annotations

import json
from pathlib import Path

from smoke.train_foundir_blur_only_flux2_overfit_ddp import (
    strength_weighted_index,
    validate_config,
    validate_external_continuation,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/foundir_rain_flux2_strength_identity_ratio_gate_v1.json"


def loaded() -> dict:
    config = json.loads(CONFIG.read_text())
    config["_config_path"] = str(CONFIG)
    return config


def test_exact_ratio_only_contract() -> None:
    config = loaded()
    validate_config(config)
    validate_external_continuation(config)
    assert config["initialization"]["start_step"] == 2000
    assert config["training"]["max_steps"] == 2250
    assert config["training"]["learning_rate"] == 2.5e-5
    policy = config["training"]["strength_identity_policy"]
    assert policy["sampling_weights"] == [3, 2, 1]
    assert policy["near_clean_target"] == "lq_identity"
    assert config["gate"]["automatic_continuation"] is False


def test_absolute_positions_follow_exact_three_two_one_cycle() -> None:
    buckets = {"heavy": [10, 11], "moderate": [20, 21], "near_clean": [30, 31]}
    selected = [strength_weighted_index(i, buckets, 17, [3, 2, 1])[1] for i in range(96)]
    assert {name: selected.count(name) for name in buckets} == {
        "heavy": 48, "moderate": 32, "near_clean": 16,
    }


def test_only_scientific_change_from_equal_gate_is_sampling_ratio() -> None:
    ratio = loaded()
    equal = json.loads((ROOT / "configs/foundir_rain_flux2_strength_identity_directional_gate_v1.json").read_text())
    ignored = {"_config_path", "run_name", "status", "authorization", "runtime", "data"}
    assert {k: v for k, v in ratio.items() if k not in ignored and k != "training"} == {
        k: v for k, v in equal.items() if k not in ignored and k != "training"
    }
    ratio_training = dict(ratio["training"])
    equal_training = dict(equal["training"])
    ratio_policy = ratio_training.pop("strength_identity_policy")
    equal_policy = equal_training.pop("strength_identity_policy")
    assert ratio_training == equal_training
    assert {k: v for k, v in ratio_policy.items() if k not in {"sampling", "sampling_weights"}} == {
        k: v for k, v in equal_policy.items() if k != "sampling"
    }


def test_runner_is_bounded_and_eight_gpu() -> None:
    runner = (ROOT / "smoke/run_foundir_rain_flux2_strength_identity_ratio_gate_v1_8gpu.sh").read_text()
    assert "--nproc_per_node=8" in runner
    assert "ratio_3_2_1" in runner and "250_steps" in runner
    assert "no_automatic_continuation" in runner
