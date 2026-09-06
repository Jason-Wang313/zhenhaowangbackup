from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_contract_binds_physical_v8_validation_without_protected_tests() -> None:
    config = json.loads(
        (ROOT / "configs/foundir_five_type_qwen_real_gate_physical_v8.json").read_text()
    )
    assert config["status"] == "frozen_foundir_training_only_qwen_real_gate_physical_v8"
    assert config["scope"] == "inference_only_no_qwen_training"
    assert config["checkpoint_family"] == "physical_v8_step2500"
    assert config["qwen_training_performed"] is False
    assert config["restoration_training_performed"] is False
    assert config["protected_foundir_official_test_accessed"] is False
    assert config["protected_synthetic_test_accessed"] is False
    assert "synthetic_test" not in config["artifacts"]
    assert config["aggregation"] == "second_highest"
    assert config["records"] == 128


def test_outputs_are_new_and_launcher_is_eight_h100_h200_gpu() -> None:
    config = json.loads(
        (ROOT / "configs/foundir_five_type_qwen_real_gate_physical_v8.json").read_text()
    )
    assert "physical_v8" in config["raw_output"]
    assert "physical_v8" in config["output"]
    runner = (
        ROOT / "smoke/run_foundir_five_type_qwen_real_gate_physical_v8_8gpu.sh"
    ).read_text()
    launcher = (
        ROOT / "smoke/launch_foundir_five_type_qwen_real_gate_physical_v8_8gpu.sh"
    ).read_text()
    for text in (runner, launcher):
        assert "0,1,2,3,4,5,6,7" in text
        assert "official" in text.lower()
        assert "synthetic" in text.lower()
    assert "H(100|200)" in runner
    assert "--nproc_per_node=8" in runner
    assert "--check-only" in runner
    assert "--verify-existing" in runner
    assert "train_qwen" not in runner
    assert "remain-on-exit" in launcher
    assert "nohup env CUDA_VISIBLE_DEVICES" in launcher
