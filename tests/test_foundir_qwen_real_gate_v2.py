from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_contract_is_inference_only_and_keeps_official_test_protected() -> None:
    config = json.loads(
        (ROOT / "configs/foundir_five_type_qwen_real_gate_v2.json").read_text()
    )
    assert config["scope"] == "inference_only_no_qwen_training"
    assert config["qwen_training_performed"] is False
    assert config["restoration_training_performed"] is False
    assert config["protected_foundir_official_test_accessed"] is False
    assert config["aggregation"] == "second_highest"
    assert config["records"] == 128
    assert config["thresholds"] == {
        "minimum_valid_output_rate": 0.99,
        "minimum_type_micro_f1": 0.80,
        "minimum_type_macro_f1": 0.70,
        "minimum_degradation_set_exact": 0.60,
        "minimum_each_type_recall": 0.50,
    }


def test_launcher_is_eight_gpu_and_never_trains_qwen() -> None:
    runner = (ROOT / "smoke/run_foundir_five_type_qwen_real_gate_v2_8gpu.sh").read_text()
    launcher = (
        ROOT / "smoke/launch_foundir_five_type_qwen_real_gate_v2_8gpu.sh"
    ).read_text()
    for text in (runner, launcher):
        assert "0,1,2,3,4,5,6,7" in text
        assert "official" in text.lower()
    assert "--nproc_per_node=8" in runner
    assert "--check-only" in runner
    assert "--verify-existing" in runner
    assert "train_qwen" not in runner
    assert "remain-on-exit" in launcher
    assert "nohup env CUDA_VISIBLE_DEVICES" in launcher
