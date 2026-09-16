from __future__ import annotations

import json
from pathlib import Path

import pytest

from smoke.train_foundir_blur_only_flux2_continuation_ddp import (
    continuation_lr_scale,
    validate_config,
)
from smoke.train_foundir_blur_only_flux2_full_ddp import RankPositionSampler, sha256


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/foundir_blur_only_flux2_continuation_v1.json"
EVAL_CONFIG = ROOT / "configs/foundir_blur_only_flux2_continuation_development_v1.json"


def loaded_config() -> dict[str, object]:
    return json.loads(CONFIG.read_text())


def test_config_locks_parent_scope_and_global_batch() -> None:
    config = loaded_config()
    validate_config(config)
    assert config["parent"]["step"] == 10000
    assert sha256(ROOT / config["parent"]["checkpoint"]) == config["parent"]["checkpoint_sha256"]
    assert sha256(ROOT / config["parent"]["config"]) == config["parent"]["config_sha256"]
    assert (ROOT / config["parent"]["optimizer"]).is_file()
    assert config["training"]["global_batch_size"] == 32
    assert config["scope"]["development_gt_optimization"] is False


@pytest.mark.parametrize(
    ("step", "expected_lr"),
    [(10000, 5e-6), (11000, 4e-6), (12000, 3e-6), (13000, 2e-6), (14000, 1e-6)],
)
def test_absolute_continuation_lr_boundaries(step: int, expected_lr: float) -> None:
    scale = continuation_lr_scale(step, 14000, 0, 0.2)
    assert 5e-6 * scale == pytest.approx(expected_lr)


def test_continuation_lr_refuses_schedule_drift() -> None:
    with pytest.raises(RuntimeError, match="outside frozen range"):
        continuation_lr_scale(9999, 14000, 0, 0.2)
    with pytest.raises(RuntimeError, match="outside frozen range"):
        continuation_lr_scale(14001, 14000, 0, 0.2)


def test_cumulative_data_positions_start_after_parent_global_batch() -> None:
    start, stop = 10000 * 32, 12000 * 32
    values = []
    for rank in range(8):
        local = list(RankPositionSampler(start=start, stop=start + 32, rank=rank, world=8))
        values.extend(local)
    assert sorted(values) == list(range(start, start + 32))
    assert stop - start == 64000


@pytest.mark.parametrize(
    "key",
    ["qwen_degradation_conditioning", "mixed_degradations", "pixel_refiner", "development_gt_optimization"],
)
def test_config_rejects_forbidden_scope_changes(key: str) -> None:
    config = loaded_config()
    config["scope"][key] = True
    with pytest.raises(RuntimeError, match="scope changed"):
        validate_config(config)


def test_h100_and_h200_preserve_global_batch_32() -> None:
    config = loaded_config()
    profiles = config["distributed"]["hardware_profiles"]
    assert 8 * profiles["H100"]["micro_batch_size_per_rank"] * profiles["H100"]["gradient_accumulation_steps"] == 32
    assert 8 * profiles["H200"]["micro_batch_size_per_rank"] * profiles["H200"]["gradient_accumulation_steps"] == 32


def test_step12_evaluator_is_bound_to_one_staged_checkpoint() -> None:
    config = json.loads(EVAL_CONFIG.read_text())
    assert config["checkpoints"] == [{
        "step": 12000,
        "path": "checkpoints/foundir_blur_only_flux2_continuation_v1/step_012000.safetensors",
    }]
    assert config["data"]["expected_records"] == 2595
    assert config["data"]["expected_source_groups"] == 165
    source = (ROOT / "smoke/evaluate_foundir_blur_only_flux2_continuation_ddp.py").read_text()
    assert 'metadata.get("config_sha256") != parent_sha' in source
    assert 'item["sha256"] = checkpoint_sha' in source


def test_runner_stages_at_12k_before_full_development() -> None:
    runner = (ROOT / "smoke/run_foundir_blur_only_flux2_continuation_v1_8gpu.sh").read_text()
    training = runner.index("--stop-step 12000")
    evaluation = runner.index("--config \"$EVAL_CONFIG\"")
    assert training < evaluation
    assert "STOP_GATE Compare step12 against step10=25.734236/0.145406" in runner
    assert "CUDA_VISIBLE_DEVICES exactly to 0,1,2,3,4,5,6,7" in runner
