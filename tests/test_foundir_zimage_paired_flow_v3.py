from __future__ import annotations

import copy
import json
import sys
from collections import Counter
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / ".deps/zimage"))
sys.path.insert(0, str(ROOT / "third_party/diffsynth-studio"))
sys.path.insert(0, str(ROOT / "third_party/z-image/src"))

from lowlight_restoration.foundir_zimage_paired_flow_v3 import (
    evaluation_checkpoint,
    load_config,
    load_frozen_rows,
    parent_checkpoint,
    plain_velocity_mse,
    sample_assignment,
    validate_config,
    validate_gpu_inventory,
)
from diffsynth.diffusion.flow_match import FlowMatchScheduler


CONFIG = ROOT / "configs/foundir_blur_only_zimage_teacher_paired_flow_continuation_v3.json"


def config_copy():
    return json.loads(CONFIG.read_text())


def test_v3_config_parent_and_fixed_crop_contract() -> None:
    config = load_config(CONFIG)
    train, visual = load_frozen_rows(config, verify_images=False)
    assert len(train) == len(visual) == 16
    assert not ({row["source_group"] for row in train} & {row["source_group"] for row in visual})
    parent = parent_checkpoint(config, verify_hash=True)
    assert parent.name == "step_000400.safetensors"
    assert evaluation_checkpoint(config, 400) == parent
    assert evaluation_checkpoint(config, 800).name == "step_000800.safetensors"
    assert evaluation_checkpoint(config, 1200).name == "step_001200.safetensors"


def test_solver_aligned_assignment_is_exactly_balanced() -> None:
    config = load_config(CONFIG)
    training = config["training"]
    total = int(training["additional_steps"]) * int(config["distributed"]["required_world_size"])
    assignments = [
        sample_assignment(index, 16, int(training["inference_grid_steps"]))
        for index in range(total)
    ]
    counts = Counter(assignments)
    assert len(counts) == 16 * 50
    assert set(counts.values()) == {8}


def test_training_grid_is_the_exact_deployed_shift_six_grid() -> None:
    config = load_config(CONFIG)
    train_scheduler = FlowMatchScheduler("Z-Image")
    train_scheduler.set_timesteps(
        config["training"]["inference_grid_steps"], shift=config["training"]["scheduler_shift"]
    )
    inference_scheduler = FlowMatchScheduler("Z-Image")
    inference_scheduler.set_timesteps(
        config["inference"]["teacher_steps"], shift=config["inference"]["scheduler_shift"]
    )
    assert torch.equal(train_scheduler.sigmas, inference_scheduler.sigmas)
    assert torch.equal(train_scheduler.timesteps, inference_scheduler.timesteps)
    assert float(train_scheduler.sigmas[0]) == 1.0


def test_plain_velocity_mse_has_no_empirical_timestep_weight() -> None:
    prediction = torch.tensor([1.0, 3.0])
    target = torch.tensor([0.0, 1.0])
    assert plain_velocity_mse(prediction, target).item() == pytest.approx(2.5)
    config = load_config(CONFIG)
    assert config["training"]["loss_weighting"] == "plain_unweighted_velocity_mse"


@pytest.mark.parametrize(
    "field",
    [
        "distillation",
        "decoder_finetuning",
        "qwen_degradation_conditioning",
        "mixed_degradations",
        "sealed_internal_accessed",
        "official_foundir_test_accessed",
    ],
)
def test_v3_forbidden_scope_fails_closed(field: str) -> None:
    config = config_copy()
    config["scope"][field] = True
    with pytest.raises(RuntimeError, match="forbidden"):
        validate_config(config)


def test_v3_alignment_and_resume_contract_fail_closed() -> None:
    mutations = []
    for key, value in (
        ("timestep_sampling", "random_1000_step_grid"),
        ("loss_weighting", "empirical_t2i_weight"),
        ("optimizer_resume", True),
        ("initial_step", 0),
        ("additional_steps", 400),
    ):
        changed = config_copy()
        changed["training"][key] = value
        mutations.append(changed)
    for config in mutations:
        with pytest.raises(RuntimeError):
            validate_config(config)


def test_v3_h100_h200_and_batch_contract() -> None:
    config = load_config(CONFIG)
    assert validate_gpu_inventory(["NVIDIA H100 80GB HBM3"] * 8, [81920] * 8, config) == "H100"
    assert validate_gpu_inventory(["NVIDIA H200"] * 8, [143000] * 8, config) == "H200"
    invalid = copy.deepcopy(config)
    invalid["training"]["global_batch_size"] = 16
    with pytest.raises(RuntimeError, match="batch"):
        validate_config(invalid)
