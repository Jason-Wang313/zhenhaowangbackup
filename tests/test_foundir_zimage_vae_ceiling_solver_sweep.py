from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / ".deps/zimage"))
sys.path.insert(0, str(ROOT / "third_party/diffsynth-studio"))

from lowlight_restoration.foundir_zimage_vae_ceiling import (
    load_config,
    load_frozen_rows,
    teacher_checkpoint,
    validate_config,
    validate_gpu_inventory,
)
from smoke.evaluate_foundir_blur_only_zimage_vae_ceiling_solver_sweep_ddp import (
    aggregate,
    aggregate_solver,
    best_solver_steps,
)


CONFIG = ROOT / "configs/foundir_blur_only_zimage_vae_ceiling_v1.json"


def config_copy():
    return json.loads(CONFIG.read_text())


def test_ceiling_solver_config_and_artifact_contract() -> None:
    config = load_config(CONFIG)
    train, visual = load_frozen_rows(config, verify_images=False)
    assert len(train) == len(visual) == 16
    assert config["evaluation"]["teacher_inference_steps"] == [1, 2, 4, 8, 16, 50]
    assert teacher_checkpoint(config, verify_hash=True).name == "step_001200.safetensors"
    assert config["scope"]["decoder_finetuning"] is False
    assert config["scope"]["distillation"] is False


@pytest.mark.parametrize(
    "field",
    [
        "training",
        "distillation",
        "decoder_finetuning",
        "qwen_degradation_conditioning",
        "mixed_degradations",
        "sealed_internal_accessed",
        "official_foundir_test_accessed",
    ],
)
def test_ceiling_solver_forbidden_scope_fails_closed(field: str) -> None:
    config = config_copy()
    config["scope"][field] = True
    with pytest.raises(RuntimeError, match="forbidden"):
        validate_config(config)


def test_ceiling_solver_protocol_changes_fail_closed() -> None:
    wrong_steps = config_copy()
    wrong_steps["evaluation"]["teacher_inference_steps"] = [1, 4, 50]
    with pytest.raises(RuntimeError, match="sweep"):
        validate_config(wrong_steps)
    wrong_checkpoint = config_copy()
    wrong_checkpoint["teacher"]["cumulative_step"] = 800
    with pytest.raises(RuntimeError, match="terminal"):
        validate_config(wrong_checkpoint)


def test_ceiling_solver_h100_h200_contract() -> None:
    config = load_config(CONFIG)
    assert validate_gpu_inventory(["NVIDIA H100 80GB HBM3"] * 8, [81920] * 8, config) == "H100"
    assert validate_gpu_inventory(["NVIDIA H200"] * 8, [143000] * 8, config) == "H200"
    invalid = copy.deepcopy(config)
    invalid["distributed"]["required_world_size"] = 4
    with pytest.raises(RuntimeError, match="eight"):
        validate_config(invalid)


def test_metric_aggregation_rejects_empty_and_nonfinite_rows() -> None:
    with pytest.raises(RuntimeError, match="empty"):
        aggregate([])
    with pytest.raises(RuntimeError, match="empty"):
        aggregate_solver([])
    ceiling_row = {
        "input_psnr": 20.0,
        "input_lpips": 0.3,
        "lq_roundtrip_psnr": 19.0,
        "lq_roundtrip_lpips": 0.31,
        "gt_roundtrip_psnr": float("nan"),
        "gt_roundtrip_lpips": 0.01,
    }
    with pytest.raises(RuntimeError, match="non-finite"):
        aggregate([ceiling_row])


def test_best_solver_steps_are_reported_per_split() -> None:
    snapshots = []
    for role in ("tiny_overfit_train", "visual_development"):
        snapshots.extend(
            [
                {"role": role, "steps": 1, "metrics": {"output_psnr": 20.0, "output_lpips": 0.25}},
                {"role": role, "steps": 4, "metrics": {"output_psnr": 21.0, "output_lpips": 0.20}},
                {"role": role, "steps": 8, "metrics": {"output_psnr": 20.5, "output_lpips": 0.18}},
            ]
        )
    result = best_solver_steps(snapshots)
    assert result["tiny_overfit_train"] == {"best_psnr_step": 4, "best_lpips_step": 8}
    assert result["visual_development"] == {"best_psnr_step": 4, "best_lpips_step": 8}
