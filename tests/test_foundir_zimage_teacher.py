from __future__ import annotations

import copy
from pathlib import Path

import pytest

from lowlight_restoration.foundir_zimage import (
    ROOT,
    load_aligned_pair,
    load_config,
    load_frozen_rows,
    validate_config,
    validate_gpu_inventory,
)


CONFIG = ROOT / "configs/foundir_blur_only_zimage_teacher_overfit_v1.json"


def test_config_and_frozen_pair_contract() -> None:
    config = load_config(CONFIG)
    train, visual = load_frozen_rows(config, verify_images=False)
    assert len(train) == len(visual) == 16
    assert not ({row["source_group"] for row in train} & {row["source_group"] for row in visual})
    lq, gt = load_aligned_pair(train[0], 512)
    assert lq.size == gt.size == (512, 512)


@pytest.mark.parametrize(
    "field",
    ["distillation", "decoder_finetuning", "qwen_degradation_conditioning", "mixed_degradations"],
)
def test_forbidden_scope_fails_closed(field: str) -> None:
    config = load_config(CONFIG)
    config = copy.deepcopy(config)
    config.pop("_config_path")
    config["scope"][field] = True
    with pytest.raises(RuntimeError, match="forbidden"):
        validate_config(config)


def test_h100_and_h200_profiles_are_supported_but_mixed_is_rejected() -> None:
    config = load_config(CONFIG)
    assert validate_gpu_inventory(["NVIDIA H100 80GB HBM3"] * 8, [81920] * 8, config) == "H100"
    assert validate_gpu_inventory(["NVIDIA H200"] * 8, [143000] * 8, config) == "H200"
    with pytest.raises(RuntimeError, match="mixed"):
        validate_gpu_inventory(
            ["NVIDIA H100 80GB HBM3"] * 4 + ["NVIDIA H200"] * 4,
            [81920] * 4 + [143000] * 4,
            config,
        )


def test_global_batch_and_teacher_order_fail_closed() -> None:
    config = load_config(CONFIG)
    config = copy.deepcopy(config)
    config.pop("_config_path")
    config["training"]["global_batch_size"] = 16
    with pytest.raises(RuntimeError, match="global batch"):
        validate_config(config)

    config = load_config(CONFIG)
    config = copy.deepcopy(config)
    config.pop("_config_path")
    config["scope"]["distillation"] = True
    with pytest.raises(RuntimeError):
        validate_config(config)
