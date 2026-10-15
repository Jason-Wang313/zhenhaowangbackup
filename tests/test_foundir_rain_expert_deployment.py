from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from lowlight_restoration.rain_expert import (
    FALLBACK_BACKEND,
    RAIN_BACKEND,
    load_rain_expert,
    restore_isolated_rain,
    select_restoration_backend,
    select_restoration_backend_for_category,
    validate_rain_expert_config,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/foundir_rain_restormer_deployment_selection_v1.json"


def test_frozen_selection_contract_and_sow_evidence() -> None:
    config = validate_rain_expert_config(CONFIG)
    evidence = config["evidence"]
    assert evidence["mean_psnr"] == pytest.approx(42.2843466542096)
    assert evidence["mean_lpips"] == pytest.approx(0.047452168240756334)
    assert evidence["mean_psnr"] >= 34.2 and evidence["mean_lpips"] <= 0.08
    assert evidence["legacy_stronger_bootstrap_gate_pass"] is False


@pytest.mark.parametrize("label", ["blur", "noise", "haze", "low_light"])
def test_every_other_single_type_keeps_existing_backend(label: str) -> None:
    assert select_restoration_backend([label]) == FALLBACK_BACKEND


@pytest.mark.parametrize(
    "labels",
    [[], ["rain", "haze"], ["blur", "rain"], ["rain", "noise"], ["blur", "noise", "rain", "haze", "low_light"]],
)
def test_empty_and_mixed_inputs_keep_existing_backend(labels: list[str]) -> None:
    assert select_restoration_backend(labels) == FALLBACK_BACKEND


def test_only_exact_singleton_rain_activates_expert() -> None:
    assert select_restoration_backend(["rain"]) == RAIN_BACKEND
    assert select_restoration_backend_for_category("10Rain", 1) == RAIN_BACKEND
    for category in ("01Blur", "05Noise", "08Haze", "14Lowlight", "13Rain_Haze"):
        assert select_restoration_backend_for_category(category, 1) == FALLBACK_BACKEND
    assert select_restoration_backend_for_category("10Rain", 2) == FALLBACK_BACKEND


def test_invalid_labels_fail_closed() -> None:
    with pytest.raises(ValueError, match="unknown"):
        select_restoration_backend(["snow"])
    with pytest.raises(ValueError, match="duplicate"):
        select_restoration_backend(["rain", "rain"])


def test_config_rejects_scope_expansion() -> None:
    raw = json.loads(CONFIG.read_text())
    raw["routing"]["allowed_expert_degradations"] = ["rain", "haze"]
    temporary = ROOT / "configs/.test_rain_expert_scope_expansion.json"
    temporary.write_text(json.dumps(raw))
    try:
        with pytest.raises(RuntimeError, match="contract changed"):
            validate_rain_expert_config(temporary)
    finally:
        temporary.unlink()


def test_adapted_checkpoint_loads_strictly_and_refuses_non_rain() -> None:
    model = load_rain_expert(CONFIG)
    assert sum(parameter.numel() for parameter in model.parameters()) == 26_126_644
    with pytest.raises(RuntimeError, match="refused"):
        restore_isolated_rain(model, torch.rand(1, 3, 16, 16), ["haze"])
