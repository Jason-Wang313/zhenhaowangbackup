import json
from pathlib import Path

from smoke.evaluate_foundir_haze_flux2_step3000_full_development_ddp import (
    EXPECTED_CONFIG_SHA,
    build_result,
    load_haze_rows,
    sha256,
    validate_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/foundir_haze_flux2_step3000_full_development_v1.json"


def test_frozen_haze_config_and_full_development_scope():
    config, parent, rows = validate_config(CONFIG, verify_images=False)
    assert sha256(CONFIG) == EXPECTED_CONFIG_SHA
    assert parent["scope"]["category"] == "08Haze"
    assert config["model"]["checkpoint_step"] == 3000
    assert len(rows) == 1580
    assert len({row["source_group"] for row in rows}) == 158
    assert config["gate"]["formal_checks"] == ["sow_psnr", "sow_lpips"]
    assert config["gate"]["automatic_follow_on"] is False


def test_haze_rows_are_source_disjoint_and_protected_splits_excluded():
    config = json.loads(CONFIG.read_text())
    rows = load_haze_rows(config)
    assert {row["category"] for row in rows} == {"08Haze"}
    assert {row["composition_size"] for row in rows} == {"1"}
    assert {row["split"] for row in rows} == {"validation"}
    assert all("official" not in row["split"] and "internal" not in row["split"] for row in rows)


def test_haze_selection_does_not_use_released_foundir_baseline():
    config = json.loads(CONFIG.read_text())
    serialized = json.dumps(config)
    assert "third_party/foundir/premodel" not in serialized
    assert "specialist_model" not in serialized
    assert config["model"]["name"] == "flux.2-klein-base-4b"


def test_result_reports_sow_absolute_and_percentage_margins():
    config = json.loads(CONFIG.read_text())
    rows = [
        {
            "source_group": "a",
            "input": {"psnr": 15.0, "lpips": 0.30},
            "restored": {"psnr": 23.0, "lpips": 0.10},
        },
        {
            "source_group": "b",
            "input": {"psnr": 16.0, "lpips": 0.25},
            "restored": {"psnr": 22.0, "lpips": 0.12},
        },
    ]
    result = build_result(
        rows, config, config_path=CONFIG, config_sha=EXPECTED_CONFIG_SHA,
        profile="H100", gpu_name="NVIDIA H100 80GB HBM3", world=8,
        wall_seconds=1.0,
    )
    assert result["automatic_metric_gate_pass"] is True
    assert result["formal_acceptance_checks"] == {"sow_psnr": True, "sow_lpips": True}
    assert abs(result["sow_margins"]["psnr_absolute_db"] - 0.4) < 1e-9
    assert abs(result["sow_margins"]["psnr_relative_percent"] - 100 * 0.4 / 22.1) < 1e-9
    assert abs(result["sow_margins"]["lpips_absolute_safety"] - 0.08) < 1e-9
    assert result["source_group_bootstrap"]["source_groups"] == 2


def test_result_fails_each_sow_metric_independently():
    config = json.loads(CONFIG.read_text())
    rows = [
        {
            "source_group": "a",
            "input": {"psnr": 20.0, "lpips": 0.20},
            "restored": {"psnr": 22.09, "lpips": 0.191},
        },
        {
            "source_group": "b",
            "input": {"psnr": 20.0, "lpips": 0.20},
            "restored": {"psnr": 22.09, "lpips": 0.191},
        },
    ]
    result = build_result(
        rows, config, config_path=CONFIG, config_sha=EXPECTED_CONFIG_SHA,
        profile="H200", gpu_name="NVIDIA H200", world=8, wall_seconds=1.0,
    )
    assert result["automatic_metric_gate_pass"] is False
    assert result["formal_acceptance_checks"] == {"sow_psnr": False, "sow_lpips": False}
