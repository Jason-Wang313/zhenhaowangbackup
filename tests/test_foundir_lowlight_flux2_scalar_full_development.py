import json
from pathlib import Path

import pytest

from smoke.evaluate_foundir_lowlight_flux2_scalar_residual_full_development_ddp import (
    EXPECTED_CONFIG_SHA,
    build_result,
    load_lowlight_rows,
    sha256,
    validate_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/foundir_lowlight_flux2_scalar_residual_full_development_v1.json"


def test_frozen_config_scope_and_visual_diversity():
    config, parent, rows = validate_config(CONFIG, verify_images=False)
    assert sha256(CONFIG) == EXPECTED_CONFIG_SHA
    assert config["inference"]["frozen_alpha"] == 0.925
    assert config["gate"]["formal_checks"] == ["sow_psnr", "sow_lpips"]
    assert config["gate"]["tail_statistics_descriptive_only"] is True
    assert parent["scope"]["category"] == "14Lowlight"
    assert len(rows) == 1400
    assert len({row["source_group"] for row in rows}) == 21
    visual = set(config["visual_record_ids"])
    assert len({row["source_group"] for row in rows if row["record_id"] in visual}) == 16


def test_lowlight_rows_are_source_disjoint_and_protected_splits_excluded():
    config = json.loads(CONFIG.read_text())
    rows = load_lowlight_rows(config)
    assert {row["category"] for row in rows} == {"14Lowlight"}
    assert {row["composition_size"] for row in rows} == {"1"}
    assert {row["split"] for row in rows} == {"validation"}
    assert all("official" not in row["split"] and "internal" not in row["split"] for row in rows)


def test_result_uses_only_sow_means_as_formal_gate():
    config = json.loads(CONFIG.read_text())
    rows = [
        {
            "source_group": "a",
            "input": {"psnr": 10.0, "lpips": 0.5},
            "base": {"psnr": 18.0, "lpips": 0.2},
            "candidate": {"psnr": 19.0, "lpips": 0.18},
        },
        {
            "source_group": "b",
            "input": {"psnr": 20.0, "lpips": 0.2},
            "base": {"psnr": 19.0, "lpips": 0.1},
            "candidate": {"psnr": 18.9, "lpips": 0.12},
        },
    ]
    result = build_result(
        rows,
        config,
        config_path=CONFIG,
        config_sha=EXPECTED_CONFIG_SHA,
        profile="H100",
        gpu_name="NVIDIA H100 80GB HBM3",
        world=8,
        wall_seconds=1.0,
    )
    assert result["automatic_metric_gate_pass"] is True
    assert result["formal_acceptance_checks"] == {"sow_psnr": True, "sow_lpips": True}
    assert result["tail_statistics_descriptive_only"] is True
    assert result["summary"]["task_metrics"]["low_light"] == result["candidate_metrics"]
    assert result["descriptive_tail"]["records_regressing_more_than_1db_vs_input"] == 1


def test_result_fails_each_sow_metric_independently():
    config = json.loads(CONFIG.read_text())
    row = {
        "source_group": "a",
        "input": {"psnr": 10.0, "lpips": 0.5},
        "base": {"psnr": 18.0, "lpips": 0.2},
        "candidate": {"psnr": 18.89, "lpips": 0.281},
    }
    result = build_result(
        [row], config, config_path=CONFIG, config_sha=EXPECTED_CONFIG_SHA,
        profile="H200", gpu_name="NVIDIA H200", world=8, wall_seconds=1.0,
    )
    assert result["automatic_metric_gate_pass"] is False
    assert result["formal_acceptance_checks"] == {"sow_psnr": False, "sow_lpips": False}


def test_launcher_records_total_runtime_and_governance():
    launcher = (ROOT / "smoke/launch_foundir_lowlight_flux2_scalar_residual_full_development_v1_8gpu.sh").read_text()
    runner = (ROOT / "smoke/run_foundir_lowlight_flux2_scalar_residual_full_development_v1_8gpu.sh").read_text()
    assert "TOTAL END-TO-END TIME" in launcher
    assert "advisor_governance.py check" in launcher
    assert "advisor_governance.py check" in runner
    assert "tail_statistics_descriptive_only" in runner
    assert "08Haze" not in runner and "10Rain" not in runner
