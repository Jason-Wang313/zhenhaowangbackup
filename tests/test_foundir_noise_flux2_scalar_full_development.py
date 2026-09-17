import json
from pathlib import Path

from smoke.evaluate_foundir_noise_flux2_scalar_residual_full_development_ddp import (
    EXPECTED_CONFIG_SHA,
    load_noise_rows,
    sha256,
    validate_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/foundir_noise_flux2_scalar_residual_full_development_v1.json"


def test_frozen_config_and_scope():
    config, parent, rows = validate_config(CONFIG, verify_images=False)
    assert sha256(CONFIG) == EXPECTED_CONFIG_SHA
    assert config["inference"]["frozen_alpha"] == 0.3
    assert config["gate"]["formal_checks"] == ["sow_psnr", "sow_lpips"]
    assert config["gate"]["tail_statistics_descriptive_only"] is True
    assert parent["scope"]["category"] == "05Noise"
    assert len(rows) == 1859
    assert len({row["source_group"] for row in rows}) == 34


def test_noise_rows_are_source_disjoint_and_protected_splits_excluded():
    config = json.loads(CONFIG.read_text())
    rows = load_noise_rows(config)
    assert {row["category"] for row in rows} == {"05Noise"}
    assert {row["composition_size"] for row in rows} == {"1"}
    assert {row["split"] for row in rows} == {"validation"}
    assert all("official" not in row["split"] and "internal" not in row["split"] for row in rows)


def test_launcher_records_total_runtime_and_governance():
    launcher = (ROOT / "smoke/launch_foundir_noise_flux2_scalar_residual_full_development_v1_8gpu.sh").read_text()
    runner = (ROOT / "smoke/run_foundir_noise_flux2_scalar_residual_full_development_v1_8gpu.sh").read_text()
    assert "TOTAL END-TO-END TIME" in launcher
    assert "advisor_governance.py check" in launcher
    assert "advisor_governance.py check" in runner
    assert "tail_statistics_descriptive_only" in runner
