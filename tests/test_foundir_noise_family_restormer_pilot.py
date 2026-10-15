from __future__ import annotations

import json
from pathlib import Path

import pytest

import smoke.train_foundir_noise_family_restormer_pilot_ddp as trainer


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_config_and_manifest_pass_preflight_contract() -> None:
    path = ROOT / "configs/foundir_noise_family_restormer_pilot_v1.json"
    config = trainer.validate_config(path)
    rows = trainer.load_rows(config)
    assert len(rows["train"]) == 2048
    assert len(rows["validation"]) == 256


def test_category_balanced_sampling_is_exact_and_deterministic() -> None:
    grouped = {
        category: {
            f"{category}-a": [{"record_id": f"{category}-a", "category": category}],
            f"{category}-b": [{"record_id": f"{category}-b", "category": category}],
        }
        for category in trainer.CATEGORIES
    }
    first = trainer.category_balanced_batch(grouped, step=10, seed=7, count=32)
    second = trainer.category_balanced_batch(grouped, step=10, seed=7, count=32)
    assert first == second
    assert [row["category"] for row, _ in first].count("05Noise") == 16
    assert [row["category"] for row, _ in first].count("16Lowlight_Noise") == 16
    for rank in range(8):
        local = first[rank::8]
        assert [row["category"] for row, _ in local].count("05Noise") == 2
        assert [row["category"] for row, _ in local].count("16Lowlight_Noise") == 2


def test_config_rejects_loss_or_gate_drift(tmp_path: Path) -> None:
    source = ROOT / "configs/foundir_noise_family_restormer_pilot_v1.json"
    config = json.loads(source.read_text())
    config["optimization"]["loss"] = "rgb_l1_plus_lpips"
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    with pytest.raises(RuntimeError, match="contract changed"):
        trainer.validate_config(path)


def test_joint_selection_score_uses_worst_category_margin(monkeypatch, tmp_path: Path) -> None:
    checkpoint = tmp_path / "step_000500.pt"
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setattr(trainer, "ROOT", tmp_path)
    monkeypatch.setattr(
        trainer,
        "source_group_bootstrap",
        lambda rows, **kwargs: {
            "unit": "source_group_equal_weight",
            "source_groups": len(rows),
            "resamples": 10,
            "confidence": 0.95,
            "group_balanced_mean": {"psnr": 20.0, "lpips": 0.1},
            "lower": {"psnr": 19.0, "lpips": 0.09},
            "upper": {"psnr": 21.0, "lpips": 0.11},
        },
    )
    rows = []
    values = {
        "05Noise": {"psnr": 40.0, "lpips": 0.07},
        "16Lowlight_Noise": {"psnr": 18.8, "lpips": 0.25},
    }
    for category, restored in values.items():
        rows.append(
            {
                "record_id": category,
                "source_group": category,
                "category": category,
                "input": {"psnr": 10.0, "lpips": 0.5},
                "restored": restored,
            }
        )
    config = {
        "data": {"validation_records_per_category": 1},
        "bootstrap": {"resamples": 10, "confidence": 0.95, "seed": 1},
        "per_category_gates": {
            "05Noise": {"mean_psnr_minimum": 39.9, "mean_lpips_maximum": 0.08},
            "16Lowlight_Noise": {
                "mean_psnr_minimum": 18.733316631721234,
                "mean_lpips_maximum": 0.2582512099761516,
            },
        },
    }
    payload = trainer.validation_payload(
        rows, config=config, config_sha="config", checkpoint=checkpoint, step=500
    )
    assert payload["joint_gate_pass"] is True
    assert payload["selection_score"] == pytest.approx(18.8 - 18.733316631721234)
