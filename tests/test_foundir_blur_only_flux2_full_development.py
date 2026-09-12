from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from smoke.evaluate_foundir_blur_only_flux2_full_development_ddp import (
    BlurDevelopmentDataset,
    aggregate_checkpoint,
    diagnose_overfitting,
    expand_cfg_condition,
    load_development_rows,
    sha256,
    validate_config,
    verify_sidecar,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/foundir_blur_only_flux2_full_development_v1.json"


def loaded_config() -> dict[str, object]:
    return json.loads(CONFIG.read_text())


def metric_row(record_id: str, group: str, psnr: float, lpips: float) -> dict[str, object]:
    return {
        "record_id": record_id,
        "source_group": group,
        "input": {"psnr": 23.0, "lpips": 0.30},
        "restored": {"psnr": psnr, "lpips": lpips},
    }


def curve_row(step: int, psnr: float, lpips: float) -> dict[str, object]:
    return {"checkpoint_step": step, "metrics": {"psnr": psnr, "lpips": lpips}}


def test_config_binds_all_five_real_checkpoints_and_scope() -> None:
    config, parent = validate_config(CONFIG)
    assert [item["step"] for item in config["checkpoints"]] == [2000, 4000, 6000, 8000, 10000]
    assert parent["model"]["name"] == "flux.2-klein-base-4b"
    assert config["scope"]["qwen_degradation_conditioning"] is False
    assert config["scope"]["mixed_degradations"] is False
    assert config["scope"]["official_foundir_test_accessed"] is False


def test_selection_is_all_isolated_blur_development_and_source_disjoint() -> None:
    rows = load_development_rows(loaded_config())
    assert len(rows) == 2595
    assert len({row["source_group"] for row in rows}) == 165
    assert {row["category"] for row in rows} == {"01Blur"}
    assert {row["composition_size"] for row in rows} == {"1"}
    assert {row["split"] for row in rows} == {"validation"}


def test_cfg_batch_order_matches_denoise_cfg_duplication() -> None:
    context = torch.tensor([[[1.0]], [[2.0]]])
    ids = torch.tensor([[[[10]]], [[[20]]]])
    batch_context, batch_ids = expand_cfg_condition(context, ids, 3)
    assert batch_context[:, 0, 0].tolist() == [1.0, 1.0, 1.0, 2.0, 2.0, 2.0]
    assert batch_ids[:, 0, 0, 0].tolist() == [10, 10, 10, 20, 20, 20]


def test_crop_is_deterministic_and_aligned(tmp_path: Path, monkeypatch) -> None:
    import smoke.evaluate_foundir_blur_only_flux2_full_development_ddp as evaluator

    lq = np.zeros((520, 530, 3), dtype=np.uint8)
    lq[:, :, 0] = np.arange(530, dtype=np.uint16).astype(np.uint8)
    gt = lq.copy()
    lq_path = tmp_path / "lq.png"
    gt_path = tmp_path / "gt.png"
    Image.fromarray(lq).save(lq_path)
    Image.fromarray(gt).save(gt_path)
    monkeypatch.setattr(evaluator, "ROOT", tmp_path)
    row = {
        "record_id": "record",
        "source_group": "group",
        "width": "530",
        "height": "520",
        "lq_path": "lq.png",
        "gt_path": "gt.png",
        "lq_sha256": sha256(lq_path),
        "gt_sha256": sha256(gt_path),
    }
    dataset = BlurDevelopmentDataset([row], crop_size=512, seed=7)
    first = dataset[0]
    second = dataset[0]
    assert first["crop_xy"] == second["crop_xy"]
    assert first["lq"].equal(first["gt"])
    assert first["lq"].equal(second["lq"])
    assert first["lq"].shape == (3, 512, 512)


def test_aggregate_uses_record_mean_and_group_bootstrap() -> None:
    config = loaded_config()
    rows = [
        metric_row("a1", "a", 26.0, 0.17),
        metric_row("a2", "a", 28.0, 0.13),
        metric_row("b1", "b", 24.0, 0.19),
    ]
    result = aggregate_checkpoint(
        rows,
        step=2000,
        checkpoint_path="checkpoint",
        checkpoint_sha="a" * 64,
        config=config,
        config_sha="b" * 64,
    )
    assert result["metrics"]["psnr"] == pytest.approx(26.0)
    assert result["metrics"]["lpips"] == pytest.approx(0.1633333333)
    assert result["source_group_bootstrap"]["source_groups"] == 2
    assert result["sow_checks"] == {"psnr": True, "lpips": True}


def test_overfitting_requires_earlier_pareto_domination() -> None:
    diagnosis = diagnose_overfitting([
        curve_row(2000, 25.0, 0.20),
        curve_row(4000, 26.0, 0.16),
        curve_row(6000, 25.8, 0.17),
        curve_row(8000, 25.7, 0.18),
        curve_row(10000, 25.5, 0.19),
    ])
    assert diagnosis["late_overfitting_detected"] is True
    assert diagnosis["best_psnr_step"] == 4000
    assert diagnosis["best_lpips_step"] == 4000
    assert diagnosis["earlier_steps_dominating_final"] == [4000, 6000, 8000]


def test_sidecar_rejects_wrong_checkpoint_or_protected_access(tmp_path: Path) -> None:
    row = {"record_id": "r", "source_group": "g"}
    payload = {
        "schema_version": 1,
        "status": "complete",
        "config_sha256": "c",
        "checkpoint_step": 2000,
        "checkpoint_sha256": "d",
        "split": "validation",
        "record_id": "r",
        "source_group": "g",
        "category": "01Blur",
        "crop_xy": [1, 2],
        "input": {"psnr": 20.0, "lpips": 0.3},
        "restored": {"psnr": 26.0, "lpips": 0.17},
        "restored_uint8_sha256": "a" * 64,
        "sealed_internal_accessed": False,
        "protected_foundir_test_accessed": False,
    }
    path = tmp_path / "sidecar.json"
    path.write_text(json.dumps(payload))
    assert verify_sidecar(path, row=row, step=2000, checkpoint_sha="d", config_sha="c") == payload
    payload["protected_foundir_test_accessed"] = True
    path.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="identity changed"):
        verify_sidecar(path, row=row, step=2000, checkpoint_sha="d", config_sha="c")


def test_launchers_are_detached_resume_safe_and_governed() -> None:
    runner = (ROOT / "smoke/run_foundir_blur_only_flux2_full_development_v1_8gpu.sh").read_text()
    launcher = (ROOT / "smoke/launch_foundir_blur_only_flux2_full_development_v1_8gpu.sh").read_text()
    assert "advisor_governance.py\" check" in runner
    assert "--diagnostic-only" in runner
    assert "--require-decision" in runner
    assert "CUDA_VISIBLE_DEVICES" in launcher
    assert "nohup setsid" in launcher
