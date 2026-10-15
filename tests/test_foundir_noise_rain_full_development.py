from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from smoke.evaluate_foundir_noise_rain_full_development_ddp import (
    load_audit_crop,
    mean_metrics,
    select_audit_rows,
    source_group_bootstrap,
    strong_gate_checks,
    verify_sidecar,
)
from smoke.evaluate_foundir_five_type_flux2_ddp import load_aligned_crop


def record(record_id: str, category: str, group: str, psnr: float, lpips: float):
    return {
        "record_id": record_id,
        "split": "validation",
        "category": category,
        "source_group": group,
        "input": {"psnr": psnr - 1.0, "lpips": lpips + 0.02},
        "restored": {"psnr": psnr, "lpips": lpips},
    }


def test_full_development_selection_is_exact_and_deterministic() -> None:
    rows = [
        {"record_id": "n1", "split": "validation", "category": "05Noise"},
        {"record_id": "r1", "split": "validation", "category": "10Rain"},
        {"record_id": "n2", "split": "train", "category": "05Noise"},
        {"record_id": "h1", "split": "validation", "category": "08Haze"},
    ]
    config = {
        "selection": {
            "categories": {"05Noise": 1, "10Rain": 1},
            "expected_records": 2,
        }
    }
    selected = select_audit_rows(rows, config)
    assert [item["record_id"] for item in selected] == ["n1", "r1"]
    with pytest.raises(RuntimeError, match="coverage changed"):
        select_audit_rows(rows[:1], config)


def test_regular_crop_exactly_matches_existing_evaluator(tmp_path, monkeypatch) -> None:
    lq = np.arange(540 * 520 * 3, dtype=np.uint32).reshape(540, 520, 3).astype(np.uint8)
    gt = np.flip(lq, axis=1).copy()
    lq_path = tmp_path / "lq.png"
    gt_path = tmp_path / "gt.png"
    Image.fromarray(lq).save(lq_path)
    Image.fromarray(gt).save(gt_path)

    import smoke.evaluate_foundir_five_type_flux2_ddp as old_evaluator
    import smoke.evaluate_foundir_noise_rain_full_development_ddp as new_evaluator

    monkeypatch.setattr(old_evaluator, "ROOT", tmp_path)
    monkeypatch.setattr(new_evaluator, "ROOT", tmp_path)
    from lowlight_restoration.flux_prompt_cache import file_sha256

    row = {
        "record_id": "regular",
        "lq_path": "lq.png",
        "gt_path": "gt.png",
        "lq_sha256": file_sha256(lq_path),
        "gt_sha256": file_sha256(gt_path),
    }
    old_lq, old_gt, old_xy = load_aligned_crop(row, crop_size=512, seed=17)
    new_lq, new_gt, new_xy, resized = load_audit_crop(row, crop_size=512, seed=17)
    assert resized is False
    assert new_xy == old_xy
    assert new_lq.equal(old_lq)
    assert new_gt.equal(old_gt)


def test_undersized_pair_is_resized_with_aligned_deterministic_geometry(
    tmp_path, monkeypatch
) -> None:
    lq = np.zeros((256, 640, 3), dtype=np.uint8)
    lq[:, :, 0] = np.arange(640, dtype=np.uint16).astype(np.uint8)
    gt = lq.copy()
    lq_path = tmp_path / "small_lq.png"
    gt_path = tmp_path / "small_gt.png"
    Image.fromarray(lq).save(lq_path)
    Image.fromarray(gt).save(gt_path)

    import smoke.evaluate_foundir_noise_rain_full_development_ddp as evaluator
    from lowlight_restoration.flux_prompt_cache import file_sha256

    monkeypatch.setattr(evaluator, "ROOT", tmp_path)
    row = {
        "record_id": "undersized",
        "lq_path": "small_lq.png",
        "gt_path": "small_gt.png",
        "lq_sha256": file_sha256(lq_path),
        "gt_sha256": file_sha256(gt_path),
    }
    first = load_audit_crop(row, crop_size=512, seed=23)
    second = load_audit_crop(row, crop_size=512, seed=23)
    assert first[3] is True
    assert first[0].shape == first[1].shape == (3, 512, 512)
    assert first[0].equal(first[1])
    assert first[0].equal(second[0])
    assert first[2] == second[2]


def test_source_group_bootstrap_is_deterministic_and_group_balanced() -> None:
    rows = [
        record("a1", "05Noise", "a", 40.0, 0.06),
        record("a2", "05Noise", "a", 42.0, 0.08),
        record("b1", "05Noise", "b", 38.0, 0.10),
    ]
    first = source_group_bootstrap(rows, resamples=200, confidence=0.95, seed=7)
    second = source_group_bootstrap(rows, resamples=200, confidence=0.95, seed=7)
    assert first == second
    assert first["source_groups"] == 2
    assert first["group_balanced_mean"]["psnr"] == pytest.approx(39.5)
    assert first["group_balanced_mean"]["lpips"] == pytest.approx(0.085)


def test_mean_metrics_and_strong_gate_require_all_safety_margins() -> None:
    rows = [
        record("a", "10Rain", "a", 36.0, 0.05),
        record("b", "10Rain", "b", 35.0, 0.06),
    ]
    metrics = mean_metrics(rows)
    assert metrics["psnr"] == pytest.approx(35.5)
    assert metrics["lpips"] == pytest.approx(0.055)
    bootstrap = {"lower": {"psnr": 34.8}, "upper": {"lpips": 0.069}}
    gate = {
        "mean_psnr_minimum": 35.2,
        "mean_lpips_maximum": 0.06,
        "group_bootstrap_psnr_lower_minimum": 34.7,
        "group_bootstrap_lpips_upper_maximum": 0.07,
    }
    assert all(strong_gate_checks(metrics, bootstrap, gate).values())
    bootstrap["upper"]["lpips"] = 0.071
    assert strong_gate_checks(metrics, bootstrap, gate)["bootstrap_lpips_upper"] is False


def test_sidecar_resume_rejects_wrong_identity_and_nonfinite_metrics(tmp_path) -> None:
    levels = {"blur": 1, "noise": 3, "rain": 1, "haze": 1, "low_light": 1}
    decision = {
        "active_types": ["noise"],
        "base": "generalist",
        "refinement": "none",
        "route": "generalist+none",
    }
    row = {
        "record_id": "n1",
        "source_group": "group-a",
        "category": "05Noise",
        "composition_size": "1",
        "qwen_json": json.dumps(levels),
    }
    payload = {
        "schema_version": 1,
        "status": "complete",
        "config_sha256": "config",
        "router_config_sha256": "router",
        "split": "validation",
        **row,
        "composition_size": 1,
        "qwen_levels": levels,
        "decision": decision,
        "crop_xy": [0, 3],
        "undersized_pair_resized": False,
        "input": {"psnr": 20.0, "lpips": 0.3},
        "restored": {"psnr": 40.0, "lpips": 0.06},
        "restored_uint8_sha256": "a" * 64,
        "sealed_internal_test_accessed": False,
        "protected_foundir_test_accessed": False,
    }
    path = tmp_path / "sidecar.json"
    path.write_text(json.dumps(payload))
    assert verify_sidecar(
        path,
        row=row,
        decision=decision,
        config_sha="config",
        router_sha="router",
    ) == payload
    payload["config_sha256"] = "wrong"
    path.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="identity changed"):
        verify_sidecar(
            path,
            row=row,
            decision=decision,
            config_sha="config",
            router_sha="router",
        )
    payload["config_sha256"] = "config"
    payload["sealed_internal_test_accessed"] = True
    path.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="identity changed"):
        verify_sidecar(
            path,
            row=row,
            decision=decision,
            config_sha="config",
            router_sha="router",
        )
    payload["sealed_internal_test_accessed"] = False
    payload["restored"]["lpips"] = float("nan")
    path.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="metrics are invalid"):
        verify_sidecar(
            path,
            row=row,
            decision=decision,
            config_sha="config",
            router_sha="router",
        )
