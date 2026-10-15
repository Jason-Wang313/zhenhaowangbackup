from __future__ import annotations

from collections import Counter

import pytest

import smoke.evaluate_foundir_blur_haze_lowlight_full_development_ddp as evaluator


def _metric_row(task: str, category: str, group: str, index: int) -> dict:
    return {
        "record_id": f"{task}-{group}-{index}",
        "split": "validation",
        "category": category,
        "source_group": group,
        "decision": {"route": "generalist+none"},
        "input": {"psnr": 20.0, "lpips": 0.3},
        "restored": {"psnr": 30.0, "lpips": 0.1},
    }


def test_selection_uses_every_predeclared_related_development_record(monkeypatch) -> None:
    expected = {"01Blur": 2, "08Haze": 1, "14Lowlight": 1}
    monkeypatch.setattr(evaluator, "EXPECTED_CATEGORIES", expected)
    rows = [
        {"record_id": "b1", "split": "validation", "category": "01Blur"},
        {"record_id": "b2", "split": "validation", "category": "01Blur"},
        {"record_id": "h1", "split": "validation", "category": "08Haze"},
        {"record_id": "l1", "split": "validation", "category": "14Lowlight"},
        {"record_id": "train", "split": "train", "category": "01Blur"},
        {"record_id": "noise", "split": "validation", "category": "05Noise"},
    ]
    selected = evaluator.select_audit_rows(
        rows, {"selection": {"expected_records": sum(expected.values())}}
    )
    assert Counter(row["category"] for row in selected) == Counter(expected)
    assert {row["record_id"] for row in selected} == {"b1", "b2", "h1", "l1"}


def test_selection_rejects_duplicate_record_ids(monkeypatch) -> None:
    monkeypatch.setattr(evaluator, "EXPECTED_CATEGORIES", {"01Blur": 2})
    rows = [
        {"record_id": "same", "split": "validation", "category": "01Blur"},
        {"record_id": "same", "split": "validation", "category": "01Blur"},
    ]
    with pytest.raises(RuntimeError, match="not unique"):
        evaluator.select_audit_rows(rows, {"selection": {"expected_records": 2}})


def test_result_keeps_isolated_task_gates_separate_from_mixture_metrics() -> None:
    rows = []
    categories = {"blur": "01Blur", "haze": "08Haze", "low_light": "14Lowlight"}
    for task, category in categories.items():
        rows.extend(
            [_metric_row(task, category, "a", 1), _metric_row(task, category, "b", 2)]
        )
    rows.append(_metric_row("mix", "09Lowlight_Haze", "c", 1))
    config = {
        "metrics": {
            "bootstrap_resamples": 20,
            "bootstrap_confidence": 0.95,
            "bootstrap_seed": 9,
            "worst_records_per_category": 1,
        },
        "strong_gates": {
            task: {
                "category": category,
                "mean_psnr_minimum": 25.0,
                "mean_lpips_maximum": 0.2,
                "group_bootstrap_psnr_lower_minimum": 25.0,
                "group_bootstrap_lpips_upper_maximum": 0.2,
            }
            for task, category in categories.items()
        },
    }
    result = evaluator.build_result(rows, config, config_sha="config")
    assert result["records"] == 7
    assert set(result["strong_gate_checks"]) == set(categories)
    assert all(result["strong_gate_pass"].values())
    assert "09Lowlight_Haze" in result["category_metrics"]
    assert result["sealed_internal_test_accessed"] is False
    assert result["protected_foundir_test_accessed"] is False
