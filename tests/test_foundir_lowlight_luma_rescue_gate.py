from __future__ import annotations

import pytest

from smoke.evaluate_foundir_lowlight_luma_rescue_gate_ddp import (
    candidate_rows,
    candidate_triggers,
    select_candidate,
)


def candidate(l2=None, l1=None):
    return {
        "qwen_level_minimum": 3,
        "l2_mean_luma_maximum": l2,
        "l1_mean_luma_maximum": l1,
    }


@pytest.mark.parametrize(
    ("level", "luma", "spec", "expected"),
    [
        (3, 0.9, candidate(), True),
        (2, 0.2, candidate(0.2), True),
        (2, 0.20001, candidate(0.2), False),
        (1, 0.075, candidate(0.2, 0.075), True),
        (1, 0.08, candidate(0.2, 0.075), False),
        (1, 0.01, candidate(0.2), False),
    ],
)
def test_candidate_trigger_is_category_blind(level, luma, spec, expected) -> None:
    assert candidate_triggers(level, luma, spec) is expected


@pytest.mark.parametrize(("level", "luma"), [(0, 0.1), (6, 0.1), (True, 0.1), (2, -0.1), (2, 1.1)])
def test_candidate_trigger_rejects_invalid_inputs(level, luma) -> None:
    with pytest.raises(RuntimeError):
        candidate_triggers(level, luma, candidate(0.2))


def metric_row(record_id: str, category: str, level: int) -> dict:
    return {
        "record_id": record_id,
        "category": category,
        "qwen_levels": {"low_light": level},
        "decision": {"route": "generalist+none"},
        "restored": {"psnr": 20.0, "lpips": 0.2},
    }


def test_candidate_rows_replaces_only_triggered_records() -> None:
    baseline = {
        "dark": metric_row("dark", "14Lowlight", 2),
        "bright": metric_row("bright", "05Noise", 2),
        "confident": metric_row("confident", "14Lowlight", 4),
    }
    rescue = {
        "dark": {
            "mean_luma": 0.1,
            "rescued": {"psnr": 30.0, "lpips": 0.1},
            "rescued_decision": {"route": "generalist+low_light"},
        },
        "bright": {
            "mean_luma": 0.5,
            "rescued": {"psnr": 10.0, "lpips": 0.5},
            "rescued_decision": {"route": "generalist+low_light"},
        },
    }
    rows, changes = candidate_rows(baseline, rescue, candidate(0.2))
    by_id = {row["record_id"]: row for row in rows}
    assert changes == 1
    assert by_id["dark"]["restored"]["psnr"] == 30.0
    assert by_id["bright"]["restored"]["psnr"] == 20.0
    assert by_id["confident"]["restored"]["psnr"] == 20.0


def summary(psnr, lpips, low_psnr, high_lpips, changes):
    categories = {
        category: {"psnr": psnr, "lpips": lpips}
        for category in ("01Blur", "05Noise", "10Rain", "08Haze", "14Lowlight")
    }
    return {
        "route_changes": changes,
        "category_metrics": categories,
        "source_group_bootstrap": {"low_light": {"lower": {"psnr": low_psnr}, "upper": {"lpips": high_lpips}}},
    }


def test_selection_rejects_non_lowlight_regression_and_selects_best_safe_margin() -> None:
    current = summary(30.0, 0.1, 18.0, 0.3, 0)
    safer = summary(30.0, 0.1, 20.0, 0.25, 10)
    unsafe = summary(29.0, 0.1, 30.0, 0.1, 1)
    result = select_candidate(
        {"current_qwen_l3": current, "safe": safer, "unsafe": unsafe},
        {
            "strong_gates": {
                "low_light": {
                    "group_bootstrap_psnr_lower_minimum": 19.4,
                    "group_bootstrap_lpips_upper_maximum": 0.27,
                }
            }
        },
    )
    assert result["candidate"] == "safe"
