from __future__ import annotations

import pytest

from smoke.evaluate_foundir_five_type_qwen_routed_integrated_ddp import (
    CONFIDENCE_GUARDED_POLICY,
    route_decision,
)


def levels(**updates: int) -> dict[str, int]:
    result = {"blur": 1, "noise": 1, "rain": 1, "haze": 1, "low_light": 1}
    result.update(updates)
    return result


@pytest.mark.parametrize(
    ("qwen_levels", "expected"),
    [
        (levels(), "generalist+none"),
        (levels(blur=3), "blur_specialization+none"),
        (levels(noise=4), "generalist+none"),
        (levels(rain=2), "generalist+weather"),
        (levels(haze=5), "generalist+weather"),
        (levels(blur=4, rain=3), "blur_specialization+weather"),
        (levels(blur=4, rain=3, low_light=2), "blur_specialization+low_light"),
    ],
)
def test_route_uses_only_actual_qwen_levels(qwen_levels: dict[str, int], expected: str) -> None:
    assert route_decision(qwen_levels)["route"] == expected


@pytest.mark.parametrize(
    "invalid",
    [
        {"blur": 2},
        levels(blur=0),
        levels(blur=6),
        levels(blur=True),
        {**levels(), "jpeg": 2},
    ],
)
def test_route_rejects_invalid_or_out_of_scope_levels(invalid: dict[str, int]) -> None:
    with pytest.raises(RuntimeError, match="invalid frozen Qwen"):
        route_decision(invalid)


V2_ROUTING = {
    "policy": CONFIDENCE_GUARDED_POLICY,
    "blur_level_minimum": 4,
    "low_light_level_minimum": 3,
    "weather_haze_level_minimum": 2,
}


@pytest.mark.parametrize(
    ("qwen_levels", "expected"),
    [
        (levels(), "generalist+none"),
        (levels(blur=3), "generalist+none"),
        (levels(blur=4), "blur_specialization+none"),
        (levels(low_light=2), "generalist+none"),
        (levels(low_light=3), "generalist+low_light"),
        (levels(rain=5), "generalist+none"),
        (levels(haze=2), "generalist+weather"),
        (levels(blur=5, haze=5), "blur_specialization+none"),
        (levels(blur=5, haze=5, low_light=3), "blur_specialization+low_light"),
    ],
)
def test_confidence_guarded_v2_route_uses_only_qwen_levels(
    qwen_levels: dict[str, int], expected: str
) -> None:
    decision = route_decision(qwen_levels, V2_ROUTING)
    assert decision["route"] == expected
    assert decision["policy"] == CONFIDENCE_GUARDED_POLICY


def test_route_rejects_unknown_policy() -> None:
    with pytest.raises(RuntimeError, match="unsupported integrated Qwen routing policy"):
        route_decision(levels(), {"policy": "unknown"})
