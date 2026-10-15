from __future__ import annotations

import pytest

from lowlight_restoration.noise_family_fail_closed_router import (
    expert_plan_for_decision,
    select_noise_family_route,
)


@pytest.mark.parametrize("route", ["noise", "noise+low_light"])
def test_exact_accepted_consensus_routes(route: str) -> None:
    decision = select_noise_family_route(route, True, route)
    assert decision.accepted is True
    assert decision.route == route
    assert decision.reason == "consensus"
    expected = ("noise",) if route == "noise" else ("noise", "low_light")
    assert expert_plan_for_decision(decision) == expected


def test_dual_refusal_always_abstains() -> None:
    decision = select_noise_family_route("noise", False, "noise")
    assert decision.accepted is False
    assert decision.route is None
    assert decision.reason == "dual_refused"
    assert expert_plan_for_decision(decision) == ()


@pytest.mark.parametrize(
    "dual,physical",
    [("noise", "noise+low_light"), ("noise+low_light", "noise")],
)
def test_disagreement_always_abstains(dual: str, physical: str) -> None:
    decision = select_noise_family_route(dual, True, physical)
    assert decision.accepted is False
    assert decision.route is None
    assert decision.reason == "selector_disagreement"
    assert expert_plan_for_decision(decision) == ()


@pytest.mark.parametrize(
    "dual,physical",
    [("blur", "blur"), ("low_light", "low_light"), (None, "noise"), ("noise", None)],
)
def test_out_of_scope_or_missing_route_abstains(
    dual: str | None, physical: str | None
) -> None:
    decision = select_noise_family_route(dual, True, physical)
    assert decision.accepted is False
    assert decision.route is None
