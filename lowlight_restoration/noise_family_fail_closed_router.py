"""Fail-closed consensus routing for the two Noise-family restoration paths."""
from __future__ import annotations

from dataclasses import dataclass


NOISE_FAMILY_ROUTES = frozenset({"noise", "noise+low_light"})
NOISE_FAMILY_EXPERT_PLANS = {
    "noise": ("noise",),
    "noise+low_light": ("noise", "low_light"),
}


@dataclass(frozen=True)
class NoiseFamilyDecision:
    """A route is present only when both frozen selectors agree."""

    route: str | None
    accepted: bool
    reason: str


def select_noise_family_route(
    dual_route: str | None,
    dual_accepted: bool,
    physical_route: str | None,
) -> NoiseFamilyDecision:
    """Accept exact in-family agreement; abstain on every other input.

    This layer intentionally does not guess.  The caller must run the two
    frozen selectors independently and pass their canonical route strings.
    """
    if dual_accepted is not True:
        return NoiseFamilyDecision(None, False, "dual_refused")
    if dual_route not in NOISE_FAMILY_ROUTES:
        return NoiseFamilyDecision(None, False, "dual_out_of_scope")
    if physical_route not in NOISE_FAMILY_ROUTES:
        return NoiseFamilyDecision(None, False, "physical_refused_or_out_of_scope")
    if dual_route != physical_route:
        return NoiseFamilyDecision(None, False, "selector_disagreement")
    return NoiseFamilyDecision(dual_route, True, "consensus")


def expert_plan_for_decision(decision: NoiseFamilyDecision) -> tuple[str, ...]:
    """Return the only permitted expert plan, or an empty plan on refusal."""
    if not decision.accepted or decision.route is None:
        return ()
    return NOISE_FAMILY_EXPERT_PLANS.get(decision.route, ())
