"""Truth-free fail-closed composition of the frozen FoundIR routing heads."""
from __future__ import annotations


ROUTE_PLANS = {
    "blur": ("blur",),
    "noise": ("noise",),
    "haze": ("haze",),
    "rain": ("rain",),
    "low_light": ("low_light",),
    "blur+noise": ("blur_noise",),
    "haze+low_light": ("haze", "low_light"),
    "rain+haze": ("rain", "haze"),
    "blur+low_light": ("low_light", "blur"),
    "noise+low_light": ("noise", "low_light"),
    "blur+noise+low_light": ("noise", "low_light", "blur"),
}
NOISE_FAMILY = frozenset(("noise", "noise+low_light"))
LOW_LIGHT_FALLBACK_MAX_EXPOSURE = 0.15


def final_eleven_route(
    *,
    compositional_route: str | None,
    compositional_accepted: bool,
    eleven_route: str | None,
    eleven_accepted: bool,
    dual_route: str | None,
    dual_accepted: bool,
    physical_route: str | None,
) -> str | None:
    """Apply only frozen, truth-free consensus rules; otherwise abstain."""
    if (
        dual_accepted is True
        and dual_route in NOISE_FAMILY
        and physical_route == dual_route
    ):
        return dual_route
    if (
        dual_accepted is True
        and eleven_accepted is True
        and dual_route == eleven_route == "low_light"
    ):
        return "low_light"
    if (
        compositional_accepted is True
        and eleven_accepted is True
        and compositional_route == eleven_route
        and compositional_route in ROUTE_PLANS
    ):
        return compositional_route
    return None


def expert_plan(route: str | None) -> tuple[str, ...]:
    """Never execute an expert after abstention or for an unknown route."""
    return ROUTE_PLANS.get(route, ())


def safe_expert_plan(
    route: str | None,
    *,
    exposure: float | None = None,
    five_proposal: str | None = None,
) -> tuple[str, ...]:
    """Return a conservative partial-restoration plan for clearly dark abstentions.

    This fallback does not claim that a mixed degradation is a low-light
    singleton. It only applies the low-light component shared by every dark
    development case below the fixed exposure bound.
    """
    plan = expert_plan(route)
    if plan:
        return plan
    if five_proposal == "low_light":
        return ("low_light",)
    if exposure is not None and exposure <= LOW_LIGHT_FALLBACK_MAX_EXPOSURE:
        return ("low_light",)
    return ()
