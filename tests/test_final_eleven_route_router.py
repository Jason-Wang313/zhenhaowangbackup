from lowlight_restoration.final_eleven_route_router import expert_plan, final_eleven_route


def test_noise_family_consensus_has_first_priority():
    route = final_eleven_route(
        compositional_route="low_light", compositional_accepted=True,
        eleven_route="low_light", eleven_accepted=True,
        dual_route="noise+low_light", dual_accepted=True,
        physical_route="noise+low_light",
    )
    assert route == "noise+low_light"
    assert expert_plan(route) == ("noise", "low_light")


def test_lowlight_requires_dual_and_eleven_agreement():
    assert final_eleven_route(
        compositional_route="abstain", compositional_accepted=False,
        eleven_route="low_light", eleven_accepted=True,
        dual_route="low_light", dual_accepted=True,
        physical_route="low_light",
    ) == "low_light"


def test_general_route_requires_exact_consensus():
    assert final_eleven_route(
        compositional_route="rain+haze", compositional_accepted=True,
        eleven_route="rain+haze", eleven_accepted=True,
        dual_route="rain+haze", dual_accepted=True,
        physical_route="rain+haze",
    ) == "rain+haze"
    assert final_eleven_route(
        compositional_route="rain+haze", compositional_accepted=True,
        eleven_route="rain", eleven_accepted=True,
        dual_route="rain+haze", dual_accepted=True,
        physical_route="rain+haze",
    ) is None


def test_abstention_never_executes_an_expert():
    assert expert_plan(None) == ()
    assert expert_plan("unknown") == ()
