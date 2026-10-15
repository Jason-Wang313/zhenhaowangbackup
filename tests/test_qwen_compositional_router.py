from lowlight_restoration.qwen_compositional_router import (
    COMPONENTS,
    COMPONENTS_BY_CLASS,
    component_targets,
    compositional_metrics,
    compositional_route,
)


def test_supported_classes_have_unique_component_sets():
    assert COMPONENTS == ("blur", "noise", "rain", "haze", "low_light")
    assert len(COMPONENTS_BY_CLASS) == len(set(COMPONENTS_BY_CLASS.values())) == 11


def test_component_targets_are_independent():
    assert component_targets(["noise+low_light", "rain+haze"]) == [
        [0, 1, 0, 0, 1],
        [0, 0, 1, 1, 0],
    ]


def test_compositional_route_requires_confident_supported_set():
    assert compositional_route([0.95, 0.03, 0.02, 0.01, 0.96], margin=0.9) == "blur+low_light"
    assert compositional_route([0.95, 0.20, 0.02, 0.01, 0.96], margin=0.9) is None
    assert compositional_route([0.95, 0.95, 0.95, 0.01, 0.01], margin=0.9) is None


def test_compositional_metrics_separate_wrong_from_refusal():
    metrics = compositional_metrics(
        [[0.95, 0.01, 0.01, 0.01, 0.01], [0.01, 0.95, 0.01, 0.01, 0.01], [0.5] * 5],
        ["blur", "blur", "haze"],
        margin=0.9,
    )
    assert metrics["overall"]["correct"] == 1
    assert metrics["overall"]["wrong"] == 1
    assert metrics["overall"]["refusal"] == 1
