from __future__ import annotations

import itertools
import json
from pathlib import Path

import pytest
import torch

from lowlight_restoration.five_expert_router import (
    BACKENDS,
    CATEGORIES,
    LABELS,
    RoutingRefusal,
    apply_frozen_expert_postprocess,
    select_five_expert_backend,
    select_five_expert_backend_for_category,
    validate_five_expert_routing_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/foundir_five_expert_routing_v1.json"


def test_frozen_config_and_all_five_artifact_identities() -> None:
    config = validate_five_expert_routing_config(CONFIG)
    assert set(config["experts"]) == set(LABELS)


@pytest.mark.parametrize("label", LABELS)
def test_each_exact_singleton_selects_only_its_frozen_expert(label: str) -> None:
    assert select_five_expert_backend([label]) == BACKENDS[label]


def test_every_non_singleton_known_combination_is_refused() -> None:
    with pytest.raises(RoutingRefusal, match="empty"):
        select_five_expert_backend([])
    for size in range(2, len(LABELS) + 1):
        for labels in itertools.combinations(LABELS, size):
            with pytest.raises(RoutingRefusal, match="mixed"):
                select_five_expert_backend(labels)


def test_order_cannot_turn_a_mixture_into_a_singleton_route() -> None:
    for labels in itertools.permutations(("rain", "haze", "blur")):
        with pytest.raises(RoutingRefusal, match="mixed"):
            select_five_expert_backend(labels)


@pytest.mark.parametrize("labels", [["rain", "rain"], ["blur", "blur", "noise"]])
def test_duplicates_are_refused(labels: list[str]) -> None:
    with pytest.raises(RoutingRefusal, match="duplicate"):
        select_five_expert_backend(labels)


@pytest.mark.parametrize("labels", [["snow"], ["rain", "snow"], ["Lowlight"], [""]])
def test_unknown_or_noncanonical_labels_are_refused(labels: list[str]) -> None:
    with pytest.raises(RoutingRefusal, match="unknown"):
        select_five_expert_backend(labels)


@pytest.mark.parametrize("category,label", CATEGORIES.items())
def test_exact_dataset_categories_route_only_singleton_records(category: str, label: str) -> None:
    assert select_five_expert_backend_for_category(category, 1) == BACKENDS[label]
    with pytest.raises(RoutingRefusal, match="non-singleton"):
        select_five_expert_backend_for_category(category, 2)


@pytest.mark.parametrize("category", ["13Rain_Haze", "00Unknown", "", "01Blur_05Noise"])
def test_unknown_and_mixed_dataset_categories_are_refused(category: str) -> None:
    with pytest.raises(RoutingRefusal, match="unknown or mixed"):
        select_five_expert_backend_for_category(category, 1)


@pytest.mark.parametrize(
    "label,alpha",
    [("blur", 0.75), ("noise", 0.30), ("low_light", 0.925)],
)
def test_scalar_expert_postprocess_is_exactly_frozen(label: str, alpha: float) -> None:
    lq = torch.tensor([[[[0.0, 0.25]], [[0.5, 0.75]], [[1.0, 0.1]]]])
    restored = torch.tensor([[[[1.0, 0.75]], [[0.0, 0.25]], [[0.5, 0.9]]]])
    output_u8, output = apply_frozen_expert_postprocess(BACKENDS[label], lq, restored)
    expected_u8 = (lq + alpha * (restored - lq)).clamp(0, 1).mul(255).round().to(torch.uint8)
    assert torch.equal(output_u8, expected_u8)
    assert torch.equal(output, expected_u8.float().div(255))


@pytest.mark.parametrize("label", ["haze", "rain"])
def test_identity_expert_postprocess_only_quantizes(label: str) -> None:
    lq = torch.zeros(1, 3, 1, 2)
    restored = torch.tensor([[[[-0.1, 1.1]], [[0.1, 0.9]], [[0.5, 0.501]]]])
    output_u8, output = apply_frozen_expert_postprocess(BACKENDS[label], lq, restored)
    expected_u8 = restored.clamp(0, 1).mul(255).round().to(torch.uint8)
    assert torch.equal(output_u8, expected_u8)
    assert torch.equal(output, expected_u8.float().div(255))


def test_postprocess_refuses_unknown_backend_bad_shapes_and_nonfinite_pixels() -> None:
    image = torch.zeros(1, 3, 8, 8)
    with pytest.raises(RoutingRefusal, match="unknown expert backend"):
        apply_frozen_expert_postprocess("made_up", image, image)
    with pytest.raises(ValueError, match="matching BCHW"):
        apply_frozen_expert_postprocess(BACKENDS["blur"], image, image[:, :, :-1])
    bad = image.clone()
    bad[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        apply_frozen_expert_postprocess(BACKENDS["rain"], bad, image)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda raw: raw["scope"].update(qwen_routing_enabled=True),
        lambda raw: raw["scope"].update(mixed_degradations_supported=True),
        lambda raw: raw["routing"].update(mixed_label_action="pick_first"),
        lambda raw: raw["experts"]["rain"].update(backend=BACKENDS["haze"]),
        lambda raw: raw["experts"]["blur"]["postprocess"].update(alpha=0.8),
    ],
)
def test_scope_or_expert_mutation_fails_closed(mutation) -> None:
    raw = json.loads(CONFIG.read_text())
    mutation(raw)
    temporary = ROOT / "configs/.test_five_expert_routing_mutation.json"
    temporary.write_text(json.dumps(raw))
    try:
        with pytest.raises(RuntimeError):
            validate_five_expert_routing_config(temporary)
    finally:
        temporary.unlink()
