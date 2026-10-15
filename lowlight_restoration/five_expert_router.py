"""Fail-closed routing contract for the five frozen isolated-degradation experts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch

from .scalar_residual import quantized_scalar_residual


ROOT = Path(__file__).resolve().parents[1]
POLICY = "fail_closed_exact_singleton_five_expert_v1"
LABELS = ("blur", "noise", "haze", "rain", "low_light")
BACKENDS = {
    "blur": "blur_flux2_step10000_alpha075",
    "noise": "noise_flux2_step400_alpha030",
    "haze": "haze_flux2_step3000",
    "rain": "rain_restormer_step5000",
    "low_light": "lowlight_flux2_step400_alpha0925",
}
CATEGORIES = {
    "01Blur": "blur",
    "05Noise": "noise",
    "08Haze": "haze",
    "10Rain": "rain",
    "14Lowlight": "low_light",
}
SCALAR_ALPHAS = {
    BACKENDS["blur"]: 0.75,
    BACKENDS["noise"]: 0.30,
    BACKENDS["low_light"]: 0.925,
}
IDENTITY_BACKENDS = {BACKENDS["haze"], BACKENDS["rain"]}


class RoutingRefusal(ValueError):
    """Raised whenever an input is not an exact, known singleton degradation."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_five_expert_backend(degradations: Sequence[str]) -> str:
    labels = tuple(str(value) for value in degradations)
    if not labels:
        raise RoutingRefusal("empty degradation declaration is refused")
    if len(labels) != len(set(labels)):
        raise RoutingRefusal("duplicate degradation labels are refused")
    unknown = sorted(set(labels) - set(LABELS))
    if unknown:
        raise RoutingRefusal(f"unknown degradation labels are refused: {unknown}")
    if len(labels) != 1:
        raise RoutingRefusal("mixed degradation declarations are refused")
    return BACKENDS[labels[0]]


def select_five_expert_backend_for_category(category: str, composition_size: int) -> str:
    if int(composition_size) != 1:
        raise RoutingRefusal("non-singleton dataset compositions are refused")
    label = CATEGORIES.get(str(category))
    if label is None:
        raise RoutingRefusal(f"unknown or mixed dataset category is refused: {category}")
    return BACKENDS[label]


def apply_frozen_expert_postprocess(
    backend: str, lq_zero_to_one: torch.Tensor, restored_zero_to_one: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the frozen expert's exact uint8 output rule.

    Returns uint8 BCHW and the same pixels normalized to [0, 1].
    """

    if (
        lq_zero_to_one.ndim != 4
        or lq_zero_to_one.shape != restored_zero_to_one.shape
        or lq_zero_to_one.shape[1] != 3
    ):
        raise ValueError("expert postprocess expects matching BCHW RGB tensors")
    if not torch.isfinite(lq_zero_to_one).all() or not torch.isfinite(restored_zero_to_one).all():
        raise ValueError("expert postprocess refuses non-finite pixels")
    if backend in SCALAR_ALPHAS:
        lq_m11 = lq_zero_to_one.float().mul(2).sub(1)
        restored_m11 = restored_zero_to_one.float().mul(2).sub(1)
        output_u8, _ = quantized_scalar_residual(
            lq_m11, restored_m11, SCALAR_ALPHAS[backend]
        )
        return output_u8, output_u8.float().div(255)
    if backend in IDENTITY_BACKENDS:
        output_u8 = restored_zero_to_one.float().clamp(0, 1).mul(255).round().to(torch.uint8)
        return output_u8, output_u8.float().div(255)
    raise RoutingRefusal(f"unknown expert backend is refused: {backend}")


def validate_five_expert_routing_config(path: Path) -> dict[str, object]:
    config = json.loads(Path(path).read_text())
    scope = config.get("scope", {})
    routing = config.get("routing", {})
    registry_record = config.get("registry", {})
    experts = config.get("experts", {})
    execution = config.get("execution", {})
    if (
        config.get("schema_version") != 1
        or config.get("status") != "user_authorized_frozen_five_expert_fail_closed_routing"
        or scope.get("isolated_single_degradation_only") is not True
        or scope.get("mixed_degradations_supported") is not False
        or scope.get("qwen_routing_enabled") is not False
        or scope.get("model_weight_updates") is not False
        or scope.get("checkpoint_selection_reopened") is not False
        or scope.get("development_gt_used_for_routing") is not False
        or scope.get("sealed_internal_accessed") is not False
        or scope.get("official_foundir_test_accessed") is not False
        or routing.get("policy") != POLICY
        or tuple(routing.get("accepted_labels", ())) != LABELS
        or routing.get("unknown_label_action") != "raise_routing_refusal"
        or routing.get("empty_label_action") != "raise_routing_refusal"
        or routing.get("duplicate_label_action") != "raise_routing_refusal"
        or routing.get("mixed_label_action") != "raise_routing_refusal"
        or routing.get("ground_truth_pixels_used") is not False
        or routing.get("qwen_prediction_used") is not False
        or set(experts) != set(LABELS)
        or execution.get("cpu_routing_tests_required") is not True
        or execution.get("routed_vs_direct_output_equivalence_required") is not True
        or execution.get("automatic_qwen_follow_on") is not False
        or execution.get("automatic_mixed_degradation_follow_on") is not False
    ):
        raise RuntimeError("five-expert routing contract changed")

    registry_path = ROOT / str(registry_record.get("path", ""))
    if (
        not registry_path.is_file()
        or file_sha256(registry_path) != registry_record.get("sha256")
    ):
        raise RuntimeError("five-expert registry identity changed")
    registry = json.loads(registry_path.read_text())
    if (
        registry.get("status")
        != "five_source_disjoint_development_evaluations_complete_quality_frozen"
        or registry.get("decision", {}).get("stop_single_type_metric_optimization") is not True
        or registry.get("eligibility", {}).get("released_foundir_generalist_or_specialists_selected")
        is not False
    ):
        raise RuntimeError("five-expert registry eligibility changed")

    registry_tasks = registry.get("tasks", {})
    for label in LABELS:
        expert = experts[label]
        frozen = registry_tasks.get(label)
        if not isinstance(expert, Mapping) or not isinstance(frozen, Mapping):
            raise RuntimeError(f"missing frozen expert contract: {label}")
        if (
            expert.get("backend") != BACKENDS[label]
            or expert.get("category") not in CATEGORIES
            or CATEGORIES[expert["category"]] != label
            or expert.get("checkpoint") != frozen.get("checkpoint")
            or expert.get("checkpoint_sha256") != frozen.get("checkpoint_sha256")
            or frozen.get("mean_sow_pass") is not True
        ):
            raise RuntimeError(f"frozen expert binding changed: {label}")
        checkpoint = ROOT / str(expert["checkpoint"])
        if not checkpoint.is_file() or file_sha256(checkpoint) != expert["checkpoint_sha256"]:
            raise RuntimeError(f"frozen expert checkpoint identity changed: {label}")
        postprocess = expert.get("postprocess", {})
        expected = {
            "blur": ("scalar_residual", 0.75),
            "noise": ("scalar_residual", 0.30),
            "haze": ("identity", None),
            "rain": ("identity", None),
            "low_light": ("scalar_residual", 0.925),
        }[label]
        if postprocess.get("kind") != expected[0]:
            raise RuntimeError(f"frozen expert postprocess changed: {label}")
        if expected[1] is None:
            if set(postprocess) != {"kind"}:
                raise RuntimeError(f"identity expert gained postprocessing: {label}")
        elif float(postprocess.get("alpha", -1)) != expected[1]:
            raise RuntimeError(f"frozen expert alpha changed: {label}")
    return config
