"""Fail-closed deployment contract for the isolated Rain Restormer expert."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch

from .task_native_restormer import Restormer


ROOT = Path(__file__).resolve().parents[1]
KNOWN_DEGRADATIONS = ("blur", "noise", "rain", "haze", "low_light")
RAIN_BACKEND = "rain_restormer_step5000"
FALLBACK_BACKEND = "existing_pipeline_unchanged"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_restoration_backend(degradations: Sequence[str]) -> str:
    """Select Rain expert only for the exact singleton label ``rain``."""

    labels = tuple(str(value) for value in degradations)
    if len(labels) != len(set(labels)):
        raise ValueError("duplicate degradation labels are forbidden")
    unknown = sorted(set(labels) - set(KNOWN_DEGRADATIONS))
    if unknown:
        raise ValueError(f"unknown degradation labels: {unknown}")
    return RAIN_BACKEND if labels == ("rain",) else FALLBACK_BACKEND


def select_restoration_backend_for_category(category: str, composition_size: int) -> str:
    """Dataset-facing equivalent of the singleton Rain routing contract."""

    if category == "10Rain" and int(composition_size) == 1:
        return RAIN_BACKEND
    return FALLBACK_BACKEND


def validate_rain_expert_config(path: Path) -> dict[str, object]:
    config = json.loads(Path(path).read_text())
    scope = config.get("scope", {})
    routing = config.get("routing", {})
    model = config.get("model", {})
    evidence = config.get("evidence", {})
    execution = config.get("execution", {})
    if (
        config.get("schema_version") != 1
        or config.get("status") != "user_authorized_isolated_rain_restormer_deployment_selection"
        or scope.get("task") != "rain"
        or scope.get("category") != "10Rain"
        or int(scope.get("isolated_composition_size", -1)) != 1
        or scope.get("other_types_modified") is not False
        or scope.get("mixed_degradations_supported_by_expert") is not False
        or scope.get("sealed_internal_accessed") is not False
        or scope.get("official_foundir_test_accessed") is not False
        or routing.get("policy") != "fail_closed_exact_singleton_rain_only"
        or routing.get("expert_backend") != RAIN_BACKEND
        or routing.get("non_rain_backend") != FALLBACK_BACKEND
        or routing.get("allowed_expert_degradations") != ["rain"]
        or set(routing.get("explicitly_forbidden_expert_degradations", []))
        != {"blur", "noise", "haze", "low_light"}
        or routing.get("mixed_degradation_fallback") != FALLBACK_BACKEND
        or routing.get("ground_truth_pixels_used_for_routing") is not False
        or model.get("architecture") != "task_native_restormer_deraining"
        or int(model.get("step", -1)) != 5000
        or int(evidence.get("validation_records", -1)) != 835
        or int(evidence.get("validation_source_groups", -1)) != 117
        or float(evidence.get("mean_psnr", -1)) < float(evidence.get("rain_sow_psnr_minimum", 1e9))
        or float(evidence.get("mean_lpips", 1e9)) > float(evidence.get("rain_sow_lpips_maximum", -1))
        or evidence.get("mean_sow_pass") is not True
        or evidence.get("legacy_stronger_bootstrap_gate_pass") is not False
        or execution.get("new_training_required") is not False
        or execution.get("new_gpu_evaluation_required") is not False
        or execution.get("input_range") != "zero_to_one_rgb"
        or execution.get("output_range") != "clamp_zero_to_one_rgb"
        or int(execution.get("spatial_dimensions_must_be_divisible_by", -1)) != 8
    ):
        raise RuntimeError("isolated Rain expert deployment contract changed")
    for path_key, sha_key in (
        ("base_checkpoint", "base_checkpoint_sha256"),
        ("adaptation_config", "adaptation_config_sha256"),
        ("checkpoint", "checkpoint_sha256"),
    ):
        artifact = ROOT / str(model[path_key])
        if not artifact.is_file() or file_sha256(artifact) != model[sha_key]:
            raise RuntimeError(f"Rain expert artifact identity changed: {artifact}")
    for path_key, sha_key in (("selection", "selection_sha256"), ("development_result", "development_result_sha256")):
        artifact = ROOT / str(evidence[path_key])
        if not artifact.is_file() or file_sha256(artifact) != evidence[sha_key]:
            raise RuntimeError(f"Rain expert evidence identity changed: {artifact}")
    selection = json.loads((ROOT / str(evidence["selection"])).read_text())
    result = json.loads((ROOT / str(evidence["development_result"])).read_text())
    if (
        selection.get("task") != "rain"
        or selection.get("selected_step") != 5000
        or selection.get("checkpoint_sha256") != model["checkpoint_sha256"]
        or selection.get("development_result_sha256") != evidence["development_result_sha256"]
        or selection.get("metrics", {}).get("psnr") != evidence["mean_psnr"]
        or selection.get("metrics", {}).get("lpips") != evidence["mean_lpips"]
        or selection.get("sealed_internal_test_accessed") is not False
        or selection.get("protected_foundir_test_accessed") is not False
        or result.get("task") != "rain"
        or result.get("step") != 5000
        or result.get("checkpoint_sha256") != model["checkpoint_sha256"]
    ):
        raise RuntimeError("Rain expert selection evidence changed")
    return config


def load_rain_expert(path: Path, *, device: torch.device | str = "cpu") -> Restormer:
    config = validate_rain_expert_config(path)
    checkpoint_path = ROOT / str(config["model"]["checkpoint"])
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (
        payload.get("status") != "foundir_task_native_adaptation_checkpoint"
        or payload.get("task") != "rain"
        or payload.get("step") != 5000
        or payload.get("config_sha256") != config["model"]["adaptation_config_sha256"]
        or payload.get("sealed_internal_test_accessed") is not False
        or payload.get("protected_foundir_test_accessed") is not False
        or not isinstance(payload.get("model"), Mapping)
    ):
        raise RuntimeError("Rain expert checkpoint payload changed")
    model = Restormer(layer_norm_type="WithBias")
    model.load_state_dict(payload["model"], strict=True)
    return model.to(device).eval().requires_grad_(False)


def restore_isolated_rain(model: Restormer, image: torch.Tensor, degradations: Sequence[str]) -> torch.Tensor:
    if select_restoration_backend(degradations) != RAIN_BACKEND:
        raise RuntimeError("Rain expert refused a non-isolated-Rain input")
    if image.ndim != 4 or image.shape[1] != 3 or image.shape[-2] % 8 or image.shape[-1] % 8:
        raise ValueError("Rain expert expects BCHW RGB with spatial dimensions divisible by 8")
    with torch.inference_mode():
        return model(image).float().clamp(0, 1)
