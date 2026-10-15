"""Runtime replay of the frozen four-way FoundIR routing consensus."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from safetensors import safe_open

from .final_eleven_route_router import final_eleven_route, safe_expert_plan
from .noise_lowlight_physical_guard import physical_features, physical_route
from .qwen_compositional_router import compositional_route
from .qwen_eleven_class_gate import CLASSES, SINGLETON_CLASSES, consensus_route
from smoke.run_foundir_qwen_compositional_router_v1 import predict_probabilities
from smoke.run_foundir_qwen_compositional_router_v3 import apply_gate
from smoke.run_foundir_qwen_five_class_gate_ddp import probabilities, sha256
from smoke.run_foundir_qwen_lowlight_noise_dual_nonlinear_v2 import (
    apply_dual,
    model_from,
)


ROOT = Path(__file__).resolve().parents[1]


def _workspace(value: str | Path) -> Path:
    path = Path(value)
    result = (path if path.is_absolute() else ROOT / path).resolve()
    result.relative_to(ROOT.resolve())
    return result


def _json(path: str | Path) -> dict:
    return json.loads(_workspace(path).read_text())


def _tensors(path: str | Path, expected_sha256: str | None = None) -> dict[str, np.ndarray]:
    resolved = _workspace(path)
    if expected_sha256 is not None and sha256(resolved) != expected_sha256:
        raise RuntimeError(f"frozen router artifact changed: {resolved}")
    with safe_open(resolved, framework="pt", device="cpu") as handle:
        return {name: handle.get_tensor(name).numpy() for name in handle.keys()}


def _softmax_head(
    features: np.ndarray,
    tensors: dict[str, np.ndarray],
    temperature: float,
) -> np.ndarray:
    standardized = (features - tensors["feature_mean"]) / tensors["feature_scale"]
    logits = standardized @ tensors["coefficient"].T + tensors["intercept"]
    return probabilities(logits, temperature)


class FrozenFinalElevenRouter:
    """Apply exactly the frozen development router to new images and Qwen features."""

    def __init__(self) -> None:
        self.consensus_config = _json(
            "configs/foundir_qwen_mixed_consensus_development_v3.json"
        )
        self.compositional_config = _json(
            "configs/foundir_qwen_compositional_router_v3.json"
        )
        self.compositional_result = _json(
            "checkpoints/foundir_qwen_compositional_router_v3/result.json"
        )
        self.dual_config = _json(
            "configs/foundir_qwen_lowlight_noise_dual_nonlinear_v2.json"
        )
        self.dual_result = _json(
            "checkpoints/foundir_qwen_lowlight_noise_dual_nonlinear_v2/result.json"
        )
        self.physical_config = _json(
            "configs/foundir_qwen_noise_lowlight_physical_holdout_v3.json"
        )

        consensus = self.consensus_config["router"]
        self.five_head = _tensors(consensus["five_head"], consensus["five_head_sha256"])
        self.eleven_head = _tensors(
            consensus["eleven_head"], consensus["eleven_head_sha256"]
        )

        base = self.dual_config["base"]
        self.base_head = _tensors(base["head"], base["head_sha256"])
        self.lowlight_model = model_from(
            _workspace(self.dual_config["lowlight_side"]["head"]),
            int(self.dual_config["features"]["feature_dim"]),
        )
        if (
            sha256(_workspace(self.dual_config["lowlight_side"]["head"]))
            != self.dual_config["lowlight_side"]["head_sha256"]
        ):
            raise RuntimeError("frozen low-light nonlinear head changed")
        noise_path = _workspace(self.dual_result["head"])
        if sha256(noise_path) != self.dual_result["head_sha256"]:
            raise RuntimeError("frozen noise nonlinear head changed")
        self.noise_model = model_from(
            noise_path, int(self.dual_config["features"]["feature_dim"])
        )

        rule_spec = self.physical_config["physical_guard"]
        rule_path = _workspace(rule_spec["rule"])
        if sha256(rule_path) != rule_spec["rule_sha256"]:
            raise RuntimeError("frozen physical guard changed")
        self.physical_rule = json.loads(rule_path.read_text())["rule"]

        self.negative_noise_margin = float(
            self.compositional_result["selected"]["negative_noise_margin"]
        )
        self.noise_margin = float(self.dual_result["margin_selection"]["margin"])

    @property
    def feature_dim(self) -> int:
        return int(self.dual_config["features"]["feature_dim"])

    def route(
        self,
        features: np.ndarray,
        image_paths: Sequence[str | Path],
    ) -> list[dict[str, object]]:
        features = np.asarray(features, dtype=np.float32)
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError(f"expected [N,{self.feature_dim}] Qwen features")
        if len(image_paths) != len(features):
            raise ValueError("image paths and Qwen features are not aligned")

        consensus = self.consensus_config["router"]
        five = _softmax_head(features, self.five_head, float(consensus["five_temperature"]))
        eleven = _softmax_head(
            features, self.eleven_head, float(consensus["eleven_temperature"])
        )
        eleven_routes = [
            consensus_route(
                five_row,
                eleven_row,
                five_threshold=float(consensus["five_threshold"]),
                eleven_threshold=float(consensus["eleven_threshold"]),
            )
            for five_row, eleven_row in zip(five, eleven, strict=True)
        ]

        z = (features - self.base_head["feature_mean"]) / self.base_head["feature_scale"]
        base = predict_probabilities(
            z, self.base_head["coefficient"], self.base_head["intercept"]
        )
        linear_conditional = predict_probabilities(
            z,
            self.base_head["conditional_noise_coefficient"][None, :],
            self.base_head["conditional_noise_intercept"],
        )[:, 0]

        comp_config = self.compositional_config["base"]
        comp_values = apply_gate(
            base,
            linear_conditional,
            route_margin=float(comp_config["route_margin"]),
            positive_margin=float(comp_config["positive_noise_margin"]),
            negative_margin=self.negative_noise_margin,
        )
        compositional_routes = [
            compositional_route(row, margin=float(comp_config["route_margin"]))
            for row in comp_values
        ]

        with torch.inference_mode():
            tensor = torch.from_numpy(z.astype(np.float32, copy=False))
            lowlight_q = torch.sigmoid(self.lowlight_model(tensor)).numpy()
            noise_q = torch.sigmoid(self.noise_model(tensor)).numpy()
        dual_values, _, _ = apply_dual(
            base,
            linear_conditional,
            lowlight_q,
            noise_q,
            self.dual_config,
            self.noise_margin,
        )
        dual_routes = [
            compositional_route(
                row, margin=float(self.dual_config["base"]["route_margin"])
            )
            for row in dual_values
        ]

        physical = [
            physical_features(_workspace(path), int(self.physical_rule["maximum_side"]))
            for path in image_paths
        ]
        results: list[dict[str, object]] = []
        for index, item in enumerate(physical):
            physical_final = dual_routes[index]
            if dual_routes[index] == "noise":
                physical_final = physical_route(
                    item,
                    float(self.physical_rule["low_exposure"]),
                    float(self.physical_rule["high_exposure"]),
                    float(self.physical_rule["dark_noise_floor"]),
                )
            route = final_eleven_route(
                compositional_route=compositional_routes[index],
                compositional_accepted=compositional_routes[index] is not None,
                eleven_route=eleven_routes[index],
                eleven_accepted=eleven_routes[index] is not None,
                dual_route=dual_routes[index],
                dual_accepted=dual_routes[index] is not None,
                physical_route=physical_final,
            )
            five_proposal = SINGLETON_CLASSES[int(np.argmax(five[index]))]
            plan = safe_expert_plan(
                route,
                exposure=float(item["exposure"]),
                five_proposal=five_proposal,
            )
            results.append(
                {
                    "route": route,
                    "expert_plan": list(plan),
                    "compositional_route": compositional_routes[index],
                    "eleven_route": eleven_routes[index],
                    "dual_route": dual_routes[index],
                    "physical_route": physical_final,
                    "five_proposal": five_proposal,
                    "five_confidence": float(np.max(five[index])),
                    "eleven_proposal": CLASSES[int(np.argmax(eleven[index]))],
                    "eleven_confidence": float(np.max(eleven[index])),
                    "exposure": float(item["exposure"]),
                    "dark_noise": float(item["dark_noise"]),
                    "fallback_applied": route is None and bool(plan),
                }
            )
        return results
