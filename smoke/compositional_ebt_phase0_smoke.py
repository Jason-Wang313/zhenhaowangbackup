#!/usr/bin/env python3
"""Zero-update Plain/Compositional EBT forward/backward smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lowlight_restoration.compositional_ebt import (
    RestorationEnergyTransformer,
    count_trainable_parameters,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--result", type=Path)
    return parser.parse_args()


def run(device: torch.device) -> dict[str, object]:
    torch.manual_seed(20261007)
    kwargs = dict(
        image_size=32,
        patch_size=8,
        hidden_size=64,
        depth=2,
        num_heads=4,
        zero_init_energy_head=False,
    )
    plain = RestorationEnergyTransformer(mode="plain", **kwargs).to(device)
    torch.manual_seed(20261007)
    compositional = RestorationEnergyTransformer(
        mode="compositional", **kwargs
    ).to(device)
    with torch.no_grad():
        plain_parameters = dict(plain.named_parameters())
        for name, parameter in compositional.named_parameters():
            parameter.copy_(plain_parameters[name])
    batch = 2
    observation = torch.rand(batch, 3, 32, 32, device=device) * 1.6 - 0.8
    target = torch.rand(batch, 3, 32, 32, device=device) * 1.6 - 0.8
    q = torch.tensor(
        [[1.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 1.0]],
        device=device,
    )

    arms: dict[str, object] = {}
    for name, model in (("plain", plain), ("compositional", compositional)):
        candidate = observation.clone().detach()
        before = model(observation, candidate, q).detach()
        step = model.descent_step(
            observation,
            candidate,
            q,
            step_size=1e-2,
            create_graph=True,
            clamp=None,
        )
        after = model(observation, step.candidate, q)
        reconstruction_loss = torch.nn.functional.smooth_l1_loss(
            step.candidate, target
        )
        reconstruction_loss.backward()
        parameter_gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        finite_parameter_gradients = bool(parameter_gradients) and all(
            bool(torch.isfinite(gradient).all()) for gradient in parameter_gradients
        )
        arms[name] = {
            "energy_before": before.cpu().tolist(),
            "energy_after": after.detach().cpu().tolist(),
            "energy_decreased_all": bool((after.detach() < before).all()),
            "candidate_gradient_abs_max": float(
                step.candidate_gradient.detach().abs().max().cpu()
            ),
            "candidate_gradient_finite": bool(
                torch.isfinite(step.candidate_gradient).all()
            ),
            "reconstruction_loss": float(reconstruction_loss.detach().cpu()),
            "parameter_gradients_finite": finite_parameter_gradients,
            "parameters_with_gradient": len(parameter_gradients),
        }

    plain_parameters = count_trainable_parameters(plain)
    compositional_parameters = count_trainable_parameters(compositional)
    passed = (
        plain_parameters == compositional_parameters
        and all(
            arm["energy_decreased_all"]
            and arm["candidate_gradient_finite"]
            and arm["candidate_gradient_abs_max"] > 0
            and arm["parameter_gradients_finite"]
            for arm in arms.values()
        )
    )
    return {
        "schema_version": 1,
        "status": "zero_update_phase0_pass" if passed else "zero_update_phase0_fail",
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "optimizer_steps": 0,
        "official_foundir_test_accessed": False,
        "sealed_internal_accessed": False,
        "triple_endpoint_accessed": False,
        "plain_trainable_parameters": plain_parameters,
        "compositional_trainable_parameters": compositional_parameters,
        "matched_parameter_count": plain_parameters == compositional_parameters,
        "arms": arms,
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    result = run(device)
    payload = json.dumps(result, indent=2, sort_keys=True)
    print(payload)
    if args.result is not None:
        args.result.parent.mkdir(parents=True, exist_ok=True)
        if args.result.exists() and args.result.read_text() != payload + "\n":
            raise RuntimeError(f"refusing to overwrite differing result: {args.result}")
        args.result.write_text(payload + "\n")
    if result["status"] != "zero_update_phase0_pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
