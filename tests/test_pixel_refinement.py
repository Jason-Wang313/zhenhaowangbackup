from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn
from safetensors import safe_open

from lowlight_restoration.lora import (
    inject_lora,
    load_restoration_checkpoint,
    save_restoration_checkpoint,
)
from lowlight_restoration.pixel_refinement import (
    BoundedPixelRefiner,
    pixel_refinement_losses,
    pixel_refiner_parameter_count,
)
from smoke.train_foundir_five_type_flux2_full_ddp import (
    resume as resume_full,
    save_checkpoint as save_full_checkpoint,
)
from smoke.train_foundir_five_type_flux2_overfit_ddp import resume_checkpoint


class PixelRefinementTests(unittest.TestCase):
    def test_zero_initialization_is_exact_identity(self) -> None:
        model = BoundedPixelRefiner(width=8, blocks=3, maximum_residual_m11=0.2)
        decoded = torch.rand(2, 3, 24, 20) * 2 - 1
        degraded = torch.rand_like(decoded) * 2 - 1
        self.assertTrue(torch.equal(model(decoded, degraded), decoded))

    def test_output_correction_is_strictly_bounded(self) -> None:
        model = BoundedPixelRefiner(width=8, blocks=2, maximum_residual_m11=0.125)
        nn.init.normal_(model.ending.weight, std=5.0)
        nn.init.normal_(model.ending.bias, std=5.0)
        decoded = torch.zeros(1, 3, 16, 16)
        degraded = torch.ones_like(decoded)
        refined = model(decoded, degraded)
        self.assertLessEqual(float((refined - decoded).abs().max()), 0.125001)

    def test_refinement_loss_has_finite_gradients(self) -> None:
        model = BoundedPixelRefiner(width=8, blocks=2)
        decoded = torch.rand(1, 3, 20, 20) * 2 - 1
        degraded = torch.rand_like(decoded) * 2 - 1
        target = torch.rand_like(decoded) * 2 - 1
        refined = model(decoded, degraded)
        losses = pixel_refinement_losses(
            refined, decoded, target, charbonnier_epsilon=1e-3
        )
        sum(losses.values()).backward()
        self.assertTrue(all(torch.isfinite(value) for value in losses.values()))
        self.assertIsNotNone(model.ending.weight.grad)
        self.assertGreater(float(model.ending.weight.grad.abs().sum()), 0.0)

    def test_parameter_count_is_lightweight(self) -> None:
        model = BoundedPixelRefiner(width=32, blocks=6)
        count = pixel_refiner_parameter_count(model)
        self.assertGreater(count, 10_000)
        self.assertLess(count, 1_000_000)

    def test_combined_checkpoint_round_trip_is_strict(self) -> None:
        flux = nn.Sequential(nn.Linear(4, 6), nn.SiLU(), nn.Linear(6, 4))
        inject_lora(flux, ["0", "2"], rank=2, alpha=2)
        refiner = BoundedPixelRefiner(width=8, blocks=2)
        with torch.no_grad():
            for parameter in flux.parameters():
                if parameter.requires_grad:
                    parameter.uniform_(-0.2, 0.2)
            refiner.ending.weight.uniform_(-0.1, 0.1)
        expected_flux = {
            key: value.detach().clone()
            for key, value in flux.state_dict().items()
            if "lora_" in key
        }
        expected_refiner = {
            key: value.detach().clone() for key, value in refiner.state_dict().items()
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "restoration.safetensors"
            save_restoration_checkpoint(flux, refiner, path, {"step": 7})
            with safe_open(path, framework="pt", device="cpu") as handle:
                self.assertEqual(
                    handle.metadata()["state_schema"],
                    "flux_lora_plus_bounded_pixel_refiner_v1",
                )
            with torch.no_grad():
                for parameter in flux.parameters():
                    if parameter.requires_grad:
                        parameter.zero_()
                for parameter in refiner.parameters():
                    parameter.zero_()
            load_restoration_checkpoint(flux, refiner, path)
        for key, value in expected_flux.items():
            self.assertTrue(torch.equal(flux.state_dict()[key], value), key)
        for key, value in expected_refiner.items():
            self.assertTrue(torch.equal(refiner.state_dict()[key], value), key)

    @staticmethod
    def trainable_toy():
        flux = nn.Sequential(nn.Linear(4, 6), nn.SiLU(), nn.Linear(6, 4))
        inject_lora(flux, ["0", "2"], rank=2, alpha=2)
        refiner = BoundedPixelRefiner(width=8, blocks=2)
        optimizer = torch.optim.AdamW(
            [
                *(parameter for parameter in flux.parameters() if parameter.requires_grad),
                *refiner.parameters(),
            ],
            lr=1e-3,
        )
        return flux, refiner, optimizer

    def test_overfit_resume_loads_one_combined_model_and_optimizer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            flux, refiner, optimizer = self.trainable_toy()
            checkpoint = output / "step_000003.safetensors"
            save_restoration_checkpoint(flux, refiner, checkpoint, {"step": 3, "config_sha256": "abc"})
            torch.save(
                {
                    "step": 3,
                    "config_sha256": "abc",
                    "optimizer": optimizer.state_dict(),
                    "training_losses": [3.0, 2.0, 1.0],
                },
                checkpoint.with_suffix(".optimizer.pt"),
            )
            expected = {key: value.clone() for key, value in refiner.state_dict().items()}
            with torch.no_grad():
                for parameter in refiner.parameters():
                    parameter.add_(1)
            step, losses = resume_checkpoint(flux, refiner, optimizer, output, "abc")
            self.assertEqual((step, losses), (3, [3.0, 2.0, 1.0]))
            for key, value in expected.items():
                self.assertTrue(torch.equal(refiner.state_dict()[key], value), key)

    def test_full_resume_loads_combined_checkpoint_and_progress(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            flux, refiner, optimizer = self.trainable_toy()
            save_full_checkpoint(
                flux,
                refiner,
                optimizer,
                output,
                step=4,
                presentations_completed=128,
                stage_history=[{"stage": "single", "global_step": 3}],
                current_stage_progress={"stage": "double", "global_step": 4},
                config_sha="def",
                trainable_parameters=123,
            )
            expected = {key: value.clone() for key, value in refiner.state_dict().items()}
            with torch.no_grad():
                for parameter in refiner.parameters():
                    parameter.add_(1)
            step, presentations, history, progress = resume_full(
                flux, refiner, optimizer, output, "def"
            )
            self.assertEqual(step, 4)
            self.assertEqual(presentations, 128)
            self.assertEqual(history[0]["stage"], "single")
            self.assertEqual(progress["stage"], "double")
            for key, value in expected.items():
                self.assertTrue(torch.equal(refiner.state_dict()[key], value), key)


if __name__ == "__main__":
    unittest.main()
