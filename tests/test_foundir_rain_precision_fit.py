from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from lowlight_restoration.flux2_endpoint_precision import (
    assert_fp32_optimizer, promote_lora_to_fp32, restore_endpoint,
)
from lowlight_restoration.lora import inject_lora, load_lora_checkpoint, save_lora_checkpoint
from smoke.train_foundir_rain_flux2_precision_fit_v1_ddp import (
    ROOT, read_contract, select_fit_rows, training_losses,
)


def pack(image):
    batch, channels, height, width = image.shape
    return image.reshape(batch, channels, -1).transpose(1, 2), torch.zeros(batch, height * width, 4, dtype=torch.long)


class ToyAE(nn.Module):
    def encode(self, image):
        return image.float() * .317

    def decode(self, latent):
        return latent * 1.5


class ToyFlux(nn.Module):
    def __init__(self):
        super().__init__()
        self.velocity = nn.Parameter(torch.tensor(.031))

    def forward(self, *, x, **kwargs):
        return torch.ones_like(x) * self.velocity


class PrecisionFitTests(unittest.TestCase):
    def test_small_updates_are_retained_and_checkpoint_roundtrip_is_fp32(self):
        model = nn.Sequential(nn.Linear(4, 4, dtype=torch.bfloat16))
        inject_lora(model, ["0"], rank=2, alpha=2)
        parameters = promote_lora_to_fp32(model)
        with torch.no_grad():
            for parameter in parameters:
                parameter.fill_(.01)
        before = [p.clone() for p in parameters]
        optimizer = torch.optim.AdamW(parameters, lr=1e-5, weight_decay=0)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            loss = model(torch.ones(2, 4)).float().sum()
        loss.backward(); optimizer.step()
        assert_fp32_optimizer(optimizer)
        self.assertTrue(all(torch.any(a != b) for a, b in zip(parameters, before)))
        self.assertIsNone(model[0].base.weight.grad)
        coarse = torch.tensor(.01, dtype=torch.bfloat16)
        self.assertEqual((coarse.float() - 1e-5).to(torch.bfloat16), coarse)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lora.safetensors"
            save_lora_checkpoint(model, path, {"step": 1})
            restored = nn.Sequential(nn.Linear(4, 4, dtype=torch.bfloat16))
            inject_lora(restored, ["0"], rank=2, alpha=2)
            loaded = promote_lora_to_fp32(restored)
            load_lora_checkpoint(restored, path)
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(parameters, loaded)))

    def test_shared_endpoint_preserves_observation_precision_and_gradients(self):
        model, ae = ToyFlux(), ToyAE()
        observation = torch.linspace(-.7, .7, 192).reshape(1, 3, 8, 8)
        context, ids = torch.zeros(1, 1, 1), torch.zeros(1, 1, 4, dtype=torch.long)
        model.train()
        result = restore_endpoint(model, ae, observation, context, ids, pack)
        expected = ae.encode(observation) - result["prediction"].float().transpose(1, 2).reshape_as(observation)
        self.assertEqual(result["predicted_clean"].dtype, torch.float32)
        self.assertTrue(torch.equal(result["predicted_clean"], expected))
        self.assertFalse(result["degraded"].requires_grad)
        self.assertTrue(torch.any(result["degraded"] != result["degraded"].bfloat16().float()))
        model.eval()
        with torch.inference_mode():
            inferred = restore_endpoint(model, ae, observation, context, ids, pack)
        self.assertTrue(torch.equal(result["decoded"], inferred["decoded"]))
        result["decoded"].square().mean().backward()
        self.assertTrue(torch.isfinite(model.velocity.grad))
        self.assertNotEqual(float(model.velocity.grad), 0)

    def test_full_parent_loss_backpropagates_through_shared_endpoint(self):
        config, parent = read_contract(ROOT / "configs/foundir_rain_flux2_precision_fit_v1.json")
        model, ae = ToyFlux(), ToyAE()
        observation = torch.ones(1, 3, 8, 8) * .23
        target = observation * .8
        result = restore_endpoint(model, ae, observation, torch.zeros(1, 1, 1), torch.zeros(1, 1, 4), pack)
        loss, terms = training_losses(result, ae.encode(target), observation, target, (target + 1) / 2,
                                     lambda a, b: (a - b).square().mean(), pack, parent["training"]["losses"])
        self.assertEqual(len(terms), 8)
        self.assertTrue(all(torch.isfinite(value) for value in terms.values()))
        loss.backward()
        self.assertTrue(torch.isfinite(model.velocity.grad))
        self.assertEqual(config["crop_size"], 512)

    def test_fit_sampling_is_order_invariant_source_unique_and_train_only(self):
        rows = [{"role": "direct_endpoint_train", "split": "train", "source_group": f"g{i//2}",
                 "record_id": f"r{i}", "width": "800", "height": "600"} for i in range(70)]
        a, b = select_fit_rows(rows), select_fit_rows(list(reversed(rows)))
        self.assertEqual(a, b)
        self.assertEqual(len({r["source_group"] for r in a}), 32)
        self.assertTrue(all(int(r["crop_x"]) + 512 <= 800 and int(r["crop_y"]) + 512 <= 600 for r in a))
        rows[0]["split"] = "test"
        with self.assertRaises(RuntimeError):
            select_fit_rows(rows)


if __name__ == "__main__":
    unittest.main()
