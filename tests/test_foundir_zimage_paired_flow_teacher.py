from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / ".deps/zimage"))
sys.path.insert(0, str(ROOT / "third_party/diffsynth-studio"))

from lowlight_restoration.foundir_zimage_paired_flow import (
    load_config,
    load_frozen_rows,
    model_paths,
    paired_flow_euler_step,
    paired_flow_state,
    paired_flow_target,
    validate_config,
    validate_gpu_inventory,
)
from diffsynth.diffusion.flow_match import FlowMatchScheduler
from smoke.evaluate_foundir_blur_only_zimage_teacher_paired_flow_ddp import restore_from_exact_lq_latent


CONFIG = ROOT / "configs/foundir_blur_only_zimage_teacher_paired_flow_overfit_v2.json"


def config_copy():
    return json.loads(CONFIG.read_text())


def test_paired_flow_endpoints_target_and_exact_euler_solution() -> None:
    gt = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    lq = torch.tensor([[[[5.0, 6.0], [7.0, 8.0]]]])
    target = paired_flow_target(gt, lq)
    assert torch.equal(paired_flow_state(gt, lq, 0.0), gt)
    assert torch.equal(paired_flow_state(gt, lq, 1.0), lq)
    assert torch.equal(target, lq - gt)
    sample = lq.clone()
    schedule = [1.0, 0.8, 0.35, 0.0]
    for sigma, next_sigma in zip(schedule, schedule[1:]):
        sample = paired_flow_euler_step(sample, target, sigma, next_sigma)
    assert torch.allclose(sample, gt)


def test_shift_six_schedule_starts_at_exact_lq_endpoint_and_ends_via_euler_at_zero() -> None:
    scheduler = FlowMatchScheduler("Z-Image")
    scheduler.set_timesteps(50, shift=6.0)
    assert float(scheduler.sigmas[0]) == pytest.approx(1.0)
    assert len(scheduler.sigmas) == 50
    assert torch.all(scheduler.sigmas[:-1] > scheduler.sigmas[1:])
    assert float(scheduler.sigmas[-1]) > 0.0
    final = paired_flow_euler_step(
        torch.ones(1), torch.ones(1), scheduler.sigmas[-1], torch.tensor(0.0)
    )
    assert torch.isfinite(final).all()


def test_evaluator_initializes_model_from_exact_lq_latent() -> None:
    class FakeScheduler:
        def set_timesteps(self, steps, shift):
            assert steps == 2 and shift == 6.0
            self.sigmas = torch.tensor([1.0, 0.5])
            self.timesteps = self.sigmas * 1000

    class FakeVAE:
        def __call__(self, value):
            return value

    class FakePipe:
        device = torch.device("cpu")
        torch_dtype = torch.float32
        units = []
        in_iteration_models = ("dit", "controlnet")
        dit = object()
        controlnet = None
        vae_encoder = FakeVAE()
        vae_decoder = FakeVAE()

        def __init__(self):
            self.scheduler = FakeScheduler()
            self.first_model_latent = None

        def load_models_to_device(self, names):
            return None

        def preprocess_image(self, image):
            del image
            return torch.full((1, 1, 1, 1), 7.0)

        def model_fn(self, *, latents, **kwargs):
            del kwargs
            if self.first_model_latent is None:
                self.first_model_latent = latents.clone()
            return torch.zeros_like(latents)

        def vae_output_to_image(self, value):
            pixel = int(value.flatten()[0])
            return Image.new("RGB", (1, 1), (pixel, pixel, pixel))

    pipe = FakePipe()
    result = restore_from_exact_lq_latent(
        pipe, Image.new("RGB", (8, 8)), "deblur", steps=2, shift=6.0
    )
    assert torch.equal(pipe.first_model_latent, torch.full((1, 1, 1, 1), 7.0))
    assert result.getpixel((0, 0)) == (7, 7, 7)


def test_config_and_frozen_crop_contract() -> None:
    config = load_config(CONFIG)
    train, visual = load_frozen_rows(config, verify_images=False)
    assert len(train) == len(visual) == 16
    assert not ({row["source_group"] for row in train} & {row["source_group"] for row in visual})
    assert config["implementation"]["controlnet"] == "disabled"
    assert "controlnet" not in config["model"]
    assert config["training"]["scheduler_shift"] == 6.0
    assert config["inference"]["cfg_scale"] == 1.0
    assert config["inference"]["initialization"] == "exact_lq_vae_latent_at_sigma_1"


@pytest.mark.parametrize(
    "field",
    [
        "distillation",
        "decoder_finetuning",
        "qwen_degradation_conditioning",
        "mixed_degradations",
        "sealed_internal_accessed",
        "official_foundir_test_accessed",
    ],
)
def test_forbidden_scope_fails_closed(field: str) -> None:
    config = config_copy()
    config["scope"][field] = True
    with pytest.raises(RuntimeError, match="forbidden"):
        validate_config(config)


def test_controlnet_shift_cfg_and_initialization_fail_closed() -> None:
    mutations = []
    with_controlnet = config_copy()
    with_controlnet["model"]["controlnet"] = {"path": "forbidden"}
    mutations.append(with_controlnet)
    wrong_shift = config_copy()
    wrong_shift["training"]["scheduler_shift"] = 3.0
    mutations.append(wrong_shift)
    wrong_cfg = config_copy()
    wrong_cfg["inference"]["cfg_scale"] = 4.0
    mutations.append(wrong_cfg)
    wrong_init = config_copy()
    wrong_init["inference"]["initialization"] = "gaussian_noise"
    mutations.append(wrong_init)
    for config in mutations:
        with pytest.raises(RuntimeError):
            validate_config(config)


def test_only_base_artifacts_are_resolved() -> None:
    paths = model_paths(load_config(CONFIG))
    assert set(paths) == {"transformer", "text_encoder", "vae", "tokenizer"}
    assert len(paths["transformer"]) == 2
    assert len(paths["text_encoder"]) == 3


def test_h100_h200_profiles_and_batch_contract() -> None:
    config = load_config(CONFIG)
    assert validate_gpu_inventory(["NVIDIA H100 80GB HBM3"] * 8, [81920] * 8, config) == "H100"
    assert validate_gpu_inventory(["NVIDIA H200"] * 8, [143000] * 8, config) == "H200"
    with pytest.raises(RuntimeError, match="mixed"):
        validate_gpu_inventory(["NVIDIA H100"] * 4 + ["NVIDIA H200"] * 4, [81920] * 8, config)
    invalid = copy.deepcopy(config)
    invalid["training"]["global_batch_size"] = 16
    with pytest.raises(RuntimeError, match="batch"):
        validate_config(invalid)
