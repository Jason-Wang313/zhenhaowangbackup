import unittest

import torch
from torch import nn

from lowlight_restoration.vlm_guided_moe import (
    ConditionedMoEMLP,
    VLMGuidedSoftMoE,
    conditioned_moe_modules,
    inject_vlm_guided_moe_into_flux_double_blocks,
    routing_load_balance_loss,
    vlm_moe_conditioning,
)


class VLMGuidedSoftMoETest(unittest.TestCase):
    def _model(self, *, top_k=None, zero_vlm=True):
        torch.manual_seed(7)
        return VLMGuidedSoftMoE(
            feature_dim=16,
            vlm_dim=12,
            num_experts=4,
            expert_hidden_dim=24,
            router_hidden_dim=10,
            top_k=top_k,
            zero_initialize_vlm_output=zero_vlm,
        )

    def test_dense_weights_are_normalized(self):
        model = self._model(zero_vlm=False)
        output, routing = model(torch.randn(3, 9, 16), torch.randn(3, 5, 12))
        self.assertEqual(output.shape, (3, 9, 16))
        torch.testing.assert_close(routing.weights.sum(-1), torch.ones(3))
        self.assertTrue(torch.all(routing.weights > 0))

    def test_top_k_keeps_exactly_two_experts(self):
        model = self._model(top_k=2, zero_vlm=False)
        _, routing = model(torch.randn(5, 7, 16), torch.randn(5, 12))
        self.assertTrue(torch.equal((routing.weights > 0).sum(-1), torch.full((5,), 2)))
        torch.testing.assert_close(routing.weights.sum(-1), torch.ones(5))

    def test_zero_initialized_vlm_branch_matches_image_only(self):
        model = self._model(zero_vlm=True).eval()
        image = torch.randn(2, 6, 16)
        vlm = torch.randn(2, 4, 12)
        with torch.no_grad():
            with_vlm, with_routing = model(image, vlm, use_vlm=True)
            image_only, image_routing = model(image, use_vlm=False)
        torch.testing.assert_close(with_vlm, image_only, rtol=0, atol=0)
        torch.testing.assert_close(with_routing.weights, image_routing.weights, rtol=0, atol=0)
        torch.testing.assert_close(with_routing.vlm_logit_adjustment, torch.zeros(2, 4))

    def test_nonzero_vlm_branch_changes_only_routing_inputs(self):
        model = self._model(zero_vlm=False).eval()
        image = torch.randn(2, 6, 16)
        vlm_a = torch.randn(2, 4, 12)
        vlm_b = torch.randn(2, 4, 12)
        _, routing_a = model(image, vlm_a)
        _, routing_b = model(image, vlm_b)
        torch.testing.assert_close(routing_a.image_logits, routing_b.image_logits)
        self.assertFalse(torch.allclose(routing_a.vlm_logit_adjustment, routing_b.vlm_logit_adjustment))
        self.assertFalse(torch.allclose(routing_a.weights, routing_b.weights))

    def test_gradients_reach_both_routers_and_experts(self):
        model = self._model(zero_vlm=False)
        image = torch.randn(4, 8, 16)
        vlm = torch.randn(4, 3, 12)
        output, routing = model(image, vlm)
        loss = output.square().mean() + 0.1 * routing_load_balance_loss(routing.weights)
        loss.backward()
        groups = {
            "image_router": model.image_router,
            "vlm_projector": model.vlm_projector,
            "experts": model.experts,
        }
        for name, module in groups.items():
            gradients = [p.grad for p in module.parameters() if p.requires_grad]
            self.assertTrue(gradients, name)
            self.assertTrue(any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in gradients), name)

    def test_shape_and_condition_errors_fail_loudly(self):
        model = self._model()
        with self.assertRaisesRegex(ValueError, "vlm_tokens are required"):
            model(torch.randn(2, 5, 16), use_vlm=True)
        with self.assertRaisesRegex(ValueError, "batch sizes differ"):
            model(torch.randn(2, 5, 16), torch.randn(3, 2, 12))
        with self.assertRaisesRegex(ValueError, "feature dimension changed"):
            model(torch.randn(2, 5, 16), torch.randn(2, 2, 11))


class _ToyDoubleBlock(nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__()
        self.hidden_size = feature_dim
        self.img_mlp = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.GELU(),
            nn.Linear(feature_dim * 2, feature_dim),
        )

    def forward(self, image_tokens):
        return self.img_mlp(image_tokens)


class _ToyFlux(nn.Module):
    def __init__(self):
        super().__init__()
        self.double_blocks = nn.ModuleList([_ToyDoubleBlock(8), _ToyDoubleBlock(8)])

    def forward(self, image_tokens):
        for block in self.double_blocks:
            image_tokens = block(image_tokens)
        return image_tokens


class FluxMoEInjectionTest(unittest.TestCase):
    def test_insertion_preserves_pretrained_function_exactly(self):
        torch.manual_seed(11)
        model = _ToyFlux().eval()
        image = torch.randn(3, 6, 8)
        with torch.no_grad():
            expected = model(image)
        injected = inject_vlm_guided_moe_into_flux_double_blocks(
            model,
            block_indices=[1],
            vlm_dim=12,
            num_experts=3,
            router_hidden_dim=7,
        )
        self.assertEqual(list(injected), ["double_blocks.1.img_mlp"])
        self.assertIsInstance(model.double_blocks[1].img_mlp, ConditionedMoEMLP)
        with torch.no_grad(), vlm_moe_conditioning(model, None, use_vlm=False):
            actual = model(image)
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)

    def test_context_clears_conditioning_after_forward(self):
        model = _ToyFlux()
        inject_vlm_guided_moe_into_flux_double_blocks(
            model,
            block_indices=[0],
            vlm_dim=12,
            num_experts=2,
            router_hidden_dim=7,
        )
        module = next(iter(conditioned_moe_modules(model).values()))
        with vlm_moe_conditioning(model, torch.randn(2, 4, 12), use_vlm=True):
            model(torch.randn(2, 5, 8))
            self.assertIsNotNone(module.last_routing)
        self.assertIsNone(module._vlm_tokens)
        self.assertFalse(module._use_vlm)
        self.assertIsNone(module.last_routing)

    def test_bad_or_duplicate_block_indices_fail(self):
        with self.assertRaisesRegex(ValueError, "non-empty and unique"):
            inject_vlm_guided_moe_into_flux_double_blocks(
                _ToyFlux(), block_indices=[0, 0], vlm_dim=12, num_experts=2, router_hidden_dim=7
            )
        with self.assertRaisesRegex(IndexError, "out of range"):
            inject_vlm_guided_moe_into_flux_double_blocks(
                _ToyFlux(), block_indices=[2], vlm_dim=12, num_experts=2, router_hidden_dim=7
            )

    def test_injected_bank_can_be_trainable_inside_frozen_backbone(self):
        model = _ToyFlux().requires_grad_(False)
        inject_vlm_guided_moe_into_flux_double_blocks(
            model,
            block_indices=[1],
            vlm_dim=12,
            num_experts=2,
            router_hidden_dim=7,
            trainable=True,
        )
        trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
        self.assertTrue(trainable_names)
        self.assertTrue(all(name.startswith("double_blocks.1.img_mlp.moe.") for name in trainable_names))


if __name__ == "__main__":
    unittest.main()
