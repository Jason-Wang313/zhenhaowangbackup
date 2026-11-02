import copy
import unittest

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from lowlight_restoration.existing_flux_experts import FLOW_CONTRACT, FrozenLoRAExpertBank, LoRAWeights, install_replay_bank
from lowlight_restoration.lora import LoRALinear
from smoke.audit_foundir_existing_flux_experts_v1 import expected_modules


class BankTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(91)
        self.base = nn.Linear(7, 5)
        self.experts = [LoRAWeights(str(i), torch.randn(3, 7), torch.randn(5, 3), 3.) for i in range(4)]
        self.bank = FrozenLoRAExpertBank(copy.deepcopy(self.base), self.experts)
        self.x = torch.randn(2, 6, 7)

    def test_one_hot_is_original_lora(self):
        for i, e in enumerate(self.experts):
            original = LoRALinear(copy.deepcopy(self.base), 3, e.alpha)
            with torch.no_grad():
                original.lora_A.weight.copy_(e.a)
                original.lora_B.weight.copy_(e.b)
            weights = F.one_hot(torch.full((2,), i), 4).float()
            self.assertTrue(torch.equal(original(self.x), self.bank(self.x, weights)))
            self.assertTrue(torch.equal(original(self.x), self.bank.forward_expert(self.x, i)))

    def test_soft_merge_is_delta_sum_not_factor_mean(self):
        weights = torch.tensor([[.1, .2, .3, .4], [.4, .3, .2, .1]])
        delta = sum(F.linear(F.linear(self.x, e.a), e.b) * weights[:, i, None, None]
                    for i, e in enumerate(self.experts))
        torch.testing.assert_close(self.bank(self.x, weights), self.base(self.x) + delta)
        wrong_a = sum(e.a for e in self.experts) / 4
        wrong_b = sum(e.b for e in self.experts) / 4
        uniform = self.bank(self.x, torch.full((2, 4), .25))
        self.assertFalse(torch.allclose(uniform, self.base(self.x) + F.linear(F.linear(self.x, wrong_a), wrong_b)))

    def test_no_checkpoint_mutation(self):
        originals = [e.a.clone() for e in self.experts]
        with torch.no_grad():
            self.bank.a[0].add_(1)
        for e, old in zip(self.experts, originals):
            self.assertTrue(torch.equal(e.a, old))

    def test_freeze_and_router_gradient(self):
        logits = torch.randn(2, 4, requires_grad=True)
        self.bank(self.x, logits.softmax(-1)).square().mean().backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(logits.grad.abs().max().item(), 0)
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in self.bank.parameters()))

    def test_explicit_routing_checkpoint_recompute(self):
        logits = torch.randn(2, 4, requires_grad=True)
        expected = torch.autograd.grad(self.bank(self.x, logits.softmax(-1)).square().mean(), logits)[0]
        actual = torch.autograd.grad(checkpoint(self.bank, self.x, logits.softmax(-1), use_reentrant=False).square().mean(), logits)[0]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_reject_endpoint(self):
        e = self.experts[0]
        with self.assertRaisesRegex(ValueError, "endpoint"):
            FrozenLoRAExpertBank(self.base, [LoRAWeights(e.name, e.a, e.b, e.alpha, "degraded_endpoint_t1_zero_noise_v1")])

    def test_reject_invalid_weights(self):
        for w in [torch.zeros(2, 4), torch.full((2, 4), float("nan")),
                  torch.tensor([[2., -1., 0., 0.]]).expand(2, -1), torch.ones(4)]:
            with self.assertRaises(ValueError):
                self.bank(self.x, w)

    def test_reject_bad_weights_and_shapes(self):
        for e in [LoRAWeights("a", torch.randn(3, 8), torch.randn(5, 3), 3),
                  LoRAWeights("a", torch.full((3, 7), float("nan")), torch.randn(5, 3), 3),
                  LoRAWeights("a", torch.randn(3, 7), torch.randn(5, 3), float("nan"))]:
            with self.assertRaises(ValueError):
                FrozenLoRAExpertBank(self.base, [e])
        with self.assertRaises(ValueError):
            FrozenLoRAExpertBank(self.base, [self.experts[0], self.experts[0]])

    def test_bfloat16_single_expert_equivalence(self):
        base = copy.deepcopy(self.base).bfloat16()
        experts = [LoRAWeights(e.name, e.a.bfloat16(), e.b.bfloat16(), 1.7) for e in self.experts]
        bank = FrozenLoRAExpertBank(base, experts)
        for i, e in enumerate(experts):
            original = LoRALinear(copy.deepcopy(base), 3, e.alpha)
            with torch.no_grad():
                original.lora_A.weight.copy_(e.a)
                original.lora_B.weight.copy_(e.b)
            weights = F.one_hot(torch.full((2,), i), 4).float()
            self.assertTrue(torch.equal(original(self.x.bfloat16()), bank(self.x.bfloat16(), weights)))

    def test_known_50_sites(self):
        self.assertEqual(len(expected_modules()), 50)
        self.assertEqual(len(set(expected_modules())), 50)
        self.assertNotIn("double_blocks.0.img_mlp.0", expected_modules())

    def test_install_keeps_original_forward_signature(self):
        model = nn.Sequential(copy.deepcopy(self.base))
        states = [(e.name, {"0.lora_A.weight": e.a, "0.lora_B.weight": e.b}, e.alpha, e.contract)
                  for e in self.experts]
        install_replay_bank(model, modules=["0"], expert_states=states, expert_index=2)
        self.assertTrue(torch.equal(model(self.x), self.bank.forward_expert(self.x, 2)))

    def test_failed_install_leaves_model_untouched(self):
        model = nn.Sequential(copy.deepcopy(self.base), nn.Linear(5, 3))
        original = model[0]
        state = {"0.lora_A.weight": torch.randn(3, 7), "0.lora_B.weight": torch.randn(5, 3),
                 "1.lora_A.weight": torch.randn(3, 99), "1.lora_B.weight": torch.randn(3, 3)}
        with self.assertRaises(ValueError):
            install_replay_bank(model, modules=["0", "1"], expert_states=[("test", state, 3, FLOW_CONTRACT)], expert_index=0)
        self.assertIs(model[0], original)
        self.assertTrue(model[0].weight.requires_grad)


if __name__ == "__main__":
    unittest.main()
