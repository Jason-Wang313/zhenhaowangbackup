import copy
import json
from pathlib import Path

import tempfile
import unittest
from functools import wraps
import torch
from torch import nn

from lowlight_restoration.joint_vlm_moe import JointVLMExpertFFN
from lowlight_restoration.token_residual_moe import attach_token_router, token_conditioning, batch_slot
from smoke.token_residual_moe_v1 import Samples, read_feature, save_feature, write_once

raises = unittest.TestCase().assertRaises


def temporary_directory(test):
    @wraps(test)
    def run():
        with tempfile.TemporaryDirectory(prefix='token-router-test-') as directory:
            test(Path(directory))
    return run


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = JointVLMExpertFFN(nn.Linear(6, 6), 6, hidden=8, arm='image_only')
        with torch.no_grad():
            self.block.experts[1].weight.add_(.2)

    def forward(self, x):
        return self.block(x)


def model():
    torch.manual_seed(8)
    net = TinyModel()
    attach_token_router(net, feature_dim=4, hidden=8)
    return net


def test_zero_insertion_preserves_trained_experts_router_and_output():
    net = TinyModel().eval()
    x = torch.randn(2, 3, 6)
    ref = net(x)
    experts, router = net.block.experts, net.block.router
    attach_token_router(net, feature_dim=4, hidden=8)
    assert net.block.experts is experts and net.block.router is router
    with token_conditioning(net, torch.randn(2, 4)):
        assert torch.equal(net(x), ref)
        assert net.block.last_adjustment.count_nonzero() == 0


def test_additive_formula_and_gradients():
    net = model()
    x, features = torch.randn(2, 3, 6), torch.randn(2, 4)
    nn.init.normal_(net.block.vlm_projector[-1].weight)
    with token_conditioning(net, features):
        output = net(x)
        expected = (net.block.router(x.mean(1)) + net.block.vlm_projector(features)).softmax(-1)
        assert torch.equal(expected, net.block.last_weights)
        output.square().mean().backward()
        assert net.block.vlm_projector[-1].weight.grad.norm() > 0
        assert net.block.vlm_projector[1].weight.grad.norm() > 0
    assert net.block.condition is None and net.block.last_weights is None


def test_reject_missing_severity_nonfinite_and_trainable_features():
    net = model()
    for bad in [None, torch.ones(2, 5), torch.full((2, 4), float('nan')),
                torch.ones(2, 4, requires_grad=True)]:
        with raises(ValueError), token_conditioning(net, bad):
            net(torch.randn(2, 3, 6))


def test_forbid_oracle_routing():
    net = model()
    net.block.labels = torch.tensor([1])
    with raises(RuntimeError), token_conditioning(net, torch.randn(1, 4)):
        net(torch.randn(1, 3, 6))


def test_disabled_vlm_exactly_recovers_image_route():
    net = model()
    nn.init.normal_(net.block.vlm_projector[-1].weight)
    x = torch.randn(2, 3, 6)
    with token_conditioning(net, None, enabled=False):
        out = net(x)
        w = net.block.router(x.mean(1)).softmax(-1)
        first, second = [e(x) for e in net.block.experts]
        assert torch.equal(out, first + w[:, 1, None, None] * (second-first))


def test_four_by_two_preserves_batch32_and_task_balance():
    slots = [batch_slot(7, m, r) for m in range(4) for r in range(8)]
    assert sorted(i for i, _ in slots) == list(range(224, 256))
    assert sum(scale for _, scale in slots) / 8 == 1
    assert sum((i//8) % 2 == 0 for i, _ in slots) == 16
    with raises(ValueError):
        batch_slot(0, 4, 0)


def test_schedule_is_exact_old_global_microbatch_sequence():
    cfg = {'steps': 25, 'seed': 123, 'sampling_start': 352000}
    row = {'record_id': 'allowed', 'role': 'direct_endpoint_train', 'split': 'train', 'source_group': 's'}
    samples = Samples('.', cfg, {'rain': [row], 'blur': [row]}, {'rain': [], 'blur_preview': []})
    for micro in range(4):
        for rank in range(8):
            spec = samples.train_spec(micro*8+rank)
            assert spec['task_id'] == micro % 2
            assert spec['position'] == ((11000*4+micro)//2)*8+rank
    with raises(ValueError):
        samples.train_spec(800)


@temporary_directory
def test_token_cache_rejects_bad_identity_and_shape(tmp_path):
    path = tmp_path / 'tokens.safetensors'
    binding = {'sample': 'a'}
    value = torch.randn(3584)
    save_feature(path, value, binding)
    assert torch.equal(read_feature(path, binding), value)
    with raises(RuntimeError):
        read_feature(path, {'sample': 'b'})
    with raises(RuntimeError):
        save_feature(path, value + 1, binding)


@temporary_directory
def test_publish_preserves_existing_results(tmp_path):
    path = tmp_path / 'result.json'
    write_once(path, {'v': 1})
    write_once(path, {'v': 1})
    with raises(RuntimeError):
        write_once(path, {'v': 2})
    assert json.loads(path.read_text()) == {'v': 1}


def test_block_checkpoint_backward_retains_condition():
    from torch.utils.checkpoint import checkpoint
    net = model()
    x = torch.randn(2, 3, 6, requires_grad=True)
    with token_conditioning(net, torch.randn(2, 4)):
        checkpoint(net, x, use_reentrant=False).square().mean().backward()
    assert net.block.vlm_projector[-1].weight.grad.norm() > 0


def test_eight_rank_weighting_equals_serial_batch_gradient():
    # Algebraic CPU check; actual multi-process Gloo test lives in the smoke script.
    net = model()
    x, f = torch.randn(32, 3, 6), torch.randn(32, 4)
    with token_conditioning(net, f):
        net(x).square().mean().backward()
    expected = [p.grad.clone() for p in net.parameters()]
    net.zero_grad()
    for r in range(8):
        for m in range(4):
            slot, scale = batch_slot(0, m, r)
            with token_conditioning(net, f[slot:slot+1]):
                (net(x[slot:slot+1]).square().mean()*scale/8).backward()
    for p, ref in zip(net.parameters(), expected):
        torch.testing.assert_close(p.grad, ref, atol=1e-6, rtol=1e-5)


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(test) for name, test in globals().items()
                              if name.startswith('test_') and callable(test))


if __name__ == '__main__':
    unittest.main()
