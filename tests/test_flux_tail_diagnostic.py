import copy
import json
from pathlib import Path
import sys

import unittest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'third_party/flux2/src'))
from flux2.sampling import denoise_cfg
from lowlight_restoration.flux_tail_diagnostic import Tail, preview, finite_difference, rms_direction, continuous_blend, checkpoint_blocks
from smoke.diagnose_foundir_flux_tail_oracle_v1 import validate_spec, aggregate, assigned_indices


class Model(torch.nn.Module):
    def forward(self, *, x, x_ids, timesteps, ctx, ctx_ids, guidance):
        return .2 * x + ctx.mean((1, 2))[:, None, None] + timesteps[:, None, None]


def inputs():
    torch.manual_seed(123)
    state = torch.randn(1, 4, 4)
    ids = torch.zeros(1, 4, 4, dtype=torch.long)
    ref = torch.randn(1, 2, 4)
    ref_ids = torch.ones(1, 2, 4, dtype=torch.long)
    context = torch.tensor([[[.1]], [[.5]]])
    context_ids = torch.zeros(2, 1, 4, dtype=torch.long)
    schedule = [1., .8, .33, .1, 0.]
    tail = Tail(Model(), None, ids, ref, ref_ids, context, context_ids, schedule)
    tail.bf16 = False
    return state, tail


def test_zero_guidance_matches_official_and_split_restart():
    state, tail = inputs()
    with torch.no_grad():
        expected = denoise_cfg(tail.model, state, tail.ids, tail.context, tail.context_ids,
                              tail.schedule, 1., tail.reference, tail.reference_ids)
        actual, cached = tail.rollout(state, 0, capture=True)
        assert torch.equal(expected, actual)
        for index, value in cached.items():
            assert torch.equal(tail.rollout(value, index), expected)
            assert not value.is_inference()


def test_preview_sign_and_jacobian():
    state = torch.randn(1, 4, 4, dtype=torch.float64, requires_grad=True)
    t = .3
    value = preview(state, state.square(), t)
    gradient = torch.autograd.grad(value.sum(), state)[0]
    assert torch.allclose(gradient, 1 - 2*t*state)
    skipped = preview(state, state.square().detach(), t)
    assert torch.equal(torch.autograd.grad(skipped.sum(), state)[0], torch.ones_like(state))


def test_real_rollout_gradient_is_not_preview_gradient():
    state, tail = inputs()
    state.requires_grad_(True)
    expected_factor = 1.
    for current, following in zip(tail.schedule, tail.schedule[1:]):
        expected_factor *= 1 + (following-current)*.2
    full = torch.autograd.grad(tail.rollout(state, 0).sum(), state)[0]
    assert torch.allclose(full, torch.full_like(full, expected_factor))
    approx = torch.autograd.grad(preview(state, tail.velocity(state, 1.), 1.).sum(), state)[0]
    assert not torch.allclose(full, approx)
    assert tail.calls['flux_forward'] > len(tail.schedule)-1  # recomputations counted


def test_finite_difference_and_descent():
    state = torch.randn(1, 4, 4, requires_grad=True)
    fn = lambda z: ((z-.1)**2).mean()
    gradient = torch.autograd.grad(fn(state), state)[0]
    checks = finite_difference(fn, state.detach(), gradient, [.001, .003, .01])
    assert all(r['positive_direction'] and r['relative_error'] < .01 for r in checks)
    direction, rms = rms_direction(gradient)
    assert rms > 0
    assert fn(state - .01*direction) < fn(state)
    assert torch.allclose(direction.square().mean().sqrt(), torch.tensor(1.))
    for bad in (torch.zeros(2), torch.tensor([float('nan')])):
        with unittest.TestCase().assertRaises(RuntimeError):
            rms_direction(bad)


def test_blend_keeps_original_alpha_and_gradient():
    raw = torch.full((1, 3, 8, 8), .2, requires_grad=True)
    image = continuous_blend(torch.zeros_like(raw), raw)
    assert torch.allclose(image, torch.full_like(image, .06))
    assert torch.allclose(torch.autograd.grad(image.sum(), raw)[0], torch.full_like(raw, .3))


def test_actual_small_flux_blocks_input_gradient_and_checkpoint_equivalence():
    from flux2.model import Flux2, Flux2Params
    torch.manual_seed(19)
    model = Flux2(Flux2Params(in_channels=4, context_in_dim=3, hidden_size=32, num_heads=4,
        depth=1, depth_single_blocks=1, axes_dim=[2, 2, 2, 2], mlp_ratio=2., use_guidance_embed=False))
    model.eval().requires_grad_(False)
    state, tail = inputs()
    tail.model = model
    tail.context = torch.randn(2, 2, 3)
    tail.context_ids = torch.zeros(2, 2, 4, dtype=torch.long)
    state.requires_grad_(True)
    before = tail.rollout(state, 2)
    expected_gradient = torch.autograd.grad(before.square().mean(), state)[0]
    checkpoint_blocks(model, tail.calls)
    after = tail.rollout(state, 2)
    gradient = torch.autograd.grad(after.square().mean(), state)[0]
    assert torch.equal(before, after)
    assert torch.equal(expected_gradient, gradient)
    assert gradient.isfinite().all() and gradient.abs().sum() > 0
    assert all(p.grad is None and not p.requires_grad for p in model.parameters())
    assert tail.calls['flux_block_forward'] > 4
    with unittest.TestCase().assertRaises(RuntimeError):
        checkpoint_blocks(model, tail.calls)


def test_frozen_scope():
    spec = json.loads((ROOT/'configs/foundir_flux_tail_oracle_v1.json').read_text())
    validate_spec(spec)
    for field, value in [('deployable', True), ('auto_train', True), ('steps', 4),
                         ('cache_indices', list(range(1, 9))), ('preview_start_indices', [49]),
                         ('rms_latent_steps', [.2]), ('finite_difference_required_passes', 0)]:
        with unittest.TestCase().assertRaises(RuntimeError):
            validate_spec(spec | {field: value})


def test_four_eight_gpu_partition_preserves_all_eight_samples():
    for world_size in (4, 8):
        shards = [assigned_indices(rank, world_size) for rank in range(world_size)]
        assert sorted(sum(shards, [])) == list(range(8))
        assert all(len(shard) == 8//world_size for shard in shards)
    assert assigned_indices(0, 4) == [0, 4]
    assert assigned_indices(3, 4) == [3, 7]
    for rank, world in [(0, 2), (4, 4), (-1, 8)]:
        with unittest.TestCase().assertRaises(ValueError):
            assigned_indices(rank, world)
    spec = json.loads((ROOT/'configs/foundir_flux_tail_oracle_v1.json').read_text())
    for count in (4, 8):
        assert count * spec['budget_wall_hours_cap_by_gpus'][str(count)] == 12
    for field, value in [('allowed_world_sizes', [8]), ('budget_wall_hours_cap_by_gpus', {'4': 1.5, '8': 1.5})]:
        with unittest.TestCase().assertRaises(RuntimeError):
            validate_spec(spec | {field: value})


def test_four_eight_gpu_aggregation_is_identical():
    rows = [dict(record_id=str(i), source_group=str(i), baseline_parity_pass=True,
            finite_difference_pass=True, arms={'historical_bf16': dict(
            metrics=dict(psnr=40.+i/10, lpips=.05+i/1000), control='historical_bf16')}) for i in range(8)]
    spec = json.loads((ROOT/'configs/foundir_flux_tail_oracle_v1.json').read_text())
    results = []
    for world_size in (4, 8):
        parts = [[rows[i] for i in assigned_indices(rank, world_size)] for rank in range(world_size)]
        restored = sorted(sum(parts, []), key=lambda row: int(row['record_id']))
        results.append(aggregate(restored, spec))
    assert results[0] == results[1]


def test_aggregate_keeps_failures_and_precision_control_separate():
    spec = json.loads((ROOT/'configs/foundir_flux_tail_oracle_v1.json').read_text())
    rows = []
    for i in range(8):
        arms = {'historical_bf16': dict(metrics=dict(psnr=40., lpips=.1), control='historical_bf16'),
                'fp32': dict(metrics=dict(psnr=40.2, lpips=.09), control='historical_bf16'),
                'guided': dict(metrics=dict(psnr=40.31, lpips=.089), control='fp32'),
                'bad': dict(metrics=dict(psnr=39., lpips=.2), control='fp32')}
        rows.append(dict(record_id=str(i), source_group=str(i), arms=arms,
                         baseline_parity_pass=True, finite_difference_pass=True))
    result = aggregate(rows, spec)
    assert result['means']['guided']['joint_target_vs_historical']
    assert abs(result['means']['guided']['psnr_gain_vs_precision_control'] - .11) < 1e-10
    assert not result['means']['bad']['joint_target_vs_historical']
    assert not result['deployable'] and not result['scientific_gate_evaluated']
    bad = copy.deepcopy(rows)
    del bad[2]['arms']['bad']
    with unittest.TestCase().assertRaises(RuntimeError):
        aggregate(bad, spec)


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(value) for name, value in globals().items()
                              if name.startswith('test_') and callable(value))


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
