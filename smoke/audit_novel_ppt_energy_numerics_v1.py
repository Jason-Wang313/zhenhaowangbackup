#!/usr/bin/env python3
"""CPU-only numerical diagnosis on a historical energy, not an image eval.

Uses a fixed synthetic repeating pattern and the frozen step-5118 checkpoint.
No dataset pixels, optimizer, inference-quality metrics or parameter updates.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.nn import functional as F

from lowlight_restoration.proposal_relative_energy_v2 import SpatialProposalRelativeEnergy
from lowlight_restoration.restoration_energy_numerics import refinement_objective

CHECKPOINT = 'checkpoints/foundir_proposal_relative_energy_noise_v2/noise_energy_v2_formal1/step_005118.pt'
EXPECTED_SHA = 'b9444ff2c3d4dedfbad2022c4d3dbb74b193fd6e64d2dcf52cb33275fa438bd1'


def file_sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def spatial_target_gradient_probe(root):
    # Execute only the actual local ranking function; avoid importing its CUDA
    # trainer and any pretrained metric/data-loading dependencies.
    path = root / 'smoke/train_foundir_proposal_relative_energy_noise_v2_ddp.py'
    parsed = ast.parse(path.read_text())
    fn = next(node for node in parsed.body if isinstance(node, ast.FunctionDef) and node.name == 'spatial_ranking')
    namespace = {'torch': torch, 'Tensor': torch.Tensor, 'F': F}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), 'exec'), namespace)
    target = torch.zeros(1, 3, 16, 16)
    candidate = torch.full_like(target, .1, requires_grad=True)
    worse = torch.full_like(target, .2)
    maps = torch.zeros(1, 3, 1, 16, 16, requires_grad=True)
    loss = namespace['spatial_ranking'](maps, (target, candidate, worse), target, kernel=8, margin=.02)
    gradient = torch.autograd.grad(loss, candidate)[0]
    return {'candidate_receives_gradient_via_quality_label_even_with_independent_scores': bool(gradient.abs().max() > 0),
            'candidate_gradient_abs_max': gradient.abs().max().item(),
            'legacy_trainer_sha256': file_sha(path),
            'meaning': 'target_side_gradient_exists; contribution_to_scientific_failure_not_measured'}


def audit(root):
    torch.set_num_threads(2)
    torch.manual_seed(20261027)
    path = root / CHECKPOINT
    actual = file_sha(path)
    if actual != EXPECTED_SHA:
        raise ValueError('historical checkpoint identity changed')
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload['step'] != 5118:
        raise ValueError('wrong historical step')
    model = SpatialProposalRelativeEnergy().eval().requires_grad_(False)
    prefix = 'energy_model.'
    model.load_state_dict({k[len(prefix):]: v for k, v in payload['model'].items() if k.startswith(prefix)}, strict=True)
    del payload
    axis = torch.arange(64, dtype=torch.float32)
    xx, yy = torch.meshgrid(axis, axis, indexing='ij')
    tile = torch.stack((.2 * torch.sin(xx / 64 * 2 * torch.pi),
                        .2 * torch.cos(yy / 64 * 2 * torch.pi),
                        .1 * torch.sin((xx + yy) / 64 * 2 * torch.pi)))[None]
    descriptor = torch.zeros(1, 5)  # a valid all-level-1 diagnostic condition
    condition = model.condition(descriptor)
    schedule = model.step_sizes(condition).detach()
    results = []
    for size in (128, 192, 256):
        observation = tile.repeat(1, 1, size // 64, size // 64)
        candidate = (observation + .02).detach().requires_grad_(True)
        density = model.score_map_from_condition(observation, candidate, condition)
        legacy = torch.autograd.grad(density.flatten(1).mean(1).sum(), candidate, retain_graph=True)[0]
        corrected = torch.autograd.grad(refinement_objective(density, reference_pixels=128**2).sum(), candidate)[0]
        step = schedule[0, 0].item()
        results.append({'size': size, 'legacy_gradient_rms': legacy.square().mean().sqrt().item(),
                        'legacy_proposed_step_abs_mean': (step * legacy).abs().mean().item(),
                        'reference_area_gradient_rms': corrected.square().mean().sqrt().item(),
                        'reference_area_proposed_step_abs_mean': (step * corrected).abs().mean().item(),
                        'explicit_area_multiplier': size * size / 128**2,
                        'mean_energy': density.mean().item()})
    return {'status': 'cpu_numerical_diagnosis_only', 'checkpoint': CHECKPOINT, 'checkpoint_sha256': actual,
            'synthetic_pattern_only': True, 'dataset_pixels_accessed': False,
            'optimizer_or_weight_updates': False, 'new_restoration_quality_claim': False,
            'step_sizes_all_absent_condition': schedule.tolist(),
            'resolution_probes': results, 'ranking_target_graph': spatial_target_gradient_probe(root),
            'caveat': 'Area correction removes explicit scaling, not learned directional error; no PSNR/LPIPS claim.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = audit(args.root)
    content = json.dumps(result, indent=2, allow_nan=False) + '\n'
    if args.output:
        with args.output.open('x') as stream:
            stream.write(content)
        print(f'CPU_ENERGY_NUMERICAL_AUDIT_WRITTEN {args.output}')
    else:
        print(content, end='')


if __name__ == '__main__':
    main()
