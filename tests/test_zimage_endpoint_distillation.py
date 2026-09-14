import json
from pathlib import Path

import torch
import pytest

from smoke.train_foundir_blur_only_zimage_endpoint_distill_ddp import load_settings, loss_terms, teacher_is_usable


def test_teacher_filter_requires_both_metrics():
    assert teacher_is_usable(20, .3, 21, .2)
    assert not teacher_is_usable(20, .3, 19, .2)
    assert not teacher_is_usable(20, .3, 21, .4)


def test_frozen_decoder_propagates_endpoint_gradient_and_rejected_teacher_has_no_effect():
    settings = dict(pixel_mse_weight=1., gt_latent_mse_weight=.1, lpips_weight=.05, teacher_latent_mse_weight=.01)
    z = torch.randn(1, 3, 8, 8, requires_grad=True)
    decoder = torch.nn.Conv2d(3, 3, 1).requires_grad_(False)
    target = torch.zeros_like(z)
    item = dict(gt_rgb=target, gt_latent=target, teacher_latent=torch.ones_like(z), teacher_usable=False)
    metric = lambda a,b: (a-b).square().mean()
    loss, terms = loss_terms(z, decoder(z), item, metric, settings)
    assert terms['teacher_latent_mse'] == 0
    loss.backward()
    assert torch.isfinite(z.grad).all() and z.grad.abs().sum() > 0
    assert all(p.grad is None for p in decoder.parameters())


def test_fp32_parameter_accumulates_small_adam_updates():
    results = []
    for dtype in (torch.bfloat16, torch.float32):
        p = torch.nn.Parameter(torch.tensor([.030029296875], dtype=dtype))
        optimizer = torch.optim.AdamW([p], lr=5e-5, weight_decay=0)
        initial = p.detach().clone()
        for _ in range(10):
            p.grad = torch.ones_like(p)
            optimizer.step()
        results.append(bool(torch.equal(initial, p)))
    assert results == [True, False]


def test_endpoint_config_protects_scope(tmp_path):
    original = Path('configs/foundir_blur_only_zimage_endpoint_distill_v1.json')
    load_settings(original)
    payload = json.loads(original.read_text())
    payload['sealed_internal_accessed'] = True
    path = tmp_path / 'invalid.json'
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='scope'):
        load_settings(path)
