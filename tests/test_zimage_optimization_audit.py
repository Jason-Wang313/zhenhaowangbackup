import torch
import pytest

from smoke.audit_foundir_blur_only_zimage_optimization import assignment, objective, gradient_summary


def test_eight_independent_probes_are_balanced():
    s = dict(arms=['latent_rgb','latent_mixed','lora_rgb','lora_mixed'], record_ids=['hard','near_clean'])
    pairs = [assignment(i,s) for i in range(8)]
    assert len(set(pairs))==8
    assert all(pairs.count((a,r))==1 for a in s['arms'] for r in s['record_ids'])
    with pytest.raises(ValueError):
        assignment(8,s)


def test_rgb_ablation_really_removes_auxiliary_gradients():
    z = torch.tensor([1.,2.],requires_grad=True)
    terms = dict(pixel_mse=z[0]**2,gt_latent_mse=z[1]**2,lpips=-z[0],teacher_latent_mse=z.sum()*0)
    e = dict(pixel_mse_weight=1.,gt_latent_mse_weight=.1,lpips_weight=.05,teacher_latent_mse_weight=.01)
    rgb_grad = torch.autograd.grad(objective(terms,'lora_rgb',e),z,retain_graph=True)[0]
    mixed_grad = torch.autograd.grad(objective(terms,'lora_mixed',e),z,retain_graph=True)[0]
    assert torch.allclose(rgb_grad,torch.tensor([2.,0.]))
    assert torch.allclose(mixed_grad,torch.tensor([1.95,.4]))
    summary = gradient_summary(z,terms,e)
    assert summary['latent_gradient_norms']['pixel_mse']==2
    assert summary['weighted_latent_gradient_norms']['gt_latent_mse']==pytest.approx(.4)
    assert summary['latent_gradient_cosines']['lpips__pixel_mse']==pytest.approx(-1)
    assert summary['latent_gradient_cosines']['gt_latent_mse__pixel_mse']==pytest.approx(0)


def test_direct_latent_can_learn_through_frozen_decoder():
    torch.manual_seed(0)
    decoder = torch.nn.Conv2d(3,3,1).requires_grad_(False)
    z = torch.nn.Parameter(torch.randn(1,3,8,8))
    target = decoder(torch.zeros_like(z)).detach()
    opt = torch.optim.AdamW([z],lr=.05,weight_decay=0)
    initial = float((decoder(z)-target).square().mean().detach())
    for _ in range(30):
        opt.zero_grad(set_to_none=True)
        loss=(decoder(z)-target).square().mean()
        loss.backward()
        opt.step()
    assert float(loss.detach())<initial*.3
    assert all(p.grad is None for p in decoder.parameters())
