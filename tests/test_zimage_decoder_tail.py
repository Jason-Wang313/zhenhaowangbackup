import copy
import torch
from safetensors.torch import save_file

from smoke.train_foundir_blur_only_zimage_decoder_tail_ddp import (
    DecoderTail, restore_student_fp32, acceptance, decoder_loss,
)


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 1)

    def forward(self, x, *args):
        return self.conv(x), None, None, None


def test_tail_only_updates_and_frozen_prefix_has_no_grad():
    d = torch.nn.Module()
    d.scaling_factor, d.shift_factor = .36, .11
    d.conv_in = torch.nn.Conv2d(3, 3, 1)
    d.blocks = torch.nn.ModuleList([Block() for _ in range(18)])
    d.conv_norm_out = torch.nn.GroupNorm(1, 3)
    d.conv_act = torch.nn.SiLU()
    d.conv_out = torch.nn.Conv2d(3, 3, 1)
    d.bfloat16()
    tail = DecoderTail(d)
    frozen = {n:p.detach().clone() for n,p in tail.named_parameters() if not p.requires_grad}
    params = [p for p in tail.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-3)
    before = d.conv_out.weight.detach().clone()
    out, anchor = tail(torch.randn(1,3,8,8), torch.randn(1,3,8,8))
    loss = decoder_loss(out, anchor, torch.zeros_like(out), lambda a,b:(a-b).square().mean(), dict(lpips_weight=.05, gt_anchor_weight=.1))
    loss.backward()
    opt.step()
    assert not torch.equal(before, d.conv_out.weight)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.dtype == torch.float32 for p in params)
    for n,p in tail.named_parameters():
        if n in frozen:
            assert p.grad is None and torch.equal(p, frozen[n])


def test_student_restore_preserves_sub_bf16_update(tmp_path):
    class Student(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.pipe = torch.nn.Module()
            self.pipe.dit = torch.nn.Linear(1,1, bias=False).bfloat16()
        def promote_trainable(self):
            self.float()
    m = Student()
    value = torch.tensor([[.030021234]], dtype=torch.float32)
    path = tmp_path/'student.safetensors'
    save_file({'weight':value}, path)
    restore_student_fp32(m, path)
    assert torch.equal(m.pipe.dit.weight, value)
    assert not m.pipe.dit.weight.requires_grad


def test_train_only_success_does_not_pass_decoder_gate():
    metrics = dict(psnr_gain_db=1., lpips_improvement=.1, output_psnr=26., output_lpips=.2)
    snapshots = [dict(step=400, role='tiny_overfit_train', metrics=copy.copy(metrics)),
                 dict(step=400, role='visual_development', metrics=copy.copy(metrics), gt_roundtrip_psnr=40., gt_roundtrip_lpips=.01),
                 dict(step=0, role='visual_development', metrics=dict(output_psnr=25., output_lpips=.3))]
    assert acceptance(snapshots)
    snapshots[1]['metrics']['psnr_gain_db'] = -1.
    assert not acceptance(snapshots)
    snapshots[1]['metrics']['psnr_gain_db'] = 1.
    snapshots[1]['gt_roundtrip_psnr'] = 30.
    assert not acceptance(snapshots)
