#!/usr/bin/env python3
"""Synthetic CPU-only two-rank second-order DDP test for the FLUX energy path."""
import sys
from pathlib import Path
from contextlib import nullcontext
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from lowlight_restoration.novel_ppt_energy import objective
from lowlight_restoration.novel_ppt_energy_nonzero_init import NonzeroInitNovelPPTEnergy
from smoke.qualify_foundir_flux_energy_v1 import training_candidates


def main():
    torch.set_num_threads(1)
    dist.init_process_group('gloo')
    try:
        assert dist.get_world_size() == 2
        torch.manual_seed(9 + dist.get_rank())
        model = NonzeroInitNovelPPTEnergy(width=4, condition_dim=8)
        wrapped = DDP(model, broadcast_buffers=False, gradient_as_bucket_view=False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        weights = dict(mse=1., lpips=2., gradient_fit=1., global_rank=.1, local_rank=.1, no_harm_mse=1., no_harm_lpips=1.)
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            for micro in range(2):
                x = torch.rand(1, 3, 16, 16)*.5
                p, target = x+.02, x+.007
                images = dict(observation=x, proposal=p, target=target, wrong_expert=x+.1)
                with wrapped.no_sync() if micro == 0 else nullcontext():
                    output = wrapped(x, p, p, torch.zeros(1, 5), training_candidates(images))
                    loss, _, _ = objective(output, p, target, lambda a,b: (a-b).square().mean((1,2,3)),
                                          residual_scale=.02, weights=weights)
                    (loss/2).backward()
            assert all(p.grad is not None and bool(p.grad.isfinite().all()) for p in model.parameters())
            optimizer.step()
        flat = torch.cat([p.detach().flatten() for p in model.parameters()])
        peers = [torch.empty_like(flat) for _ in range(2)]
        dist.all_gather(peers, flat)
        assert torch.equal(peers[0], peers[1]) and model.field.head.weight.detach().abs().max() > 0
        if dist.get_rank() == 0:
            print('FLUX_ENERGY_CPU_DDP_PASS ranks=2 steps=2 accumulation=2 all_parameters_equal=true real_pixels_accessed=false')
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
