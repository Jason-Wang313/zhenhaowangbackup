#!/usr/bin/env python3
"""Two CPU ranks: full energy double-backward and accumulation; synthetic only."""
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from lowlight_restoration.novel_ppt_energy import NovelPPTEnergy, objective


def main():
    torch.set_num_threads(1)
    dist.init_process_group('gloo')
    try:
        rank, world = dist.get_rank(), dist.get_world_size()
        if world != 2:
            raise RuntimeError('this synthetic smoke requires two CPU ranks')
        torch.manual_seed(20261027 + rank)
        model = NovelPPTEnergy(width=4, condition_dim=8)
        wrapped = DDP(model, broadcast_buffers=False, gradient_as_bucket_view=False)
        optimizer = torch.optim.AdamW(wrapped.parameters(), lr=1e-3)
        weights = dict(mse=1., lpips=2., gradient_fit=1., global_rank=.1, local_rank=.1, no_harm_mse=1., no_harm_lpips=1.)
        perceptual = lambda a, b: (a-b).square().mean((1, 2, 3))
        from contextlib import nullcontext
        for step in range(2):
            optimizer.zero_grad()
            for micro in range(2):
                x = torch.rand(1, 3, 16, 16)*.5
                p, anchor, target = x+.02, x+.01, x+.007
                c = torch.tensor([[0., .25, 0., 0., 0.]])
                with wrapped.no_sync() if micro == 0 else nullcontext():
                    output = wrapped(x, p, anchor, c, (target, p, anchor, x))
                    loss, _, _ = objective(output, anchor, target, perceptual, residual_scale=.02, weights=weights)
                    (loss/2).backward()
            if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
                raise RuntimeError('invalid second-order DDP gradient')
            optimizer.step()
        weights = model.field.head.weight.detach().flatten()
        peers = [torch.empty_like(weights) for _ in range(world)]
        dist.all_gather(peers, weights)
        if not bool(weights.abs().max() > 0) or not torch.equal(peers[0], peers[1]):
            raise RuntimeError('ranks did not learn identical nonzero potential')
        if rank == 0:
            print('NOVEL_PPT_CPU_DDP_PASS ' + json.dumps({'ranks': 2, 'steps': 2,
                  'accumulation': 2, 'second_order_gradients': True, 'identical_trained_weights': True,
                  'dataset_pixels_accessed': False, 'GPU_used': False}), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
