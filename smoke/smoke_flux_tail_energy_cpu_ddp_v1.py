#!/usr/bin/env python3
"""Synthetic two-rank test: score/derivative supervision and DDP accumulation."""
from pathlib import Path
from contextlib import nullcontext
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from lowlight_restoration.flux_tail_learned_energy import TailQualityEnergy,quality_targets,learning_loss


def main():
    torch.set_num_threads(1);dist.init_process_group('gloo')
    try:
        assert dist.get_world_size()==2
        torch.manual_seed(81+dist.get_rank())
        model=TailQualityEnergy(width=4,condition_dim=8)
        wrapped=DDP(model,broadcast_buffers=False,gradient_as_bucket_view=False)
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4)
        def perceptual(a,b):return (a-b).square().mean((1,2,3),keepdim=True)
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            for micro in range(2):
                x=torch.rand(2,3,16,16)*.4;proposal=x*.8;candidate=proposal+.01*torch.randn_like(proposal)
                labels=quality_targets(candidate,x*.7,perceptual)
                with wrapped.no_sync() if micro==0 else nullcontext():
                    out=wrapped(x,proposal,candidate,torch.rand(2,5),torch.full((2,1),.05))
                    loss,_=learning_loss(out,labels);(loss/2).backward()
            assert all(p.grad is not None and bool(p.grad.isfinite().all()) for p in model.parameters())
            optimizer.step()
        flat=torch.cat([p.detach().flatten() for p in model.parameters()]);parts=[torch.empty_like(flat) for _ in range(2)]
        dist.all_gather(parts,flat);assert torch.equal(parts[0],parts[1])
        if dist.get_rank()==0:print('TAIL_ENERGY_CPU_DDP_PASS ranks=2 accumulation=2 equal_parameters=true real_pixels_accessed=false')
    finally:dist.destroy_process_group()


if __name__=='__main__':main()
