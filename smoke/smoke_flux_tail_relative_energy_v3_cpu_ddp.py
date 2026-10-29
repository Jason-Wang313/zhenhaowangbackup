#!/usr/bin/env python3
"""Two-rank real second-order relative-loss and exact per-rank resume test."""
from contextlib import nullcontext
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from lowlight_restoration.flux_tail_learned_energy import TailQualityEnergy,quality_targets
from lowlight_restoration.flux_tail_relative_energy_v3 import relative_loss
from smoke.flux_tail_checkpoint_v2 import cpu_tree,rng_state,restore_rng


def main():
    torch.set_num_threads(1);dist.init_process_group('gloo')
    try:
        rank=dist.get_rank();assert dist.get_world_size()==2
        torch.manual_seed(67+rank)
        model=TailQualityEnergy(width=4,condition_dim=8)
        wrapped=DDP(model,broadcast_buffers=False,gradient_as_bucket_view=False)
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4)
        weights=dict(relative=1.,absolute=.1,gradient=1.,direction=1.,relative_beta=.05)
        def update():
            optimizer.zero_grad(set_to_none=True)
            for micro in range(2):
                x=torch.rand(2,3,16,16)*.4;p=x*.9;y=p+.01*torch.randn_like(p)
                labels=quality_targets(y,x*.8,lambda a,b:(a-b).square().mean((1,2,3),keepdim=True))
                with wrapped.no_sync() if micro==0 else nullcontext():
                    loss,_,_=relative_loss(wrapped(x,p,y,torch.zeros(2,5),torch.full((2,1),.2)),labels,weights)
                    (loss/2).backward()
            assert all(p.grad is not None and bool(p.grad.isfinite().all()) for p in model.parameters())
            optimizer.step()
        update();saved=cpu_tree(model.state_dict());opt_saved=cpu_tree(optimizer.state_dict());rng=rng_state('cpu')
        update();expected=cpu_tree(model.state_dict())
        model.load_state_dict(saved);optimizer.load_state_dict(opt_saved);restore_rng(rng,'cpu');update()
        assert all(torch.equal(v,model.state_dict()[k]) for k,v in expected.items())
        flat=torch.cat([p.detach().flatten() for p in model.parameters()]);parts=[torch.empty_like(flat) for _ in range(2)]
        dist.all_gather(parts,flat);assert torch.equal(parts[0],parts[1])
        if rank==0:print('TAIL_RELATIVE_V3_CPU_DDP_PASS second_order=true per_rank_resume_exact=true ranks=2')
    finally:dist.destroy_process_group()


if __name__=='__main__':main()
