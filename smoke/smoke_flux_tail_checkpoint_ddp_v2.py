#!/usr/bin/env python3
"""Two-rank, real second-order DDP: interrupt/resume must equal uninterrupted training."""
from pathlib import Path
import sys
import tempfile
from contextlib import nullcontext
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from lowlight_restoration.flux_tail_learned_energy import TailQualityEnergy,quality_targets,learning_loss
from smoke import flux_tail_checkpoint_v2 as saves
from smoke.qualify_foundir_flux_tail_energy_v1 import parameter_hash


def main():
    torch.set_num_threads(1);dist.init_process_group('gloo');rank=dist.get_rank();world=dist.get_world_size()
    assert world==2
    temporary=tempfile.TemporaryDirectory(prefix='tail-energy-ddp-resume-') if rank==0 else None
    paths=[temporary.name if temporary else None];dist.broadcast_object_list(paths,src=0);root=Path(paths[0])
    receipt=dict(spec=dict(learning_rate=1e-4),config_sha256='synthetic',source_sha256={'test':'only'})
    def build():
        torch.manual_seed(101)
        model=TailQualityEnergy(width=4,condition_dim=8)
        wrapped=DDP(model,broadcast_buffers=False,gradient_as_bucket_view=False)
        return model,wrapped,torch.optim.AdamW(model.parameters(),lr=1e-4)
    def step(wrapped,opt):
        opt.zero_grad(set_to_none=True)
        for micro in range(2):
            x=torch.rand(2,3,16,16)*.4;p=x*.8;y=p+.01*torch.randn_like(p)
            labels=quality_targets(y,x*.7,lambda a,b:(a-b).square().mean((1,2,3),keepdim=True))
            with wrapped.no_sync() if micro==0 else nullcontext():
                out=wrapped(x,p,y,torch.rand(2,5),torch.full((2,1),.05))
                loss,_=learning_loss(out,labels);(loss/2).backward()
        opt.step()
    try:
        model,wrapped,opt=build();step(wrapped,opt)
        parts=[None]*world;dist.all_gather_object(parts,saves.rng_state('cpu'))
        if rank==0:saves.commit_step(root,model,opt,step=1,world=world,rank_rng=parts,receipt=receipt,
            log=[dict(step=1)],initial_hash='initial',elapsed_seconds=1.,peak_memory_gib=[0.]*world)
        dist.barrier();step(wrapped,opt);reference=parameter_hash(model)
        del model,wrapped,opt
        model,wrapped,opt=build();state,_=saves.load_latest(root,receipt,world)
        model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer'])
        saves.restore_rng(state['rank_rng'][rank],'cpu');step(wrapped,opt)
        assert parameter_hash(model)==reference
        parts=[None]*world;dist.all_gather_object(parts,parameter_hash(model));assert len(set(parts))==1
        if rank==0:print('TAIL_ENERGY_V2_DDP_RESUME_PASS ranks=2 second_order=true exact_parameters=true adam_rng_restored=true')
        dist.barrier()
    finally:
        dist.destroy_process_group()
        if temporary:temporary.cleanup()


if __name__=='__main__':main()
