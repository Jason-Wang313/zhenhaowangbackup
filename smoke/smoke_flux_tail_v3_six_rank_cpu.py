#!/usr/bin/env python3
"""Three torchrun agents x two CPU ranks: unequal-pair DDP and durable resume.

This validates real Gloo collectives on one CPU host, NOT CUDA/NCCL networking.
Synthetic data only; uses production pair assignment, loss and checkpoint code.
"""
from contextlib import nullcontext
import argparse
import copy
import json
import os
from pathlib import Path
import random
import sys
from datetime import timedelta
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from lowlight_restoration.flux_tail_learned_energy import TailQualityEnergy,quality_targets
from lowlight_restoration.flux_tail_relative_energy_v3 import relative_loss
from smoke import flux_tail_distributed_v3 as cluster
from smoke import flux_tail_relative_checkpoint_v3 as checkpoints
from smoke.flux_tail_checkpoint_v2 import cpu_tree,rng_state,restore_rng,atomic_json
from smoke.qualify_foundir_flux_tail_energy_v1 import parameter_hash
from smoke.run_foundir_flux_tail_relative_energy_v3 import spec


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(1)
    dist.init_process_group('gloo',timeout=timedelta(seconds=120))
    try:
        rank=dist.get_rank();world=dist.get_world_size();assert world==6
        assert int(os.environ['LOCAL_WORLD_SIZE'])==2
        groups=[None]*world;dist.all_gather_object(groups,int(os.environ['GROUP_RANK']))
        assert groups==[0,0,1,1,2,2]
        args.output.mkdir(parents=True,exist_ok=True)
        cluster.shared_storage_probe(args.output/'visibility')
        torch.manual_seed(89);random.seed(89+rank)
        model=TailQualityEnergy(width=4,condition_dim=8).double()
        reference=copy.deepcopy(model) if rank==0 else None
        wrapped=DDP(model,broadcast_buffers=False,gradient_as_bucket_view=False)
        opt=torch.optim.AdamW(model.parameters(),lr=1e-4)
        weights=spec()['loss'];receipt=dict(spec=spec(),world_size=world,mode='qualification',
            protocol_sha256='synthetic-six-rank',execution_sha256='same-six-rank-mapping')
        def pair(slot,noise):
            generator=torch.Generator().manual_seed(8000+slot)
            x=torch.rand(2,3,16,16,generator=generator,dtype=torch.float64)*.4
            p=x*.9;y=p+.01*torch.randn(p.shape,generator=generator,dtype=torch.float64)+noise*.001
            labels=quality_targets(y,x*.8,lambda a,b:(a-b).square().mean((1,2,3),keepdim=True))
            return (x,p,y,torch.zeros(2,5,dtype=torch.float64),torch.full((2,1),.2,dtype=torch.float64)),labels
        def update(step,compare=False):
            opt.zero_grad(set_to_none=True)
            noise=float(torch.rand(()))+random.random();noises=[None]*world
            dist.all_gather_object(noises,noise)
            slots=cluster.training_slots(step,rank,world);value=0.
            for micro,slot in enumerate(slots):
                inputs,labels=pair(slot,noise)
                with wrapped.no_sync() if micro<len(slots)-1 else nullcontext():
                    loss,_,_=relative_loss(wrapped(*inputs),labels,weights)
                    (loss*cluster.pair_loss_weight(world)).backward()
                value+=float(loss.detach())*cluster.pair_loss_weight(world)
            mean=torch.tensor(value,dtype=torch.float64);dist.all_reduce(mean);mean/=world
            if compare and rank==0:
                reference.zero_grad(set_to_none=True);wanted=0.
                for slot in range(32):
                    inputs,labels=pair(slot,noises[slot%world])
                    loss,_,_=relative_loss(reference(*inputs),labels,weights)
                    (loss/32).backward();wanted+=float(loss.detach())/32
                assert abs(float(mean)-wanted)<1e-10
                for a,b in zip(model.parameters(),reference.parameters(),strict=True):
                    torch.testing.assert_close(a.grad,b.grad,atol=1e-10,rtol=1e-9)
            dist.barrier();opt.step()
            return dict(step=step,mean=float(mean))
        row=update(1,compare=True)
        ranks=[None]*world
        dist.all_gather_object(ranks,dict(rng=rng_state('cpu'),parameter_sha256=parameter_hash(model),peak_gib=0.))
        if rank==0:checkpoints.commit(args.output,model,opt,1,receipt,ranks,[row],0.)
        dist.barrier();update(2);expected=cpu_tree(model.state_dict());expected_opt=cpu_tree(opt.state_dict())
        state,_=checkpoints.load(args.output,receipt)
        model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);restore_rng(state['rank_states'][rank]['rng'],'cpu')
        update(2)
        assert all(torch.equal(v,model.state_dict()[k]) for k,v in expected.items())
        for key,value in expected_opt['state'].items():
            for field,tensor in value.items():assert torch.equal(tensor,opt.state_dict()['state'][key][field])
        for bad in [receipt|dict(world_size=4),receipt|dict(execution_sha256='different-hardware-mapping')]:
            try:checkpoints.load(args.output,bad)
            except RuntimeError:pass
            else:raise AssertionError('incompatible resume was accepted')
        hashes=[None]*world;dist.all_gather_object(hashes,parameter_hash(model));assert len(set(hashes))==1
        if rank==0:
            result=dict(status='pass',world_size=6,torchrun_agents=3,local_ranks=2,
                        pairs_by_rank=[6,6,5,5,5,5],serial_mean_and_gradients_match=True,
                        durable_model_adam_rng_resume_exact=True,hardware_mapping_change_rejected=True,
                        backend='gloo',physical_hosts=1,cuda_qualified=False)
            atomic_json(args.output/'result.json',result)
            print('TAIL_RELATIVE_V3_SIX_RANK_CPU_PASS '+json.dumps(result),flush=True)
        dist.barrier()
    finally:dist.destroy_process_group()


if __name__=='__main__':main()
