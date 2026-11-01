#!/usr/bin/env python3
"""Eight-process CPU/Gloo batch40 relative-loss gradients and exact resume."""
import copy
from contextlib import nullcontext
from pathlib import Path
import random
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'tests'))

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from test_token_residual_moe_v1 import model,token_conditioning
from smoke.token_moe_retention_v1 import batch_slot,retention_objective,TERMS
from smoke.token_residual_moe_v1 import write_once
from lowlight_restoration.token_moe_retention_state import save_state,resume,rng_state


def objective(output,index):
    # Different task/sample scales exercise relative-loss weighting, not just
    # DDP on a single shared scalar. All six terms retain a model gradient.
    terms={k:output.square().mean()*(i+1) for i,k in enumerate(TERMS)}
    reference={k:float((i+1)*(.2+index*.01)) for i,k in enumerate(TERMS)}
    return retention_objective(terms,reference,{k:1. for k in TERMS},
        dict(weighted_reference_floor=1e-4,rgb_mse_floor=4e-6,lpips_floor=1e-3,regression_weight=5.))[0]


def step(net,wrapped,opt,x,f,rank):
    opt.zero_grad()
    for micro in range(5):
        slot,scale=batch_slot(0,micro,rank)
        with wrapped.no_sync() if micro<4 else nullcontext():
            with token_conditioning(net,f[slot:slot+1]):
                (objective(wrapped(x[slot:slot+1]),slot)*scale).backward()
    gradients=[p.grad.clone() for p in net.parameters()]
    opt.step()
    return gradients


def worker(rank,directory):
    torch.set_num_threads(1)
    path=Path(directory)
    dist.init_process_group('gloo',init_method=(path/'rendezvous').as_uri(),rank=rank,world_size=8)
    try:
        net=model(); wrapped=DDP(net)
        opt=torch.optim.AdamW(net.parameters(),lr=1e-3,weight_decay=0)
        g=torch.Generator().manual_seed(213)
        x,f=torch.randn(40,3,6,generator=g),torch.randn(40,4,generator=g)
        first_grad=step(net,wrapped,opt,x,f,rank)
        first_weights=[p.detach().clone() for p in net.parameters()]
        first_states=copy.deepcopy(list(opt.state.values()))
        states=[None]*8; dist.all_gather_object(states,rng_state())
        binding={'mode':'qualification','test':'batch40','global_batch':40}
        if rank==0:
            write_once(path/'launch.json',{'binding':binding})
            save_state(net,opt,path,step=1,binding=binding,rank_rng=states)
        dist.barrier()
        draw=(random.random(),np.random.rand(),torch.rand(9))
        step(net,wrapped,opt,x,f,rank)
        new=model(); new_opt=torch.optim.AdamW(new.parameters(),lr=1e-3,weight_decay=0)
        assert resume(new,new_opt,path/'step_000001.resume.json',binding,rank,25)==1
        for value,expected in zip(new.parameters(),first_weights): torch.testing.assert_close(value,expected,rtol=0,atol=0)
        for actual,expected in zip(new_opt.state.values(),first_states):
            for key in ('step','exp_avg','exp_avg_sq'): torch.testing.assert_close(actual[key],expected[key],rtol=0,atol=0)
        assert random.random()==draw[0] and np.random.rand()==draw[1]
        torch.testing.assert_close(torch.rand(9),draw[2],rtol=0,atol=0)
        new_wrapped=DDP(new); step(new,new_wrapped,new_opt,x,f,rank)
        for a,b in zip(net.parameters(),new.parameters()): torch.testing.assert_close(a,b,rtol=1e-6,atol=1e-7)
        for a,b in zip(opt.state.values(),new_opt.state.values()):
            for key in ('step','exp_avg','exp_avg_sq'): torch.testing.assert_close(a[key],b[key],rtol=1e-6,atol=1e-7)
        if rank==0:
            serial=model(); serial_opt=torch.optim.AdamW(serial.parameters(),lr=1e-3,weight_decay=0)
            for iteration in range(2):
                serial_opt.zero_grad()
                for slot in range(40):
                    with token_conditioning(serial,f[slot:slot+1]):
                        (objective(serial(x[slot:slot+1]),slot)/40).backward()
                if iteration==0:
                    for a,p in zip(first_grad,serial.parameters()): torch.testing.assert_close(a,p.grad,rtol=1e-5,atol=1e-6)
                serial_opt.step()
            for a,b in zip(net.parameters(),serial.parameters()): torch.testing.assert_close(a,b,rtol=2e-5,atol=2e-6)
            difference=max(float((a-b).abs().max()) for a,b in zip(net.parameters(),new.parameters()))
            print(f'RETENTION_CPU_8RANK_BATCH40_GRADIENT_ADAM_RNG_RESUME_PASS max_abs={difference}',flush=True)
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__=='__main__':
    with tempfile.TemporaryDirectory(prefix='token-retention-gloo-') as directory:
        mp.spawn(worker,args=(directory,),nprocs=8,join=True)
