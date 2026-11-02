"""Two-process CPU DDP: saved LoRA/Adam/RNG reproduces the next update exactly."""
import argparse
from contextlib import nullcontext
import os
from pathlib import Path
import sys
import random
from unittest import mock
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from lowlight_restoration.lora import inject_lora
from smoke import rain_flow_formal_stage1_state_v1 as s


def new():
    torch.manual_seed(17)
    m=torch.nn.Sequential();m.add_module("fc",torch.nn.Linear(3,2))
    inject_lora(m,["fc"],2,2.,0.)
    wrapped=DDP(m)
    opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=1e-5,weight_decay=0.,fused=False)
    return m,wrapped,opt


def update(wrapped,opt):
    opt.zero_grad()
    for micro in range(4):
        with wrapped.no_sync() if micro<3 else nullcontext():
            (wrapped(torch.randn(4,3)).square().mean()/4).backward()
    opt.step();opt.zero_grad()


def probe_simulation(rank):
    from smoke import train_foundir_rain_flow_formal_stage1_v1 as trainer
    def fake(*args):
        random.random();torch.rand(10)
    with mock.patch.object(trainer,"probe_one",side_effect=fake):
        trainer.probe(None,None,None,None,None,None,rank,"cpu",50)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--output",type=Path,required=True);args=parser.parse_args()
    dist.init_process_group("gloo");rank=dist.get_rank();world=dist.get_world_size()
    if world!=2:raise ValueError("CPU smoke requires two workers")
    args.output.mkdir(exist_ok=True)
    model,wrapped,opt=new();torch.manual_seed(100+rank)
    for step in range(2):
        if step:update(wrapped,opt)
        rng=[None]*2;dist.all_gather_object(rng,s.rng_state("cpu"))
        if rank==0:s.save_step(args.output,model,opt,step=step,world=2,rank_rng=rng,receipt_hash="cpu_ddp_test",history=[{}]*step)
        dist.barrier()
    state,_=s.load_latest(args.output,"cpu_ddp_test",2)
    probe_simulation(rank)
    update(wrapped,opt);expected=s.trainable_state(model);expected_opt=s.cpu_tree(opt.state_dict())
    model2,wrapped2,opt2=new();s.restore(model2,opt2,state,rank,"cpu");probe_simulation(rank);update(wrapped2,opt2)
    actual=s.trainable_state(model2)
    assert all(torch.equal(expected[n],actual[n]) for n in expected)
    for index,entry in expected_opt["state"].items():
        for key,value in entry.items():assert torch.equal(value,opt2.state_dict()["state"][index][key])
    hashes=[None]*2;dist.all_gather_object(hashes,s.parameter_hash(actual));assert len(set(hashes))==1
    if rank==0:print("RAIN_FLOW_FORMAL_STAGE1_CPU_DDP_RESUME_PASS world=2 accumulation=4 next_update_bitwise_equal=true",flush=True)
    dist.destroy_process_group()


if __name__=="__main__":main()
