"""CPU DDP: old step200 state -> new bound receipt -> steps201/202 equal uninterrupted."""
import argparse
from contextlib import nullcontext
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from lowlight_restoration.lora import inject_lora
from smoke import rain_native50_continue_state_v1 as s
from smoke import rain_flow_formal_stage1_state_v1 as old


def new():
    torch.manual_seed(17);m=torch.nn.Sequential();m.add_module('fc',torch.nn.Linear(3,2))
    inject_lora(m,['fc'],2,2.,0.)
    return m,DDP(m),torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=1e-5,weight_decay=0.,fused=False)


def update(w,opt):
    opt.zero_grad()
    for micro in range(4):
        with w.no_sync() if micro<3 else nullcontext():(w(torch.randn(4,3)).square().mean()/4).backward()
    opt.step();opt.zero_grad()


def equal(m,opt,expected,expected_opt):
    assert all(torch.equal(t,expected[n]) for n,t in s.trainable_state(m).items())
    assert opt.state_dict()['param_groups']==expected_opt['param_groups']
    for index,entry in expected_opt['state'].items():
        for key,value in entry.items():assert torch.equal(value,opt.state_dict()['state'][index][key])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    dist.init_process_group('gloo');rank=dist.get_rank();world=dist.get_world_size();assert world==2
    args.output.mkdir(exist_ok=True);parent=args.output/'parent';cont=args.output/'continue'
    parent.mkdir(exist_ok=True);cont.mkdir(exist_ok=True)
    model,wrapped,opt=new();torch.manual_seed(100+rank)
    for _ in range(200):update(wrapped,opt)
    rng=[None]*2;dist.all_gather_object(rng,s.rng_state('cpu'))
    if rank==0:old.save_step(parent,model,opt,step=200,world=2,rank_rng=rng,receipt_hash='parent',history=[{}]*200)
    dist.barrier()
    record=__import__('json').loads((parent/'steps'/'step_000200.json').read_text())
    saved=torch.load(parent/record['file'],weights_only=True,map_location='cpu')
    update(wrapped,opt);expected201=s.trainable_state(model);opt201=s.cpu_tree(opt.state_dict())
    update(wrapped,opt);expected202=s.trainable_state(model);opt202=s.cpu_tree(opt.state_dict())
    # Simulate staged cache work / model loading consuming RNG before restoration.
    torch.rand(123);m2,w2,o2=new();s.restore(m2,o2,saved,rank,'cpu')
    rng=[None]*2;dist.all_gather_object(rng,s.rng_state('cpu'))
    if rank==0:s.save_step(cont,m2,o2,step=200,world=2,rank_rng=rng,receipt_hash='new',history=saved['history'])
    dist.barrier();migrated,_=s.load_latest(cont,'new',2)
    assert migrated['next_global_slot']==6400
    m3,w3,o3=new();s.restore(m3,o3,migrated,rank,'cpu');update(w3,o3);equal(m3,o3,expected201,opt201)
    rng=[None]*2;dist.all_gather_object(rng,s.rng_state('cpu'))
    if rank==0:s.save_step(cont,m3,o3,step=201,world=2,rank_rng=rng,receipt_hash='new',history=[{}]*201)
    dist.barrier();again,_=s.load_latest(cont,'new',2)
    m4,w4,o4=new();s.restore(m4,o4,again,rank,'cpu');update(w4,o4);equal(m4,o4,expected202,opt202)
    hashes=[None]*2;dist.all_gather_object(hashes,s.parameter_hash(s.trainable_state(m4)));assert len(set(hashes))==1
    if rank==0:print('RAIN_NATIVE50_CONTINUE_CPU_DDP_PASS old_step200_new_receipt_step201_and202_bitwise_equal=true',flush=True)
    dist.destroy_process_group()


if __name__=='__main__':main()
