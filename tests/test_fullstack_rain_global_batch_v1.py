"""Run with six CPU torchrun workers; compare Adam against a single global batch."""
import copy
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.distributed as dist
from lowlight_restoration.rain_global_batch_v1 import positions, sum_gradients


def update(model, optimizer, x, y, indices, distributed):
    optimizer.zero_grad(set_to_none=True)
    for i in indices:
        ((model(x[i:i+1])-y[i:i+1]).square().mean()/32).backward()
    if distributed:
        sum_gradients(model.parameters())
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    optimizer.step()


def main():
    torch.set_num_threads(1); dist.init_process_group('gloo')
    rank, world = dist.get_rank(), dist.get_world_size()
    assert world == 6
    all_positions = [p for r in range(world) for p in positions(800,r,world)]
    assert sorted(all_positions) == list(range(25600,25632))
    assert [len(positions(800,r,world)) for r in range(world)] == [6,6,5,5,5,5]
    torch.manual_seed(92)
    model = torch.nn.Sequential(torch.nn.Linear(3,7),torch.nn.Tanh(),torch.nn.Linear(7,2)).double()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.)
    x, y = torch.randn(32,3,dtype=torch.float64), torch.randn(32,2,dtype=torch.float64)
    for _ in range(5): update(model,optimizer,x,y,range(32),False)
    reference = copy.deepcopy(model)
    refopt = torch.optim.AdamW(reference.parameters(),lr=1e-5,weight_decay=0.)
    refopt.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    for step in range(800,803):
        update(model,optimizer,x,y,[p-32*step for p in positions(step,rank,world)],True)
        update(reference,refopt,x,y,range(32),False)
        for a,b in zip(model.parameters(),reference.parameters()):
            torch.testing.assert_close(a,b,rtol=0,atol=1e-12)
        for key, state in optimizer.state_dict()['state'].items():
            for name,value in state.items():
                torch.testing.assert_close(value,refopt.state_dict()['state'][key][name],rtol=0,atol=1e-12)
        # Restore model and all Adam moments between steps, rather than fresh Adam.
        saved_model=copy.deepcopy(model.state_dict());saved_opt=copy.deepcopy(optimizer.state_dict())
        model.load_state_dict(saved_model);optimizer.load_state_dict(saved_opt)
    if rank == 0:print('RAIN_GLOBAL32_SIX_PROCESS_PASS coverage=32 counts=6,6,5,5,5,5 Adam_and_gradients_match=true',flush=True)
    dist.destroy_process_group()


if __name__=='__main__':main()
