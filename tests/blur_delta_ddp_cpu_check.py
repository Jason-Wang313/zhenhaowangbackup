"""Real two-process SUM / batch8 Adam equivalence; run with torchrun CPU."""
import os
import torch
import torch.distributed as dist
from lowlight_restoration.lora import inject_lora

torch.set_num_threads(1)
dist.init_process_group('gloo')
rank = int(os.environ['RANK'])
torch.manual_seed(71)
model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Tanh(), torch.nn.Linear(4, 3))
inject_lora(model, ['0', '2'], 2, 2.)
import copy
reference = copy.deepcopy(model)
parameters = [p for p in model.parameters() if p.requires_grad]
baseline = [p for p in reference.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(parameters, lr=1e-5, weight_decay=0.)
expected = torch.optim.AdamW(baseline, lr=1e-5, weight_decay=0.)
for step in range(2):
    optimizer.zero_grad()
    expected.zero_grad()
    x, y = torch.randn(8, 3), torch.randn(8, 3)
    ((reference(x)-y).square().mean()).backward()
    for index in range(rank, 8, 2):
        ((model(x[index:index+1])-y[index:index+1]).square().mean()/8).backward()
    for p in parameters:
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
    for p, q in zip(parameters, baseline):
        torch.testing.assert_close(p.grad, q.grad, rtol=1e-5, atol=1e-7)
    torch.nn.utils.clip_grad_norm_(parameters, 1.)
    torch.nn.utils.clip_grad_norm_(baseline, 1.)
    optimizer.step()
    expected.step()
    for p, q in zip(parameters, baseline):
        torch.testing.assert_close(p, q, rtol=1e-6, atol=1e-7)
dist.barrier()
if rank == 0:
    print('BLUR_DELTA_TWO_RANK_CPU_ADAM_PASS steps=2 batch=8')
dist.destroy_process_group()
