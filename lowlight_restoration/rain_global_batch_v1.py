"""Exact global32 sample coverage with uneven local counts; no hidden padding."""


def positions(step, rank, world):
    if not 800 <= step < 1000 or not 1 <= world <= 32 or not 0 <= rank < world:
        raise ValueError('outside fixed step800..1000/global32 continuation')
    return [32*step+i for i in range(rank, 32, world)]


def sum_gradients(parameters):
    """Caller backpropagates each real sample loss/32; SUM, never average again."""
    import torch
    import torch.distributed as dist
    parameters = list(parameters)
    if not parameters:
        raise ValueError('no trainable parameters')
    bad = torch.tensor([int(any(p.grad is None for p in parameters))],
                       device=parameters[0].device)
    dist.all_reduce(bad, op=dist.ReduceOp.MAX)
    if bad.item():
        raise ValueError('a rank has missing parameter gradients')
    flat = torch.cat([p.grad.detach().reshape(-1) for p in parameters])
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    if not torch.isfinite(flat).all():
        raise ValueError('nonfinite global gradient')
    offset = 0
    for p in parameters:
        p.grad.copy_(flat[offset:offset+p.numel()].view_as(p))
        offset += p.numel()
