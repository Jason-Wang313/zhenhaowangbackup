#!/usr/bin/env python3
"""Two-rank CPU DDP: global32 accumulation, native-horizon toy gradient, exact resume.

Tiny synthetic tensors only; not a GPU FLUX qualification. Eight-RNG checkpoint
schema is exercised with repeated two-rank RNG fixtures, not claimed eight-rank execution.
"""
import copy
from contextlib import nullcontext
import os
from pathlib import Path
import sys
import tempfile
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from lowlight_restoration.frozen_expert_router import FrozenExpertRouter, run_steps
from lowlight_restoration.five_expert_flux import ORDER
from smoke import frozen_expert_router_state_v1 as state


def sample(slot):
    g = torch.Generator().manual_seed(7100+slot)
    return torch.randn(1, 6, generator=g), torch.randn(1, 4, generator=g), torch.tensor([slot%5])


def loss(model, slot):
    x, v, target = sample(slot); logits, weights, _, _ = model(x, v)
    generated = run_steps(torch.ones(2, 1, 2), weights, torch.linspace(1, 0, 51).tolist(),
        ORDER[int(weights.detach().argmax())],
        lambda s, w, i, t: torch.sin(s[:1]*.02+(w*torch.arange(5)).sum()*.1+t*.01), checkpointed=True)
    return generated.square().mean()+.1*torch.nn.functional.cross_entropy(logits, target)


def update(model, optimizer, step, rank=None):
    optimizer.zero_grad(set_to_none=True)
    micros = 32 if rank is None else 16
    for micro in range(micros):
        slot = step*32 + (micro if rank is None else 2*micro+rank)
        with model.no_sync() if rank is not None and micro < micros-1 else nullcontext():
            (loss(model, slot)/micros).backward()
    optimizer.step()


def main():
    torch.set_num_threads(1); dist.init_process_group('gloo')
    rank = dist.get_rank()
    if dist.get_world_size() != 2: raise ValueError('two CPU ranks required')
    torch.manual_seed(4)
    model = FrozenExpertRouter(6, 4, 8)
    serial = copy.deepcopy(model); serial_opt = torch.optim.AdamW(serial.parameters(), lr=.001, weight_decay=0)
    ddp = DDP(model, broadcast_buffers=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=0)
    directories = [tempfile.mkdtemp(prefix='frozen-router-ddp-') if rank == 0 else None]
    dist.broadcast_object_list(directories, src=0); output = Path(directories[0])

    def save(step):
        rng = [None]*2; dist.all_gather_object(rng, state.io.rng_state('cpu'))
        if rank == 0:
            state.save(output, model, optimizer, step=step, history=[{'step': i+1} for i in range(step)],
                rank_rng=[rng[i%2] for i in range(8)], receipt_hash='synthetic-two-rank')
        dist.barrier()

    save(0); update(ddp, optimizer, 0, rank); save(1)
    update(serial, serial_opt, 0)
    for a, b in zip(model.parameters(), serial.parameters()):
        torch.testing.assert_close(a, b, atol=2e-7, rtol=2e-6)
    saved = state.load(output, 'synthetic-two-rank')
    update(ddp, optimizer, 1, rank)
    reference = state.model_state(model); expected_opt = state.io.cpu_tree(optimizer.state_dict())
    # Rebuild the module, optimizer and reducer as a fresh worker would.
    restored = FrozenExpertRouter(6, 4, 8)
    restored_ddp = DDP(restored, broadcast_buffers=False)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=.001, weight_decay=0)
    state.restore(restored, restored_optimizer, saved, rank, 'cpu')
    update(restored_ddp, restored_optimizer, 1, rank)
    for n, p in restored.named_parameters():
        if not torch.equal(p.detach(), reference[n]): raise ValueError('resume next step differs')
    for i, values in restored_optimizer.state_dict()['state'].items():
        for n, t in values.items():
            if not torch.equal(t, expected_opt['state'][i][n]): raise ValueError('Adam differs after resume')
    hashes = [None]*2; dist.all_gather_object(hashes, state.io.parameter_hash(state.model_state(restored)))
    if hashes[0] != hashes[1]: raise ValueError('rank parameter mismatch')
    if rank == 0: print('FROZEN_ROUTER_CPU_DDP_PASS ranks=2 global_batch=32 exact_resume=true GPU_qualified=false', flush=True)
    dist.destroy_process_group()


if __name__ == '__main__': main()
