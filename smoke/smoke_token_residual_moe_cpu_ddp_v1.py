#!/usr/bin/env python3
"""Eight-process CPU/Gloo batch32, exact state restore and numerical continuation."""
import copy
import argparse
from contextlib import nullcontext
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests'))

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from test_token_residual_moe_v1 import model
from lowlight_restoration.token_residual_moe import batch_slot, token_conditioning
from lowlight_restoration.joint_moe_state import save_state, rng_state
from smoke.train_foundir_token_residual_moe_v1 import resume_training
from smoke.token_residual_moe_v1 import write_once


def step(net, wrapped, optimizer, x, f, rank):
    optimizer.zero_grad()
    for micro in range(4):
        slot, scale = batch_slot(0, micro, rank)
        with wrapped.no_sync() if micro < 3 else nullcontext():
            with token_conditioning(net, f[slot:slot+1]):
                (wrapped(x[slot:slot+1]).square().mean() * scale).backward()
    optimizer.step()


def worker(rank, directory, mode):
    torch.set_num_threads(1)
    path = Path(directory)
    dist.init_process_group('gloo', init_method=(path / 'rendezvous').as_uri(), rank=rank, world_size=8)
    try:
        net = model()
        wrapped = DDP(net)
        optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0)
        g = torch.Generator().manual_seed(213)
        x, f = torch.randn(32, 3, 6, generator=g), torch.randn(32, 4, generator=g)
        step(net, wrapped, optimizer, x, f, rank)
        first_weights = [p.detach().clone() for p in net.parameters()]
        first_states = copy.deepcopy(list(optimizer.state.values()))
        states = [None] * 8
        dist.all_gather_object(states, rng_state())
        binding = {'config_sha256': 'toy', 'source_sha256': {'toy': 'test'}, 'data': 'test'}
        if rank == 0:
            write_once(path / 'launch.json', {'binding': binding})
            save_state(net, optimizer, path, step=1, mode=mode, arm='token_residual',
                       config_sha='toy', sources=binding['source_sha256'], rank_rng=states)
        dist.barrier()
        expected_rng_draw = torch.rand(9)
        step(net, wrapped, optimizer, x, f, rank)
        resumed = model()
        opt = torch.optim.AdamW(resumed.parameters(), lr=1e-3, weight_decay=0)
        assert resume_training(resumed, opt, path / 'step_000001.resume.json', binding, rank,
                               mode=mode, total_steps=1000 if mode == 'formal' else 25) == 1
        for value, expected in zip(resumed.parameters(), first_weights):
            torch.testing.assert_close(value, expected, atol=0, rtol=0)
        for actual, expected in zip(opt.state.values(), first_states):
            for key in ('step', 'exp_avg', 'exp_avg_sq'):
                torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)
        torch.testing.assert_close(torch.rand(9), expected_rng_draw, rtol=0, atol=0)
        wrapped_resume = DDP(resumed)
        step(resumed, wrapped_resume, opt, x, f, rank)
        for a, b in zip(net.parameters(), resumed.parameters()):
            # A rebuilt DDP reducer can change FP32 reduction ordering. State
            # restored above is bit-exact; continuation is numerical, not bitwise.
            torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-7)
        for a, b in zip(optimizer.state.values(), opt.state.values()):
            for key in ('step', 'exp_avg', 'exp_avg_sq'):
                torch.testing.assert_close(a[key], b[key], rtol=1e-6, atol=1e-7)
        if rank == 0:
            serial = model()
            serial_opt = torch.optim.AdamW(serial.parameters(), lr=1e-3, weight_decay=0)
            for _ in range(2):
                serial_opt.zero_grad()
                with token_conditioning(serial, f):
                    serial(x).square().mean().backward()
                serial_opt.step()
            for a, b in zip(net.parameters(), serial.parameters()):
                torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)
            difference = max(float((a-b).abs().max()) for a,b in zip(net.parameters(), resumed.parameters()))
            print(f'TOKEN_ROUTER_CPU_8RANK_DDP_BATCH32_STATE_RESTORE_CONTINUATION_PASS mode={mode} max_abs={difference}', flush=True)
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('qualification', 'formal'), default='qualification')
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='token-router-gloo-') as directory:
        mp.spawn(worker, args=(directory, args.mode), nprocs=8, join=True)
