#!/usr/bin/env python3
"""Two-process CPU test of detached-prefix DDP accumulation and exact resume."""
from datetime import timedelta
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests'))
sys.path.insert(0, str(ROOT / 'third_party/flux2/src'))
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from contextlib import nullcontext
from test_rain_onpolicy_midpoint_v1 import Toy, AE, Metric, sampler, optimizer
from lowlight_restoration import rain_onpolicy_midpoint as logic
from smoke import rain_onpolicy_continue_state_v1 as state


def worker(rank, folder):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://' + folder + '/rendezvous',
                            rank=rank, world_size=2, timeout=timedelta(seconds=120))
    try:
        torch.manual_seed(7)
        model = Toy()
        wrapped = DDP(model)
        opt = optimizer(model)
        # Give every Adam parameter a genuine initialized state, then emulate the
        # already-validated step600 parent counter. This is not a quality test.
        inputs = logic.detached_prefix(model, sampler)
        loss, _ = logic.objective(wrapped, AE(), inputs, torch.zeros(1, 3, 2, 2), Metric(),
                                  {'gt_rgb_mse': 10., 'gt_lpips': .05})
        loss.backward()
        opt.step()
        opt.zero_grad()
        for entry in opt.state.values():
            entry['step'].fill_(600)
        torch.manual_seed(103 + rank)

        def update():
            opt.zero_grad()
            for micro in range(2):
                inputs = logic.detached_prefix(model, sampler)
                target = torch.rand(1, 3, 2, 2) * .1
                with wrapped.no_sync() if micro == 0 else nullcontext():
                    loss, _ = logic.objective(wrapped, AE(), inputs, target, Metric(),
                                              {'gt_rgb_mse': 10., 'gt_lpips': .05})
                    (loss / 2).backward()
            opt.step()
            opt.zero_grad()

        rngs = [None] * 2
        dist.all_gather_object(rngs, state.base.rng_state('cpu'))
        if rank == 0:
            state.save_step(folder, model, opt, step=600, world=2, rank_rng=rngs,
                            receipt_hash='cpu_ddp', history=[{}] * 600)
        dist.barrier()
        update()
        expected = state.base.trainable_state(model)
        expected_opt = state.base.cpu_tree(opt.state_dict())
        saved, _ = state.load_latest(folder, 'cpu_ddp', 2)
        state.base.restore(model, opt, saved, rank, 'cpu')
        update()
        actual = state.base.trainable_state(model)
        assert all(torch.equal(actual[k], v) for k, v in expected.items())
        for idx, entries in expected_opt['state'].items():
            assert all(torch.equal(v, opt.state_dict()['state'][idx][key]) for key, v in entries.items())
        hashes = [None] * 2
        dist.all_gather_object(hashes, state.base.parameter_hash(actual))
        assert len(set(hashes)) == 1
        rngs = [None] * 2
        dist.all_gather_object(rngs, state.base.rng_state('cpu'))
        if rank == 0:
            state.save_step(folder, model, opt, step=601, world=2, rank_rng=rngs,
                            receipt_hash='cpu_ddp', history=[{}] * 601)
        dist.barrier()
        assert state.load_latest(folder, 'cpu_ddp', 2)[0]['next_global_slot'] == 19232
        if rank == 0:
            print('RAIN_ONPOLICY_CONTINUE_CPU_DDP_EXACT_RESUME_PASS two_processes=true CUDA_tested=false', flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    with tempfile.TemporaryDirectory(prefix='rain-onpolicy-ddp-') as folder:
        torch.multiprocessing.start_processes(worker, args=(folder,), nprocs=2, start_method='fork', join=True)
