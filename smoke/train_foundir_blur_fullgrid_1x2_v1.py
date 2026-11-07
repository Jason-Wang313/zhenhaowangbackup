#!/usr/bin/env python3
"""All15 deployment views per original source, four epochs, full1792 ->9472."""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'third_party/flux2/src'))
from smoke import run_foundir_fullstack_blur_coverage_v2 as parent
from smoke import blur_fullgrid_1x2_state_v1 as state
from smoke import blur_fullgrid_1x2_data_v1 as data
from smoke import blur_fullgrid_1x2_snapshots_v1 as snapshots
from smoke import train_foundir_blur_fullgrid_v1 as previous
from smoke.existing_flux_replay_contract_v1 import read, sha, write_new as write
from lowlight_restoration.rain_global_batch_v1 import sum_gradients

OUT = data.OUT
REPORT = ROOT/'reports/foundir_blur_fullgrid_1x2_v1_train_result_2026-11-07.json'
START, END, WORLD = state.START, state.END, state.WORLD
MILESTONES = (START, START+1, START+2, *state.SNAPSHOTS)
RECIPE, SOURCES, need, core = data.RECIPE, data.SOURCES, parent.need, parent.core
setup, load_model, load_vae_prompt, prediction = parent.setup, parent.load_model, parent.load_vae_prompt, parent.prediction
prepare, bound = data.prepare, data.bound


def run():
    import torch
    import torch.distributed as dist
    from lowlight_restoration.flux2_endpoint_precision import assert_fp32_optimizer
    from smoke.official_vlm_v4_metric_identity_fix import checked_metric_models
    begun = time.monotonic(); b = bound(); local, device = setup(b)
    cache_manifest = read(data.CACHE_OUT/'cache_manifest.json')
    need(cache_manifest['receipt_sha256'] == b['migration_parent_receipt_sha256']
         and set(cache_manifest['markers']) == {data.key(i) for i in b['views']}, 'all15360 caches required')
    rank, node = int(os.environ['RANK']), int(os.environ['FOUNDIR_FULLSTACK_NODE'])
    need(int(os.environ['WORLD_SIZE']) == WORLD and node == 3 and rank == local, 'single node3 two GPUs required')
    dist.init_process_group('nccl', device_id=device, timeout=timedelta(minutes=10))
    ping = torch.tensor(float(rank+1), device=device); dist.all_reduce(ping); need(float(ping) == 3., 'NCCL sum failed')
    runtime = {'rank': rank, 'node': node, 'host': __import__('socket').gethostname(),
               'torch': str(torch.__version__), 'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
               'GPU': torch.cuda.get_device_name(device), 'receipt_sha256': sha(OUT/'receipt.json')}
    prior_runtime = read(previous.OUT/f'runtime{state.PARENT_RNG_RANKS[rank]}.json')
    need(all(runtime[k] == prior_runtime[k] for k in ('torch', 'cuda', 'cudnn', 'GPU')),
         'migration numerical runtime differs from parent node3')
    signatures = [None]*WORLD; dist.all_gather_object(signatures, runtime)
    need(len({s['host'] for s in signatures}) == 1, 'one physical node required')
    for s in signatures:
        need(all(s[k] == runtime[k] for k in ('torch', 'cuda', 'cudnn')), 'numerical runtime mismatch')
    write(OUT/f'runtime{rank}.json', runtime)
    print(f'BLUR_FULLGRID_1X2_NCCL_VERIFIED rank={rank} world=2', flush=True)
    model = load_model(b, device); ae, prompt = load_vae_prompt(b, device)
    need(not any(isinstance(m, torch.nn.Dropout) and m.p > 0 for m in model.modules()), 'unexpected stochastic dropout')
    metrics, _ = checked_metric_models(device, b['metric_identity']); metrics['lpips'].as_loss = True
    parameters = [p for p in model.parameters() if p.requires_grad]; optimizer = parent.optimizer_for(model)
    receipt = sha(OUT/'receipt.json'); saved, _ = state.latest(OUT, receipt); fresh = saved is None
    if fresh:
        saved, _ = previous.state.load_step(previous.OUT, START, b['migration_parent_receipt_sha256'])
        need(saved['parameter_hash'] == b['migration_parent_checkpoint']['parameter_hash'], 'wrong parent3687')
        state.migrate(model, optimizer, saved, rank, device)
    else: state.restore(model, optimizer, saved, rank, device)
    start, history = saved['step'], saved['history']; del saved
    starts = [None]*WORLD; dist.all_gather_object(starts, (start, fresh)); need(len(set(starts)) == 1, 'rank start differs')
    print(f'BLUR_FULLGRID_1X2_RESTORED rank={rank} step={start} model_Adam_sampler_preserved=true same_world_RNG_restored={not fresh} parent_RNG_rank={state.PARENT_RNG_RANKS[rank] if fresh else rank} topology_migration_not_bitwise={fresh}', flush=True)

    def save(step):
        rngs = [None]*WORLD; dist.all_gather_object(rngs, parent.state.rng_state(device))
        if step in MILESTONES:
            hashes = [None]*WORLD; dist.all_gather_object(hashes, parent.state.parameter_hash(parent.state.trainable_state(model)))
            need(len(set(hashes)) == 1, 'rank parameter mismatch')
        if rank == 0:
            marker = parent.state.save(OUT, model, optimizer, step=step, world=WORLD, rank_rng=rngs, receipt_sha=receipt, history=history)
            print('BLUR_FULLGRID_1X2_SAVED '+json.dumps(marker), flush=True)
        dist.barrier()

    if fresh: save(START)
    if start in state.SNAPSHOTS: snapshots.evaluate(model,ae,prompt,metrics,b,start,rank,device)
    model.train()
    for step in range(start, END):
        tick = time.monotonic(); optimizer.zero_grad(set_to_none=True)
        totals = torch.zeros(3, dtype=torch.float64, device=device)
        for index in state.indices(step, rank):
            item = b['views'][index]; marker = data.CACHE_OUT/'cache'/(data.key(item)+'.json')
            need(sha(marker) == cache_manifest['markers'][data.key(item)], 'view reference cache changed')
            tensors = data.cached(item, device); gt = data.rgb(item, 'gt', device)
            need(tensors is not None, 'view cache missing')
            pred = prediction(model, ae, prompt, tensors); loss, terms = core.objective(pred, gt, metrics['lpips'])
            need(torch.isfinite(loss) and loss.requires_grad, 'invalid loss'); (loss/8).backward()
            totals += torch.tensor([float(loss.detach()), float(terms['mse'].detach()), float(terms['lpips'].detach())], device=device, dtype=torch.float64)/8
            del tensors, gt, pred, loss, terms
        sum_gradients(parameters)
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True); need(float(norm) > 0, 'zero gradient')
        optimizer.step(); assert_fp32_optimizer(optimizer); optimizer.zero_grad(set_to_none=True)
        need(all(torch.isfinite(p).all() for p in parameters), 'nonfinite parameters')
        dist.all_reduce(totals)
        stats = torch.tensor([time.monotonic()-tick, torch.cuda.max_memory_allocated(device)/1024**3], device=device)
        dist.all_reduce(stats, op=dist.ReduceOp.MAX)
        entry = {'step': step+1, 'loss': float(totals[0]), 'mse': float(totals[1]), 'lpips': float(totals[2]),
                 'gradient_norm': float(norm), 'next_global_slot': (step+1)*8, 'max_rank_seconds': float(stats[0]),
                 'peak_allocated_GiB': float(stats[1]), 'world': WORLD, 'global_batch': 8}
        history.append(entry)
        if rank == 0: print('BLUR_FULLGRID_1X2_TRAIN '+json.dumps(entry), flush=True)
        save(step+1)
        if step+1 in state.SNAPSHOTS: snapshots.evaluate(model,ae,prompt,metrics,b,step+1,rank,device)
        if step+1 == START+2:
            # Real two-rank CUDA restore: preserve exact state before continuing step3690.
            saved, _ = state.latest(OUT, receipt); before = parent.state.parameter_hash(parent.state.trainable_state(model))
            state.restore(model, optimizer, saved, rank, device)
            need(parent.state.parameter_hash(parent.state.trainable_state(model)) == before, 'CUDA restore changed model')
            del saved; print(f'BLUR_FULLGRID_1X2_CUDA_RESTORE_VERIFIED rank={rank} step={START+2}', flush=True)
    write(OUT/f'worker{rank}.json', {'rank': rank, 'node': node, 'step': END, 'exit_code': 0,
          'receipt_sha256': receipt, 'parameter_hash': parent.state.parameter_hash(parent.state.trainable_state(model)),
          'seconds': time.monotonic()-begun, 'peak_allocated_GiB': torch.cuda.max_memory_allocated(device)/1024**3})
    dist.barrier(); dist.destroy_process_group(); print(f'BLUR_FULLGRID_1X2_END rank={rank} status=0', flush=True)


def finish():
    begin = time.monotonic(); b = bound(); saved, marker = state.latest(OUT, sha(OUT/'receipt.json'))
    need(saved is not None and saved['step'] == END, 'fixed9472 incomplete')
    workers = []
    for rank in range(WORLD):
        w = read(OUT/f'worker{rank}.json')
        need(w['rank'] == rank and w['step'] == END and w['exit_code'] == 0
             and w['receipt_sha256'] == sha(OUT/'receipt.json') and w['parameter_hash'] == saved['parameter_hash'], 'rank incomplete')
        workers.append(w)
    need(saved['history'][:START] == previous.state.load_step(previous.OUT, START, b['migration_parent_receipt_sha256'])[0]['history'], 'parent history changed')
    result = {'status': 'fixed1024_all15_views_four_epochs_training_complete', 'recipe': RECIPE, 'workers': workers,
              'final_checkpoint': marker, 'history': saved['history'], 'training_fit_snapshots': snapshots.verify(b), 'official_Blur_pass': False,
              'actual_final_SHA_Adam_two_RNG_sampler_verified': True, 'checkpoint_receipts_contiguous': END-START+1,
              'all_large_checkpoints_rehashed': False, 'same_world_state_preserved': False, 'model_Adam_history_sampler_preserved': True, 'parent_world': 4, 'world': 2, 'bitwise_four_rank_continuation': False,
              'seconds': time.monotonic()-begin, 'exit_code': 0, 'GPU_hours': 0}
    write(REPORT, result); print('BLUR_FULLGRID_1X2_TRAIN_VERIFIED '+json.dumps({k:v for k,v in result.items() if k not in ('history','workers','recipe')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('phase', choices=['prepare','run','finish'])
    globals()[parser.parse_args().phase]()
