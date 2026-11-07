#!/usr/bin/env python3
"""Fixed1024 sources, original model/Adam step200 ->1280, global8 on three nodes."""
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
from smoke import blur_exposure_state_v1 as state
from smoke.existing_flux_replay_contract_v1 import read, sha, write_new as write
from lowlight_restoration.rain_global_batch_v1 import sum_gradients

OUT = ROOT/'checkpoints/foundir_blur_exposure_v1'
REPORT = ROOT/'reports/foundir_blur_exposure_v1_train_result_2026-11-07.json'
FIT = ROOT/'reports/foundir_blur_fixed8_fit_v1_result_2026-11-07.json'
START, END, WORLD = state.START, state.END, state.WORLD
MILESTONES = (200, 201, 202, 512, 1024, 1280)
RECIPE = {**parent.RECIPE, 'version': 3, 'start_step': START, 'train_steps': END,
          'new_updates': END-START, 'world': WORLD, 'nodes': [0, 1, 2], 'node': None,
          'global_batch': 8, 'final_exposures_per_source': 10, 'migration_seed': 2026110714,
          'initialization': 'original1024_step200_model_and_complete_Adam; never_fixed8_weights',
          'migration': '2_to6; exact_model_Adam_sampler; reseeded_rank_RNG; not_bitwise',
          'resume': 'same_world6_only_with_all_rank_RNG',
          'evaluation': 'fixed_final1280_complete_official150_only; no_checkpoint_search',
          'gradient_reduction': 'each_sample_loss/8_then_flat_SUM_no_extra_division'}
SOURCES = ['smoke/train_foundir_blur_exposure_v1.py', 'smoke/blur_exposure_state_v1.py',
           'smoke/run_foundir_fullstack_blur_exposure_v1.sh', 'lowlight_restoration/rain_global_batch_v1.py']
need = parent.need
core = parent.core
setup, load_model, load_vae_prompt, prediction = parent.setup, parent.load_model, parent.load_vae_prompt, parent.prediction


def prepare():
    started = time.monotonic(); old = parent.bound()
    fit = read(FIT); need(fit['status'] == 'fixed8_fresh100_training_fit_complete', 'fixed8 evidence required')
    value = parent.state.load(parent.OUT/'pilot', sha(parent.OUT/'receipt.json'), 2)
    need(value['step'] == START and value['next_global_slot'] == 1600, 'original1024 completed200 required')
    need(len(value['optimizer']['state']) == 100, 'complete original Adam required')
    for rank in (0, 1):
        done = read(parent.OUT/'pilot'/f'train_done{rank}.json')
        need(done['step'] == START and done['exit_code'] == 0 and done['parameter_hash'] == value['parameter_hash'], 'parent incomplete')
    markers = {}
    for item in old['rows']:
        p = parent.OUT/'cache'/(parent.parent.key(item)+'.json'); r = read(p)
        need(r['item'] == item and r['receipt_sha256'] == sha(parent.OUT/'receipt.json')
             and r['GT_input'] is False and (parent.OUT/r['tensor']).is_file(), 'cache missing or changed')
        markers[parent.parent.key(item)] = sha(p)
    for name, digest in old['assets'].items(): need(sha(ROOT/name) == digest, 'original asset changed')
    need(shutil.disk_usage(ROOT).free > 500*1024**3, 'need500GiB; preserve old assets')
    b = {k: old[k] for k in ('rows', 'assets', 'model', 'lora', 'prompt', 'metric_identity')}
    b.update(recipe=RECIPE, parent_receipt_sha256=sha(parent.OUT/'receipt.json'),
             parent_checkpoint=read(parent.OUT/'pilot/steps/step_000200.json'),
             fixed8_evidence_sha256=sha(FIT), cache_marker_hashes=markers,
             sources={n: sha(ROOT/n) for n in sorted(set(old['sources']) | set(SOURCES))})
    write(OUT/'receipt.json', b)
    result = {'status': 'destination_CPU_preflight_pass', 'seconds': time.monotonic()-started,
              'exit_code': 0, 'GPU_hours': 0, 'actual_parent_SHA_Adam_RNG_verified': True,
              'parent_parameter_hash': value['parameter_hash'], 'original_rank_RNG_preserved_in_parent': 2,
              'new_runtime_rank_RNG_reseeded': 6, 'bitwise_parent_resume': False,
              'cache_metadata_checked': 1024, 'cache_tensors_rehashed_at_use': True,
              'training_sources': 1024, 'total_steps': END, 'new_updates': END-START,
              'final_exposures_per_source': 10, 'local_counts': [len(state.indices(START, r)) for r in range(WORLD)],
              'official_pixels_accessed': False, 'receipt_sha256': sha(OUT/'receipt.json')}
    write(ROOT/'reports/foundir_blur_exposure_v1_preflight_2026-11-07.json', result)
    print('BLUR_EXPOSURE_PREFLIGHT_PASS '+json.dumps(result), flush=True)


def bound():
    b = read(OUT/'receipt.json'); old = parent.bound()
    need(b['recipe'] == RECIPE and b['rows'] == old['rows'] and b['assets'] == old['assets'], 'scope changed')
    need(b['parent_receipt_sha256'] == sha(parent.OUT/'receipt.json') and b['fixed8_evidence_sha256'] == sha(FIT), 'parent evidence changed')
    need(b['parent_checkpoint'] == read(parent.OUT/'pilot/steps/step_000200.json'), 'parent checkpoint marker changed')
    for name, digest in b['sources'].items(): need(sha(ROOT/name) == digest, 'source changed: '+name)
    return b


def run():
    import torch
    import torch.distributed as dist
    from lowlight_restoration.flux2_endpoint_precision import assert_fp32_optimizer
    from smoke.official_vlm_v4_metric_identity_fix import checked_metric_models
    begun = time.monotonic(); b = bound(); local, device = setup(b)
    rank, node = int(os.environ['RANK']), int(os.environ['FOUNDIR_FULLSTACK_NODE'])
    need(int(os.environ['WORLD_SIZE']) == WORLD and node in (0, 1, 2) and rank == 2*node+local, 'three nodes two GPUs required')
    dist.init_process_group('nccl', device_id=device, timeout=timedelta(minutes=10))
    ping = torch.tensor(float(rank+1), device=device); dist.all_reduce(ping); need(float(ping) == 21., 'NCCL sum failed')
    runtime = {'rank': rank, 'node': node, 'host': __import__('socket').gethostname(),
               'torch': str(torch.__version__), 'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
               'GPU': torch.cuda.get_device_name(device), 'receipt_sha256': sha(OUT/'receipt.json')}
    signatures = [None]*WORLD; dist.all_gather_object(signatures, runtime)
    need(len({s['host'] for s in signatures}) == 3, 'three distinct physical nodes required')
    for s in signatures:
        need(all(s[k] == runtime[k] for k in ('torch', 'cuda', 'cudnn')), 'numerical runtime mismatch')
    write(OUT/f'runtime{rank}.json', runtime)
    print(f'BLUR_EXPOSURE_NCCL_VERIFIED rank={rank} world=6', flush=True)
    model = load_model(b, device); ae, prompt = load_vae_prompt(b, device)
    need(not any(isinstance(m, torch.nn.Dropout) and m.p > 0 for m in model.modules()), 'unexpected stochastic dropout')
    metrics, _ = checked_metric_models(device, b['metric_identity']); metrics['lpips'].as_loss = True
    parameters = [p for p in model.parameters() if p.requires_grad]; optimizer = parent.optimizer_for(model)
    receipt = sha(OUT/'receipt.json'); saved, _ = state.latest(OUT, receipt); fresh = saved is None
    if fresh:
        saved = parent.state.load(parent.OUT/'pilot', b['parent_receipt_sha256'], 2)
        need(saved['step'] == START and saved['parameter_hash'] == b['parent_checkpoint']['parameter_hash'], 'wrong migration parent')
        state.restore_parameters_adam(model, optimizer, saved)
        random.seed(RECIPE['migration_seed']+rank); torch.manual_seed(RECIPE['migration_seed']+rank)
        torch.cuda.manual_seed(RECIPE['migration_seed']+rank)
    else: state.restore(model, optimizer, saved, rank, device)
    start, history = saved['step'], saved['history']; del saved
    starts = [None]*WORLD; dist.all_gather_object(starts, (start, fresh)); need(len(set(starts)) == 1, 'rank start differs')
    print(f'BLUR_EXPOSURE_RESTORED rank={rank} step={start} model_Adam_sampler_preserved=true topology_migration_not_bitwise={fresh}', flush=True)

    def save(step):
        rngs = [None]*WORLD; dist.all_gather_object(rngs, parent.state.rng_state(device))
        if step in MILESTONES:
            hashes = [None]*WORLD; dist.all_gather_object(hashes, parent.state.parameter_hash(parent.state.trainable_state(model)))
            need(len(set(hashes)) == 1, 'rank parameter mismatch')
        if rank == 0:
            marker = parent.state.save(OUT, model, optimizer, step=step, world=WORLD, rank_rng=rngs, receipt_sha=receipt, history=history)
            print('BLUR_EXPOSURE_SAVED '+json.dumps(marker), flush=True)
        dist.barrier()

    if fresh: save(START)
    model.train()
    for step in range(start, END):
        tick = time.monotonic(); optimizer.zero_grad(set_to_none=True)
        totals = torch.zeros(3, dtype=torch.float64, device=device)
        for index in state.indices(step, rank):
            item = b['rows'][index]; marker = parent.OUT/'cache'/(parent.parent.key(item)+'.json')
            need(sha(marker) == b['cache_marker_hashes'][parent.parent.key(item)], 'reference cache changed')
            data = parent.cached(item, device); gt = parent.rgb(item, 'gt', device)
            pred = prediction(model, ae, prompt, data); loss, terms = core.objective(pred, gt, metrics['lpips'])
            need(torch.isfinite(loss) and loss.requires_grad, 'invalid loss'); (loss/8).backward()
            totals += torch.tensor([float(loss.detach()), float(terms['mse'].detach()), float(terms['lpips'].detach())], device=device, dtype=torch.float64)/8
            del data, gt, pred, loss, terms
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
        if rank == 0: print('BLUR_EXPOSURE_TRAIN '+json.dumps(entry), flush=True)
        save(step+1)
        if step+1 == 202:
            # Real six-rank CUDA restore: preserve exact state before continuing step203.
            saved, _ = state.latest(OUT, receipt); before = parent.state.parameter_hash(parent.state.trainable_state(model))
            state.restore(model, optimizer, saved, rank, device)
            need(parent.state.parameter_hash(parent.state.trainable_state(model)) == before, 'CUDA restore changed model')
            del saved; print(f'BLUR_EXPOSURE_CUDA_RESTORE_VERIFIED rank={rank} step=202', flush=True)
    write(OUT/f'worker{rank}.json', {'rank': rank, 'node': node, 'step': END, 'exit_code': 0,
          'receipt_sha256': receipt, 'parameter_hash': parent.state.parameter_hash(parent.state.trainable_state(model)),
          'seconds': time.monotonic()-begun, 'peak_allocated_GiB': torch.cuda.max_memory_allocated(device)/1024**3})
    dist.barrier(); dist.destroy_process_group(); print(f'BLUR_EXPOSURE_END rank={rank} status=0', flush=True)


def finish():
    begin = time.monotonic(); b = bound(); saved, marker = state.latest(OUT, sha(OUT/'receipt.json'))
    need(saved is not None and saved['step'] == END, 'fixed1280 incomplete')
    workers = []
    for rank in range(WORLD):
        w = read(OUT/f'worker{rank}.json')
        need(w['rank'] == rank and w['step'] == END and w['exit_code'] == 0
             and w['receipt_sha256'] == sha(OUT/'receipt.json') and w['parameter_hash'] == saved['parameter_hash'], 'rank incomplete')
        workers.append(w)
    need(saved['history'][:START] == parent.state.load(parent.OUT/'pilot', b['parent_receipt_sha256'], 2)['history'], 'old training history changed')
    result = {'status': 'fixed1024_ten_epochs_training_complete', 'recipe': RECIPE, 'workers': workers,
              'final_checkpoint': marker, 'history': saved['history'], 'official_Blur_pass': False,
              'actual_final_SHA_Adam_six_RNG_sampler_verified': True, 'checkpoint_receipts_contiguous': END-START+1,
              'all_large_checkpoints_rehashed': False, 'bitwise_two_to_six_resume': False,
              'seconds': time.monotonic()-begin, 'exit_code': 0, 'GPU_hours': 0}
    write(REPORT, result); print('BLUR_EXPOSURE_TRAIN_VERIFIED '+json.dumps({k:v for k,v in result.items() if k not in ('history','workers','recipe')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('phase', choices=['prepare','run','finish'])
    globals()[parser.parse_args().phase]()
