#!/usr/bin/env python3
"""Can the current native delta mapping fit eight fixed training inputs? Fresh100 only."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'third_party/flux2/src'))
from smoke import run_foundir_fullstack_blur_coverage_v2 as t
from smoke.blur_fixed8_fit_contract_v1 import STEPS, SNAPSHOTS, select_items, summarize
from smoke.existing_flux_replay_contract_v1 import read, sha, write_new as write

OUT = ROOT/'checkpoints/foundir_blur_fixed8_fit_v1'
REPORT = ROOT/'reports/foundir_blur_fixed8_fit_v1_result_2026-11-07.json'
SOURCES = ['smoke/train_foundir_blur_fixed8_fit_v1.py', 'smoke/blur_fixed8_fit_contract_v1.py',
           'smoke/run_foundir_fullstack_blur_fixed8_fit_v1.sh']
need = t.need
RECIPE = {'steps': STEPS, 'snapshots': list(SNAPSHOTS), 'train': 8, 'global_batch': 8,
          'world': 2, 'node': 3, 'learning_rate': 1e-5, 'loss': '10*MSE_RGB01 + .1*LPIPS',
          'weight_decay': 0., 'clip_grad': 1., 'initialization': 'original_Blur_step10000_fresh_Adam',
          'selection': 'first8_original1024_order_no_metric_filter', 'crop': 512,
          'exposures_per_source': 100, 'official_pixels': False, 'scope': 'training_fit_only',
          'quality_gate': False, 'automatic_official_or_continuation': False}


def prepare():
    started = time.monotonic(); b = t.bound(); items = select_items(b['rows'])
    actual = ROOT/'reports/foundir_blur_actual_updates_v1_result_2026-11-07.json'
    need(read(actual)['status'] == 'two_actual_historical_updates_reproduced', 'actual updates audit required')
    for item in items:
        for side in ('lq', 'gt'):
            need(t.parent.crop(item, side).shape == (512, 512, 3), 'fixed crop changed')
        need(t.cached(item, 'cpu') is not None, 'original reference cache required')
    for name, digest in b['assets'].items(): need(sha(ROOT/name) == digest, 'asset changed')
    from smoke.official_vlm_v4_metric_identity_fix import checked_metric_models
    checked_metric_models('cpu', b['metric_identity'])
    need(shutil.disk_usage(ROOT).free > 55*1024**3, 'need55GiB; preserve old assets')
    value = {'recipe': RECIPE, 'items': items, 'train_receipt_sha256': sha(t.OUT/'receipt.json'),
             'actual_update_result_sha256': sha(actual),
             'initial_parameter_hash': read(t.OUT/'pilot/steps/step_000000.json')['parameter_hash'],
             'sources': {n: sha(ROOT/n) for n in sorted(set(b['sources']) | set(SOURCES))}}
    write(OUT/'receipt.json', value)
    result = {'status': 'destination_CPU_preflight_pass', 'seconds': time.monotonic()-started,
              'exit_code': 0, 'GPU_hours': 0, 'sources': 8, 'images_hashed_and_crop_checked': 16,
              'reference_caches_checked': 8, 'recipe': RECIPE, 'receipt_sha256': sha(OUT/'receipt.json')}
    write(ROOT/'reports/foundir_blur_fixed8_fit_v1_preflight_2026-11-07.json', result)
    print('BLUR_FIXED8_PREFLIGHT_PASS '+json.dumps(result), flush=True)


def bound():
    b = read(OUT/'receipt.json'); original = t.bound()
    need(b['recipe'] == RECIPE and b['items'] == select_items(original['rows']), 'fixed scope changed')
    need(b['train_receipt_sha256'] == sha(t.OUT/'receipt.json'), 'parent changed')
    need(b['actual_update_result_sha256'] == sha(ROOT/'reports/foundir_blur_actual_updates_v1_result_2026-11-07.json'), 'audit changed')
    for name, digest in b['sources'].items(): need(sha(ROOT/name) == digest, 'source changed: '+name)
    return b, original


def run():
    import torch
    import torch.distributed as dist
    from datetime import timedelta
    from lowlight_restoration.flux2_endpoint_precision import assert_fp32_optimizer
    from smoke.official_vlm_v4_metric_identity_fix import checked_metric_models, producer
    begun = time.monotonic(); b, original = bound(); rank, device = t.setup(original)
    dist.init_process_group('nccl', timeout=timedelta(minutes=15))
    need(dist.get_world_size() == 2 and dist.get_rank() == rank, 'single node two ranks only')
    receipt = sha(OUT/'receipt.json'); model = t.load_model(original, device)
    ae, prompt = t.load_vae_prompt(original, device)
    metrics, _ = checked_metric_models(device, original['metric_identity']); metrics['lpips'].as_loss = True
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = t.optimizer_for(model)
    saved = t.state.load(OUT, receipt, 2)
    start, history = (0, []) if saved is None else (saved['step'], saved['history'])
    need(start <= STEPS, 'fixed100 exceeded')
    if saved is not None: t.state.restore(model, optimizer, saved, rank, device)
    else: need(t.state.parameter_hash(t.state.trainable_state(model)) == b['initial_parameter_hash'], 'fresh original initialization required')
    del saved

    def save(step):
        rngs, hashes = [None]*2, [None]*2
        dist.all_gather_object(rngs, t.state.rng_state(device))
        digest = t.state.parameter_hash(t.state.trainable_state(model)); dist.all_gather_object(hashes, digest)
        need(len(set(hashes)) == 1, 'rank parameters diverged')
        if rank == 0:
            marker = t.state.save(OUT, model, optimizer, step=step, world=2, rank_rng=rngs,
                                  receipt_sha=receipt, history=history)
            print('BLUR_FIXED8_SAVED '+json.dumps(marker), flush=True)
        dist.barrier()

    def evaluate(step):
        rng = t.state.rng_state(device); before = t.state.parameter_hash(t.state.trainable_state(model))
        need(before == read(OUT/'steps'/f'step_{step:06d}.json')['parameter_hash'], 'snapshot differs from saved step')
        model.eval(); local = []
        try:
            for index in range(rank, 8, 2):
                item = b['items'][index]; data = t.cached(item, device)
                with torch.no_grad():
                    pred = t.prediction(model, ae, prompt, data); need(torch.isfinite(pred).all(), 'nonfinite snapshot')
                    array = t.core.uint8_output(pred)
                    if step == 0: need(torch.equal(pred, data['lq']), 'initial delta not exact LQ')
                    file = OUT/'evaluations'/f'step_{step:06d}'/(t.parent.key(item)+'.png')
                    producer.png_once(file, array[0].permute(1, 2, 0).cpu().numpy())
                    gt = (t.rgb(item, 'gt', device)+1)*.5
                    lq = t.core.uint8_output(data['lq']).float()/255
                    values = {name: {k: float(m(image, gt).mean()) for k, m in metrics.items()}
                              for name, image in [('lq', lq), ('candidate', array.float()/255)]}
                record = {'index': index, 'item': item, 'rank': rank, 'step': step,
                          'parameter_hash': before, 'receipt_sha256': receipt, 'scope': 'training_fit_only',
                          'metrics': values, 'output': {'path': str(file.relative_to(OUT)), 'sha256': sha(file)}}
                write(file.with_suffix('.json'), record); local.append(record)
                del data, pred, array, gt, lq
            gathered = [None]*2; dist.all_gather_object(gathered, local)
            summary = summarize([r for group in gathered for r in group], b['items'], step, before, receipt)
            need(t.state.parameter_hash(t.state.trainable_state(model)) == before, 'evaluation changed parameters')
            if rank == 0:
                write(OUT/'evaluations'/f'step_{step:06d}_summary.json', summary)
                print('BLUR_FIXED8_EVALUATION '+json.dumps(summary), flush=True)
        finally:
            t.state.restore_rng(rng, device); model.train()
        dist.barrier()

    if not list((OUT/'steps').glob('step_*.json')): save(0)
    if start in SNAPSHOTS: evaluate(start)
    for step in range(start, STEPS):
        tick = time.monotonic(); optimizer.zero_grad(set_to_none=True); model.train()
        totals = torch.zeros(3, device=device, dtype=torch.float64)
        for index in t.core.sample_indices(step, rank, count=8):
            item = b['items'][index]; data = t.cached(item, device); gt = t.rgb(item, 'gt', device)
            prediction = t.prediction(model, ae, prompt, data)
            loss, terms = t.core.objective(prediction, gt, metrics['lpips'])
            need(torch.isfinite(loss) and loss.requires_grad, 'invalid training loss')
            (loss/8).backward()
            totals += torch.tensor([float(loss.detach()), float(terms['mse'].detach()), float(terms['lpips'].detach())], device=device, dtype=torch.float64)/8
            del data, gt, prediction, loss, terms
        for p in parameters:
            need(p.grad is not None, 'missing LoRA gradient'); dist.all_reduce(p.grad)
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
        need(float(norm) > 0, 'zero gradient'); optimizer.step(); assert_fp32_optimizer(optimizer)
        optimizer.zero_grad(set_to_none=True)
        need(all(torch.isfinite(p).all() for p in parameters), 'nonfinite parameters')
        dist.all_reduce(totals)
        speed = torch.tensor(time.monotonic()-tick, device=device); dist.all_reduce(speed, op=dist.ReduceOp.MAX)
        entry = {'step': step+1, 'loss': float(totals[0]), 'mse': float(totals[1]), 'lpips': float(totals[2]),
                 'gradient_norm': float(norm), 'next_global_slot': (step+1)*8, 'max_rank_seconds': float(speed)}
        history.append(entry)
        if rank == 0: print('BLUR_FIXED8_TRAIN '+json.dumps(entry), flush=True)
        save(step+1)
        if step+1 in SNAPSHOTS: evaluate(step+1)
    write(OUT/f'worker{rank}.json', {'rank': rank, 'step': STEPS, 'receipt_sha256': receipt,
          'parameter_hash': t.state.parameter_hash(t.state.trainable_state(model)),
          'seconds': time.monotonic()-begun, 'peak_allocated_GiB': torch.cuda.max_memory_allocated(device)/1024**3,
          'exit_code': 0, 'scope': 'training_fit_only'})
    dist.barrier(); dist.destroy_process_group(); print(f'BLUR_FIXED8_END rank={rank} status=0', flush=True)


def finish():
    begun = time.monotonic(); b, _ = bound(); receipt = sha(OUT/'receipt.json')
    saved = t.state.load(OUT, receipt, 2); need(saved['step'] == STEPS, 'fixed100 incomplete')
    for rank in (0, 1):
        w = read(OUT/f'worker{rank}.json')
        need(w['rank'] == rank and w['step'] == STEPS and w['receipt_sha256'] == receipt
             and w['parameter_hash'] == saved['parameter_hash'] and w['exit_code'] == 0, 'worker incomplete')
    snapshots = {}; all_records = []
    for step in SNAPSHOTS:
        records = [read(OUT/'evaluations'/f'step_{step:06d}'/(t.parent.key(item)+'.json')) for item in b['items']]
        digest = read(OUT/'steps'/f'step_{step:06d}.json')['parameter_hash']
        summary = summarize(records, b['items'], step, digest, receipt)
        need(summary == read(OUT/'evaluations'/f'step_{step:06d}_summary.json'), 'independent snapshot means differ')
        for r in records: need(sha(OUT/r['output']['path']) == r['output']['sha256'], 'PNG changed')
        snapshots[str(step)] = summary; all_records.extend(records)
    need(snapshots['0']['means']['candidate'] == snapshots['0']['means']['lq'], 'initialization not LQ')
    fit = read(ROOT/'reports/foundir_blur_coverage_fit_v2_result_2026-11-06.json')
    previous = []
    for item in b['items']:
        matches = [r for r in fit['records'] if r['item'] == item]; need(len(matches) == 1, 'previous same8 missing')
        previous.append(matches[0])
    import math
    old_means = {k: math.fsum(r['metrics_training_fit_only']['candidate'][k] for r in previous)/8 for k in ('psnr', 'lpips', 'ssim')}
    result = {'status': 'fixed8_fresh100_training_fit_complete', 'scope': 'training_fit_only', 'quality_gate': False,
              'official_Blur_pass': False, 'official_pixels_accessed': False, 'recipe': RECIPE,
              'snapshots': snapshots, 'records': all_records, 'training_history': saved['history'],
              'prior1024_step200_same8_means': old_means, 'receipt_sha256': receipt,
              'seconds': time.monotonic()-begun, 'exit_code': 0, 'GPU_hours': 0}
    write(REPORT, result)
    print('BLUR_FIXED8_VERIFIED '+json.dumps({'final': snapshots[str(STEPS)], 'prior1024_step200_same8': old_means}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('phase', choices=['prepare', 'run', 'finish'])
    globals()[parser.parse_args().phase]()
