#!/usr/bin/env python3
"""Fixed200 diagnostic: can one training frame fit all16 contexts simultaneously?"""
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
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'third_party/flux2/src'))
from smoke import run_foundir_fullstack_blur_coverage_v2 as t
from smoke import train_foundir_blur_multiview_v1 as parent
from smoke import blur_singleframe_state_v1 as state
from smoke.blur_singleframe_contract_v1 import views, key, summarize
from smoke import official_vlm_router_v4_contract as geometry
from smoke.existing_flux_replay_contract_v1 import read, sha, write_new as write

OUT = ROOT/'checkpoints/foundir_blur_singleframe_fit_v1'
CONTEXT = ROOT/'checkpoints/foundir_blur_multiview_context_v1'
REPORT = ROOT/'reports/foundir_blur_singleframe_fit_v1_result_2026-11-07.json'
START, END, WORLD, SNAPSHOTS = state.START, state.END, state.WORLD, state.SNAPSHOTS
SOURCES = ['smoke/train_foundir_blur_singleframe_fit_v1.py', 'smoke/blur_singleframe_state_v1.py',
           'smoke/blur_singleframe_contract_v1.py', 'smoke/run_foundir_fullstack_blur_singleframe_fit_v1.sh']
need = t.need
RECIPE = {'start_step': START, 'end_step': END, 'new_steps': 200, 'snapshots': list(SNAPSHOTS),
          'node': 3, 'world': WORLD, 'sources': 1, 'views': 16, 'global_batch': 8,
          'exposures_per_view': 100, 'selection': 'first_original1024_source_no_metric_filter',
          'sample_order': '16_views_epoch_shuffle_without_replacement_seed61903',
          'initialization': 'complete1792_model_Adam; world4_to2_explicit_new_rank_RNG_seed61903',
          'bitwise_parent_continuation': False, 'same_world2_resume': True,
          'learning_rate': 1e-5, 'weight_decay': 0., 'clip_grad': 1.,
          'loss': '10*MSE_RGB01 + .1*LPIPS', 'input_normalization': 'official_CPU_FP32',
          'scope': 'training_fit_only', 'quality_gate': False, 'may_seed_formal': False,
          'official_pixels': False, 'automatic_official_or_continuation': False,
          'failure_stop': 'nonfinite/zero_gradient/hash_mismatch/OOM/initial_pixel_mismatch; no_auto_retry'}


def png(path):
    import numpy as np
    from PIL import Image
    with Image.open(path) as image:
        return np.array(image.convert('RGB'), copy=True)


def prepare():
    begun = time.monotonic(); original = parent.bound(); items = views(original['rows'])
    saved, marker = parent.state.load_step(parent.OUT, START, sha(parent.OUT/'receipt.json'))
    need(saved['step'] == START and len(saved['optimizer']['state']) == 100, 'complete1792 required')
    context = read(CONTEXT/'receipt.json')
    old = read(CONTEXT/'records'/(t.parent.key(original['rows'][0])+'.json'))
    need(context['items'][0] == original['rows'][0] and old['item'] == original['rows'][0]
         and old['parameter_hash'] == saved['parameter_hash'], 'same frame and frozen1792 required')
    initial_outputs = {}
    for item in items:
        if item['kind'] == 'deployment_grid':
            r = read(CONTEXT/'tiles'/(item['tile_key']+'.json'))
            need(r['identity']['item'] == original['rows'][0] and r['GT_input'] is False,
                 'context tile identity changed')
            spec = r['png']
        else:
            spec = old['uncached_crop']['output']
        need(sha(CONTEXT/spec['path']) == spec['sha256'], 'prior frozen1792 output changed')
        initial_outputs[key(item)] = spec
    need(sha(CONTEXT/old['output']['path']) == old['output']['sha256'], 'prior full image changed')
    for side in ('lq', 'gt'):
        row = items[0]['row']; file = ROOT/row[side+'_path']
        need(sha(file) == row[side+'_sha256'] and png(file).shape == (1080, 1920, 3), 'training file changed')
    for name, digest in original['assets'].items(): need(sha(ROOT/name) == digest, 'asset changed')
    from smoke.official_vlm_v4_metric_identity_fix import checked_metric_models
    checked_metric_models('cpu', original['metric_identity'])
    need(shutil.disk_usage(ROOT).free > 100*1024**3, 'need100GiB while preserving all assets')
    receipt = {k: original[k] for k in ('assets', 'model', 'lora', 'prompt', 'metric_identity')}
    receipt.update(recipe=RECIPE, items=items, parent_checkpoint=marker,
                   parent_receipt_sha256=sha(parent.OUT/'receipt.json'),
                   context_receipt_sha256=sha(CONTEXT/'receipt.json'),
                   initial_outputs=initial_outputs, initial_full_output=old['output'],
                   sources={n: sha(ROOT/n) for n in sorted(set(original['sources']) | set(SOURCES))})
    write(OUT/'receipt.json', receipt)
    value = {'status': 'destination_CPU_preflight_pass', 'seconds': time.monotonic()-begun,
             'recipe': RECIPE, 'initial16_PNG_and_full_SHA_verified': True,
             'parent_model_Adam_four_rank_RNG_SHA_verified': True,
             'receipt_sha256': sha(OUT/'receipt.json'), 'exit_code': 0, 'GPU_hours': 0}
    write(ROOT/'reports/foundir_blur_singleframe_fit_v1_preflight_2026-11-07.json', value)
    print('BLUR_SINGLEFRAME_PREFLIGHT_PASS '+json.dumps(value), flush=True)


def bound():
    b = read(OUT/'receipt.json')
    need(b['recipe'] == RECIPE and b['items'] == views(parent.bound()['rows']), 'fixed scope changed')
    need(b['parent_receipt_sha256'] == sha(parent.OUT/'receipt.json')
         and b['parent_checkpoint'] == read(parent.OUT/'steps'/f'step_{START:06d}.json')
         and b['context_receipt_sha256'] == sha(CONTEXT/'receipt.json'), 'parent changed')
    for name, digest in b['sources'].items(): need(sha(ROOT/name) == digest, 'source changed: '+name)
    for spec in [*b['initial_outputs'].values(), b['initial_full_output']]:
        need(sha(CONTEXT/spec['path']) == spec['sha256'], 'initial comparison changed')
    return b


def rgb(item, side, device):
    from smoke.evaluate_foundir_official_vlm_router_v4 import tensor
    return tensor(t.parent.crop(item, side), device)


def cached(item, device):
    import torch
    from safetensors.torch import load_file
    marker = OUT/'cache'/(key(item)+'.json')
    if not marker.exists(): return None
    r = read(marker); file = OUT/r['tensor']
    need(r['item'] == item and r['receipt_sha256'] == sha(OUT/'receipt.json') and r['GT_input'] is False,
         'cache identity changed')
    need(file.resolve().parent == (OUT/'tensors').resolve() and sha(file) == r['sha256'], 'cache SHA/path changed')
    data = load_file(str(file), device=str(device))
    need(set(data) == {'lq', 'latent', 'reference_velocity', 'reference_decode'}, 'cache schema changed')
    need(all(v.dtype == torch.float32 and torch.isfinite(v).all() for v in data.values()), 'invalid cache')
    need(data['lq'].shape == data['reference_decode'].shape == (1, 3, 512, 512)
         and data['latent'].shape == data['reference_velocity'].shape, 'cache shapes changed')
    return data


def cache_views(b, model, ae, prompt, rank, device):
    import torch
    from safetensors.torch import save_file
    from flux2.sampling import batched_prc_img
    model.eval()
    for item in b['items'][rank::2]:
        if cached(item, device) is not None: continue
        with torch.no_grad():
            lq = rgb(item, 'lq', device); latent = ae.encode(lq).float()
            data = {'lq': lq, 'latent': latent,
                    'reference_velocity': t.core.native_velocity(model, latent, prompt, batched_prc_img),
                    'reference_decode': ae.decode(latent).float()}
        need(all(v.dtype == torch.float32 and torch.isfinite(v).all() for v in data.values()), 'invalid reference')
        directory = OUT/'tensors'; directory.mkdir(parents=True, exist_ok=True)
        file = directory/(key(item)+f'_{time.time_ns()}.safetensors')
        save_file({k: v.cpu().contiguous() for k, v in data.items()}, str(file))
        with file.open('rb') as handle: os.fsync(handle.fileno())
        write(OUT/'cache'/(key(item)+'.json'), {'item': item, 'rank': rank, 'GT_input': False,
              'receipt_sha256': sha(OUT/'receipt.json'), 'tensor': str(file.relative_to(OUT)), 'sha256': sha(file)})
        print(f'BLUR_SINGLEFRAME_CACHE rank={rank} view={item["view"]}', flush=True)


def run():
    import numpy as np
    import torch
    import torch.distributed as dist
    from lowlight_restoration.flux2_endpoint_precision import assert_fp32_optimizer
    from smoke.official_vlm_v4_metric_identity_fix import checked_metric_models, producer
    begun = time.monotonic(); b = bound(); rank, device = t.setup(b)
    need(int(os.environ['FOUNDIR_FULLSTACK_NODE']) == 3 and int(os.environ['WORLD_SIZE']) == 2,
         'physical node3 two ranks only')
    dist.init_process_group('nccl', device_id=device, timeout=timedelta(minutes=15))
    need(dist.get_world_size() == WORLD and dist.get_rank() == rank, 'rank topology differs')
    receipt = sha(OUT/'receipt.json'); model = t.load_model(b, device)
    ae, prompt = t.load_vae_prompt(b, device)
    need(not any(isinstance(m, torch.nn.Dropout) and m.p > 0 for m in model.modules()), 'unexpected dropout')
    cache_views(b, model, ae, prompt, rank, device); dist.barrier()
    metrics, _ = checked_metric_models(device, b['metric_identity']); metrics['lpips'].as_loss = True
    optimizer = t.optimizer_for(model); parameters = [p for p in model.parameters() if p.requires_grad]
    saved, _ = state.latest(OUT, receipt); fresh = saved is None
    if fresh:
        saved, _ = parent.state.load_step(parent.OUT, START, b['parent_receipt_sha256'])
        state.restore_parameters_adam(model, optimizer, saved)
        random.seed(61903+rank); torch.manual_seed(61903+rank); torch.cuda.manual_seed(61903+rank)
    else:
        state.restore(model, optimizer, saved, rank, device)
    start, history = saved['step'], saved['history']; del saved
    starts = [None]*WORLD; dist.all_gather_object(starts, (start, fresh))
    need(len(set(starts)) == 1, 'rank resume differs')
    print(f'BLUR_SINGLEFRAME_RESTORED rank={rank} step={start} model_Adam_preserved=true new_world2_RNG={fresh}', flush=True)

    def save(step):
        rngs, hashes = [None]*WORLD, [None]*WORLD
        dist.all_gather_object(rngs, t.state.rng_state(device))
        digest = t.state.parameter_hash(t.state.trainable_state(model)); dist.all_gather_object(hashes, digest)
        need(len(set(hashes)) == 1, 'rank parameters diverged')
        if rank == 0:
            marker = t.state.save(OUT, model, optimizer, step=step, world=WORLD, rank_rng=rngs,
                                  receipt_sha=receipt, history=history)
            print('BLUR_SINGLEFRAME_SAVED '+json.dumps(marker), flush=True)
        dist.barrier()

    def tensor01(array):
        return torch.from_numpy(array.copy()).permute(2, 0, 1).unsqueeze(0).to(device).float()/255

    def score(array, gt):
        with torch.no_grad():
            image, target = tensor01(array), tensor01(gt)
            return {k: float(m(image, target).mean()) for k, m in metrics.items()}

    def evaluate(step):
        rng = t.state.rng_state(device); before = t.state.parameter_hash(t.state.trainable_state(model))
        need(before == read(OUT/'steps'/f'step_{step:06d}.json')['parameter_hash'], 'snapshot differs from step')
        model.eval(); records = []
        try:
            for index in range(rank, 16, WORLD):
                item = b['items'][index]; data = cached(item, device)
                with torch.no_grad():
                    pred = t.prediction(model, ae, prompt, data)
                    need(torch.isfinite(pred).all(), 'nonfinite snapshot')
                    array = t.core.uint8_output(pred)[0].permute(1, 2, 0).cpu().numpy()
                if step == START:
                    need(np.array_equal(array, png(CONTEXT/b['initial_outputs'][key(item)]['path'])),
                         f'initial frozen1792 pixel mismatch at view{index}')
                file = OUT/'evaluations'/f'step_{step:06d}'/(key(item)+'.png')
                producer.png_once(file, array)
                # Prediction is fixed before GT scoring; never use GT to pick outputs.
                gt = t.parent.crop(item, 'gt'); lq = t.parent.crop(item, 'lq')
                r = {'index': index, 'item': item, 'rank': rank, 'step': step,
                     'parameter_hash': before, 'receipt_sha256': receipt, 'scope': 'training_fit_only',
                     'metrics': {'lq': score(lq, gt), 'candidate': score(array, gt)},
                     'output': {'path': str(file.relative_to(OUT)), 'sha256': sha(file)}}
                write(file.with_suffix('.json'), r); records.append(r)
                del data, pred, array
            gathered = [None]*WORLD; dist.all_gather_object(gathered, records)
            records = sorted([r for group in gathered for r in group], key=lambda r: r['index'])
            summary = summarize(records, b['items'], step, before, receipt)
            if rank == 0:
                tiles = {r['item']['tile_key']: png(OUT/r['output']['path']) for r in records[:15]}
                row = b['items'][0]['row']; full = geometry.stitch(row, tiles)
                if step == START:
                    need(np.array_equal(full, png(CONTEXT/b['initial_full_output']['path'])), 'initial full image differs')
                file = OUT/'evaluations'/f'step_{step:06d}_full.png'; producer.png_once(file, full)
                gt, lq = png(ROOT/row['gt_path']), png(ROOT/row['lq_path'])
                summary['full'] = {'lq': score(lq, gt), 'candidate': score(full, gt),
                                   'output': {'path': str(file.relative_to(OUT)), 'sha256': sha(file)}}
                write(OUT/'evaluations'/f'step_{step:06d}_summary.json', summary)
                print('BLUR_SINGLEFRAME_EVALUATION '+json.dumps(summary), flush=True)
            need(t.state.parameter_hash(t.state.trainable_state(model)) == before, 'evaluation changed weights')
        finally:
            t.state.restore_rng(rng, device); model.train()
        dist.barrier()

    if fresh: save(START)
    if start in SNAPSHOTS: evaluate(start)
    for step in range(start, END):
        tick = time.monotonic(); optimizer.zero_grad(set_to_none=True); model.train()
        totals = torch.zeros(3, dtype=torch.float64, device=device)
        for index in state.indices(step, rank):
            item = b['items'][index]; data = cached(item, device); gt = rgb(item, 'gt', device)
            pred = t.prediction(model, ae, prompt, data); loss, terms = t.core.objective(pred, gt, metrics['lpips'])
            need(torch.isfinite(loss) and loss.requires_grad, 'invalid loss'); (loss/8).backward()
            totals += torch.tensor([float(loss.detach()), float(terms['mse'].detach()), float(terms['lpips'].detach())],
                                   device=device, dtype=torch.float64)/8
            del data, gt, pred, loss, terms
        for p in parameters:
            need(p.grad is not None, 'missing LoRA gradient'); dist.all_reduce(p.grad)
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
        need(float(norm) > 0, 'zero gradient'); optimizer.step(); assert_fp32_optimizer(optimizer)
        optimizer.zero_grad(set_to_none=True)
        need(all(torch.isfinite(p).all() for p in parameters), 'nonfinite parameters')
        dist.all_reduce(totals)
        speed = torch.tensor(time.monotonic()-tick, device=device); dist.all_reduce(speed, op=dist.ReduceOp.MAX)
        entry = {'step': step+1, 'diagnostic_step': step+1-START, 'scope': 'training_fit_only',
                 'loss': float(totals[0]), 'mse': float(totals[1]), 'lpips': float(totals[2]),
                 'gradient_norm': float(norm), 'next_global_slot': (step+1)*8,
                 'max_rank_seconds': float(speed), 'world': WORLD, 'global_batch': 8}
        history.append(entry)
        if rank == 0: print('BLUR_SINGLEFRAME_TRAIN '+json.dumps(entry), flush=True)
        save(step+1)
        if step+1 in SNAPSHOTS: evaluate(step+1)
    write(OUT/f'worker{rank}.json', {'rank': rank, 'step': END, 'exit_code': 0,
          'receipt_sha256': receipt, 'parameter_hash': t.state.parameter_hash(t.state.trainable_state(model)),
          'seconds': time.monotonic()-begun, 'peak_allocated_GiB': torch.cuda.max_memory_allocated(device)/1024**3,
          'scope': 'training_fit_only'})
    dist.barrier(); dist.destroy_process_group(); print(f'BLUR_SINGLEFRAME_END rank={rank} status=0', flush=True)


def finish():
    import numpy as np
    begun = time.monotonic(); b = bound(); receipt = sha(OUT/'receipt.json')
    saved, marker = state.latest(OUT, receipt)
    need(saved is not None and saved['step'] == END, 'fixed200 incomplete')
    old, _ = parent.state.load_step(parent.OUT, START, b['parent_receipt_sha256'])
    need(saved['history'][:START] == old['history'], 'parent history changed')
    del old
    workers = [read(OUT/f'worker{rank}.json') for rank in range(WORLD)]
    for rank, w in enumerate(workers):
        need(w['rank'] == rank and w['step'] == END and w['exit_code'] == 0
             and w['receipt_sha256'] == receipt and w['parameter_hash'] == saved['parameter_hash'], 'worker incomplete')
    snapshots, all_records = {}, []
    for step in SNAPSHOTS:
        records = [read(OUT/'evaluations'/f'step_{step:06d}'/(key(i)+'.json')) for i in b['items']]
        digest = read(OUT/'steps'/f'step_{step:06d}.json')['parameter_hash']
        summary = summarize(records, b['items'], step, digest, receipt)
        actual = read(OUT/'evaluations'/f'step_{step:06d}_summary.json')
        need(summary == {k: v for k, v in actual.items() if k != 'full'}, 'independent snapshot means differ')
        for r in records: need(sha(OUT/r['output']['path']) == r['output']['sha256'], 'PNG changed')
        spec = actual['full']['output']; need(sha(OUT/spec['path']) == spec['sha256'], 'full PNG changed')
        full = geometry.stitch(b['items'][0]['row'],
                               {r['item']['tile_key']: png(OUT/r['output']['path']) for r in records[:15]})
        need(np.array_equal(full, png(OUT/spec['path'])), 'full stitching differs')
        if step == START:
            for r in records:
                need(np.array_equal(png(OUT/r['output']['path']), png(CONTEXT/b['initial_outputs'][key(r['item'])]['path'])),
                     'initial1792 output reproduction failed')
        snapshots[str(step)] = actual; all_records.extend(records)
    result = {'status': 'singleframe16_context_fixed200_training_fit_complete', 'recipe': RECIPE,
              'scope': 'training_fit_only', 'quality_gate': False, 'may_seed_formal': False,
              'official_Blur_pass': False, 'official_pixels_accessed': False, 'workers': workers,
              'snapshots': snapshots, 'records': all_records, 'new_training_history': saved['history'][START:],
              'final_checkpoint': marker, 'receipt_sha256': receipt, 'parent_history_unchanged': True,
              'complete_checkpoint_receipts': END-START+1, 'all_large_checkpoints_rehashed': False,
              'seconds': time.monotonic()-begun, 'exit_code': 0, 'GPU_hours': 0}
    write(REPORT, result)
    print('BLUR_SINGLEFRAME_VERIFIED '+json.dumps({s: v['full'] for s, v in snapshots.items()}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('phase', choices=['prepare', 'run', 'finish'])
    globals()[parser.parse_args().phase]()
