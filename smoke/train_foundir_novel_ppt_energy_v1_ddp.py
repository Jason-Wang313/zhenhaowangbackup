#!/usr/bin/env python3
"""Full PPT retry: frozen strong baseline, action ranking AND energy gradients."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from lowlight_restoration.novel_ppt_energy import NovelPPTEnergy, objective
from smoke.dual_retry_runtime import CONFIG, ROOT, bound_json, initialize, sha, sources, write_once
from smoke.train_foundir_proposal_relative_energy_noise_v1_ddp import (
    fixed_validation_rows, load_audit_crop, load_frozen_proposal, load_task_rows,
    load_training_patch, sample_global_batch,
)
from smoke.train_foundir_proposal_relative_energy_noise_v2_ddp import build_lpips, run_frozen
from smoke.train_foundir_complete_candidate_energy_noise_v1_ddp import (
    load_residual_basis, load_negative_models, severity_from_rows,
)


def contract():
    config = json.loads(CONFIG.read_text()); spec = config['ppt']
    base = bound_json(spec['basis_config'], spec['basis_config_sha256'])
    candidates = bound_json(spec['candidate_config'], spec['candidate_config_sha256'])
    if sha(ROOT/base['data']['manifest']) != base['data']['manifest_sha256']:
        raise RuntimeError('Noise manifest changed')
    for binding in (base['frozen_proposal'], candidates['frozen_residual_basis']):
        if sha(ROOT/binding['checkpoint']) != binding['checkpoint_sha256']:
            raise RuntimeError('frozen model checkpoint changed')
    rows = load_task_rows(base)  # internal split metadata only, NEVER its pixels
    # Strengthen existing source isolation with exact target-hash exclusion.
    blocked = {r['gt_sha256'] for r in rows['validation'] + rows['internal_test']}
    if any(r['gt_sha256'] in blocked for r in rows['train']):
        raise RuntimeError('training clean hash overlaps a held-out split')
    validation = fixed_validation_rows(rows['validation'], base['data']['validation_subset_records'], base['data']['validation_crop_seed'])
    return config, base, candidates, rows['train'], validation


def validate_receipt(path, config_sha, binding):
    receipt = json.loads(path.read_text())
    if not (receipt['status'] == 'complete' and receipt['mode'] == 'qualification'
            and receipt['engineering_pass'] and receipt['steps'] == 25
            and receipt['config_sha256'] == config_sha and receipt['source_sha256'] == binding):
        raise RuntimeError('formal requires a matching successful engineering qualification')
    checkpoint = path.parent / receipt['checkpoint']
    if sha(checkpoint) != receipt['checkpoint_sha256']:
        raise RuntimeError('qualification checkpoint identity changed')


def frozen_outputs(proposal_model, basis, observation, negative_models=None):
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        # Match the historical Fidelity contract, including FP32 proposal
        # addition in the residual basis (BF16 here silently changes quality).
        proposal = run_frozen(proposal_model, observation).add(1).mul(.5)
        fidelity, perceptual = basis(observation, proposal)
        negatives = []
        if negative_models is not None:
            negatives = [model(observation).clamp(0, 1) for model in negative_models.values()]
            negatives.append(proposal_model(proposal).clamp(0, 1))
    # Operations outside inference_mode create normal tensors for autograd.
    values = [value.float().mul(2).sub(1) for value in (proposal, fidelity, perceptual, *negatives)]
    return values


def evaluate(model, proposal_model, basis, rows, base, perceptual, device, rank):
    model.eval()
    local = []
    for index in range(rank, len(rows), 8):
        row = rows[index]
        if row['split'] != 'validation':
            raise RuntimeError('evaluation requires allowed development rows only')
        lq, gt, _, _ = load_audit_crop(row, crop_size=256, seed=base['data']['validation_crop_seed'])
        observation, target = lq[None].to(device), gt[None].to(device)
        proposal, anchor, _ = frozen_outputs(proposal_model, basis, (observation+1)*.5)
        descriptor = severity_from_rows([row], device)
        with torch.enable_grad():
            trajectory, _, _ = model.trajectory(observation, proposal, anchor, descriptor, create_graph=False)
        trajectory = tuple(t.detach() for t in trajectory)
        selected, selected_index, _ = model.select(observation, proposal, anchor, descriptor, trajectory)
        values = dict(zip(('proposal', 'fidelity', 'energy_1', 'energy_2', 'energy_3', 'selected'),
                          (proposal, anchor, *trajectory, selected), strict=True))
        record = {'record_id': row['record_id'], 'source_group': row['source_group'], 'selected_step': int(selected_index)}
        with torch.no_grad():
            for name, value in values.items():
                if not bool(torch.isfinite(value).all()):
                    raise RuntimeError('nonfinite evaluation output')
                mse = ((value-target)*.5).square().mean()
                record[name] = {'psnr': float(-10*torch.log10(mse.clamp_min(1e-12))),
                                'lpips': float(perceptual(value, target).mean())}
        local.append(record)
    gathered = [None]*8; dist.all_gather_object(gathered, local)
    model.train()
    if rank:
        return None
    records = sorted([r for shard in gathered for r in shard], key=lambda r: r['record_id'])
    if len(records) != len(rows) or {r['record_id'] for r in records} != {r['record_id'] for r in rows}:
        raise RuntimeError('evaluation records missing or duplicated')
    metrics = {name: {metric: sum(r[name][metric] for r in records)/len(records) for metric in ('psnr', 'lpips')}
               for name in ('proposal', 'fidelity', 'energy_1', 'energy_2', 'energy_3', 'selected')}
    return {'records': len(records), 'metrics': metrics, 'rows': records, 'GT_used_for_selection': False,
            'selected_psnr_gain_over_fidelity': metrics['selected']['psnr']-metrics['fidelity']['psnr'],
            'selected_lpips_change_from_fidelity': metrics['selected']['lpips']-metrics['fidelity']['lpips']}


def save_checkpoint(output, model, optimizer, step, mode, config_sha, binding, rank):
    rng = {'cpu': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state().cpu(), 'python': random.getstate()}
    gathered = [None]*8; dist.all_gather_object(gathered, rng)
    if rank:
        return None
    path = output/f'step_{step:06d}.pt'
    temporary = path.with_suffix('.partial')
    if path.exists() or temporary.exists():
        raise RuntimeError('preserve existing checkpoint')
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    if any(not bool(torch.isfinite(v).all()) for v in state.values()):
        raise RuntimeError('nonfinite model state')
    for value in optimizer.state.values():
        for name in ('exp_avg', 'exp_avg_sq'):
            if name not in value or not bool(torch.isfinite(value[name]).all()):
                raise RuntimeError('invalid Adam state')
    with temporary.open('xb') as stream:
        torch.save({'model': state, 'optimizer': optimizer.state_dict(), 'step': step, 'mode': mode,
                    'config_sha256': config_sha, 'source_sha256': binding,
                    'rank_rng': gathered, 'next_sample_step': step+1, 'world_size': 8}, stream)
    os.replace(temporary, path)
    return {'checkpoint': path.name, 'checkpoint_sha256': sha(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', default='qualification_v1')
    parser.add_argument('--mode', choices=('qualification', 'formal'), default='qualification')
    parser.add_argument('--qualification', type=Path)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    config, base, candidate_spec, training, validation = contract()
    spec, binding, config_sha = config['ppt'], sources(), sha(CONFIG)
    if args.preflight:
        print('NOVEL_PPT_PREFLIGHT_PASS ' + json.dumps({'train_records': len(training),
              'train_sources': len({r['source_group'] for r in training}), 'validation_records': len(validation),
              'config_sha256': config_sha, 'source_sha256': binding, 'images_opened': 0}))
        return
    if args.mode == 'formal':
        if args.qualification is None:
            raise RuntimeError('formal requires qualification receipt; never auto-start')
        validate_receipt(args.qualification, config_sha, binding)
    config, rank, device, output, gpu_names = initialize('ppt', args.run_id)
    started = time.monotonic()
    try:
        os.environ.update(TORCH_HOME=str(ROOT/'models/torch'), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        random.seed(spec['seed']); torch.manual_seed(spec['seed']); torch.cuda.manual_seed_all(spec['seed'])
        if rank == 0:
            write_once(output/'launch.json', {'config': config, 'mode': args.mode, 'config_sha256': config_sha,
                       'source_sha256': binding, 'fresh_initialization': True,
                       'qualification_checkpoint_used_for_initialization': False,
                       'protected_pixels_accessed': False})
        proposal_model = load_frozen_proposal(base, device)
        basis = load_residual_basis(candidate_spec, base, device)
        negative_models = load_negative_models(candidate_spec, device)
        model = NovelPPTEnergy(spec['width'], spec['condition_dim'], spec['residual_scale_minus1_to1'],
                               spec['fixed_coordinate_step'], spec['coordinate_bound']).to(device)
        wrapped = DDP(model, device_ids=[device.index], broadcast_buffers=False, gradient_as_bucket_view=False)
        optimizer = torch.optim.AdamW(wrapped.parameters(), lr=spec['learning_rate'], weight_decay=spec['weight_decay'])
        perceptual = build_lpips(device)
        groups = {}
        for row in training:
            if row['split'] != 'train':
                raise RuntimeError('non-training optimization row')
            groups.setdefault(row['source_group'], []).append(row)
        total = spec['qualification_steps'] if args.mode == 'qualification' else spec['primary_steps']
        eval_rows = validation[:spec['qualification_development_records']] if args.mode == 'qualification' else validation
        result = evaluate(model, proposal_model, basis, eval_rows, base, perceptual, device, rank)
        if rank == 0:
            if any(r['energy_3'] != r['fidelity'] or r['selected_step'] != 0 for r in result['rows']):
                raise RuntimeError('fresh zero-potential model must preserve fidelity exactly')
            write_once(output/'validation_000000.json', result)
        verified, times, last_checkpoint = set(), [], None
        initial_head = model.field.head.weight.detach().clone()
        model.train()
        for step in range(1, total+1):
            t0 = time.monotonic()
            if step <= spec['warmup_steps']:
                lr = spec['learning_rate']*step/spec['warmup_steps']
            else:
                progress = (step-spec['warmup_steps'])/(spec['primary_steps']-spec['warmup_steps'])
                lr = spec['minimum_learning_rate'] + .5*(spec['learning_rate']-spec['minimum_learning_rate'])*(1+math.cos(math.pi*progress))
            for group in optimizer.param_groups:
                group['lr'] = lr
            optimizer.zero_grad(set_to_none=True)
            selected = sample_global_batch(groups, task='noise', step=step, seed=spec['seed'], count=32)
            totals = {}
            for micro in range(4):
                row, crop_seed = selected[rank*4+micro]
                if row['split'] != 'train':
                    raise RuntimeError('protected/nontraining row before decode')
                lq, gt = load_training_patch(row, crop_size=spec['crop'], augmentation_seed=crop_seed, verified=verified)
                observation01, target = lq[None].to(device), gt[None].to(device).mul(2).sub(1)
                proposal, anchor, perception, *negatives = frozen_outputs(proposal_model, basis, observation01, negative_models)
                observation = observation01.mul(2).sub(1)
                descriptor = severity_from_rows([row], device)
                candidates = (target, observation, proposal, anchor, perception, .5*(proposal+anchor),
                              (proposal+1.25*(anchor-proposal)).clamp(-1, 1), *negatives)
                from contextlib import nullcontext
                context = wrapped.no_sync() if micro < 3 else nullcontext()
                with context:
                    output_graph = wrapped(observation, proposal, anchor, descriptor, candidates)
                    loss, terms, diagnostics = objective(output_graph, anchor, target, perceptual,
                            residual_scale=spec['residual_scale_minus1_to1'], weights=spec['loss'],
                            ranking_lpips_weight=spec['ranking_lpips_weight'])
                    if not bool(torch.isfinite(loss)):
                        raise RuntimeError('nonfinite energy loss')
                    (loss/4).backward()
                for key, value in {'loss': loss, **terms, **diagnostics}.items():
                    totals[key] = totals.get(key, 0.) + float(value.detach())/4
                del output_graph, loss, terms, diagnostics, candidates
            parameters = list(model.parameters())
            if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in parameters):
                raise RuntimeError('missing/nonfinite second-order parameter gradient')
            norm = torch.nn.utils.clip_grad_norm_(parameters, spec['gradient_clip'])
            if not bool(torch.isfinite(norm)):
                raise RuntimeError('nonfinite gradient norm')
            optimizer.step()
            tensor = torch.tensor([*totals.values(), float(norm)], device=device, dtype=torch.float64)
            dist.all_reduce(tensor); tensor /= 8
            torch.cuda.synchronize(); times.append(time.monotonic()-t0)
            if rank == 0 and (step == 1 or step % 5 == 0 or step == total):
                print('NOVEL_PPT_TRAIN ' + json.dumps({'step': step, 'total_steps': total, 'lr': lr,
                      **dict(zip([*totals, 'preclip_gradient_norm'], tensor.tolist(), strict=True)),
                      'step_seconds': times[-1], 'peak_memory_gib': torch.cuda.max_memory_allocated()/1024**3}), flush=True)
            if step == total or (args.mode == 'formal' and (step % 250 == 0 or step in spec['milestones'])):
                last_checkpoint = save_checkpoint(output, model, optimizer, step, args.mode, config_sha, binding, rank)
                dist.barrier()
            if step == total or (args.mode == 'formal' and step in spec['milestones']):
                result = evaluate(model, proposal_model, basis, eval_rows, base, perceptual, device, rank)
                if rank == 0:
                    write_once(output/f'validation_{step:06d}.json', result)
                    print('NOVEL_PPT_VALIDATION ' + json.dumps({'step': step, **{k: v for k, v in result.items() if k != 'rows'}}), flush=True)
        updated = bool((model.field.head.weight.detach()-initial_head).abs().max() > 0)
        check = torch.tensor(int(updated), device=device); dist.all_reduce(check, op=dist.ReduceOp.MIN)
        if not bool(check):
            raise RuntimeError('energy potential did not update on every rank')
        if rank == 0:
            hours = (time.monotonic()-started)/3600
            contribution = (result['selected_psnr_gain_over_fidelity'] > 0 and result['selected_lpips_change_from_fidelity'] <= 0)
            absolute = (result['metrics']['selected']['psnr'] >= spec['absolute_goal']['psnr']
                        and result['metrics']['selected']['lpips'] <= spec['absolute_goal']['lpips'])
            write_once(output/'result.json', {'status': 'complete', 'mode': args.mode, 'engineering_pass': True,
                       'steps': total, 'config_sha256': config_sha, 'source_sha256': binding, **last_checkpoint,
                       'gpu_names': gpu_names, 'wall_hours': hours, 'allocated_gpu_hours': hours*8,
                       'median_step_seconds': sorted(times)[len(times)//2],
                       'validation_records': len(eval_rows), 'validation': result['metrics'],
                       'energy_contribution_on_this_subset': contribution,
                       'full_development_absolute_gate_pass': absolute if args.mode == 'formal' else None,
                       'scientific_success_established': contribution and absolute if args.mode == 'formal' else False,
                       'protected_pixels_accessed': False, 'qualification_may_seed_formal': False,
                       'new_paper_novelty_established': False})
            print('NOVEL_PPT_COMPLETE', flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
