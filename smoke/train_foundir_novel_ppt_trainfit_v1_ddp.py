#!/usr/bin/env python3
"""Train-only learnability check: unchanged PPT energy, 32 fixed crops, 300 steps."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import gc
import hashlib
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
from torch.nn import functional as F
from lowlight_restoration.novel_ppt_energy import NovelPPTEnergy, objective
from smoke import train_foundir_novel_ppt_energy_v1_ddp as parent
from smoke.dual_retry_runtime import bound_json, initialize, sha, sources, source_disjoint_training_probe, write_once

CONFIG = ROOT / 'configs/foundir_novel_ppt_trainfit_v1.json'
EXTRA_SOURCES = (
    'smoke/train_foundir_novel_ppt_trainfit_v1_ddp.py',
    'smoke/run_foundir_novel_ppt_trainfit_v1_8gpu.sh',
    'smoke/start_foundir_novel_ppt_trainfit_v1_tmux.sh',
)


def fixed_records(rows, seed, count=32):
    selected = source_disjoint_training_probe(rows, count, seed)
    return [(row, int.from_bytes(hashlib.sha256(
        f'{seed}:fixed-crop:{row["record_id"]}'.encode()).digest()[:8], 'big')) for row in selected]


def unchanged_learning_rate(spec, step):
    if step <= spec['warmup_steps']:
        return spec['learning_rate'] * step / spec['warmup_steps']
    progress = (step-spec['warmup_steps'])/(spec['primary_steps']-spec['warmup_steps'])
    return spec['minimum_learning_rate'] + .5*(spec['learning_rate']-spec['minimum_learning_rate'])*(1+math.cos(math.pi*progress))


def training_candidates(case):
    p, a = case['proposal'], case['fidelity']
    return (case['target'], case['observation'], p, a, case['perceptual'],
            .5*(p+a), (p+1.25*(a-p)).clamp(-1, 1), *case['negatives'])


def near_candidates(case):
    p, a, perceptual = case['proposal'], case['fidelity'], case['perceptual']
    return {
        'fidelity': a, 'perceptual': perceptual,
        'half_fidelity': .5*(p+a), 'over_fidelity': (p+1.25*(a-p)).clamp(-1, 1),
        'blend_75_25': .75*a+.25*perceptual, 'blend_50_50': .5*a+.5*perceptual,
    }


def ranking_counts(scores, qualities, fidelity_quality, *, relative_gap=.1, quality_tie=1e-6, score_tie=1e-9):
    if len(scores) != len(qualities) or len(scores) < 2:
        raise ValueError('matched candidate scores/qualities required')
    if not all(math.isfinite(float(v)) for v in (*scores, *qualities, fidelity_quality)):
        raise ValueError('nonfinite ranking diagnostic')
    result = dict(all_pairs=0, all_correct_credit=0., close_pairs=0, close_correct_credit=0.)
    for i in range(len(scores)):
        for j in range(i+1, len(scores)):
            dq, ds = qualities[i]-qualities[j], scores[i]-scores[j]
            if abs(dq) <= quality_tie:
                continue
            credit = .5 if abs(ds) <= score_tie else float(dq*ds > 0)
            result['all_pairs'] += 1; result['all_correct_credit'] += credit
            if abs(dq) <= relative_gap*max(fidelity_quality, quality_tie):
                result['close_pairs'] += 1; result['close_correct_credit'] += credit
    return result


def contract():
    cfg = json.loads(CONFIG.read_text())
    bound_json(cfg['parent_config'], cfg['parent_sha256'])
    parent_cfg, base, candidate_spec, training, _ = parent.contract()
    if sha(ROOT/cfg['qualification_receipt']) != cfg['qualification_receipt_sha256']:
        raise RuntimeError('successful qualification receipt changed')
    parent.validate_receipt(ROOT/cfg['qualification_receipt'], cfg['parent_sha256'], sources())
    if not (cfg['records'] == cfg['source_groups'] == cfg['global_batch'] == 32
            and cfg['steps'] == 300 and cfg['world_size'] == 8
            and cfg['micro_batch'] == 1 and cfg['accumulation'] == 4
            and cfg['crop'] == parent_cfg['ppt']['crop'] == 256
            and cfg['milestones'] == [0, 100, 200, 300]):
        raise RuntimeError('bounded train-fit contract changed')
    selected = fixed_records(training, cfg['selection_seed'], cfg['records'])
    return cfg, parent_cfg, base, candidate_spec, selected


def tensor_hash(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def make_cache(selected, base, candidate_spec, device, rank):
    proposal_model = parent.load_frozen_proposal(base, device)
    basis = parent.load_residual_basis(candidate_spec, base, device)
    negatives = parent.load_negative_models(candidate_spec, device)
    cache, identities, verified = [], [], set()
    for row, crop_seed in selected[rank*4:(rank+1)*4]:
        if row['split'] != 'train':
            raise RuntimeError('non-training pixels prohibited')
        lq, gt = parent.load_training_patch(row, crop_size=256, augmentation_seed=crop_seed, verified=verified)
        observation01 = lq[None].to(device)
        proposal, anchor, perception, *other = parent.frozen_outputs(proposal_model, basis, observation01, negatives)
        case = dict(record_id=row['record_id'], source_group=row['source_group'], crop_seed=crop_seed,
                    observation=observation01.mul(2).sub(1), target=gt[None].to(device).mul(2).sub(1),
                    proposal=proposal, fidelity=anchor, perceptual=perception, negatives=tuple(other),
                    descriptor=parent.severity_from_rows([row], device))
        values = {'observation': case['observation'], 'target': case['target'],
                  'descriptor': case['descriptor'], 'proposal': proposal, 'fidelity': anchor,
                  'perceptual': perception, **{f'negative_{i}': v for i, v in enumerate(other)}}
        if any(v.requires_grad or not bool(torch.isfinite(v).all()) for v in values.values()):
            raise RuntimeError('frozen cache has gradients or nonfinite values')
        identities.append(dict(record_id=row['record_id'], source_group=row['source_group'], crop_seed=crop_seed,
                               tensors_sha256={k: tensor_hash(v) for k, v in values.items()}))
        cache.append(case)
    del proposal_model, basis, negatives
    gc.collect(); torch.cuda.empty_cache()
    return cache, identities


def evaluate_training_fit(model, cache, perceptual, cfg, spec, rank):
    model.eval()
    local = []
    for case in cache:
        x, p, a, target, descriptor = (case[k] for k in ('observation', 'proposal', 'fidelity', 'target', 'descriptor'))
        with torch.enable_grad():
            trajectory, gradients, coordinates = model.trajectory(x, p, a, descriptor, create_graph=False)
        trajectory = tuple(v.detach() for v in trajectory)
        selected, choice, _ = model.select(x, p, a, descriptor, trajectory)
        values = dict(zip(('proposal', 'fidelity', 'energy_1', 'energy_2', 'energy_3', 'selected'),
                          (p, a, *trajectory, selected), strict=True))
        row = {'record_id': case['record_id'], 'source_group': case['source_group'], 'selected_step': int(choice)}
        with torch.no_grad():
            for name, value in values.items():
                if not bool(torch.isfinite(value).all()):
                    raise RuntimeError('nonfinite train-fit output')
                mse = ((value-target)*.5).square().mean()
                row[name] = dict(mse=float(mse), psnr=float(-10*torch.log10(mse.clamp_min(1e-12))),
                                 lpips=float(perceptual(value, target).mean()))
            wanted = (target-a)/spec['residual_scale_minus1_to1']
            row['gradient_cosine'] = float(torch.stack([
                F.cosine_similarity(-g.flatten(1), (wanted-u).flatten(1), dim=1, eps=1e-8).mean()
                for g, u in zip(gradients, coordinates, strict=True)]).mean())
            row['update_abs_mean_minus1_to1'] = float((trajectory[-1]-a).abs().mean())
            fixed = near_candidates(case)
            scores, qualities = [], []
            row['near_candidates'] = {}
            for name, candidate in fixed.items():
                score = float(model.score_map(x, p, a, candidate, descriptor).mean())
                cost = float(((candidate-target)/spec['residual_scale_minus1_to1']).square().mean()
                             + spec['ranking_lpips_weight']*perceptual(candidate, target).mean())
                scores.append(score); qualities.append(cost)
                row['near_candidates'][name] = dict(energy=score, measured_composite_cost=cost)
            row['ranking'] = ranking_counts(scores, qualities, qualities[0],
                    relative_gap=cfg['close_ranking_relative_gap'], quality_tie=cfg['quality_tie_epsilon'],
                    score_tie=cfg['score_tie_epsilon'])
        local.append(row)
    gathered = [None]*8; dist.all_gather_object(gathered, local)
    model.train()
    if rank:
        return None
    rows = sorted((r for shard in gathered for r in shard), key=lambda r: r['record_id'])
    if len(rows) != 32 or len({r['record_id'] for r in rows}) != 32:
        raise RuntimeError('incomplete fixed training-set evaluation')
    metrics = {name: {metric: sum(r[name][metric] for r in rows)/32 for metric in ('mse', 'psnr', 'lpips')}
               for name in values}
    ranking = {key: sum(r['ranking'][key] for r in rows) for key in rows[0]['ranking']}
    for prefix in ('all', 'close'):
        ranking[prefix+'_accuracy'] = (ranking[prefix+'_correct_credit']/ranking[prefix+'_pairs']
                                        if ranking[prefix+'_pairs'] else None)
    return dict(split='train', records=32, source_groups=32, rows=rows, metrics=metrics, ranking=ranking,
                gradient_cosine=sum(r['gradient_cosine'] for r in rows)/32,
                update_abs_mean_minus1_to1=sum(r['update_abs_mean_minus1_to1'] for r in rows)/32,
                selected_psnr_gain_over_fidelity=metrics['selected']['psnr']-metrics['fidelity']['psnr'],
                selected_lpips_change_from_fidelity=metrics['selected']['lpips']-metrics['fidelity']['lpips'],
                selected_relative_mse_reduction=1-metrics['selected']['mse']/max(metrics['fidelity']['mse'], 1e-12),
                GT_used_for_selection=False, generalization_established=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', default='trainfit_v1')
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    cfg, parent_cfg, base, candidate_spec, selected = contract()
    binding, config_sha = sources(EXTRA_SOURCES), sha(CONFIG)
    identity = [{'record_id': r['record_id'], 'source_group': r['source_group'], 'split': r['split'],
                 'lq_sha256': r['lq_sha256'], 'gt_sha256': r['gt_sha256'], 'crop_seed': seed} for r, seed in selected]
    selection_sha = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    if args.preflight:
        print('PPT_TRAINFIT_PREFLIGHT_PASS '+json.dumps(dict(config_sha256=config_sha, source_sha256=binding,
              records=len(selected), sources=len({r['source_group'] for r, _ in selected}),
              selection_sha256=selection_sha, images_opened=0)), flush=True)
        return
    if not args.run_id.startswith(cfg['run_id_prefix']):
        raise RuntimeError('training-fit run ID must start trainfit_; never use a formal or qualification ID')
    _, rank, device, output, gpu_names = initialize('ppt', args.run_id)
    spec, started = parent_cfg['ppt'], time.monotonic()
    try:
        os.environ.update(TORCH_HOME=str(ROOT/'models/torch'), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        random.seed(spec['seed']); torch.manual_seed(spec['seed']); torch.cuda.manual_seed_all(spec['seed'])
        if rank == 0:
            write_once(output/'launch.json', dict(config=cfg, config_sha256=config_sha, source_sha256=binding,
                       selection_sha256=selection_sha, selected_train_records=identity,
                       parent_config_sha256=cfg['parent_sha256'], qualification_weights_loaded=False))
        cache, cache_ids = make_cache(selected, base, candidate_spec, device, rank)
        gathered = [None]*8; dist.all_gather_object(gathered, cache_ids)
        if rank == 0:
            write_once(output/'fixed_cache_identity.json', dict(records=[r for shard in gathered for r in shard],
                       selection_sha256=selection_sha))
            print('PPT_TRAINFIT_CACHE_READY records=32 source_groups=32 fixed_crops=1', flush=True)
        model = NovelPPTEnergy(spec['width'], spec['condition_dim'], spec['residual_scale_minus1_to1'],
                               spec['fixed_coordinate_step'], spec['coordinate_bound']).to(device)
        wrapped = DDP(model, device_ids=[device.index], broadcast_buffers=False, gradient_as_bucket_view=False)
        optimizer = torch.optim.AdamW(wrapped.parameters(), lr=spec['learning_rate'], weight_decay=spec['weight_decay'])
        initial_head = model.field.head.weight.detach().clone()
        perceptual = parent.build_lpips(device)
        initial = evaluate_training_fit(model, cache, perceptual, cfg, spec, rank)
        if rank == 0:
            if any(any(r[name] != r['fidelity'] for name in ('energy_1', 'energy_2', 'energy_3', 'selected'))
                   or r['selected_step'] != 0 for r in initial['rows']):
                raise RuntimeError('initial energy does not preserve Fidelity')
            write_once(output/'training_fit_000000.json', initial)
        times, last_checkpoint = [], None
        for step in range(1, cfg['steps']+1):
            t0 = time.monotonic()
            lr = unchanged_learning_rate(spec, step)
            for group in optimizer.param_groups:
                group['lr'] = lr
            optimizer.zero_grad(set_to_none=True)
            order = list(range(4)); random.Random(cfg['selection_seed']+step).shuffle(order)
            totals = {}
            for micro, index in enumerate(order):
                case = cache[index]
                with wrapped.no_sync() if micro < 3 else nullcontext():
                    graph = wrapped(case['observation'], case['proposal'], case['fidelity'],
                                    case['descriptor'], training_candidates(case))
                    loss, terms, diagnostics = objective(graph, case['fidelity'], case['target'], perceptual,
                            residual_scale=spec['residual_scale_minus1_to1'], weights=spec['loss'],
                            ranking_lpips_weight=spec['ranking_lpips_weight'])
                    if not bool(torch.isfinite(loss)):
                        raise RuntimeError('nonfinite training-fit loss')
                    (loss/4).backward()
                for key, value in {'loss': loss, **terms, **diagnostics}.items():
                    totals[key] = totals.get(key, 0.)+float(value.detach())/4
                del graph, loss, terms, diagnostics
            if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
                raise RuntimeError('missing/nonfinite training-fit gradient')
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), spec['gradient_clip'])
            if not bool(torch.isfinite(norm)):
                raise RuntimeError('nonfinite gradient norm')
            optimizer.step()
            tensor = torch.tensor([*totals.values(), float(norm)], dtype=torch.float64, device=device)
            dist.all_reduce(tensor); tensor /= 8
            torch.cuda.synchronize(); times.append(time.monotonic()-t0)
            if rank == 0 and (step == 1 or step % 10 == 0):
                print('PPT_TRAINFIT_TRAIN '+json.dumps(dict(step=step, total_steps=cfg['steps'], lr=lr,
                      **dict(zip([*totals, 'preclip_gradient_norm'], tensor.tolist(), strict=True)),
                      step_seconds=times[-1], peak_memory_gib=torch.cuda.max_memory_allocated()/1024**3)), flush=True)
            if step in cfg['milestones']:
                last_checkpoint = parent.save_checkpoint(output, model, optimizer, step, 'trainfit_only',
                                                         config_sha, binding, rank)
                dist.barrier()
                result = evaluate_training_fit(model, cache, perceptual, cfg, spec, rank)
                if rank == 0:
                    if [r['record_id'] for r in result['rows']] != [r['record_id'] for r in initial['rows']]:
                        raise RuntimeError('fixed training evaluation records changed')
                    write_once(output/f'training_fit_{step:06d}.json', result)
                    print('PPT_TRAINFIT_EVALUATION '+json.dumps(dict(step=step,
                          **{k: v for k, v in result.items() if k != 'rows'})), flush=True)
        updated = torch.tensor(int(bool((model.field.head.weight.detach()-initial_head).abs().max() > 0)), device=device)
        dist.all_reduce(updated, op=dist.ReduceOp.MIN)
        if not bool(updated):
            raise RuntimeError('energy potential did not update on every rank')
        if rank == 0:
            hours = (time.monotonic()-started)/3600
            write_once(output/'result.json', dict(status='complete', engineering_pass=True, steps=cfg['steps'],
                       mode='trainfit_only', config_sha256=config_sha, source_sha256=binding,
                       **last_checkpoint, gpu_names=gpu_names, wall_hours=hours, allocated_gpu_hours=hours*8,
                       median_step_seconds=sorted(times)[len(times)//2], final_training_fit=result,
                       initial_ranking=initial['ranking'], generalization_established=False,
                       scientific_success_established=False, may_seed_formal=False, auto_launch_formal=False,
                       development_pixels_accessed=False, protected_pixels_accessed=False))
            print('PPT_TRAINFIT_COMPLETE', flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
