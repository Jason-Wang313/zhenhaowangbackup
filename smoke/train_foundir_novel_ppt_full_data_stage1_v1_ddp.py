#!/usr/bin/env python3
"""Qualified PPT recipe on the full train pool, capped at 1706; true resume."""
from __future__ import annotations

import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import random
import re
import socket
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from lowlight_restoration.novel_ppt_energy import NovelPPTEnergy, objective
from smoke import novel_ppt_full_data_state_v1 as stateio
from smoke import train_foundir_novel_ppt_energy_v1_ddp as parent
from smoke.dual_retry_runtime import bound_json, sha, sources, write_once
from smoke.train_foundir_novel_ppt_energy_v1_ddp import (
    evaluate, frozen_outputs, load_frozen_proposal, load_residual_basis,
    load_negative_models, build_lpips, sample_global_batch, load_training_patch,
    severity_from_rows,
)


def contract():
    stage = json.loads(stateio.CONFIG.read_text())
    bound_json(stage['parent_config'], stage['parent_sha256'])
    config, base, candidates, training, validation = parent.contract()
    receipt = ROOT/stage['qualification_receipt']
    if sha(receipt) != stage['qualification_receipt_sha256']:
        raise RuntimeError('successful qualification receipt changed')
    parent.validate_receipt(receipt, stage['parent_sha256'], sources())
    if (stage['steps'] != 1706 or stage['learning_rate_horizon'] != config['ppt']['primary_steps']
            or stage['checkpoint_every'] != 100 or stage['qualification_or_trainfit_may_seed']
            or stage['auto_continue_to_8530'] or stage['world_size'] != 8
            or stage['global_batch'] != config['ppt']['global_batch']
            or stage['crop'] != config['ppt']['crop']):
        raise RuntimeError('frozen stage1 contract changed')
    if (len(training) != stage['train_records']
            or len({r['source_group'] for r in training}) != stage['train_source_groups']
            or len(validation) != stage['development_records']):
        raise RuntimeError('frozen data counts changed')
    return stage, config, base, candidates, training, validation


def initialize(stage, run_id):
    rank, local = int(os.environ['RANK']), int(os.environ['LOCAL_RANK'])
    if (int(os.environ['WORLD_SIZE']) != 8 or int(os.environ['LOCAL_WORLD_SIZE']) != 8
            or torch.cuda.device_count() != 8):
        raise RuntimeError('requires exactly one eight-GPU node')
    torch.cuda.set_device(local)
    device = torch.device('cuda', local)
    prop = torch.cuda.get_device_properties(device)
    if prop.total_memory < 75*1024**3 or not any(n in prop.name for n in ('H100', 'H200')):
        raise RuntimeError('requires H100/H200 >=75 GiB')
    dist.init_process_group('nccl', device_id=device, timeout=timedelta(minutes=30))
    names, uuids, hosts = [None]*8, [None]*8, [None]*8
    dist.all_gather_object(names, prop.name)
    dist.all_gather_object(uuids, str(prop.uuid))
    dist.all_gather_object(hosts, socket.gethostname())
    if len(set(names)) != 1 or len(set(hosts)) != 1 or len(set(uuids)) != 8:
        raise RuntimeError('requires eight distinct homogeneous GPUs on one node')
    output = ROOT/stage['output_root']/run_id
    error = [None]
    if rank == 0:
        try:
            output.mkdir(parents=True, exist_ok=False)
        except Exception as exc:
            error[0] = repr(exc)
    dist.broadcast_object_list(error, src=0)
    if error[0]:
        raise RuntimeError(error[0])
    return rank, device, output, names, uuids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', default='stage1_v1')
    parser.add_argument('--resume-receipt', type=Path)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.run_id):
        raise ValueError('invalid fresh run ID')
    stage, config, base, candidate_spec, training, validation = contract()
    spec, binding, config_sha = config['ppt'], stateio.binding(), sha(stateio.CONFIG)
    resume_state, resume_receipt = None, None
    if args.resume_receipt:
        if os.environ.get('PPT_PREVIOUS_RUN_STOPPED') != '1' and not args.preflight:
            raise RuntimeError('confirm previous job stopped before resuming')
        resume_state, resume_receipt = stateio.read_resume(args.resume_receipt, config_sha, binding)
    if args.preflight:
        print('PPT_FULL_DATA_PREFLIGHT_PASS ' + json.dumps({'train_records': len(training),
              'train_sources': len({r['source_group'] for r in training}), 'validation_records': len(validation),
              'config_sha256': config_sha, 'source_sha256': binding, 'images_opened': 0,
              'resume_step': resume_state['step'] if resume_state else None,
              'stop_step': stage['steps'], 'lr_horizon': spec['primary_steps']}))
        return
    rank, device, output, gpu_names, gpu_uuids = initialize(stage, args.run_id)
    started = time.monotonic()
    try:
        os.environ.update(TORCH_HOME=str(ROOT/'models/torch'), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        random.seed(spec['seed']); torch.manual_seed(spec['seed']); torch.cuda.manual_seed_all(spec['seed'])
        np.random.seed(spec['seed'])
        if rank == 0:
            write_once(output/'launch.json', {'config': config, 'stage_config': stage,
                       'mode': stateio.MODE, 'config_sha256': config_sha,
                       'source_sha256': binding, 'fresh_initialization': resume_state is None,
                       'resume_receipt': str(args.resume_receipt) if args.resume_receipt else None,
                       'resume_receipt_sha256': sha(args.resume_receipt) if args.resume_receipt else None,
                       'previous_job_stop_is_user_ack_not_remote_verification': bool(args.resume_receipt),
                       'qualification_or_trainfit_weights_loaded': False,
                       'hostname': socket.gethostname(), 'gpu_names': gpu_names, 'gpu_uuids': gpu_uuids,
                       'protected_pixels_accessed': False})
            print('PPT_FULL_DATA_LOADING_FROZEN_MODELS', flush=True)
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
        total, eval_rows = stage['steps'], validation
        start_step = 0
        if resume_state is not None:
            stateio.apply_state(model, optimizer, resume_state, rank, device)
            start_step = resume_state['step']
        # Development inference and module loading must not consume training RNG.
        train_rng = stateio.capture_rng(device)
        result = evaluate(model, proposal_model, basis, eval_rows, base, perceptual, device, rank)
        stateio.restore_rng(train_rng, device)
        if rank == 0:
            if start_step == 0 and any(r['energy_3'] != r['fidelity'] or r['selected_step'] != 0 for r in result['rows']):
                raise RuntimeError('fresh zero-potential model must preserve fidelity exactly')
            write_once(output/f'validation_{start_step:06d}.json', result)
            print('PPT_FULL_DATA_INITIAL_VALIDATION ' + json.dumps({'step': start_step,
                  **{k: v for k, v in result.items() if k != 'rows'}}), flush=True)
        verified, times, last_checkpoint = set(), [], resume_receipt
        if resume_receipt:
            last_checkpoint = {**resume_receipt, 'receipt': str(args.resume_receipt)}
        del resume_state
        initial_head = model.field.head.weight.detach().clone()
        model.train()
        for step in range(start_step+1, total+1):
            t0 = time.monotonic()
            lr = stateio.learning_rate(spec, step)
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
            if rank == 0 and (step == start_step+1 or step % 5 == 0 or step == total):
                print('PPT_FULL_DATA_TRAIN ' + json.dumps({'step': step, 'total_steps': total, 'lr': lr,
                      **dict(zip([*totals, 'preclip_gradient_norm'], tensor.tolist(), strict=True)),
                      'step_seconds': times[-1], 'peak_memory_gib': torch.cuda.max_memory_allocated()/1024**3}), flush=True)
            if step == total or step % stage['checkpoint_every'] == 0:
                last_checkpoint = stateio.save(output, model, optimizer, step, config_sha, binding, device, rank)
                if rank == 0:
                    print('PPT_FULL_DATA_CHECKPOINT ' + json.dumps(last_checkpoint), flush=True)
            if step == total:
                train_rng = stateio.capture_rng(device)
                result = evaluate(model, proposal_model, basis, eval_rows, base, perceptual, device, rank)
                stateio.restore_rng(train_rng, device)
                if rank == 0:
                    write_once(output/f'validation_{step:06d}.json', result)
                    print('PPT_FULL_DATA_VALIDATION ' + json.dumps({'step': step, **{k: v for k, v in result.items() if k != 'rows'}}), flush=True)
        updated = start_step == total or bool((model.field.head.weight.detach()-initial_head).abs().max() > 0)
        check = torch.tensor(int(updated), device=device); dist.all_reduce(check, op=dist.ReduceOp.MIN)
        if not bool(check):
            raise RuntimeError('energy potential did not update on every rank')
        if rank == 0:
            hours = (time.monotonic()-started)/3600
            contribution = (result['selected_psnr_gain_over_fidelity'] > 0 and result['selected_lpips_change_from_fidelity'] <= 0)
            absolute = (result['metrics']['selected']['psnr'] >= spec['absolute_goal']['psnr']
                        and result['metrics']['selected']['lpips'] <= spec['absolute_goal']['lpips'])
            write_once(output/'result.json', {'status': 'complete', 'mode': stateio.MODE, 'engineering_pass': True,
                       'steps': total, 'resumed_from_step': start_step,
                       'steps_this_attempt': total-start_step, 'auto_continue_to_8530': False,
                       'config_sha256': config_sha, 'source_sha256': binding, **last_checkpoint,
                       'gpu_names': gpu_names, 'wall_hours': hours, 'allocated_gpu_hours': hours*8,
                       'median_step_seconds': sorted(times)[len(times)//2] if times else None,
                       'validation_records': len(eval_rows), 'validation': result['metrics'],
                       'energy_contribution_on_this_subset': contribution,
                       'full_development_absolute_gate_pass': absolute,
                       'stage1_joint_metric_gate_pass': contribution and absolute,
                       'generalization_beyond_this_noise_development_set_established': False,
                       'protected_pixels_accessed': False, 'qualification_may_seed_formal': False,
                       'new_paper_novelty_established': False})
            print('PPT_FULL_DATA_COMPLETE', flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
