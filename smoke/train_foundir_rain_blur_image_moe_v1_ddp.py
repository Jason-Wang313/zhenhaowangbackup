#!/usr/bin/env python3
"""Rain-initialized image-only two-task MoE and matched dense control."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'third_party/flux2/src'))
from smoke import train_foundir_rain_flux2_precision_full_v1_ddp as full
from smoke import train_foundir_rain_flux2_precision_resume_v1_ddp as rain_resume
from lowlight_restoration.flux2_endpoint_precision import promote_lora_to_fp32, restore_endpoint, assert_fp32_optimizer
from lowlight_restoration.lora import inject_lora, load_lora_checkpoint
from lowlight_restoration.rain_blur_image_moe import (
    inject_image_moe, image_moe_modules, training_routes, warmup_fraction,
    router_classification_loss, clear_routing_graphs, trainable_state,
)
from lowlight_restoration.rain_blur_moe_data import load_protocol, RainBlurDataset

old, write_json = full.old, full.write_json
CONFIG_SHA256 = '8988e2374ad736f3e406945f1d9db83ae14587d0aba28a04dbf9cdc900ab5f12'


def source_hashes():
    paths = [Path(__file__), ROOT / 'lowlight_restoration/rain_blur_image_moe.py',
             ROOT / 'lowlight_restoration/rain_blur_moe_data.py',
             ROOT / 'smoke/run_foundir_rain_blur_image_moe_v1_8gpu.sh',
             ROOT / 'smoke/start_foundir_rain_blur_image_moe_v1_tmux.sh']
    return {**rain_resume.source_hashes(), **{str(p.relative_to(ROOT)): old.sha256(p) for p in paths}}


def read_contract(path):
    if old.sha256(path) != CONFIG_SHA256:
        raise RuntimeError('frozen Rain/Blur MoE contract changed')
    config = json.loads(path.read_text())
    if old.sha256(ROOT / config['parent_full_config']) != config['parent_full_config_sha256']:
        raise RuntimeError('parent configuration changed')
    rain_config, parent = full.read_contract(ROOT / config['parent_full_config'])
    train, _, dev = full.load_protocol(rain_config, parent)
    if old.sha256(ROOT / config['source_manifest']) != config['source_manifest_sha256']:
        raise RuntimeError('source manifest changed')
    training, development, metadata = load_protocol(ROOT, train, dev, config['source_manifest'])
    if metadata != config['data_contract']:
        raise RuntimeError('two-task split/crop contract changed')
    return config, parent, training, development, metadata


def preflight(path):
    config, parent, training, development, metadata = read_contract(path)
    old.verify_artifacts(parent)
    for key in ('rain_result', 'rain_lora', 'rain_metrics'):
        if old.sha256(ROOT / config[key]) != config[key + '_sha256']:
            raise RuntimeError(f'Rain prerequisite changed: {key}')
    result = json.loads((ROOT / config['rain_result']).read_text())
    if not (result['status'] == 'complete' and result['development_gate_pass']
            and result['source_sha256'] == rain_resume.source_hashes()):
        raise RuntimeError('working Rain prerequisite no longer matches')
    all_rows = training['rain'] + training['blur'] + development['rain'] + development['blur_full']
    paths = sorted({r[k + '_path'] for r in all_rows for k in ('lq', 'gt')})
    def check_path(relative):
        if not (ROOT / relative).is_file():
            raise RuntimeError(f'missing data: {relative}')
    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(check_path, paths))
    def check_bytes(row):
        for side in ('lq', 'gt'):
            if old.sha256(ROOT / row[side + '_path']) != row[side + '_sha256']:
                raise RuntimeError(f'development bytes changed: {row["record_id"]}')
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(check_bytes, development['rain'] + development['blur_full']))
    payload = {'status': 'rain_blur_image_moe_preflight_pass', 'config_sha256': CONFIG_SHA256,
               'data': metadata, 'protected_pixels_accessed': False,
               'training_hash_policy': 'lazy verify before decode/after stat change',
               'gpu_qualification_completed': False}
    print('RAIN_BLUR_MOE_PREFLIGHT ' + json.dumps(payload), flush=True)
    return payload


def evaluate(model, ae, rows, context, context_ids, pack, metric, *, rank, world,
             device, output, split, step, force_expert=None):
    model.eval()
    modules = image_moe_modules(model)
    for module in modules.values():
        if module.labels is not None or module.teacher_fraction != 0:
            raise RuntimeError('training labels leaked into evaluation')
        module.force_expert = force_expert
    local = []
    suffix = 'image' if force_expert is None else f'constant_expert_{force_expert}'
    try:
        for index in range(rank, len(rows), world):
            row = rows[index]
            lq, gt = [x[None].to(device) for x in old.load_crop(row, 512)]
            with torch.inference_mode():
                result = restore_endpoint(model, ae, lq, context, context_ids, pack)['decoded']
                if not bool(torch.isfinite(result).all()):
                    raise RuntimeError('nonfinite evaluation image')
                quantized = ((result + 1) * 127.5).round().clamp(0, 255) / 127.5 - 1
                entry = {'record_id': row['record_id'], 'source_group': row['source_group'],
                         'input_psnr': old.psnr(gt, lq), 'output_psnr': old.psnr(gt, quantized),
                         'input_lpips': old.lpips_value(metric, gt, lq),
                         'output_lpips': old.lpips_value(metric, gt, quantized),
                         'expert_weights': {name: m.last_weights[0].cpu().tolist()
                                            for name, m in modules.items() if m.router is not None}}
            local.append(entry)
            if index < 2 and force_expert is None:
                old.save_triptych(output / 'images' / split / f'{step:06d}_{index:03d}.png',
                                  row['record_id'], lq, quantized, gt)
    finally:
        for module in modules.values():
            module.force_expert = None
        clear_routing_graphs(model)
    combined = [None] * world
    dist.all_gather_object(combined, local)
    model.train()
    if rank != 0:
        return None
    records = sorted([r for shard in combined for r in shard], key=lambda r: r['record_id'])
    if len(records) != len(rows) or len({r['record_id'] for r in records}) != len(rows):
        raise RuntimeError('evaluation lost/duplicated records')
    for row in records:
        if not all(math.isfinite(row[k]) for k in ('input_psnr', 'output_psnr', 'input_lpips', 'output_lpips')):
            raise RuntimeError('nonfinite metric')
    summary = {'split': split, 'step': step, 'records': len(records), 'route': suffix,
               'psnr': float(np.mean([r['output_psnr'] for r in records])),
               'lpips': float(np.mean([r['output_lpips'] for r in records])),
               'mean_expert_weights': {name: np.mean([r['expert_weights'][name] for r in records], axis=0).tolist()
                                      for name in records[0]['expert_weights']},
               'rows': records, 'gt_used_for_routing': False}
    write_json(output / 'metrics' / f'{split}_{step:06d}_{suffix}.json', summary)
    print('RAIN_BLUR_MOE_METRICS ' + json.dumps({k: v for k, v in summary.items() if k != 'rows'}), flush=True)
    return summary


def check_rain_parity(actual, config):
    reference = json.loads((ROOT / config['rain_metrics']).read_text())
    a = {r['record_id']: r for r in actual['rows']}
    b = {r['record_id']: r for r in reference['rows']}
    if a.keys() != b.keys():
        raise RuntimeError('initial Rain parity used different records')
    errors = {}
    for metric in ('psnr', 'lpips'):
        if any(a[k]['source_group'] != b[k]['source_group'] or
               abs(a[k]['input_' + metric] - b[k]['input_' + metric]) > 1e-6 for k in a):
            raise RuntimeError('initial Rain inputs/crops changed')
        errors[metric] = max(abs(a[k]['output_' + metric] - b[k]['output_' + metric]) for k in a)
        if errors[metric] > config['evaluation']['parity_max_' + metric + '_error']:
            raise RuntimeError(f'MoE insertion changed working Rain outputs: {errors}')
    if actual['psnr'] < 34.2 or actual['lpips'] > .08:
        raise RuntimeError('initial Rain gate was lost')
    return {'status': 'working_rain_preserved_before_training', 'maximum_metric_errors': errors}


def save_checkpoint(model, optimizer, output, step, config, arm, mode):
    from safetensors.torch import save_file
    from safetensors import safe_open
    path = output / f'step_{step:06d}.safetensors'
    state = trainable_state(model)
    metadata = {'step': str(step), 'config_sha256': CONFIG_SHA256, 'arm': arm, 'mode': mode,
                'rain_lora_sha256': config['rain_lora_sha256']}
    save_file(state, str(path), metadata=metadata)
    with safe_open(path, framework='pt', device='cpu') as handle:
        if set(handle.keys()) != set(state) or any(handle.get_slice(k).get_dtype() != 'F32' for k in state):
            raise RuntimeError('checkpoint trainable-state mismatch')
    del state
    with path.with_suffix('.optimizer.pt').open('xb') as handle:
        torch.save({'step': step, 'config_sha256': CONFIG_SHA256, 'arm': arm, 'mode': mode,
                    'optimizer': optimizer.state_dict(), 'next_sample_position': step * 32,
                    'parameter_names': [n for n, p in model.named_parameters() if p.requires_grad]}, handle)
    return {'step': step, 'weights_sha256': old.sha256(path),
            'optimizer_sha256': old.sha256(path.with_suffix('.optimizer.pt'))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/foundir_rain_blur_image_moe_v1.json')
    parser.add_argument('--mode', choices=('qualification', 'formal'), default='qualification')
    parser.add_argument('--arm', choices=('moe', 'dense'), default='moe')
    parser.add_argument('--run-id', default='qualification_v1')
    parser.add_argument('--qualification-result', type=Path)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    if args.preflight:
        preflight(args.config)
        return
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.run_id):
        raise RuntimeError('unsafe run ID')
    config, parent, training, development, data = read_contract(args.config)
    sources = source_hashes()
    if args.mode == 'formal':
        if args.qualification_result is None:
            raise RuntimeError('formal requires its same-arm passing qualification')
        q = json.loads(args.qualification_result.read_text())
        if not (q['status'] == 'complete' and q['engineering_pass'] and q['arm'] == args.arm
                and q['mode'] == 'qualification' and q['config_sha256'] == CONFIG_SHA256
                and q['source_sha256'] == sources and q['data'] == data and q['initial_rain_parity_pass']):
            raise RuntimeError('qualification receipt mismatch')
    rank, world, local = int(os.environ['RANK']), int(os.environ['WORLD_SIZE']), int(os.environ['LOCAL_RANK'])
    if world != 8 or int(os.environ.get('LOCAL_WORLD_SIZE', 8)) != 8 or torch.cuda.device_count() != 8:
        raise RuntimeError('requires one homogeneous eight-GPU node')
    torch.cuda.set_device(local)
    device = torch.device('cuda', local)
    prop = torch.cuda.get_device_properties(device)
    if prop.total_memory < 75 * 1024**3 or not any(n in prop.name for n in ('H100', 'H200')):
        raise RuntimeError('requires H100/H200 >=75 GiB')
    dist.init_process_group('nccl', device_id=device, timeout=timedelta(minutes=30))
    output = ROOT / 'checkpoints/foundir_rain_blur_image_moe_v1' / args.arm / args.run_id
    started = time.monotonic()
    try:
        names = [None] * world
        dist.all_gather_object(names, prop.name)
        if len(set(names)) != 1:
            raise RuntimeError('mixed GPU hardware')
        error = [None]
        if rank == 0:
            try:
                if output.exists():
                    raise RuntimeError('existing output directory; preserve it, do not overwrite')
                preflight(args.config)
                output.mkdir(parents=True)
                write_json(output / 'receipt.json', {'config': config, 'config_sha256': CONFIG_SHA256,
                           'source_sha256': sources, 'data': data, 'arm': args.arm, 'mode': args.mode})
            except Exception as exc:
                error[0] = str(exc)
        dist.broadcast_object_list(error, src=0)
        if error[0]:
            raise RuntimeError(error[0])
        seed = config['training']['seed']
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
            TORCH_HOME=str(ROOT / parent['runtime']['torch_home']),
            KLEIN_4B_BASE_MODEL_PATH=str(ROOT / parent['model']['flow_weights']),
            AE_MODEL_PATH=str(ROOT / parent['model']['autoencoder_weights']))
        from flux2.util import load_ae, load_flow_model
        from flux2.sampling import batched_prc_img
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
        context, context_ids, _, _ = old.encode_prompt(parent, device)
        ae = load_ae(parent['model']['name'], device=device).eval().requires_grad_(False)
        model = load_flow_model(parent['model']['name'], device=device).eval().requires_grad_(False)
        targets = inject_lora(model, parent['lora']['target_patterns'], rank=32, alpha=32, dropout=0)
        if len(targets) != 50:
            raise RuntimeError('Rain LoRA targets changed')
        promote_lora_to_fp32(model)
        load_lora_checkpoint(model, ROOT / config['rain_lora'])
        modules = inject_image_moe(model, indices=config['architecture']['blocks'],
                                   experts=2 if args.arm == 'moe' else 1)
        trainable = [p for p in model.parameters() if p.requires_grad]
        count = sum(p.numel() for p in trainable)
        if count != config['architecture']['trainable_parameters'][args.arm]:
            raise RuntimeError('trainable parameter budget changed')
        if any(p.dtype != torch.float32 for p in trainable):
            raise RuntimeError('trainable parameters must be FP32')
        expert_params, router_params = [], []
        for module in modules.values():
            expert_params.extend(module.experts.parameters())
            if module.router is not None:
                router_params.extend(module.router.parameters())
        groups = [{'params': expert_params, 'lr': config['training']['expert_lr']}]
        if router_params:
            groups.append({'params': router_params, 'lr': config['training']['router_lr']})
        optimizer = torch.optim.AdamW(groups, weight_decay=0, fused=True)
        wrapped = DDP(model, device_ids=[local], broadcast_buffers=False, gradient_as_bucket_view=True)
        teacher = old.load_teacher(parent, device)
        metric = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).to(device).eval().requires_grad_(False)
        evaluation_args = dict(rank=rank, world=world, device=device, output=output)
        history = []
        checkpoints = []
        def assess(step, final=False):
            summaries = {}
            keys = ['rain', 'blur_full' if final and args.mode == 'formal' else 'blur_preview']
            for split in keys:
                summaries[split] = evaluate(model, ae, development[split], context, context_ids,
                    batched_prc_img, metric, split=split, step=step, **evaluation_args)
            if rank == 0:
                history.append(summaries)
            return summaries
        initial = assess(0)
        error = [None]
        if rank == 0:
            try:
                parity = check_rain_parity(initial['rain'], config)
                write_json(output / 'initial_rain_parity.json', parity)
                print('RAIN_BLUR_MOE_INITIAL_PARITY ' + json.dumps(parity), flush=True)
            except Exception as exc:
                error[0] = str(exc)
        dist.broadcast_object_list(error, src=0)
        if error[0]:
            raise RuntimeError(error[0])
        stages = config['modes'][args.mode]
        dataset = RainBlurDataset(ROOT, training, steps=stages['steps'], rank=rank, seed=seed)
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2,
                 prefetch_factor=2, multiprocessing_context='spawn', pin_memory=True)
        iterator = iter(loader)
        times, seen, observed = [], {'rain': set(), 'blur': set()}, {'experts.0': False, 'experts.1': args.arm == 'dense', 'router': args.arm == 'dense'}
        sampled_initial = {n: p.detach().flatten()[:64].cpu().clone() for n, p in model.named_parameters() if p.requires_grad}
        for step in range(stages['steps']):
            tick = time.monotonic()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            fraction = warmup_fraction(step, stages['hard_steps'], stages['transition_end']) if args.arm == 'moe' else 0
            totals = {}
            for micro in range(4):
                batch = next(iterator)
                if int(batch['position'][0]) != step * 32 + micro * world + rank:
                    raise RuntimeError('task/data position changed')
                task = int(batch['task_id'][0])
                seen['rain' if task == 0 else 'blur'].add(batch['record_id'][0])
                lq, gt = batch['lq'].to(device, non_blocking=True), batch['gt'].to(device, non_blocking=True)
                labels = batch['task_id'].to(device)
                with torch.no_grad():
                    clean = ae.encode(gt.float()).float()
                    teacher_output = teacher((lq + 1) * .5).clamp(0, 1) if task == 0 else (gt + 1) * .5
                weights = dict(parent['training']['losses'])
                if task == 1:
                    weights['teacher_l1_weight'] = 0.0
                with wrapped.no_sync() if micro < 3 else nullcontext():
                    with training_routes(model, labels, fraction):
                        prediction = restore_endpoint(wrapped, ae, lq, context, context_ids, batched_prc_img)
                        loss, terms = full.training_losses(prediction, clean, lq, gt, teacher_output,
                                                           metric, batched_prc_img, weights)
                        ce = router_classification_loss(model, labels)
                        objective = loss + config['training']['router_ce_weight'] * ce
                        if not bool(torch.isfinite(objective)):
                            raise RuntimeError('nonfinite objective')
                        (objective / 4).backward()
                    clear_routing_graphs(model)
                for name, value in {'objective': objective, 'router_ce': ce, **terms}.items():
                    totals[name] = totals.get(name, 0.0) + float(value.detach()) / 4
            norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            if not math.isfinite(float(norm)):
                raise RuntimeError('nonfinite gradient')
            optimizer.step()
            assert_fp32_optimizer(optimizer)
            vector = torch.tensor(list(totals.values()), device=device, dtype=torch.float64)
            dist.all_reduce(vector); vector /= world
            torch.cuda.synchronize()
            times.append(time.monotonic() - tick)
            completed = step + 1
            if rank == 0 and (completed == 1 or completed % 10 == 0 or completed in stages['milestones']):
                print('RAIN_BLUR_MOE_TRAIN ' + json.dumps({'step': completed, 'max_steps': stages['steps'],
                    'arm': args.arm, 'teacher_routing_fraction': fraction,
                    'losses': dict(zip(totals, vector.tolist())), 'grad_norm': float(norm),
                    'seconds': times[-1], 'peak_memory_gib': torch.cuda.max_memory_allocated() / 1024**3}), flush=True)
            if completed in stages['milestones']:
                if rank == 0:
                    checkpoints.append(save_checkpoint(model, optimizer, output, completed, config, args.arm, args.mode))
                dist.barrier()
                last = assess(completed, final=completed == stages['steps'])
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                changed = not torch.equal(parameter.detach().flatten()[:64].cpu(), sampled_initial[name])
                for group in observed:
                    if group in name:
                        observed[group] |= changed
        if not all(observed.values()):
            raise RuntimeError(f'no observed update in required branch: {observed}')
        diagnostics = {}
        if args.arm == 'moe':
            for expert in (0, 1):
                for split in ('rain', 'blur_preview'):
                    diagnostics[f'{split}_expert_{expert}'] = evaluate(model, ae, development[split],
                        context, context_ids, batched_prc_img, metric, split=split, step=stages['steps'],
                        force_expert=expert, **evaluation_args)
        all_seen = [None] * world
        dist.all_gather_object(all_seen, seen)
        if rank == 0:
            elapsed = time.monotonic() - started
            final_gate = (args.mode == 'formal' and last['rain']['psnr'] >= 34.2 and last['rain']['lpips'] <= .08
                          and last['blur_full']['psnr'] >= 25.9 and last['blur_full']['lpips'] <= .18)
            result = {'status': 'complete', 'engineering_pass': True, 'initial_rain_parity_pass': True,
                'arm': args.arm, 'mode': args.mode, 'steps': stages['steps'], 'config_sha256': CONFIG_SHA256,
                'source_sha256': sources, 'data': data, 'history': history, 'checkpoints': checkpoints,
                'constant_expert_diagnostics': diagnostics, 'both_task_gates_pass': final_gate,
                'qualification_is_engineering_only': args.mode == 'qualification',
                'trainable_parameters': count, 'sampled_updates': observed,
                'sample_presentations': stages['steps'] * 32,
                'unique_records_seen': {task: len(set.union(*(s[task] for s in all_seen))) for task in seen},
                'wall_clock_hours': elapsed / 3600, 'allocated_gpu_hours': elapsed * world / 3600,
                'median_training_step_seconds': float(np.median(times[1:])),
                'peak_memory_gib': torch.cuda.max_memory_allocated() / 1024**3,
                'gpu_name': prop.name, 'protected_pixels_accessed': False,
                'automatic_next_launch': False, 'scientific_moe_advantage_requires_dense_control': True}
            write_json(output / 'result.json', result)
            print('RAIN_BLUR_MOE_END ' + json.dumps({k: v for k, v in result.items()
                  if k not in ('history', 'constant_expert_diagnostics', 'source_sha256', 'data')}), flush=True)
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
