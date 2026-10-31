#!/usr/bin/env python3
"""Only right-diagram VLM residual routing; four nodes x two GPUs."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import random
import re
import socket
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'third_party/flux2/src'))

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from safetensors.torch import load_file

from smoke.token_residual_moe_v1 import (CONFIG, TASKS, Samples, contract, verify_assets, verify_qualification,
    source_hashes, sha, digest, write_once, feature_binding, read_feature, save_feature)
from lowlight_restoration.token_residual_moe import attach_token_router, token_conditioning, batch_slot
from lowlight_restoration.joint_vlm_moe import install_joint_moe, enable_block_checkpointing, restoration_loss
from lowlight_restoration.joint_moe_state import save_state, rng_state
from lowlight_restoration.blur_single_endpoint import restore_rng, check_adam
from lowlight_restoration.rain_blur_image_moe import load_expert_state, router_classification_loss
from lowlight_restoration.lora import inject_lora
from lowlight_restoration.flux2_endpoint_precision import promote_lora_to_fp32, restore_endpoint, assert_fp32_optimizer


def rank0(action):
    result = [None]
    if dist.get_rank() == 0:
        try:
            action()
        except Exception as error:
            result[0] = repr(error)
    dist.broadcast_object_list(result, src=0)
    if result[0]:
        raise RuntimeError(result[0])


def distributed_start(binding, output):
    rank, local = int(os.environ['RANK']), int(os.environ['LOCAL_RANK'])
    if (int(os.environ['WORLD_SIZE']), int(os.environ['LOCAL_WORLD_SIZE']), torch.cuda.device_count()) != (8, 2, 2):
        raise RuntimeError('requires FOUR nodes, TWO visible GPUs per node, eight ranks')
    torch.cuda.set_device(local)
    device = torch.device('cuda', local)
    prop = torch.cuda.get_device_properties(device)
    if prop.total_memory < 75 * 1024**3 or not any(n in prop.name for n in ('H100', 'H200')):
        raise RuntimeError('requires H100/H200 >=75GiB per GPU')
    dist.init_process_group('nccl', device_id=device, timeout=timedelta(minutes=30))
    from importlib.metadata import version
    versions = {name: version(name) for name in
                ('torch', 'torchvision', 'transformers', 'safetensors', 'numpy', 'Pillow', 'torchmetrics')}
    local_info = {'binding': binding, 'hostname': socket.gethostname(), 'gpu': prop.name,
                  'versions': versions, 'cuda_runtime': torch.version.cuda}
    infos = [None] * 8
    dist.all_gather_object(infos, local_info)
    if any(r['binding'] != binding or r['versions'] != infos[0]['versions']
           or r['cuda_runtime'] != torch.version.cuda for r in infos):
        raise RuntimeError('cross-rank code/config/runtime mismatch')
    if len({r['hostname'] for r in infos}) != 4:
        raise RuntimeError('four distinct nodes required')
    for start in (0, 2, 4, 6):
        if (infos[start]['hostname'], infos[start]['gpu']) != (infos[start+1]['hostname'], infos[start+1]['gpu']):
            raise RuntimeError('each node must expose two same-model GPUs')
    # Every node must see each other node's freshly generated marker.
    import uuid
    nonce = uuid.uuid4().hex
    nonces = [None] * 8
    dist.all_gather_object(nonces, nonce)
    path = output / 'shared_checks' / f'{rank}_{nonce}.json'
    write_once(path, {'rank': rank, 'nonce': nonce})
    dist.barrier()
    for r, n in enumerate(nonces):
        p = output / 'shared_checks' / f'{r}_{n}.json'
        if json.loads(p.read_text()) != {'rank': r, 'nonce': n}:
            raise RuntimeError('workspace is not shared across nodes')
    return rank, local, device, infos


def extract(samples, specs, cache_dir, config_sha, qwen, qcfg, rank, device, binding):
    assigned = specs[rank::8]
    missing = [s for s in assigned if read_feature(cache_dir / (s['key'] + '.safetensors'),
                   feature_binding(s, config_sha, qwen)) is None]
    if missing:
        from smoke.train_qwen_multidegradation_lora_ddp import load_model
        from lowlight_restoration.qwen_lora import load_qwen_adapter
        from lowlight_restoration.qwen_five_class_gate import image_embedding_sequence
        from PIL import Image
        model, processor, _ = load_model(qcfg, variant='merger_trainable', device=device)
        load_qwen_adapter(model, ROOT / qwen['qwen']['adapter'])
        model.eval().requires_grad_(False)
        for index, spec in enumerate(missing):
            lq, _ = samples.pair(spec)
            array = ((lq + 1) * 127.5).round().clamp(0, 255).byte().permute(1, 2, 0).numpy()
            inputs = processor.image_processor(images=[Image.fromarray(array)], return_tensors='pt')
            with torch.inference_mode():
                seq = image_embedding_sequence(model.get_image_features(
                    inputs.pixel_values.to(device), inputs.image_grid_thw.to(device)))
                if len(seq) != 1 or seq[0].ndim != 2 or seq[0].shape[-1] != 3584:
                    raise RuntimeError('unexpected Qwen visual merger tokens')
                feature = seq[0].float().mean(0).cpu()
            if not torch.isfinite(feature).all():
                raise RuntimeError('nonfinite Qwen feature')
            save_feature(cache_dir / (spec['key'] + '.safetensors'), feature,
                         feature_binding(spec, config_sha, qwen))
            if index % 20 == 0 or index + 1 == len(missing):
                print(f'TOKEN_FEATURE rank={rank} completed={index+1}/{len(missing)}', flush=True)
        del model, processor
    dist.barrier()
    def finish():
        identities = {}
        for spec in specs:
            path = cache_dir / (spec['key'] + '.safetensors')
            if read_feature(path, feature_binding(spec, config_sha, qwen)) is None:
                raise RuntimeError('incomplete token cache')
            identities[spec['key']] = sha(path)
        write_once(cache_dir / 'receipt.json', {'binding': binding, 'files': identities,
            'records': len(specs), 'GT_input_to_VLM': False, 'exact_restoration_crop': True})
        print('TOKEN_FEATURES_COMPLETE', flush=True)
    rank0(finish)


def parameter_groups(model, parent):
    groups = {name: [] for name in ('shared_lora', 'experts', 'router', 'vlm')}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dtype != torch.float32:
            raise RuntimeError('trainable parameters must remain FP32')
        if '.lora_A.' in name or '.lora_B.' in name:
            key = 'shared_lora'
        elif '.experts.' in name:
            key = 'experts'
        elif '.vlm_projector.' in name:
            key = 'vlm'
        elif '.router.' in name:
            key = 'router'
        else:
            raise RuntimeError('unexpected trainable parameter: ' + name)
        groups[key].append(p)
    if any(not values for values in groups.values()):
        raise RuntimeError('missing trainable component')
    rates = {'shared_lora': 'shared_lora_lr', 'experts': 'expert_lr', 'router': 'router_lr', 'vlm': 'router_lr'}
    return [{'params': values, 'lr': parent['training'][rates[key]], 'name': key}
            for key, values in groups.items()]


def resume_training(model, optimizer, path, binding, rank, *, mode, total_steps):
    receipt = json.loads(path.read_text())
    if not (receipt['mode'] == mode and receipt['arm'] == 'token_residual'
            and receipt['config_sha256'] == binding['config_sha256']
            and receipt['source_sha256'] == binding['source_sha256']
            and receipt['world_size'] == 8 and 0 < receipt['step'] <= total_steps
            and receipt['next_sample_position'] == receipt['step'] * 32):
        raise RuntimeError('only matching same-mode 4x2 state may resume; qualification cannot seed formal')
    launch = json.loads((path.parent / 'launch.json').read_text())
    if launch['binding'] != binding:
        raise RuntimeError('resume data/cache/initialization mismatch')
    for key in ('weights', 'optimizer'):
        if Path(receipt[key]).name != receipt[key] or sha(path.parent / receipt[key]) != receipt[key + '_sha256']:
            raise RuntimeError('resume artifact hash mismatch')
    saved = torch.load(path.parent / receipt['optimizer'], map_location='cpu', weights_only=True)
    for key in ('step', 'mode', 'arm', 'config_sha256', 'source_sha256', 'world_size', 'next_sample_position'):
        if saved[key] != receipt[key]:
            raise RuntimeError('resume receipt/state mismatch')
    if saved['parameter_names'] != [n for n, p in model.named_parameters() if p.requires_grad] or len(saved['rank_rng']) != 8:
        raise RuntimeError('resume parameter order/RNG mismatch')
    expected = [{k: v for k, v in g.items() if k != 'params'} for g in optimizer.param_groups]
    load_expert_state(model, path.parent / receipt['weights'])
    optimizer.load_state_dict(saved['optimizer'])
    if expected != [{k: v for k, v in g.items() if k != 'params'} for g in optimizer.param_groups]:
        raise RuntimeError('optimizer settings changed')
    check_adam(optimizer, receipt['step'])
    restore_rng(saved['rank_rng'][rank])
    return receipt['step']


def resume_qualification(model, optimizer, path, binding, rank):
    return resume_training(model, optimizer, path, binding, rank, mode='qualification', total_steps=25)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('features', 'train', 'preflight'), required=True)
    parser.add_argument('--run-id', default='qualification_4x2_v1')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--config', type=Path, default=ROOT / CONFIG)
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.run_id):
        raise ValueError('unsafe run ID')
    cfg, parent, base, qwen, qcfg, training, development, data = contract(ROOT, args.config)
    config_relative = str(args.config.resolve().relative_to(ROOT))
    config_sha, sources = sha(args.config), source_hashes(ROOT, config_relative)
    mode = cfg['mode']
    qualification = verify_qualification(ROOT, cfg)
    if qualification and qualification['qualified_data'] != data:
        raise RuntimeError('formal training pool changed from qualified source-disjoint protocol')
    samples = Samples(ROOT, cfg, training, development)
    specs = samples.all_specs()
    binding = {'config_sha256': config_sha, 'source_sha256': sources,
               'data': data, 'samples_sha256': digest(specs)}
    cache_dir = ROOT / cfg['output_root'] / 'token_cache' / digest(binding)
    output = ROOT / cfg['output_root'] / args.run_id
    if args.phase == 'preflight':
        verify_assets(ROOT, cfg, base, qwen)
        if mode == 'formal':
            import shutil
            if shutil.disk_usage(ROOT).free < cfg['minimum_free_gib'] * 1024**3:
                raise RuntimeError('formal requires 160GiB free for new checkpoints; do not delete old assets')
        print(json.dumps({'status': 'cpu_preflight_pass', 'binding': binding,
            'sample_presentations': cfg['steps'] * 32,
            'validation_records': sum(s['kind'] == 'validation' for s in specs), 'mode': mode,
            'GPU_started': False, 'protected_pixels_accessed': False}), flush=True)
        return
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        TORCH_HOME=str(ROOT / base['runtime']['torch_home']),
        KLEIN_4B_BASE_MODEL_PATH=str(ROOT / base['model']['flow_weights']),
        AE_MODEL_PATH=str(ROOT / base['model']['autoencoder_weights']))
    phase_binding = {**binding, 'run_id': args.run_id, 'phase': args.phase,
                     'resume': str(args.resume) if args.resume else None}
    rank, local, device, hardware = distributed_start(phase_binding, ROOT / cfg['output_root'])
    started = time.monotonic()
    try:
        rank0(lambda: verify_assets(ROOT, cfg, base, qwen))
        if args.phase == 'features':
            if mode == 'formal' and not args.resume:
                def check_space():
                    import shutil
                    if shutil.disk_usage(ROOT).free < cfg['minimum_free_gib'] * 1024**3:
                        raise RuntimeError('formal requires 160GiB free; no automatic deletion')
                rank0(check_space)
            extract(samples, specs, cache_dir, config_sha, qwen, qcfg, rank, device, binding)
            return
        cache_receipt = json.loads((cache_dir / 'receipt.json').read_text())
        if cache_receipt['binding'] != binding or cache_receipt['records'] != len(specs):
            raise RuntimeError('complete matching token cache required')
        verified_features = set()
        def feature(spec):
            path = cache_dir / (spec['key'] + '.safetensors')
            if spec['key'] not in verified_features:
                if sha(path) != cache_receipt['files'][spec['key']]:
                    raise RuntimeError('cached feature bytes changed')
                verified_features.add(spec['key'])
            value = read_feature(path, feature_binding(spec, config_sha, qwen))
            if value is None:
                raise RuntimeError('missing feature; no severity-label substitution')
            return value[None].to(device)
        def prepare():
            output.mkdir(parents=True, exist_ok=False)
            write_once(output / 'launch.json', {'binding': binding, 'hardware': hardware,
                'initialization': 'image_only_step11000_weights_fresh_Adam',
                'resume': str(args.resume) if args.resume else None, 'mode': mode,
                'qualification': qualification})
            if mode == 'formal' and args.resume:
                baseline = json.loads((args.resume.parent / 'baseline.json').read_text())
                if sha(ROOT / baseline['path']) != baseline['sha256']:
                    raise RuntimeError('original same-environment baseline changed')
                write_once(output / 'baseline.json', baseline)
        rank0(prepare)
        random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
        torch.cuda.manual_seed_all(cfg['seed'])
        from flux2.util import load_flow_model, load_ae
        from flux2.sampling import batched_prc_img
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
        prompt = load_file(str(ROOT / base['model']['prompt_context_cache']))
        context, ids = prompt['cfg_context'][:1].to(device), prompt['cfg_ids'][:1].to(device)
        del prompt
        ae = load_ae(base['model']['name'], device=device).eval().requires_grad_(False)
        model = load_flow_model(base['model']['name'], device=device).eval()
        names = inject_lora(model, base['lora']['target_patterns'], rank=32, alpha=32, dropout=0)
        if len(names) != 50:
            raise RuntimeError('shared LoRA structure changed')
        promote_lora_to_fp32(model)
        install_joint_moe(model, arm='image_only', indices=cfg['ffn_blocks'])
        load_expert_state(model, ROOT / cfg['initial_weights'])
        model.eval()
        probe = samples.train_spec(rank)
        lq = samples.pair(probe)[0][None].to(device)
        with torch.inference_mode():
            expected = restore_endpoint(model, ae, lq, context, ids, batched_prc_img)['decoded']
        modules = attach_token_router(model, cfg['feature_dim'], cfg['router_hidden'])
        model.eval()
        with torch.inference_mode(), token_conditioning(model, feature(probe)):
            actual = restore_endpoint(model, ae, lq, context, ids, batched_prc_img)['decoded']
        error = (actual - expected).abs().max()
        dist.all_reduce(error, op=dist.ReduceOp.MAX)
        if not torch.isfinite(error) or float(error) > 1e-5:
            raise RuntimeError('zero VLM insertion did not preserve the loaded image-only MoE')
        del expected, actual, lq
        enable_block_checkpointing(model)
        groups = parameter_groups(model, parent)
        optimizer = torch.optim.AdamW(groups, weight_decay=0, fused=True)
        metric = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).to(device).eval().requires_grad_(False)
        initial_vlm = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if '.vlm_projector.' in n}
        start_step = resume_training(model, optimizer, args.resume, binding, rank,
            mode=mode, total_steps=cfg['steps']) if args.resume else 0
        restored_rng = rng_state()
        wrapped = DDP(model, device_ids=[local], broadcast_buffers=False, gradient_as_bucket_view=True)
        params = [p for g in groups for p in g['params']]
        def evaluate(step, *, full=False):
            dev_specs = samples.evaluation_specs(specs, full=full)
            model.eval(); rows = []
            for spec in dev_specs[rank::8]:
                lq, gt = [v[None].to(device) for v in samples.pair(spec)]
                with torch.inference_mode(), token_conditioning(model, feature(spec)):
                    decoded = restore_endpoint(model, ae, lq, context, ids, batched_prc_img)['decoded']
                    quantized = ((decoded + 1) * 127.5).round().clamp(0, 255) / 127.5 - 1
                    mse = float((((gt.float() - quantized.float()) * .5).square()).mean())
                    psnr = 100. if mse <= 1e-12 else -10. * math.log10(mse)
                    lpips = float(metric((quantized + 1) * .5, (gt + 1) * .5))
                    if not math.isfinite(psnr + lpips):
                        raise RuntimeError('nonfinite evaluation')
                    report_split = 'blur_preview' if spec['split'] == 'blur_full' and not full else spec['split']
                    rows.append({'record_id': spec['row']['record_id'], 'source_group': spec['row']['source_group'],
                        'split': report_split,
                        'psnr': psnr, 'lpips': lpips,
                        'weights': {k: m.last_weights[0].cpu().tolist() for k, m in modules.items()}})
            gathered = [None] * 8
            dist.all_gather_object(gathered, rows)
            def save_metrics():
                all_rows = sorted([r for part in gathered for r in part], key=lambda r: r['record_id'])
                if (len(all_rows) != len(dev_specs)
                        or {r['record_id'] for r in all_rows} != {s['row']['record_id'] for s in dev_specs}):
                    raise RuntimeError('evaluation record mismatch')
                evaluated_splits = sorted({r['split'] for r in all_rows})
                summary = {split: {m: float(np.mean([r[m] for r in all_rows if r['split'] == split]))
                          for m in ('psnr', 'lpips')} for split in evaluated_splits}
                if full and mode == 'formal':
                    preview_ids = {r['record_id'] for r in samples.development['blur_preview']}
                    summary['blur_preview'] = {m: float(np.mean([r[m] for r in all_rows
                        if r['record_id'] in preview_ids])) for m in ('psnr', 'lpips')}
                write_once(output / f'metrics_{step:06d}.json', {'step': step, 'rows': all_rows,
                    'summary': summary, 'GT_used_for_routing': False, 'full_Blur_gate': full and mode == 'formal'})
                print('TOKEN_MOE_METRICS ' + json.dumps({'step': step, **summary}), flush=True)
            rank0(save_metrics)
        evaluate(start_step, full=start_step in cfg['full_evaluation_steps'])
        if mode == 'formal' and not args.resume:
            def bind_baseline():
                path = output / 'metrics_000000.json'
                write_once(output / 'baseline.json', {'path': str(path.relative_to(ROOT)), 'sha256': sha(path)})
            rank0(bind_baseline)
        restore_rng(restored_rng)
        times, checkpoints = [], []
        for step in range(start_step, cfg['steps']):
            tick = time.monotonic()
            model.train(); optimizer.zero_grad(set_to_none=True)
            total, max_adjustment = 0., 0.
            for micro in range(4):
                offset, scale = batch_slot(step, micro, rank)
                spec = samples.train_spec(offset)
                lq, gt = [v[None].to(device) for v in samples.pair(spec)]
                condition = feature(spec)
                weights = dict(parent['training']['losses'])
                weights['lpips'] = cfg['rain_lpips_weight'] if spec['task_id'] == 0 else cfg['blur_lpips_weight']
                with torch.no_grad():
                    clean = ae.encode(gt.float()).float()
                with wrapped.no_sync() if micro < 3 else nullcontext():
                    with token_conditioning(model, condition):
                        result = restore_endpoint(wrapped, ae, lq, context, ids, batched_prc_img)
                        loss, terms = restoration_loss(result, clean, gt, metric, batched_prc_img, weights)
                        # Historical training-only auxiliary CE, never a hard routing input.
                        labels = torch.tensor([spec['task_id']], device=device)
                        objective = loss + parent['training']['router_ce_weight'] * router_classification_loss(model, labels)
                        if not torch.isfinite(objective):
                            raise RuntimeError('nonfinite loss')
                        max_adjustment = max(max_adjustment, max(float(m.last_adjustment.detach().abs().max()) for m in modules.values()))
                        (objective * scale).backward()
                        total += float(objective.detach()) * scale
            if any(p.grad is None for p in params):
                raise RuntimeError('disconnected trainable parameter')
            norm = torch.nn.utils.clip_grad_norm_(params, 1.)
            if not math.isfinite(float(norm)):
                raise RuntimeError('nonfinite gradient')
            vlm_grad = math.sqrt(sum(float(p.grad.float().square().sum()) for n, p in model.named_parameters() if '.vlm_projector.' in n))
            optimizer.step(); assert_fp32_optimizer(optimizer)
            torch.cuda.synchronize(); times.append(time.monotonic() - tick)
            values = torch.tensor([total, max_adjustment, vlm_grad], device=device)
            dist.all_reduce(values); values /= 8
            completed = step + 1
            if rank == 0:
                print('TOKEN_MOE_TRAIN ' + json.dumps({'step': completed, 'steps': cfg['steps'], 'mode': mode,
                    'loss': float(values[0]), 'mean_rank_max_vlm_adjustment': float(values[1]),
                    'vlm_grad_norm_after_clip': float(values[2]), 'seconds': times[-1],
                    'peak_memory_gib': torch.cuda.max_memory_allocated() / 1024**3}), flush=True)
            if completed % cfg['save_every'] == 0 or completed in cfg['milestones']:
                states = [None] * 8
                dist.all_gather_object(states, rng_state())
                def save():
                    checkpoints.append(save_state(model, optimizer, output, step=completed,
                        mode=mode, arm='token_residual', config_sha=config_sha,
                        sources=sources, rank_rng=states))
                rank0(save)
            if completed in cfg['milestones']:
                before_eval = rng_state()
                evaluate(completed, full=completed in cfg['full_evaluation_steps'])
                restore_rng(before_eval)
        changed = any(not torch.equal(p.detach().cpu(), initial_vlm[n])
                      for n, p in model.named_parameters() if n in initial_vlm)
        observations = [None] * 8
        dist.all_gather_object(observations, {'changed': changed,
            'peak_memory_gib': torch.cuda.max_memory_allocated() / 1024**3})
        def finish():
            if not all(v['changed'] for v in observations):
                raise RuntimeError('VLM score branch did not update')
            hours = (time.monotonic() - started) / 3600
            scientific = {'absolute_quality_gate_pass': None, 'gain_vs_start': None}
            if mode == 'formal':
                reference = json.loads((output / 'baseline.json').read_text())
                if sha(ROOT / reference['path']) != reference['sha256']:
                    raise RuntimeError('initial full baseline identity changed')
                baseline = json.loads((ROOT / reference['path']).read_text())['summary']
                final = json.loads((output / f'metrics_{cfg["steps"]:06d}.json').read_text())['summary']
                scientific['absolute_quality_gate_pass'] = (final['rain']['psnr'] >= 34.2
                    and final['rain']['lpips'] <= .08 and final['blur_full']['psnr'] >= 25.9
                    and final['blur_full']['lpips'] <= .18)
                scientific['gain_vs_start'] = {split: {m: final[split][m]-baseline[split][m]
                    for m in ('psnr', 'lpips')} for split in ('rain', 'blur_full')}
            write_once(output / 'result.json', {'status': 'complete' if mode == 'formal' else 'engineering_pass',
                'engineering_pass': True, 'mode': mode, 'binding': binding, **scientific,
                'steps': cfg['steps'], 'start_step': start_step, 'initial_parity_max_abs': float(error),
                'checkpoints': checkpoints, 'observations': observations,
                'median_training_seconds': float(np.median(times)) if times else None, 'training_wall_hours': hours,
                'training_allocated_gpu_hours': hours * 8,
                'scientific_gate_proven': False, 'VLM_advantage_proven': False,
                'automatic_formal': False, 'protected_pixels_accessed': False})
            print('TOKEN_MOE_FORMAL_COMPLETE' if mode == 'formal' else 'TOKEN_MOE_QUALIFICATION_PASS', flush=True)
        rank0(finish)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
