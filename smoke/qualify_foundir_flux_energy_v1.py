#!/usr/bin/env python3
"""Sequential offline FLUX cache -> energy-only 25-step engineering check.

There is no Restormer model, direct refiner, formal training, or sampler guidance.
The 32 training-source scores are NOT development/generalization results.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import random
import socket
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from smoke import flux_energy_qualification_contract_v1 as contract


def hardware():
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 8:
        raise RuntimeError('requires exactly eight visible GPUs')
    devices = [torch.cuda.get_device_properties(i) for i in range(8)]
    if (len({p.name for p in devices}) != 1 or len({str(p.uuid) for p in devices}) != 8
            or any(p.total_memory < 75 * 1024**3 or not any(k in p.name for k in ('H100', 'H200')) for p in devices)):
        raise RuntimeError('requires one homogeneous eight-H100/H200 node')
    for i in range(8):
        free, total = torch.cuda.mem_get_info(i)
        if total - free > 2 * 1024**3:
            raise RuntimeError(f'GPU {i} is not idle; do not overlap/kill existing jobs')
    print('HARDWARE_IDLE_PASS ' + json.dumps([p.name for p in devices]), flush=True)


def setup_ddp():
    import torch
    import torch.distributed as dist
    if int(os.environ['WORLD_SIZE']) != 8 or int(os.environ['LOCAL_WORLD_SIZE']) != 8:
        raise RuntimeError('single node, eight ranks required')
    local = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local)
    device = torch.device('cuda', local)
    dist.init_process_group('nccl', device_id=device, timeout=timedelta(minutes=30))
    hosts = [None] * 8
    dist.all_gather_object(hosts, socket.gethostname())
    if len(set(hosts)) != 1:
        raise RuntimeError('cross-node execution is not authorized')
    return dist.get_rank(), device


def prompts(output, receipt):
    import torch
    from flux2.text_encoder import Qwen3Embedder
    from flux2.sampling import batched_prc_txt
    from safetensors.torch import save_file
    from transformers import AutoModelForCausalLM, AutoTokenizer, FineGrainedFP8Config
    if torch.cuda.device_count() != 1:
        raise RuntimeError('prompt cache phase must expose only GPU 0')
    spec = receipt['spec']
    for filename, expected in receipt['artifacts'].items():
        if filename.startswith(spec['qwen'] + '/'):
            contract.bound(filename, expected)
    encoder = Qwen3Embedder.__new__(Qwen3Embedder)
    torch.nn.Module.__init__(encoder)
    encoder.model = AutoModelForCausalLM.from_pretrained(
        str(contract.path(spec['qwen'])), dtype=torch.bfloat16, device_map='cuda:0',
        quantization_config=FineGrainedFP8Config(dequantize=True), local_files_only=True).eval().requires_grad_(False)
    encoder.tokenizer = AutoTokenizer.from_pretrained(str(contract.path(spec['qwen'])), local_files_only=True)
    encoder.max_length = 512
    files = {}
    for task, prompt in spec['prompts'].items():
        with torch.inference_mode():
            encoded = encoder(['', prompt]).to(torch.bfloat16)
            contexts, ids = batched_prc_txt(encoded)
        if contexts.shape != (2, 512, 7680) or ids.shape != (2, 512, 4) or not bool(contexts.isfinite().all()):
            raise RuntimeError('unexpected/nonfinite Qwen prompt tensors')
        name = f'prompt_{task}.safetensors'
        if (output / name).exists():
            raise FileExistsError(name)
        save_file({'cfg_context': contexts.cpu().contiguous(), 'cfg_ids': ids.cpu().contiguous()},
                  str(output / name), metadata={'prompt': prompt, 'config_sha256': receipt['config_sha256']})
        files[name] = contract.sha(output / name)
    contract.write_new(output / 'prompts.json', dict(files=files, prompts=spec['prompts'],
                       config_sha256=receipt['config_sha256'], qwen_artifacts={k: v for k, v in receipt['artifacts'].items()
                       if k.startswith(spec['qwen'] + '/')}, encoding='offline FP8 checkpoint dequantized to BF16'))
    print('FLUX_ENERGY_PROMPTS_COMPLETE', flush=True)


def prompt_receipt(output, receipt):
    info = json.loads((output / 'prompts.json').read_text())
    if info['prompts'] != receipt['spec']['prompts'] or info['config_sha256'] != receipt['config_sha256']:
        raise RuntimeError('prompt identity mismatch')
    for name, expected in info['files'].items():
        if contract.sha(output / name) != expected:
            raise RuntimeError('prompt tensors changed')
    receipt['prompt_receipt_sha256'] = contract.sha(output / 'prompts.json')
    return info


def restore(model, ae, image, context, ids, seed):
    """Batch-one historical native reference inference; 50 steps, CFG 1, t_ref 10."""
    import torch
    from flux2.sampling import batched_prc_img, denoise_cfg, get_schedule, scatter_ids
    with torch.inference_mode():
        reference = ae.encode(image).to(torch.bfloat16)
        reference_tokens, reference_ids = batched_prc_img(reference)
        reference_ids = reference_ids.clone()
        reference_ids[..., 0] = 10
        generator = torch.Generator(device=image.device).manual_seed(seed)
        noise = torch.randn(reference.shape, generator=generator, device=image.device, dtype=torch.bfloat16)
        tokens, token_ids = batched_prc_img(noise)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            restored = denoise_cfg(model, tokens, token_ids, context, ids,
                                   timesteps=get_schedule(50, tokens.shape[1]), guidance=1.,
                                   img_cond_seq=reference_tokens, img_cond_seq_ids=reference_ids)
        latent = torch.cat(scatter_ids(restored, token_ids)).squeeze(2)
        return ae.decode(latent).float().clamp(-1, 1)


def image_crop(row):
    import numpy as np
    import torch
    from PIL import Image
    if row['split'] != 'train' or row['category'] != '05Noise':
        raise RuntimeError('refusing nontraining image access')
    x, y = contract.crop_xy(row)
    result = {}
    for key, field in (('observation', 'lq'), ('target', 'gt')):
        filename = contract.bound(row[field + '_path'], row[field + '_sha256'])
        with Image.open(filename) as handle:
            if handle.size != (int(row['width']), int(row['height'])):
                raise RuntimeError('image dimensions changed')
            array = np.array(handle.convert('RGB').crop((x, y, x+512, y+512)), copy=True)
        result[key] = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    return result


def deployed_proposal(observation, raw):
    from lowlight_restoration.scalar_residual import quantized_scalar_residual
    _, raw_quantized = quantized_scalar_residual(observation, raw, 1.)
    return quantized_scalar_residual(observation, raw_quantized, .3)


def cache(output, receipt):
    import torch
    import torch.distributed as dist
    from safetensors.torch import load_file, save_file
    from lowlight_restoration.lora import inject_lora, load_lora_checkpoint
    from lowlight_restoration.scalar_residual import quantized_scalar_residual
    from flux2.util import load_flow_model, load_ae
    rank, device = setup_ddp()
    started = time.monotonic()
    try:
        spec, plan = receipt['spec'], receipt['plan']
        prompt_receipt(output, receipt)
        if rank == 0:
            for field in ('flow', 'ae', 'wrong_expert'):
                contract.bound(spec[field], spec[field + '_sha256'])
            contract.bound(plan['baseline_checkpoint'], plan['baseline_checkpoint_sha256'])
        dist.barrier()
        os.environ['KLEIN_4B_BASE_MODEL_PATH'] = str(contract.path(spec['flow']))
        os.environ['AE_MODEL_PATH'] = str(contract.path(spec['ae']))
        model = load_flow_model('flux.2-klein-base-4b', device=device)
        names = inject_lora(model, ['double_blocks.*.img_attn.qkv', 'double_blocks.*.img_attn.proj',
                                  'single_blocks.*.linear1', 'single_blocks.*.linear2'], rank=32, alpha=32, dropout=0.)
        if len(names) != 50:
            raise RuntimeError('FLUX LoRA architecture changed')
        model.eval().requires_grad_(False)
        ae = load_ae('flux.2-klein-base-4b', device=device).eval().requires_grad_(False)
        rows = list(enumerate(receipt['selected']))[rank::8]
        images = {index: image_crop(row) for index, row in rows}
        seeds = {index: int(contract.digest(2026091604, 'noise_scalar_full_development_v1', row['record_id'])[:16], 16)
                 for index, row in rows}
        for task, checkpoint in (('noise', plan['baseline_checkpoint']), ('haze', spec['wrong_expert'])):
            load_lora_checkpoint(model, contract.path(checkpoint))
            contexts = load_file(str(output / f'prompt_{task}.safetensors'), device=str(device))
            for index, row in rows:
                image = images[index]['observation'][None].to(device).float().div(127.5).sub(1)
                raw = restore(model, ae, image, contexts['cfg_context'], contexts['cfg_ids'], seeds[index])
                if not bool(raw.isfinite().all()):
                    raise RuntimeError('nonfinite frozen FLUX output')
                value, _ = deployed_proposal(image, raw) if task == 'noise' else quantized_scalar_residual(image, raw, 1.)
                images[index]['proposal' if task == 'noise' else 'wrong_expert'] = value[0].cpu().contiguous()
                print(f'CACHE rank={rank} index={index} task={task}', flush=True)
        for index, row in rows:
            name = f'cache_{index:03d}.safetensors'
            save_file(images[index], str(output / name))
            contract.write_new(output / f'cache_{index:03d}.json', dict(
                row=row, crop_xy=contract.crop_xy(row), seed=seeds[index],
                config_sha256=receipt['config_sha256'], file=name, sha256=contract.sha(output / name),
                input_uint8_sha256=hashlib.sha256(images[index]['observation'].numpy().tobytes()).hexdigest(),
                baseline_checkpoint_sha256=plan['baseline_checkpoint_sha256'],
                wrong_expert_sha256=spec['wrong_expert_sha256'], prompt_receipt_sha256=receipt['prompt_receipt_sha256']))
        dist.barrier()
        if rank == 0:
            seconds = time.monotonic() - started
            contract.write_new(output / 'cache_complete.json', dict(records=32, seconds=seconds,
                pairs_per_second=32/seconds, wall_hours=seconds/3600, allocated_gpu_hours=8*seconds/3600,
                checkpoint_loaded_twice=True, protected_pixels_accessed=False,
                config_sha256=receipt['config_sha256'], source_sha256=receipt['source_sha256']))
            print('FLUX_ENERGY_CACHE_COMPLETE ' + str(seconds), flush=True)
    finally:
        dist.destroy_process_group()


def load_cached(output, receipt, index):
    import torch
    from safetensors.torch import load_file
    row = receipt['selected'][index]
    meta = json.loads((output / f'cache_{index:03d}.json').read_text())
    contract.validate_cache_metadata(meta, row, receipt)
    filename = output / f'cache_{index:03d}.safetensors'
    if meta['file'] != filename.name or contract.sha(filename) != meta['sha256']:
        raise RuntimeError('cached tensors changed')
    tensors = load_file(str(filename))
    if set(tensors) != {'observation', 'target', 'proposal', 'wrong_expert'} or any(
            v.dtype != torch.uint8 or v.shape != (3, 512, 512) for v in tensors.values()):
        raise RuntimeError('bad cache schema')
    if hashlib.sha256(tensors['observation'].numpy().tobytes()).hexdigest() != meta['input_uint8_sha256']:
        raise RuntimeError('observation hash changed')
    return tensors, meta


def synchronized_patch(tensors, *, seed, size=256):
    """Identical deterministic crop/D4 on observation, target and every candidate."""
    import torch
    generator = random.Random(seed)
    y, x = generator.randrange(512-size+1), generator.randrange(512-size+1)
    turns, flip = generator.randrange(4), generator.randrange(2)
    result = {}
    for key, image in tensors.items():
        value = image[:, y:y+size, x:x+size]
        value = torch.rot90(value, turns, (-2, -1))
        if flip:
            value = value.flip(-1)
        result[key] = value[None].float().div(127.5).sub(1).contiguous()
    return result


def training_candidates(images):
    from torch.nn import functional as F
    x, p, target = images['observation'], images['proposal'], images['target']
    damaged = F.avg_pool2d(F.pad(p, (2, 2, 2, 2), mode='reflect'), 5, stride=1)
    return (target, x, p, .5*(x+p), (x + 1.25*(p-x)).clamp(-1, 1), damaged, images['wrong_expert'])


def evaluate_training_cache(model, cached, receipt, perceptual, device, *, initial=False):
    import torch
    results = []
    model.eval()
    for tensors, meta in cached:
        # Target is deliberately not passed into trajectory() or select().
        x, p = [tensors[k][None].to(device).float().div(127.5).sub(1) for k in ('observation', 'proposal')]
        condition = torch.tensor([contract.descriptor(meta['row'])], device=device)
        with torch.enable_grad():
            trajectory, _, _ = model.trajectory(x, p, p, condition, create_graph=False)
        chosen, selected, scores = model.select(x, p, p, condition, trajectory)
        if initial and (selected.item() != 0 or any(not torch.equal(v, p) for v in trajectory)):
            raise RuntimeError('initial energy must preserve frozen FLUX exactly')
        with torch.no_grad():
            target = tensors['target'][None].to(device).float().div(127.5).sub(1)
            metrics = {}
            for name, image in [('observation', x), ('baseline', p), *[(f'energy_{i+1}', t) for i, t in enumerate(trajectory)], ('selected', chosen)]:
                quantized = image.detach().add(1).mul(127.5).round().clamp(0, 255).div(127.5).sub(1)
                mse = ((quantized-target)/2).square().mean()
                psnr = -10 * torch.log10(mse.clamp_min(1e-12))
                lp = perceptual(quantized, target).mean()
                if not bool(torch.isfinite(psnr) & torch.isfinite(lp)):
                    raise RuntimeError('nonfinite evaluation')
                metrics[name] = dict(psnr=psnr.item(), lpips=lp.item())
        results.append(dict(record_id=meta['row']['record_id'], source_group=meta['row']['source_group'],
                            split='train', crop_xy=meta['crop_xy'], input_uint8_sha256=meta['input_uint8_sha256'],
                            selected_step=selected.item(), energy_scores=scores.detach().cpu().tolist(), metrics=metrics))
    return results


def train(output, receipt):
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    from lowlight_restoration.novel_ppt_energy import objective
    from lowlight_restoration.novel_ppt_energy_nonzero_init import NonzeroInitNovelPPTEnergy
    rank, device = setup_ddp()
    started = time.monotonic()
    try:
        spec = receipt['spec']
        cache_receipt = json.loads((output / 'cache_complete.json').read_text())
        if (cache_receipt['records'] != 32 or cache_receipt['config_sha256'] != receipt['config_sha256']
                or cache_receipt['source_sha256'] != receipt['source_sha256']):
            raise RuntimeError('cache did not complete under the current contract')
        prompt_receipt(output, receipt)
        # Four samples per rank, one sample per source. Each step visits all 32 sources.
        cached = [load_cached(output, receipt, i) for i in range(rank, 32, 8)]
        torch.manual_seed(spec['seed'])
        torch.cuda.manual_seed_all(spec['seed'])
        model = NonzeroInitNovelPPTEnergy(**spec['energy']).to(device)
        wrapped = DDP(model, device_ids=[device.index], broadcast_buffers=False, gradient_as_bucket_view=False)
        optimizer = torch.optim.AdamW(wrapped.parameters(), lr=spec['learning_rate'], weight_decay=spec['weight_decay'])
        perceptual = LearnedPerceptualImagePatchSimilarity(net_type='alex').to(device).eval().requires_grad_(False).net
        initial = evaluate_training_cache(model, cached, receipt, perceptual, device, initial=True)
        parts = [None] * 8
        dist.all_gather_object(parts, initial)
        if rank == 0:
            contract.write_new(output / 'trainfit_000000.json', {'split': 'train', 'not_generalization': True,
                                'rows': sum(parts, [])})
        timings = []
        for step in range(1, 26):
            tick = time.monotonic()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            lr = spec['learning_rate'] * min(step/spec['warmup_steps'], 1.)
            for group in optimizer.param_groups:
                group['lr'] = lr
            stats = {}
            for micro, (tensors, meta) in enumerate(cached):
                seed = int(contract.digest(spec['seed'], step, meta['row']['record_id'])[:16], 16)
                images = {k: v.to(device) for k, v in synchronized_patch(tensors, seed=seed).items()}
                x, p, target = [images[k] for k in ('observation', 'proposal', 'target')]
                condition = torch.tensor([contract.descriptor(meta['row'])], device=device)
                with wrapped.no_sync() if micro < 3 else nullcontext():
                    graph = wrapped(x, p, p, condition, training_candidates(images))
                    loss, terms, diagnostics = objective(graph, p, target, perceptual,
                         residual_scale=spec['energy']['residual_scale'], weights=spec['loss'],
                         ranking_lpips_weight=spec['ranking_lpips_weight'])
                    if not bool(torch.isfinite(loss)):
                        raise RuntimeError('nonfinite loss')
                    (loss / 4).backward()
                for key, value in {'loss': loss, **terms, **diagnostics}.items():
                    stats[key] = stats.get(key, 0.) + float(value.detach())/4
                del graph, loss, terms, diagnostics
            if any(p.grad is None or not bool(p.grad.isfinite().all()) for p in model.parameters()):
                raise RuntimeError('missing/nonfinite second-order gradient')
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), spec['gradient_clip'], error_if_nonfinite=True)
            optimizer.step()
            torch.cuda.synchronize()
            timings.append(time.monotonic()-tick)
            if rank == 0:
                print('FLUX_ENERGY_TRAIN ' + json.dumps(dict(step=step, total_steps=25, lr=lr,
                      rank0_seconds=timings[-1], rank0_grad_norm=float(norm), rank0_stats=stats)), flush=True)
        if not bool(model.field.head.weight.detach().abs().max() > 0):
            raise RuntimeError('energy head did not update')
        # Compare entire parameter states, not only the final layer, across all ranks.
        h = hashlib.sha256()
        for name, value in model.state_dict().items():
            h.update(name.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
        hashes = [None] * 8
        dist.all_gather_object(hashes, h.hexdigest())
        if len(set(hashes)) != 1:
            raise RuntimeError('DDP weights diverged')
        if rank == 0:
            checkpoint = output / 'energy_step_000025.pt'
            torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), step=25,
                            config_sha256=receipt['config_sha256'], source_sha256=receipt['source_sha256'],
                            may_seed_formal=False), checkpoint)
            restored = torch.load(checkpoint, map_location='cpu', weights_only=True)
            for key, value in model.state_dict().items():
                if not bool(value.isfinite().all()) or not torch.equal(value.detach().cpu(), restored['model'][key]):
                    raise RuntimeError('saved checkpoint mismatch')
            for state in restored['optimizer']['state'].values():
                if float(state['step']) != 25 or any(not bool(state[k].isfinite().all()) for k in ('exp_avg', 'exp_avg_sq')):
                    raise RuntimeError('saved Adam state is incomplete/nonfinite')
        dist.barrier()
        final = evaluate_training_cache(model, cached, receipt, perceptual, device)
        dist.all_gather_object(parts, final)
        peaks = [None] * 8
        dist.all_gather_object(peaks, torch.cuda.max_memory_allocated(device)/1024**3)
        if rank == 0:
            rows = sum(parts, [])
            if len(rows) != 32 or len({r['record_id'] for r in rows}) != 32:
                raise RuntimeError('incomplete engineering evaluation')
            means = {name: {metric: sum(r['metrics'][name][metric] for r in rows)/32 for metric in ('psnr', 'lpips')}
                     for name in rows[0]['metrics']}
            elapsed = time.monotonic()-started
            result = dict(status='engineering_complete', engineering_pass=True, steps=25,
                split='train', training_sources=32, scientific_gate_evaluated=False,
                generalization_success_established=False, sampling_guidance_authorized_by_result=False,
                qualification_may_seed_formal=False, means=means, rows=rows,
                checkpoint_sha256=contract.sha(output / 'energy_step_000025.pt'),
                rank_parameter_sha256=hashes, peak_memory_gib=peaks,
                rank0_mean_train_step_seconds=sum(timings)/25, wall_hours=elapsed/3600,
                allocated_gpu_hours=8*elapsed/3600, protected_pixels_accessed=False,
                config_sha256=receipt['config_sha256'], source_sha256=receipt['source_sha256'])
            contract.write_new(output / 'result.json', result)
            print('FLUX_ENERGY_QUALIFICATION_COMPLETE ' + json.dumps({k: v for k, v in result.items() if k not in ('rows', 'source_sha256')}), flush=True)
    finally:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('preflight', 'prepare', 'hardware', 'prompts', 'cache', 'train'))
    parser.add_argument('--run-id', default='qualification_v1')
    args = parser.parse_args()
    if args.phase == 'hardware':
        hardware(); return
    output = contract.run_path(args.run_id)
    if args.phase in ('preflight', 'prepare'):
        receipt = contract.preflight()
        if args.phase == 'prepare':
            output.mkdir(parents=True, exist_ok=False)
            contract.write_new(output / 'preflight.json', receipt)
        print('FLUX_ENERGY_PREFLIGHT_PASS ' + json.dumps({k: receipt[k] for k in
              ('config_sha256', 'eligible_records', 'eligible_sources', 'images_opened', 'protected_pixels_accessed')}), flush=True)
        return
    receipt = contract.load_receipt(output)
    # Direct phase invocation is fail-closed too, not only the shell launcher.
    if args.phase == 'prompts' and any(output.glob('prompt*')):
        raise FileExistsError('prompt phase already has artifacts; use a fresh run ID')
    if args.phase == 'cache' and any(output.glob('cache*')):
        raise FileExistsError('cache phase already has artifacts; use a fresh run ID')
    if args.phase == 'train' and any((output / name).exists() for name in
                                     ('trainfit_000000.json', 'energy_step_000025.pt', 'result.json')):
        raise FileExistsError('training phase already has artifacts; use a fresh run ID')
    globals()[args.phase](output, receipt)


if __name__ == '__main__':
    main()
