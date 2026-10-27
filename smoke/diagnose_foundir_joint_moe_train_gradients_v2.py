#!/usr/bin/env python3
"""V2 frozen MoE diagnosis: integer-only gradient sampling, no optimizer steps.

V1 remains unchanged because the concurrent PPT qualification binds its hash.
This isolated revision changes only diagnostic indexing and provenance.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'third_party/flux2/src'))

import torch
import torch.distributed as dist
from torch.nn import functional as F
from lowlight_restoration.joint_vlm_moe import install_joint_moe, enable_block_checkpointing, joint_conditioning, restoration_loss
from lowlight_restoration.rain_blur_image_moe import load_expert_state, router_classification_loss
from lowlight_restoration.lora import inject_lora, load_lora_checkpoint
from lowlight_restoration.flux2_endpoint_precision import promote_lora_to_fp32, restore_endpoint
from lowlight_restoration.rain_full_data import crop_pair
from smoke import train_foundir_rain_blur_joint_vlm_moe_v2_ddp as parent
from smoke.dual_retry_runtime import CONFIG, ROOT, bound_json, initialize, sha, sources, source_disjoint_training_probe, write_once


def probe_parameters(model):
    params = dict(model.named_parameters())
    shared = sorted(n for n in params if '.lora_B.' in n)[0]
    names = [shared] + [f'double_blocks.{block}.img_mlp.experts.{expert}.2.weight'
                        for block in (0, 4) for expert in (0, 1)]
    if any(name not in params for name in names):
        raise RuntimeError('declared representative gradient tensors missing')
    # Keep all tensors in the model but only ask autograd for declared probes.
    return names, [params[n] for n in names]


def probe_indices(numel, *, device=None, limit=16384):
    """Evenly spaced integer coordinates; FP32 linspace can round n-1 to n."""
    if numel < 1 or limit < 1:
        raise ValueError('positive tensor size and sampling limit required')
    count = min(numel, limit)
    if count == 1:
        return torch.zeros(1, device=device, dtype=torch.int64)
    return torch.div(torch.arange(count, device=device, dtype=torch.int64) * (numel-1),
                     count-1, rounding_mode='floor')


def gradient_summary(terms, names, params):
    all_terms, sketches = {}, {}
    for term, value in terms.items():
        gradients = torch.autograd.grad(value, params, retain_graph=True, allow_unused=True)
        entries, sketch = {}, []
        for name, p, g in zip(names, params, gradients, strict=True):
            if g is None:
                entries[name] = {'l2': 0., 'unused': True}
                sketch.append(torch.zeros(min(p.numel(), 16384), device=p.device))
            else:
                if not bool(torch.isfinite(g).all()):
                    raise RuntimeError(f'nonfinite gradient: {term} {name}')
                entries[name] = {'l2': float(g.norm()), 'unused': False}
                indices = probe_indices(g.numel(), device=g.device)
                sketch.append(g.detach().flatten()[indices])
        all_terms[term] = entries
        sketches[term] = torch.cat(sketch).double()
        del gradients
    cosine = {}
    for i, first in enumerate(terms):
        for second in list(terms)[i+1:]:
            a, b = sketches[first], sketches[second]
            denominator = a.norm() * b.norm()
            cosine[f'{first}__{second}'] = float(a.dot(b) / denominator) if denominator > 1e-30 else None
    return all_terms, cosine


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', default='diagnostic_v2')
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    cfg = json.loads(CONFIG.read_text()); spec = cfg['moe']
    bound_json(spec['parent_config'], spec['parent_sha256'])
    config, pc, base, training, development, conditions, metadata = parent.read_contract(ROOT / spec['parent_config'])
    for name in ('flow_weights', 'autoencoder_weights', 'prompt_context_cache'):
        if sha(ROOT/base['model'][name]) != base['model'][name+'_sha256']:
            raise RuntimeError(f'frozen FLUX artifact changed: {name}')
    if sha(ROOT/pc['rain_lora']) != pc['rain_lora_sha256']:
        raise RuntimeError('original Rain initialization changed')
    probes = {task: source_disjoint_training_probe(training[task], spec['sources_per_task'], spec['seed'])
              for task in ('rain', 'blur')}
    for arm, binding in spec['arms'].items():
        if sha(ROOT / binding['checkpoint']) != binding['sha256']:
            raise RuntimeError(f'{arm} final checkpoint changed')
    bindings = sources([*parent.source_hashes(), 'third_party/flux2/src/flux2/util.py',
                        'smoke/diagnose_foundir_joint_moe_train_gradients_v2.py',
                        'smoke/run_foundir_moe_gradient_diagnostic_v2_8gpu.sh',
                        'smoke/start_foundir_moe_gradient_diagnostic_v2_tmux.sh'])
    if args.preflight:
        print('MOE_RETRY_PREFLIGHT_PASS ' + json.dumps({'records_per_task': {k: len(v) for k, v in probes.items()},
              'source_sha256': bindings, 'config_sha256': sha(CONFIG), 'images_opened': 0}))
        return
    cfg, rank, device, output, names = initialize('moe', args.run_id)
    started = time.monotonic()
    try:
        if rank == 0:
            write_once(output/'launch.json', {'config_sha256': sha(CONFIG), 'source_sha256': bindings,
                       'selected_train_records': {k: [r['record_id'] for r in v] for k, v in probes.items()},
                       'data': metadata, 'optimizer_steps': 0})
        os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TORCH_HOME=str(ROOT/base['runtime']['torch_home']),
                          KLEIN_4B_BASE_MODEL_PATH=str(ROOT/base['model']['flow_weights']), AE_MODEL_PATH=str(ROOT/base['model']['autoencoder_weights']))
        from flux2.util import load_ae, load_flow_model
        from flux2.sampling import batched_prc_img
        from safetensors.torch import load_file
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
        cache = load_file(str(ROOT/base['model']['prompt_context_cache']))
        context, context_ids = cache['cfg_context'][:1].to(device), cache['cfg_ids'][:1].to(device)
        del cache
        ae = load_ae(base['model']['name'], device=device).eval().requires_grad_(False)
        metric = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).to(device).eval().requires_grad_(False)
        local = []
        for arm, binding in spec['arms'].items():
            torch.manual_seed(spec['seed']); torch.cuda.manual_seed_all(spec['seed'])
            model = load_flow_model(base['model']['name'], device=device).eval()
            inject_lora(model, base['lora']['target_patterns'], rank=32, alpha=32, dropout=0)
            promote_lora_to_fp32(model); load_lora_checkpoint(model, ROOT/pc['rain_lora'])
            install_joint_moe(model, arm=arm)
            load_expert_state(model, ROOT/binding['checkpoint'])
            enable_block_checkpointing(model); model.train()
            probe_names, parameters = probe_parameters(model)
            unchanged = [p.detach().flatten()[::4096].clone() for p in parameters]
            verified = {}
            for task_id, task in enumerate(('rain', 'blur')):
                for index in range(rank, len(probes[task]), 8):
                    row = probes[task][index]
                    if row['split'] != 'train':
                        raise RuntimeError('only train pixels permitted')
                    lq, gt = crop_pair(ROOT, row, crop_size=512, position=index,
                                       seed=spec['seed']+task_id, verified=verified)
                    lq, gt = lq[None].to(device), gt[None].to(device)
                    condition = torch.tensor([conditions[row['record_id']]], device=device)
                    with torch.no_grad(), torch.autocast('cuda', enabled=False):
                        clean = ae.encode(gt.float()).float()
                    with joint_conditioning(model, condition):
                        result = restore_endpoint(model, ae, lq, context, context_ids, batched_prc_img)
                        _, terms = restoration_loss(result, clean, gt, metric, batched_prc_img, config['training']['losses'])
                        terms['router_ce'] = router_classification_loss(model, torch.tensor([task_id], device=device))
                        coefficients = {**config['training']['losses'], 'router_ce': config['training']['router_ce_weight']}
                        weighted = {k: terms[k]*coefficients[k] for k in terms}
                        grad, cosines = gradient_summary(weighted, probe_names, parameters)
                        mse = ((result['decoded'].detach()-gt)*.5).square().mean()
                        entry = {'arm': arm, 'task': task, 'record_id': row['record_id'], 'source_group': row['source_group'],
                                 'losses_raw': {k: float(v.detach()) for k, v in terms.items()},
                                 'coefficients': coefficients, 'weighted_gradient_norms': grad,
                                 'fixed_coordinate_sketch_cosines_not_full_gradient': cosines,
                                 'train_crop_psnr_unquantized': float(-10*torch.log10(mse.clamp_min(1e-12)))}
                    if not all(torch.equal(p.detach().flatten()[::4096], saved) for p, saved in zip(parameters, unchanged, strict=True)):
                        raise RuntimeError('read-only probe parameter changed')
                    local.append(entry)
                    metric.reset()
                    print('MOE_TRAIN_GRADIENT_RECORD ' + json.dumps({'rank': rank, 'arm': arm, 'task': task, 'index': index}), flush=True)
                    del result, terms, weighted, clean, lq, gt
            del model, parameters, unchanged
            gc.collect(); torch.cuda.empty_cache(); dist.barrier()
        gathered = [None]*8; dist.all_gather_object(gathered, local)
        if rank == 0:
            rows = [r for shard in gathered for r in shard]
            if len(rows) != 2*2*spec['sources_per_task']:
                raise RuntimeError('incomplete train-only diagnosis')
            hours = (time.monotonic()-started)/3600
            write_once(output/'result.json', {'status': 'complete', 'engineering_pass': True,
                       'rows': rows, 'config_sha256': sha(CONFIG), 'source_sha256': bindings,
                       'gpu_names': names, 'wall_hours': hours, 'allocated_gpu_hours': 8*hours,
                       'optimizer_steps': 0, 'development_pixels_accessed': False,
                       'protected_pixels_accessed': False,
                       'scientific_interpretation': 'representative_train_gradients_not_loss_weight_selection_or_generalization'})
            print('MOE_TRAIN_GRADIENT_COMPLETE', flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
