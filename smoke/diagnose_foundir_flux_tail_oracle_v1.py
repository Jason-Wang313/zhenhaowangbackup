#!/usr/bin/env python3
"""TRAIN-ONLY GT oracle. No weights updated; never claim deployable performance."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from smoke import flux_energy_qualification_contract_v1 as contract

CONFIG = ROOT / 'configs/foundir_flux_tail_oracle_v1.json'
FILES = ['configs/foundir_flux_tail_oracle_v1.json',
         'lowlight_restoration/flux_tail_diagnostic.py',
         'lowlight_restoration/flux2_trajectory_endpoint.py',
         'smoke/diagnose_foundir_flux_tail_oracle_v1.py',
         'smoke/run_foundir_flux_tail_oracle_v1_8gpu.sh',
         'smoke/start_foundir_flux_tail_oracle_v1_tmux.sh',
         'smoke/tmux_launcher_lib.sh']


def binding():
    return {**contract.source_binding(), **{name: contract.sha(ROOT / name) for name in FILES}}


def validate_spec(spec):
    if (spec['mode'] != 'training_source_gt_only_mechanism_diagnostic'
            or spec['cache_indices'] != list(range(8)) or spec['world_size'] != 8
            or spec['crop'] != 512 or spec['steps'] != 50 or spec['auto_train']
            or not spec['uses_gt_for_guidance'] or spec['deployable']
            or spec['preview_start_indices'] != [40, 46, 48]
            or spec['rollout_start_indices'] != [48]
            or spec['rms_latent_steps'] != [.005, .02, .08]
            or spec['objective_lpips_weights'] != [0., 1.]
            or spec['finite_difference_epsilons'] != [.001, .003, .01]
            or spec['finite_difference_max_relative_error'] != .2
            or spec['finite_difference_required_passes'] != 2
            or spec['budget_wall_hours_cap'] != 1.5
            or spec['budget_allocated_gpu_hours_cap'] != 12
            or spec['psnr_gain_target'] != .3
            or spec['lpips_relative_reduction_target'] != .09928107050377874):
        raise RuntimeError('frozen diagnostic scope changed')


def parent_receipt(spec):
    from smoke.qualify_foundir_flux_energy_v1 import prompt_receipt
    parent = contract.path(spec['parent_cache'])
    contract.bound(str(parent / 'preflight.json'), spec['parent_preflight_sha256'])
    contract.bound(str(parent / 'prompts.json'), spec['parent_prompts_sha256'])
    receipt = contract.load_receipt(parent)
    prompt_receipt(parent, receipt)
    return parent, receipt


def prepare(output, *, write):
    spec = json.loads(CONFIG.read_text())
    validate_spec(spec)
    parent, receipt = parent_receipt(spec)
    # Pinned parent receipt records the full 685,495-row source/GT exclusion
    # audit. No new sampling or protected image access is performed here.
    selected = [receipt['selected'][i] for i in spec['cache_indices']]
    if len({r['source_group'] for r in selected}) != 8 or any(
            r['split'] != 'train' or r['category'] != '05Noise' for r in selected):
        raise RuntimeError('training-source restriction failed')
    artifacts = {}
    for index, row in zip(spec['cache_indices'], selected):
        name = f'cache_{index:03d}'
        meta_path = parent / (name + '.json')
        meta = json.loads(meta_path.read_text())
        contract.validate_cache_metadata(meta, row, receipt)
        if meta['file'] != name + '.safetensors':
            raise RuntimeError('unexpected cache path')
        data_path = contract.bound(str(parent / meta['file']), meta['sha256'])
        artifacts[str(data_path.relative_to(ROOT))] = meta['sha256']
        artifacts[str(meta_path.relative_to(ROOT))] = contract.sha(meta_path)
    # Bind only artifacts used by this diagnostic; Qwen/Haze are not loaded.
    for field in ('flow', 'ae'):
        filename = receipt['spec'][field]
        expected = receipt['spec'][field + '_sha256']
        print('VERIFY ' + field, flush=True)
        contract.bound(filename, expected)
        artifacts[filename] = expected
    for filename in (receipt['plan']['baseline_checkpoint'],
                     'models/torch/hub/checkpoints/alexnet-owt-7be5be79.pth'):
        expected = receipt['artifacts'][filename]
        contract.bound(filename, expected)
        artifacts[filename] = expected
    result = dict(spec=spec, config_sha256=contract.sha(CONFIG), source_sha256=binding(),
                  artifacts=artifacts, selected=selected, protected_pixels_accessed=False,
                  images_decoded=0, parent_preflight_sha256=spec['parent_preflight_sha256'])
    if write:
        output.mkdir(parents=True, exist_ok=False)
        contract.write_new(output / 'preflight.json', result)
    print('TAIL_ORACLE_PREFLIGHT_PASS records=8 sources=8 pixels_decoded=0', flush=True)
    return result


def load_run(output):
    receipt = json.loads((output / 'preflight.json').read_text())
    validate_spec(receipt['spec'])
    if receipt['config_sha256'] != contract.sha(CONFIG) or receipt['source_sha256'] != binding():
        raise RuntimeError('source/config changed after prepare')
    return receipt


def aggregate(rows, spec):
    if len(rows) != 8 or len({r['record_id'] for r in rows}) != 8 or len({r['source_group'] for r in rows}) != 8:
        raise RuntimeError('incomplete or duplicated diagnostic records')
    keys = set(rows[0]['arms'])
    if any(set(r['arms']) != keys for r in rows):
        raise RuntimeError('incomplete diagnostic arms')
    means = {key: {m: sum(r['arms'][key]['metrics'][m] for r in rows)/len(rows)
                   for m in ('psnr', 'lpips')} for key in sorted(keys)}
    baseline = means['historical_bf16']
    for key, value in means.items():
        control = rows[0]['arms'][key]['control']
        reference = means[control]
        value.update(psnr_gain_vs_historical=value['psnr']-baseline['psnr'],
                     lpips_reduction_vs_historical=1-value['lpips']/max(baseline['lpips'], 1e-12),
                     control=control, psnr_gain_vs_precision_control=value['psnr']-reference['psnr'],
                     lpips_reduction_vs_precision_control=1-value['lpips']/max(reference['lpips'], 1e-12))
        value['joint_target_vs_historical'] = (value['psnr_gain_vs_historical'] >= spec['psnr_gain_target']
            and value['lpips_reduction_vs_historical'] >= spec['lpips_relative_reduction_target'])
    # Report every fixed arm. Do not hide bad arms behind a GT-selected average.
    return dict(status='mechanism_diagnostic_complete', rows=rows, means=means,
                baseline_parity_pass=all(r['baseline_parity_pass'] for r in rows),
                finite_difference_pass=all(r['finite_difference_pass'] for r in rows),
                split='train', uses_gt_for_guidance=True, deployable=False,
                generalization_established=False, energy_trained=False,
                scientific_gate_evaluated=False, protected_pixels_accessed=False)


def run(output):
    import torch
    import torch.distributed as dist
    from safetensors.torch import load_file
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    from flux2.sampling import batched_prc_img, get_schedule
    from flux2.util import load_flow_model, load_ae
    from lowlight_restoration.lora import inject_lora, load_lora_checkpoint
    from lowlight_restoration.flux_tail_diagnostic import Tail, continuous_blend, finite_difference, rms_direction, checkpoint_blocks
    from smoke.qualify_foundir_flux_energy_v1 import setup_ddp, restore, load_cached, deployed_proposal

    receipt = load_run(output)
    spec = receipt['spec']
    parent, previous = parent_receipt(spec)
    rank, device = setup_ddp()
    started = time.monotonic()
    try:
        for filename, expected in receipt['artifacts'].items():
            if filename.startswith(spec['parent_cache']) or rank == 0:
                contract.bound(filename, expected)
        dist.barrier()
        tensors, meta = load_cached(parent, previous, rank)
        observation, target = [tensors[k][None].to(device).float().div(127.5).sub(1)
                               for k in ('observation', 'target')]
        os.environ['KLEIN_4B_BASE_MODEL_PATH'] = str(contract.path(previous['spec']['flow']))
        os.environ['AE_MODEL_PATH'] = str(contract.path(previous['spec']['ae']))
        model = load_flow_model('flux.2-klein-base-4b', device=device)
        names = inject_lora(model, ['double_blocks.*.img_attn.qkv', 'double_blocks.*.img_attn.proj',
                                   'single_blocks.*.linear1', 'single_blocks.*.linear2'], 32, 32, 0.)
        if len(names) != 50:
            raise RuntimeError('LoRA layout changed')
        load_lora_checkpoint(model, contract.path(previous['plan']['baseline_checkpoint']))
        model.eval().requires_grad_(False)
        original_dtype = next(model.parameters()).dtype
        original_parameter_dtypes = {name: value.dtype for name, value in model.named_parameters()}
        original_buffer_dtypes = {name: value.dtype for name, value in model.named_buffers()}
        if original_dtype != torch.bfloat16:
            raise RuntimeError('historical model precision differs from expected BF16')
        ae = load_ae('flux.2-klein-base-4b', device=device).eval().requires_grad_(False)
        lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex').to(device).eval().requires_grad_(False).net
        contexts = load_file(str(parent / 'prompt_noise.safetensors'), device=str(device))
        seed = meta['seed']
        # No inference_mode in the new entry. All caches are ordinary tensors.
        with torch.no_grad():
            reference_latent = ae.encode(observation).to(torch.bfloat16)
            reference, reference_ids = batched_prc_img(reference_latent)
            reference_ids = reference_ids.clone()
            reference_ids[..., 0] = 10
            noise = torch.randn(reference_latent.shape,
                generator=torch.Generator(device=device).manual_seed(seed), device=device, dtype=torch.bfloat16)
            tokens, ids = batched_prc_img(noise)
            tail = Tail(model, ae, ids, reference, reference_ids,
                        contexts['cfg_context'], contexts['cfg_ids'], get_schedule(50, tokens.shape[1]))
            checkpoint_blocks(model, tail.calls)
            baseline_raw = restore(model, ae, observation, contexts['cfg_context'], contexts['cfg_ids'], seed)
            historical_uint8, historical_image = deployed_proposal(observation, baseline_raw)
            final, states = tail.rollout(tokens, 0, capture=True)
            zero_raw = tail.decode(final)
            zero_uint8, _ = deployed_proposal(observation, zero_raw)
            parity = torch.equal(zero_uint8, historical_uint8) and torch.equal(
                historical_uint8[0].cpu(), tensors['proposal'])
            contract.write_new(output / f'baseline_check_rank{rank}.json', dict(
                passed=bool(parity), new_vs_original_changed_values=int((zero_uint8 != historical_uint8).sum()),
                original_vs_cache_changed_values=int((historical_uint8[0].cpu() != tensors['proposal']).sum()),
                new_vs_original_raw_max_abs=float((zero_raw-baseline_raw).abs().max()), seed=seed))
        if not parity:
            raise RuntimeError('zero-guidance or cached historical uint8 baseline mismatch; stop before oracle')
        print(f'TAIL_BASELINE_PASS rank={rank} seed={seed}', flush=True)

        def metrics(raw):
            with torch.no_grad():
                _, image = deployed_proposal(observation, raw)
                mse = ((image-target)/2).square().mean()
                value = dict(psnr=float(-10*torch.log10(mse.clamp_min(1e-12))),
                             lpips=float(lpips(image, target).mean()))
                if not all(__import__('math').isfinite(v) for v in value.values()):
                    raise RuntimeError('nonfinite final metric')
                return value

        arms = {'historical_bf16': dict(metrics=metrics(baseline_raw), control='historical_bf16')}
        controls = {}
        denominators = {}
        with torch.no_grad():
            for index in spec['preview_start_indices']:
                raw = tail.decode(tail.rollout(states[index].float(), index))
                controls[index] = metrics(raw)
                blended = continuous_blend(observation, raw)
                denominators[index] = (float(((blended-target)/2).square().mean().clamp_min(1e-8)),
                                       float(lpips(blended, target).mean().clamp_min(1e-6)))
                arms[f'fp32_state_control_{index}'] = dict(metrics=controls[index], control='historical_bf16')

        def objective(state, index, mode, weight):
            image = continuous_blend(observation, tail.image(state, index, mode))
            mse_scale, lp_scale = denominators[index]
            mse = ((image-target)/2).square().mean()
            loss = mse / mse_scale
            if weight:
                loss = loss + weight * lpips(image, target).mean() / lp_scale
            return loss

        # FP32 FD isolates differentiation correctness from BF16 rounding.
        # It validates preview AND actual short-rollout gradients, not skip-J.
        model.float()
        tail.bf16 = False
        original_tf32 = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        finite_checks = {}
        for mode in ('preview_full', 'rollout_full'):
            state = states[48].float().detach().clone().requires_grad_(True)
            fn = lambda value, mode=mode: objective(value, 48, mode, 1.)
            loss = fn(state)
            grad = torch.autograd.grad(loss, state)[0]
            tail.calls['objective_backward'] += 1
            checks = finite_difference(fn, state.detach(), grad.detach(), spec['finite_difference_epsilons'])
            passed = sum(r['positive_direction'] and r['relative_error'] <= spec['finite_difference_max_relative_error']
                         for r in checks) >= spec['finite_difference_required_passes']
            finite_checks[mode] = dict(passed=passed, checks=checks)
            print('TAIL_FINITE_DIFFERENCE ' + json.dumps(dict(rank=rank, mode=mode, passed=passed, checks=checks)), flush=True)
            del loss, grad, state
        contract.write_new(output / f'gradient_check_rank{rank}.json', finite_checks)
        if not all(r['passed'] for r in finite_checks.values()):
            raise RuntimeError('FP32 finite-difference qualification failed; do not interpret oracle')
        for name, value in model.named_parameters():
            value.data = value.data.to(original_parameter_dtypes[name])
        for name, value in model.named_buffers():
            value.data = value.data.to(original_buffer_dtypes[name])
        tail.bf16 = True
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = original_tf32
        torch.cuda.empty_cache()
        # Modified latent ALWAYS causes a fresh velocity evaluation on rollout.
        for index in spec['preview_start_indices']:
            modes = ['preview_full', 'preview_skip']
            if index in spec['rollout_start_indices']:
                modes.append('rollout_full')
            for mode in modes:
                for weight in spec['objective_lpips_weights']:
                    tick = time.monotonic()
                    calls_before = dict(tail.calls)
                    state = states[index].float().detach().clone().requires_grad_(True)
                    loss = objective(state, index, mode, weight)
                    before = float(loss.detach())
                    gradient = torch.autograd.grad(loss, state)[0]
                    tail.calls['objective_backward'] += 1
                    direction, gradient_rms = rms_direction(gradient)
                    del loss, gradient
                    for amplitude in spec['rms_latent_steps']:
                        with torch.no_grad():
                            changed = state.detach() - amplitude * direction
                            after = float(objective(changed, index, mode, weight))
                            actual_raw = tail.decode(tail.rollout(changed, index))
                            key = f'{mode}_i{index}_w{weight:g}_rms{amplitude:g}'
                            arms[key] = dict(metrics=metrics(actual_raw), control=f'fp32_state_control_{index}',
                                timestep=tail.schedule[index], objective_before=before, objective_after=after,
                                gradient_rms=gradient_rms, rms_update=amplitude,
                                input_gradient_model_precision='bf16', latent_accumulation='fp32')
                            print('TAIL_ORACLE_ARM ' + json.dumps(dict(rank=rank, arm=key, **arms[key])), flush=True)
                    torch.cuda.synchronize()
                    contract.write_new(output / f'progress_rank{rank}_{mode}_{index}_{weight:g}.json',
                        dict(seconds=time.monotonic()-tick, calls={k: tail.calls[k]-calls_before[k] for k in tail.calls}))
                    del state, direction
        if any(p.requires_grad or p.grad is not None for p in model.parameters()):
            raise RuntimeError('frozen FLUX acquired parameter gradients')
        if any(p.requires_grad or p.grad is not None for p in ae.parameters()):
            raise RuntimeError('frozen VAE acquired parameter gradients')
        torch.cuda.synchronize()
        row = dict(record_id=meta['row']['record_id'], source_group=meta['row']['source_group'],
                   split='train', crop_xy=meta['crop_xy'], seed=seed,
                   baseline_parity_pass=True, finite_difference_pass=True, finite_checks=finite_checks,
                   vlm_levels=contract.descriptor(meta['row']), vlm_used_for_guidance=False,
                   arms=arms, calls=tail.calls, historical_extra_flux_forward=50,
                   historical_extra_vae_encode=1, historical_extra_vae_decode=1,
                   initial_vae_encode=1, seconds=time.monotonic()-started,
                   peak_memory_gib=torch.cuda.max_memory_allocated(device)/1024**3,
                   source_sha256=receipt['source_sha256'], config_sha256=receipt['config_sha256'])
        contract.write_new(output / f'rank{rank}.json', row)
        parts = [None]*8
        dist.all_gather_object(parts, row)
        if rank == 0:
            result = aggregate(parts, spec)
            elapsed = time.monotonic()-started
            result.update(wall_hours=elapsed/3600, allocated_gpu_hours=8*elapsed/3600,
                          source_sha256=receipt['source_sha256'], config_sha256=receipt['config_sha256'],
                          note='GT-guided finite search, NOT global upper bound or learned-energy evidence')
            contract.write_new(output / 'result.json', result)
            print('TAIL_ORACLE_COMPLETE ' + json.dumps(dict(wall_hours=elapsed/3600,
                allocated_gpu_hours=8*elapsed/3600, fixed_arms=len(arms), deployable=False)), flush=True)
    finally:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('preflight', 'prepare', 'hardware', 'run', 'verify'))
    parser.add_argument('--run-id', default='mechanism_v1')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.run_id):
        raise ValueError('invalid fresh run ID')
    spec = json.loads(CONFIG.read_text())
    validate_spec(spec)
    output = contract.path(spec['output_root']) / args.run_id
    if args.phase == 'hardware':
        from smoke.qualify_foundir_flux_energy_v1 import hardware
        hardware()
    elif args.phase in ('preflight', 'prepare'):
        prepare(output, write=args.phase == 'prepare')
    elif args.phase == 'verify':
        receipt = load_run(output)
        result = json.loads((output / 'result.json').read_text())
        if (result['status'] != 'mechanism_diagnostic_complete' or not result['baseline_parity_pass']
                or not result['finite_difference_pass'] or result['deployable']
                or result['source_sha256'] != receipt['source_sha256']):
            raise RuntimeError('incomplete/invalid diagnostic')
        print('TAIL_ORACLE_RESULT_VERIFIED scientific_success_not_claimed')
    else:
        if (any(output.glob('rank*.json')) or any(output.glob('gradient_check*.json'))
                or any(output.glob('baseline_check*.json')) or (output/'result.json').exists()):
            raise FileExistsError('preserve partial diagnostic; use a fresh ID')
        run(output)


if __name__ == '__main__':
    main()
