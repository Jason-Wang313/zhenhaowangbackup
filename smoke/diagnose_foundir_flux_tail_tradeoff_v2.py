#!/usr/bin/env python3
"""Frozen training-source LPIPS/PSNR diagnosis. GT-guided, not deployable."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from smoke import diagnose_foundir_flux_tail_oracle_v1 as base
from smoke import flux_energy_qualification_contract_v1 as contract

CONFIG = ROOT / 'configs/foundir_flux_tail_tradeoff_v2.json'
FILES = ['configs/foundir_flux_tail_tradeoff_v2.json',
         'lowlight_restoration/flux_tail_tradeoff.py',
         'smoke/diagnose_foundir_flux_tail_tradeoff_v2.py',
         'smoke/run_foundir_flux_tail_tradeoff_v2_gpu.sh',
         'smoke/start_foundir_flux_tail_tradeoff_v2_tmux.sh']


def binding():
    return {**base.binding(), **{name: contract.sha(ROOT/name) for name in FILES}}


def validate_spec(spec):
    expected = dict(mode='training_gt_only_lpips_tradeoff_diagnostic',
        allowed_world_sizes=[4, 8], cache_indices=list(range(8)), start_index=48, total_steps=50,
        surrogates=['continuous', 'double_round_ste'], directions=['mse', 'lpips', 'legacy_sum', 'balanced'],
        rms_latent_steps=[.0005, .001, .002, .005, .01], trajectory_direction='balanced',
        trajectory_max_updates=8, trajectory_milestones=[1, 2, 4, 8],
        trajectory_acceptance='gt_quantized_both_metrics_no_harm', stop_on_joint_target=True,
        psnr_gain_target=.3, lpips_relative_reduction_target=.09928107050377874,
        budget_wall_seconds_by_gpus={'4': 3600, '8': 1800}, budget_gpu_hours_cap=4,
        auto_train=False, deployable=False,
        parent_result='checkpoints/foundir_flux_tail_oracle_v1/mechanism_4gpu_v1/result.json',
        parent_result_sha256='4985f367cca8bc1fb730dee173d111eb4929fc6ebda4b9d0db6da223d523f7b1',
        output_root='checkpoints/foundir_flux_tail_tradeoff_v2')
    if spec != expected:
        raise RuntimeError('frozen tradeoff diagnostic scope changed')


def prepare(output, gpus, write):
    spec = json.loads(CONFIG.read_text())
    validate_spec(spec)
    previous_result = json.loads(contract.bound(spec['parent_result'], spec['parent_result_sha256']).read_text())
    if not (previous_result['baseline_parity_pass'] and previous_result['finite_difference_pass']):
        raise RuntimeError('requires qualified V1 mechanism diagnostic')
    for filename, expected in previous_result['source_sha256'].items():
        contract.bound(filename, expected)
    prior = base.prepare(output, write=False, gpus=gpus)
    receipt = dict(spec=spec, world_size=gpus, base_receipt=prior,
                   source_sha256=binding(), config_sha256=contract.sha(CONFIG),
                   protected_pixels_accessed=False, images_decoded=0)
    if write:
        output.mkdir(parents=True, exist_ok=False)
        contract.write_new(output/'preflight.json', receipt)
    print(f'TRADEOFF_PREFLIGHT_PASS gpus={gpus} records=8 pixels_decoded=0', flush=True)
    return receipt


def load_run(output, gpus):
    receipt = json.loads((output/'preflight.json').read_text())
    validate_spec(receipt['spec'])
    if (receipt['world_size'] != gpus or receipt['config_sha256'] != contract.sha(CONFIG)
            or receipt['source_sha256'] != binding()):
        raise RuntimeError('source/config/world changed after prepare')
    return receipt


def arm_names(spec):
    names = {'historical_bf16', 'fp32_state_control_48'}
    for surrogate in spec['surrogates']:
        for direction in spec['directions']:
            for amplitude in spec['rms_latent_steps']:
                names.add(f'probe_{surrogate}_{direction}_{amplitude:g}')
        for step in spec['trajectory_milestones']:
            names.add(f'trajectory_{surrogate}_step{step}')
    return names


def aggregate(rows, spec):
    if sorted(r['cache_index'] for r in rows) != list(range(8)):
        raise RuntimeError('missing/duplicate sample')
    for row in rows:
        if (row['split'] != 'train' or not row['baseline_parity_pass'] or not row['ste_forward_exact']
                or set(row['arms']) != arm_names(spec)):
            raise RuntimeError('incomplete or invalid tradeoff record')
    # Reuse raw arithmetic, but never import its V1 finite-difference flags:
    # STE is a surrogate derivative, not a finite-difference-correct derivative.
    proxy = [{**r, 'finite_difference_pass': False} for r in rows]
    result = base.aggregate(proxy, spec)
    result.pop('finite_difference_pass')
    result.update(status='tradeoff_diagnostic_complete', rows=rows,
                  ste_is_surrogate_not_true_derivative=True, ste_forward_exact=True,
                  prior_fp32_gradient_check_pass=True, uses_gt_for_trial_acceptance=True)
    return result

def diagnose_record(output, receipt, parent, previous, rank, device, sample_index):
    import torch
    from safetensors.torch import load_file
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    from flux2.sampling import batched_prc_img, get_schedule
    from flux2.util import load_flow_model, load_ae
    from lowlight_restoration.lora import inject_lora, load_lora_checkpoint
    from lowlight_restoration.flux_tail_diagnostic import Tail, continuous_blend, checkpoint_blocks
    from smoke.qualify_foundir_flux_energy_v1 import restore, load_cached, deployed_proposal

    from lowlight_restoration.flux_tail_tradeoff import surrogate_image, objective_directions, choose_safe_trial, joint_target
    spec = receipt['spec']
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats(device)
    tensors, meta = load_cached(parent, previous, sample_index)
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
        contract.write_new(output / f'baseline_check_sample{sample_index:03d}_rank{rank}.json', dict(
            passed=bool(parity), new_vs_original_changed_values=int((zero_uint8 != historical_uint8).sum()),
            original_vs_cache_changed_values=int((historical_uint8[0].cpu() != tensors['proposal']).sum()),
            new_vs_original_raw_max_abs=float((zero_raw-baseline_raw).abs().max()), seed=seed))
    if not parity:
        raise RuntimeError('zero-guidance or cached historical uint8 baseline mismatch; stop before oracle')
    print(f'TAIL_BASELINE_PASS rank={rank} sample_index={sample_index} seed={seed}', flush=True)

    def image_metrics(image):
        mse = ((image-target)/2).square().mean()
        values = dict(psnr=float(-10*torch.log10(mse.clamp_min(1e-12))),
                      lpips=float(lpips(image, target).mean()))
        if not all(math.isfinite(x) for x in values.values()):
            raise RuntimeError('nonfinite quality metric')
        return values

    def evaluate_raw(raw):
        with torch.no_grad():
            _, quantized = deployed_proposal(observation, raw)
            ste = surrogate_image(observation, raw, 'double_round_ste')
            if not torch.equal(ste, quantized):
                raise RuntimeError('STE forward must exactly equal deployed double quantization')
            return dict(metrics=image_metrics(quantized),
                        continuous_metrics=image_metrics(continuous_blend(observation, raw)),
                        control='fp32_state_control_48')

    index = spec['start_index']
    initial = states[index].float().detach()
    with torch.no_grad():
        control_raw = tail.image(initial, index, 'rollout_full')
        baseline = evaluate_raw(baseline_raw)
        baseline['control'] = 'historical_bf16'
        control = evaluate_raw(control_raw)
        control['control'] = 'historical_bf16'
        scales = {}
        for surrogate in spec['surrogates']:
            img = surrogate_image(observation, control_raw, surrogate)
            scales[surrogate] = (float(((img-target)/2).square().mean().clamp_min(1e-8)),
                                 float(lpips(img, target).mean().clamp_min(1e-6)))
    arms = dict(historical_bf16=baseline, fp32_state_control_48=control)
    gradient_records = []
    trajectories = {}

    def gradients(state, surrogate):
        value = state.detach().clone().requires_grad_(True)
        raw = tail.image(value, index, 'rollout_full')
        image = surrogate_image(observation, raw, surrogate)
        mse = ((image-target)/2).square().mean()/scales[surrogate][0]
        perception = lpips(image, target).mean()/scales[surrogate][1]
        legacy = torch.autograd.grad(mse+perception, value, retain_graph=True)[0]
        gm = torch.autograd.grad(mse, value, retain_graph=True)[0]
        gp = torch.autograd.grad(perception, value)[0]
        tail.calls['objective_backward'] += 3
        directions, record = objective_directions(gm, gp, legacy_gradient=legacy)
        record.update(surrogate=surrogate, normalized_mse=float(mse.detach()),
                      normalized_lpips=float(perception.detach()))
        return directions, record

    for surrogate in spec['surrogates']:
        directions, gradient_record = gradients(initial, surrogate)
        gradient_records.append(gradient_record)
        print('TRADEOFF_GRADIENTS ' + json.dumps(dict(sample_index=sample_index, **gradient_record)), flush=True)
        # All 40 predeclared one-shot arms. No best-arm selection for deployment.
        for name in spec['directions']:
            direction = directions[name]
            for amplitude in spec['rms_latent_steps']:
                with torch.no_grad():
                    changed = initial if direction is None else initial-amplitude*direction
                    raw = tail.image(changed, index, 'rollout_full')
                    key = f'probe_{surrogate}_{name}_{amplitude:g}'
                    arms[key] = evaluate_raw(raw) | dict(surrogate=surrogate, direction=name,
                        amplitude=amplitude, degenerate_direction=direction is None)
                print('TRADEOFF_ARM ' + json.dumps(dict(sample_index=sample_index, arm=key, **arms[key])), flush=True)
        del directions

        state = initial.clone()
        current = dict(control)
        current['control'] = 'fp32_state_control_48'
        history = []
        accepted_updates = 0
        stop_reason = None
        for step in range(1, spec['trajectory_max_updates']+1):
            if stop_reason is None:
                directions, record = gradients(state, surrogate)
                direction = directions['balanced']
                trials = []
                if direction is not None:
                    for amplitude in spec['rms_latent_steps']:
                        with torch.no_grad():
                            raw = tail.image(state-amplitude*direction, index, 'rollout_full')
                            trials.append(evaluate_raw(raw) | dict(amplitude=amplitude))
                chosen = choose_safe_trial(trials, current['metrics'], baseline['metrics'])
                if chosen is None:
                    stop_reason = 'degenerate_direction' if direction is None else 'no_joint_safe_trial_in_frozen_grid'
                else:
                    state = (state-trials[chosen]['amplitude']*direction).detach()
                    current = dict(trials[chosen])
                    accepted_updates += 1
                    if joint_target(current['metrics'], baseline['metrics']):
                        stop_reason = 'gt_only_joint_target_reached'
                history.append(dict(step=step, gradient=record, trials=trials, chosen_trial=chosen,
                                    accepted_updates=accepted_updates, stop_reason=stop_reason))
                contract.write_new(output/f'trajectory_sample{sample_index:03d}_{surrogate}_step{step}.json', history[-1])
                del directions, direction
            if step in spec['trajectory_milestones']:
                key = f'trajectory_{surrogate}_step{step}'
                arms[key] = current | dict(accepted_updates=accepted_updates, stop_reason=stop_reason,
                                          uses_gt_for_acceptance=True)
                print('TRADEOFF_TRAJECTORY ' + json.dumps(dict(sample_index=sample_index, arm=key, **arms[key])), flush=True)
        trajectories[surrogate] = history

    if any(p.grad is not None or p.requires_grad for module in (model, ae, lpips) for p in module.parameters()):
        raise RuntimeError('frozen weights must not acquire gradients')
    torch.cuda.synchronize()
    row = dict(cache_index=sample_index, rank=rank, record_id=meta['row']['record_id'],
        source_group=meta['row']['source_group'], split='train', seed=seed, crop_xy=meta['crop_xy'],
        baseline_parity_pass=True, ste_forward_exact=True, arms=arms,
        gradient_records=gradient_records, trajectories=trajectories,
        calls=tail.calls, historical_extra_flux_forward=50, historical_extra_vae_encode=1,
        historical_extra_vae_decode=1, initial_vae_encode=1,
        seconds=time.monotonic()-started, peak_memory_gib=torch.cuda.max_memory_allocated(device)/1024**3,
        vlm_levels=contract.descriptor(meta['row']), vlm_used_for_guidance=False,
        config_sha256=receipt['config_sha256'], source_sha256=receipt['source_sha256'])
    contract.write_new(output/f'sample{sample_index:03d}.json', row)
    return row


def run(output, gpus):
    import gc
    import torch
    import torch.distributed as dist
    receipt = load_run(output, gpus)
    parent, previous = base.parent_receipt(receipt['base_receipt']['spec'])
    rank, device = base.setup_ddp(gpus)
    started = time.monotonic()
    try:
        for filename, expected in receipt['base_receipt']['artifacts'].items():
            if rank == 0 or filename.startswith(previous['spec']['output_root']):
                contract.bound(filename, expected)
        dist.barrier()
        local = []
        for index in base.assigned_indices(rank, gpus):
            local.append(diagnose_record(output, receipt, parent, previous, rank, device, index))
            gc.collect()
            torch.cuda.empty_cache()
        contract.write_new(output/f'rank{rank}.json', dict(rank=rank, world_size=gpus,
                           cache_indices=base.assigned_indices(rank, gpus), seconds=time.monotonic()-started))
        parts = [None]*gpus
        dist.all_gather_object(parts, local)
        if rank == 0:
            rows = sorted(sum(parts, []), key=lambda r: r['cache_index'])
            result = aggregate(rows, receipt['spec'])
            elapsed = time.monotonic()-started
            result.update(world_size=gpus, wall_hours=elapsed/3600, allocated_gpu_hours=gpus*elapsed/3600,
                          config_sha256=receipt['config_sha256'], source_sha256=receipt['source_sha256'])
            contract.write_new(output/'result.json', result)
            print('TRADEOFF_COMPLETE ' + json.dumps(dict(world_size=gpus, wall_hours=elapsed/3600,
                  allocated_gpu_hours=gpus*elapsed/3600, fixed_arms=len(rows[0]['arms']), deployable=False)), flush=True)
    finally:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('hardware', 'preflight', 'prepare', 'run', 'verify'))
    parser.add_argument('--run-id', default='tradeoff_4gpu_v1')
    parser.add_argument('--gpus', type=int, choices=(4, 8), default=4)
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.run_id):
        raise ValueError('invalid fresh run ID')
    spec = json.loads(CONFIG.read_text())
    validate_spec(spec)
    output = contract.path(spec['output_root'])/args.run_id
    if args.phase == 'hardware':
        base.hardware(args.gpus)
    elif args.phase in ('preflight', 'prepare'):
        prepare(output, args.gpus, args.phase == 'prepare')
    elif args.phase == 'verify':
        receipt = load_run(output, args.gpus)
        result = json.loads((output/'result.json').read_text())
        rebuilt = aggregate(result['rows'], spec)
        if (result['status'] != 'tradeoff_diagnostic_complete' or result['world_size'] != args.gpus
                or result['means'] != rebuilt['means'] or result['source_sha256'] != receipt['source_sha256']
                or result['config_sha256'] != receipt['config_sha256']
                or result['deployable'] or result['energy_trained'] or result['scientific_gate_evaluated']):
            raise RuntimeError('incomplete/invalid tradeoff result')
        print('TRADEOFF_RESULT_VERIFIED not_energy_or_generalization_success')
    else:
        if any(p.name != 'preflight.json' for p in output.iterdir()):
            raise FileExistsError('preserve partial run; choose a fresh ID')
        run(output, args.gpus)


if __name__ == '__main__':
    main()
