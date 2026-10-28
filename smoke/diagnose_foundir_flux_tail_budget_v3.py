#!/usr/bin/env python3
"""GT-only factorial diagnosis: direction diversity x spending excess PSNR."""
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
from smoke import diagnose_foundir_flux_tail_tradeoff_v2 as v2
from smoke import flux_energy_qualification_contract_v1 as contract

CONFIG = ROOT/'configs/foundir_flux_tail_budget_v3.json'
FILES = ['configs/foundir_flux_tail_budget_v3.json',
         'lowlight_restoration/flux_tail_budget.py',
         'smoke/diagnose_foundir_flux_tail_budget_v3.py',
         'smoke/run_foundir_flux_tail_budget_v3_gpu.sh',
         'smoke/start_foundir_flux_tail_budget_v3_tmux.sh']


def binding():
    return {**v2.binding(), **{name: contract.sha(ROOT/name) for name in FILES}}


def validate_spec(spec):
    expected = dict(mode='training_gt_only_psnr_budget_diagnostic',
        parent_result='checkpoints/foundir_flux_tail_tradeoff_v2/tradeoff_4gpu_v1/result.json',
        parent_result_sha256='78e442d5f3a4cda82c15b2903b02d66a2ab790ac167df8bc12e9a8bcf1858625',
        output_root='checkpoints/foundir_flux_tail_budget_v3', allowed_world_sizes=[4, 8],
        cache_indices=list(range(8)), start_index=48, total_steps=50, surrogate='continuous',
        direction_sets=dict(balanced=[.5], mixed=[0., .25, .5, .75, 1.]),
        acceptance_rules=['monotone', 'psnr_budget'],
        selection='minimum_psnr_target_shortfall_then_lpips_then_highest_psnr_then_smallest_step',
        rms_latent_steps=[.0005, .001, .002, .005, .01], max_updates=16,
        milestones=[1, 2, 4, 8, 16], stop_on_joint_target=False,
        psnr_gain_target=.3, lpips_relative_reduction_target=.09928107050377874,
        budget_wall_seconds_by_gpus={'4': 3600, '8': 1800}, budget_gpu_hours_cap=4,
        auto_train=False, auto_expand=False, deployable=False)
    if spec != expected:
        raise RuntimeError('frozen V3 scope changed')


def prepare(output, gpus, write):
    spec = json.loads(CONFIG.read_text())
    validate_spec(spec)
    previous_result = json.loads(contract.bound(spec['parent_result'], spec['parent_result_sha256']).read_text())
    if (previous_result['status'] != 'tradeoff_diagnostic_complete'
            or not previous_result['baseline_parity_pass'] or not previous_result['ste_forward_exact']):
        raise RuntimeError('requires complete V2 diagnostic')
    for filename, expected in previous_result['source_sha256'].items():
        contract.bound(filename, expected)
    prior = v2.prepare(output, gpus, write=False)
    receipt = dict(spec=spec, world_size=gpus, base_receipt=prior['base_receipt'],
                   source_sha256=binding(), config_sha256=contract.sha(CONFIG),
                   parent_result_sha256=spec['parent_result_sha256'],
                   protected_pixels_accessed=False, images_decoded=0)
    if write:
        output.mkdir(parents=True, exist_ok=False)
        contract.write_new(output/'preflight.json', receipt)
    print(f'BUDGET_PREFLIGHT_PASS gpus={gpus} records=8 pixels_decoded=0', flush=True)
    return receipt


def load_run(output, gpus):
    receipt = json.loads((output/'preflight.json').read_text())
    validate_spec(receipt['spec'])
    if (receipt['world_size'] != gpus or receipt['config_sha256'] != contract.sha(CONFIG)
            or receipt['source_sha256'] != binding()):
        raise RuntimeError('source/config/world changed after prepare')
    return receipt


def policies(spec):
    return [(f'{direction}_{rule}', direction, rule)
            for direction in spec['direction_sets'] for rule in spec['acceptance_rules']]


def arm_names(spec):
    return {'historical_bf16', 'fp32_state_control_48'} | {
        f'{name}_step{step}' for name, _, _ in policies(spec) for step in spec['milestones']}


def validate_trajectory(row, spec, name, direction_set, rule):
    from lowlight_restoration.flux_tail_budget import choose_trial
    baseline = row['arms']['historical_bf16']['metrics']
    current = row['arms']['fp32_state_control_48']['metrics']
    history = row['trajectories'][name]
    if not 1 <= len(history) <= spec['max_updates']:
        raise RuntimeError('invalid trajectory length')
    accepted = 0
    for step, item in enumerate(history, 1):
        if item['step'] != step or item['current_before'] != current:
            raise RuntimeError('invalid trajectory chain')
        identities = [(t['lpips_weight'], t['amplitude']) for t in item['trials']]
        expected = [(w, a) for w in spec['direction_sets'][direction_set]
                    if w not in item['degenerate_weights'] for a in spec['rms_latent_steps']]
        if identities != expected:
            raise RuntimeError('missing/extra trial')
        chosen, decisions = choose_trial(item['trials'], current, baseline, rule, spec['psnr_gain_target'])
        if chosen != item['chosen_trial'] or decisions != item['decision']:
            raise RuntimeError('GT selection mismatch')
        if chosen is not None:
            current = item['trials'][chosen]['metrics']
            accepted += 1
            if item['stop_reason'] is not None:
                raise RuntimeError('must not stop early at target')
        elif step != len(history) or item['stop_reason'] != 'no_eligible_trial_in_frozen_grid':
            raise RuntimeError('invalid stopping rule')
        if item['current_after'] != current or item['accepted_updates'] != accepted:
            raise RuntimeError('invalid accepted trajectory')
        if step in spec['milestones']:
            arm = row['arms'][f'{name}_step{step}']
            if arm['metrics'] != current or arm['accepted_updates'] != accepted:
                raise RuntimeError('invalid milestone')
    if len(history) < spec['max_updates'] and history[-1]['chosen_trial'] is not None:
        raise RuntimeError('truncated trajectory')
    for step in spec['milestones']:
        arm = row['arms'][f'{name}_step{step}']
        if step > len(history) and (arm['metrics'] != current or arm['accepted_updates'] != accepted):
            raise RuntimeError('stopped state changed')


def aggregate(rows, spec):
    validate_spec(spec)
    if sorted(r['cache_index'] for r in rows) != list(range(8)):
        raise RuntimeError('missing/duplicate sample')
    for row in rows:
        if (row['split'] != 'train' or not row['baseline_parity_pass'] or not row['ste_forward_exact']
                or set(row['arms']) != arm_names(spec)):
            raise RuntimeError('invalid sample/arm set')
        for name, direction_set, rule in policies(spec):
            validate_trajectory(row, spec, name, direction_set, rule)
    result = base.aggregate([{**r, 'finite_difference_pass': False} for r in rows], spec)
    result.pop('finite_difference_pass')
    result.update(status='psnr_budget_diagnostic_complete', rows=rows,
                  ste_forward_exact=True, ste_used_for_guidance=False,
                  uses_gt_for_trial_acceptance=True, all_trial_decisions_reverified=True,
                  no_global_oracle_bound_claimed=True)
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

    from lowlight_restoration.flux_tail_tradeoff import surrogate_image
    from lowlight_restoration.flux_tail_budget import mixed_directions, choose_trial
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
        baseline = evaluate_raw(baseline_raw)
        baseline['control'] = 'historical_bf16'
        control_raw = tail.image(initial, index, 'rollout_full')
        control = evaluate_raw(control_raw)
        control['control'] = 'historical_bf16'
        image = continuous_blend(observation, control_raw)
        scales = (float(((image-target)/2).square().mean().clamp_min(1e-8)),
                  float(lpips(image, target).mean().clamp_min(1e-6)))
    arms = dict(historical_bf16=baseline, fp32_state_control_48=control)
    trajectories, costs = {}, {}

    def gradients(state, weights):
        value = state.detach().clone().requires_grad_(True)
        raw = tail.image(value, index, 'rollout_full')
        image = continuous_blend(observation, raw)
        mse = ((image-target)/2).square().mean()/scales[0]
        perception = lpips(image, target).mean()/scales[1]
        gm = torch.autograd.grad(mse, value, retain_graph=True)[0]
        gp = torch.autograd.grad(perception, value)[0]
        tail.calls['objective_backward'] += 2
        directions, record = mixed_directions(gm, gp, weights)
        record.update(normalized_mse=float(mse.detach()), normalized_lpips=float(perception.detach()))
        return directions, record

    for name, direction_set, rule in policies(spec):
        tick, counts_before = time.monotonic(), dict(tail.calls)
        state, current = initial.clone(), dict(control)
        current['control'] = 'fp32_state_control_48'
        history, accepted, stop_reason = [], 0, None
        for step in range(1, spec['max_updates']+1):
            if stop_reason is None:
                directions, gradient_record = gradients(state, spec['direction_sets'][direction_set])
                trials, degenerate = [], []
                for weight, direction in directions.items():
                    if direction is None:
                        degenerate.append(weight)
                        continue
                    for amplitude in spec['rms_latent_steps']:
                        with torch.no_grad():
                            raw = tail.image(state-amplitude*direction, index, 'rollout_full')
                            trials.append(evaluate_raw(raw) | dict(amplitude=amplitude, lpips_weight=weight))
                before = current['metrics']
                chosen, decision = choose_trial(trials, before, baseline['metrics'], rule, spec['psnr_gain_target'])
                if chosen is None:
                    stop_reason = 'no_eligible_trial_in_frozen_grid'
                else:
                    selected = trials[chosen]
                    state = (state-selected['amplitude']*directions[selected['lpips_weight']]).detach()
                    current = dict(selected)
                    accepted += 1
                entry = dict(step=step, gradient=gradient_record, trials=trials,
                    degenerate_weights=degenerate, chosen_trial=chosen, decision=decision,
                    current_before=before, current_after=current['metrics'],
                    accepted_updates=accepted, stop_reason=stop_reason)
                history.append(entry)
                contract.write_new(output/f'trajectory_sample{sample_index:03d}_{name}_step{step}.json', entry)
                print('BUDGET_UPDATE ' + json.dumps(dict(sample_index=sample_index, policy=name, step=step,
                    accepted_updates=accepted, metrics=current['metrics'], stop_reason=stop_reason,
                    trial_count=len(trials))), flush=True)
                del directions, direction
            if step in spec['milestones']:
                key = f'{name}_step{step}'
                arms[key] = current | dict(accepted_updates=accepted, stop_reason=stop_reason,
                    attempted_updates=len(history), uses_gt_for_acceptance=True, policy=name)
        trajectories[name] = history
        costs[name] = dict(seconds=time.monotonic()-tick,
                          calls={k: tail.calls[k]-counts_before[k] for k in tail.calls},
                          attempted_updates=len(history), accepted_updates=accepted,
                          rejected_trial_count=sum(len(h['trials'])-(h['chosen_trial'] is not None) for h in history))

    if any(p.grad is not None or p.requires_grad for module in (model, ae, lpips) for p in module.parameters()):
        raise RuntimeError('frozen weights must not acquire gradients')
    torch.cuda.synchronize()
    row = dict(cache_index=sample_index, rank=rank, record_id=meta['row']['record_id'],
        source_group=meta['row']['source_group'], split='train', seed=seed, crop_xy=meta['crop_xy'],
        baseline_parity_pass=True, ste_forward_exact=True, arms=arms, trajectories=trajectories,
        costs=costs, calls=tail.calls, historical_extra_flux_forward=50, historical_extra_vae_encode=1,
        historical_extra_vae_decode=1, initial_vae_encode=1, timestep=tail.schedule[index],
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
            print('BUDGET_COMPLETE ' + json.dumps(dict(world_size=gpus, wall_hours=elapsed/3600,
                  allocated_gpu_hours=gpus*elapsed/3600, fixed_arms=len(rows[0]['arms']), deployable=False)), flush=True)
    finally:
        dist.destroy_process_group()



def verify(output, gpus):
    receipt = load_run(output, gpus)
    result = json.loads((output/'result.json').read_text())
    rebuilt = aggregate(result['rows'], receipt['spec'])
    for key in ('status', 'means', 'baseline_parity_pass', 'ste_forward_exact',
                'ste_used_for_guidance', 'uses_gt_for_trial_acceptance', 'all_trial_decisions_reverified',
                'deployable', 'energy_trained', 'scientific_gate_evaluated',
                'generalization_established', 'protected_pixels_accessed'):
        if result[key] != rebuilt[key]:
            raise RuntimeError('invalid result field: '+key)
    if (result['world_size'] != gpus or result['source_sha256'] != receipt['source_sha256']
            or result['config_sha256'] != receipt['config_sha256']):
        raise RuntimeError('invalid result identity')
    for row in result['rows']:
        index = row['cache_index']
        saved = json.loads((output/f'sample{index:03d}.json').read_text())
        if (saved != row or row['rank'] != index % gpus
                or row['source_sha256'] != receipt['source_sha256']
                or row['config_sha256'] != receipt['config_sha256']):
            raise RuntimeError('sample identity mismatch')
        for name, history in row['trajectories'].items():
            for item in history:
                step = item['step']
                if json.loads((output/f'trajectory_sample{index:03d}_{name}_step{step}.json').read_text()) != item:
                    raise RuntimeError('trial audit mismatch')
    for rank in range(gpus):
        row = json.loads((output/f'rank{rank}.json').read_text())
        if row['world_size'] != gpus or row['cache_indices'] != base.assigned_indices(rank, gpus):
            raise RuntimeError('rank partition mismatch')
    print('BUDGET_RESULT_VERIFIED not_energy_or_generalization_success', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('hardware', 'preflight', 'prepare', 'run', 'verify'))
    parser.add_argument('--run-id', default='budget_4gpu_v1')
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
        verify(output, args.gpus)
    else:
        if any(p.name != 'preflight.json' for p in output.iterdir()):
            raise FileExistsError('preserve partial run; choose a fresh ID')
        run(output, args.gpus)


if __name__ == '__main__':
    main()
