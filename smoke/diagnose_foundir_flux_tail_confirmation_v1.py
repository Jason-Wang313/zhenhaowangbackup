#!/usr/bin/env python3
"""Fixed V3 policy on the other 24 TRAIN sources. GT-only, not energy training."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from smoke import diagnose_foundir_flux_tail_budget_v3 as v3
from smoke import diagnose_foundir_flux_tail_oracle_v1 as base
from smoke import flux_energy_qualification_contract_v1 as contract

CONFIG = ROOT/'configs/foundir_flux_tail_confirmation_v1.json'
FILES = ['configs/foundir_flux_tail_confirmation_v1.json',
         'smoke/diagnose_foundir_flux_tail_confirmation_v1.py',
         'smoke/run_foundir_flux_tail_confirmation_v1_gpu.sh',
         'smoke/start_foundir_flux_tail_confirmation_v1_tmux.sh']


def binding():
    return {**v3.binding(), **{name: contract.sha(ROOT/name) for name in FILES}}


def validate_spec(spec):
    # V3's immutable config validates every numerical/acceptance setting. Only
    # the dataset and the already-selected single policy change in this entry.
    prior = json.loads(v3.CONFIG.read_text())
    v3.validate_spec(prior)
    expected = prior | dict(mode='training_source_disjoint_gt_confirmation',
        parent_result='checkpoints/foundir_flux_tail_budget_v3/budget_4gpu_v1/result.json',
        parent_result_sha256='06eed91ab7c2c0942b221a14d4c295d1e269bf57b17a8d79354ec9be074d24e4',
        output_root='checkpoints/foundir_flux_tail_confirmation_v1',
        cache_indices=list(range(8, 32)), excluded_diagnostic_indices=list(range(8)),
        primary_endpoint='mixed_psnr_budget_step16',
        direction_sets=dict(mixed=[0., .25, .5, .75, 1.]), acceptance_rules=['psnr_budget'])
    if spec != expected:
        raise RuntimeError('frozen 24-source confirmation scope changed')


def assigned_indices(rank, world_size):
    if world_size not in (4, 8) or not 0 <= rank < world_size:
        raise ValueError('requires one 4/8-GPU node')
    return list(range(8+rank, 32, world_size))


def validate_selection(selected):
    if len(selected) != 32 or any(r['split'] != 'train' or r['category'] != '05Noise' for r in selected):
        raise RuntimeError('requires the original 32 training Noise sources')
    # This enforces within-confirmation uniqueness AND exclusion of all eight
    # diagnostic sources/targets/observations without opening their pixels.
    for field in ('record_id', 'source_group', 'gt_sha256', 'lq_sha256'):
        if any(not r[field] for r in selected) or len({r[field] for r in selected}) != 32:
            raise RuntimeError('duplicate/overlapping training identity: '+field)


def prepare(output, gpus, write):
    assigned_indices(0, gpus)
    spec = json.loads(CONFIG.read_text())
    validate_spec(spec)
    prior_result = json.loads(contract.bound(spec['parent_result'], spec['parent_result_sha256']).read_text())
    if (prior_result['status'] != 'psnr_budget_diagnostic_complete'
            or not prior_result['baseline_parity_pass']
            or not prior_result['all_trial_decisions_reverified']
            or not prior_result['means'][spec['primary_endpoint']]['joint_target_vs_historical']):
        raise RuntimeError('requires the completed, passing V3 GT diagnostic')
    for filename, expected in prior_result['source_sha256'].items():
        contract.bound(filename, expected)
    parent_spec = json.loads(base.CONFIG.read_text())
    base.validate_spec(parent_spec)
    parent, previous = base.parent_receipt(parent_spec)
    validate_selection(previous['selected'])
    artifacts, identities = {}, []
    for index in spec['cache_indices']:
        row = previous['selected'][index]
        name = f'cache_{index:03d}'
        meta_path = parent/(name+'.json')
        meta = json.loads(meta_path.read_text())
        contract.validate_cache_metadata(meta, row, previous)
        if meta['file'] != name+'.safetensors':
            raise RuntimeError('unexpected cache path')
        data = contract.bound(str(parent/meta['file']), meta['sha256'])
        artifacts[str(data.relative_to(ROOT))] = meta['sha256']
        artifacts[str(meta_path.relative_to(ROOT))] = contract.sha(meta_path)
        identities.append(dict(cache_index=index, row=row, seed=meta['seed'], crop_xy=meta['crop_xy']))
    for field in ('flow', 'ae'):
        filename = previous['spec'][field]
        expected = previous['spec'][field+'_sha256']
        print('VERIFY '+field, flush=True)
        contract.bound(filename, expected)
        artifacts[filename] = expected
    for filename in (previous['plan']['baseline_checkpoint'],
                     'models/torch/hub/checkpoints/alexnet-owt-7be5be79.pth'):
        expected = previous['artifacts'][filename]
        contract.bound(filename, expected)
        artifacts[filename] = expected
    receipt = dict(spec=spec, world_size=gpus, parent_spec=parent_spec,
        config_sha256=contract.sha(CONFIG), source_sha256=binding(), artifacts=artifacts,
        identities=identities, protected_pixels_accessed=False, images_decoded=0,
        source_and_target_disjoint_from_v3=True,
        data_role='training_source_mechanism_confirmation_not_heldout_generalization')
    if write:
        output.mkdir(parents=True, exist_ok=False)
        contract.write_new(output/'preflight.json', receipt)
    print(f'CONFIRMATION_PREFLIGHT_PASS gpus={gpus} records=24 sources=24 excluded_sources=8 pixels_decoded=0', flush=True)
    return receipt


def load_run(output, gpus):
    receipt = json.loads((output/'preflight.json').read_text())
    validate_spec(receipt['spec'])
    if (receipt['world_size'] != gpus or receipt['source_sha256'] != binding()
            or receipt['config_sha256'] != contract.sha(CONFIG)
            or receipt['parent_spec'] != json.loads(base.CONFIG.read_text())):
        raise RuntimeError('source/config/world changed after prepare')
    return receipt


def aggregate(rows, spec):
    validate_spec(spec)
    if sorted(r['cache_index'] for r in rows) != spec['cache_indices']:
        raise RuntimeError('missing/duplicate/unexpected confirmation sample')
    for field in ('record_id', 'source_group'):
        if len({r[field] for r in rows}) != 24:
            raise RuntimeError('duplicate confirmation '+field)
    for row in rows:
        if (row['split'] != 'train' or not row['baseline_parity_pass'] or not row['ste_forward_exact']
                or set(row['arms']) != v3.arm_names(spec)
                or set(row['trajectories']) != {'mixed_psnr_budget'}):
            raise RuntimeError('invalid sample/arm/policy set')
        for key, arm in row['arms'].items():
            expected_control = 'fp32_state_control_48' if key.startswith('mixed_') else 'historical_bf16'
            if arm['control'] != expected_control or any(
                    not math.isfinite(arm['metrics'][m]) for m in ('psnr', 'lpips')):
                raise RuntimeError('invalid metrics/precision control')
        v3.validate_trajectory(row, spec, 'mixed_psnr_budget', 'mixed', 'psnr_budget')
    means = {key: {m: sum(r['arms'][key]['metrics'][m] for r in rows)/24
                   for m in ('psnr', 'lpips')} for key in sorted(v3.arm_names(spec))}
    baseline = dict(means['historical_bf16'])
    for key, value in means.items():
        control = rows[0]['arms'][key]['control']
        reference = means[control]
        value.update(psnr_gain_vs_historical=value['psnr']-baseline['psnr'],
            lpips_reduction_vs_historical=1-value['lpips']/max(baseline['lpips'], 1e-12),
            control=control, psnr_gain_vs_precision_control=value['psnr']-reference['psnr'],
            lpips_reduction_vs_precision_control=1-value['lpips']/max(reference['lpips'], 1e-12))
        value['joint_target_vs_historical'] = (
            value['psnr_gain_vs_historical'] >= spec['psnr_gain_target']
            and value['lpips_reduction_vs_historical'] >= spec['lpips_relative_reduction_target'])
    primary = spec['primary_endpoint']
    per_image = []
    for row in sorted(rows, key=lambda r: r['cache_index']):
        before, after = [row['arms'][key]['metrics'] for key in ('historical_bf16', primary)]
        gain = after['psnr']-before['psnr']
        reduction = 1-after['lpips']/max(before['lpips'], 1e-12)
        per_image.append(dict(cache_index=row['cache_index'], psnr_gain=gain,
            lpips_relative_reduction=reduction,
            joint_target=bool(gain >= spec['psnr_gain_target']
                              and reduction >= spec['lpips_relative_reduction_target']),
            both_no_harm=bool(gain >= 0 and reduction >= 0)))
    return dict(status='training_gt_confirmation_complete', rows=rows, means=means,
        primary_endpoint=primary, primary_mean_joint_target=means[primary]['joint_target_vs_historical'],
        primary_per_image=per_image, primary_per_image_pass_count=sum(p['joint_target'] for p in per_image),
        baseline_parity_pass=True, ste_forward_exact=True, ste_used_for_guidance=False,
        uses_gt_for_guidance=True, uses_gt_for_trial_acceptance=True,
        all_trial_decisions_reverified=True, no_global_oracle_bound_claimed=True,
        records=24, sources=24, split='train', deployable=False, energy_trained=False,
        scientific_gate_evaluated=False, generalization_established=False, protected_pixels_accessed=False)


def validate_row_identity(row, receipt):
    index = row['cache_index']
    expected = {i['cache_index']: i for i in receipt['identities']}[index]
    if (row['rank'] != (index-8) % receipt['world_size']
            or row['config_sha256'] != receipt['config_sha256']
            or row['source_sha256'] != receipt['source_sha256']
            or any(row[k] != expected['row'][k] for k in ('record_id', 'source_group', 'split'))
            or any(row[k] != expected[k] for k in ('seed', 'crop_xy'))):
        raise RuntimeError('sample identity mismatch')


def run(output, gpus):
    import gc
    import torch
    import torch.distributed as dist
    receipt = load_run(output, gpus)
    parent, previous = base.parent_receipt(receipt['parent_spec'])
    rank, device = base.setup_ddp(gpus)
    started = time.monotonic()
    try:
        if rank == 0:
            for filename, expected in receipt['artifacts'].items():
                contract.bound(filename, expected)
        dist.barrier()
        for index in assigned_indices(rank, gpus):
            # Reuse the exact hash-bound V3 gradient/rollout/quantization code.
            row = v3.diagnose_record(output, receipt, parent, previous, rank, device, index)
            validate_row_identity(row, receipt)
            del row
            gc.collect()
            torch.cuda.empty_cache()
        contract.write_new(output/f'rank{rank}.json', dict(rank=rank, world_size=gpus,
            cache_indices=assigned_indices(rank, gpus), seconds=time.monotonic()-started))
        dist.barrier()
        if rank == 0:
            # Avoid all-gathering large per-candidate JSON traces through NCCL.
            rows = [json.loads((output/f'sample{i:03d}.json').read_text()) for i in receipt['spec']['cache_indices']]
            result = aggregate(rows, receipt['spec'])
            elapsed = time.monotonic()-started
            result.update(world_size=gpus, wall_hours=elapsed/3600, allocated_gpu_hours=gpus*elapsed/3600,
                config_sha256=receipt['config_sha256'], source_sha256=receipt['source_sha256'])
            contract.write_new(output/'result.json', result)
            print('CONFIRMATION_COMPLETE '+json.dumps(dict(world_size=gpus,
                wall_hours=elapsed/3600, allocated_gpu_hours=gpus*elapsed/3600,
                primary_endpoint=result['primary_endpoint'],
                primary_mean_joint_target=result['primary_mean_joint_target'], deployable=False)), flush=True)
        dist.barrier()
    finally:
        dist.destroy_process_group()


def verify(output, gpus):
    receipt = load_run(output, gpus)
    result = json.loads((output/'result.json').read_text())
    rebuilt = aggregate(result['rows'], receipt['spec'])
    for key, value in rebuilt.items():
        if result[key] != value:
            raise RuntimeError('invalid result field: '+key)
    if (result['world_size'] != gpus or result['source_sha256'] != receipt['source_sha256']
            or result['config_sha256'] != receipt['config_sha256']):
        raise RuntimeError('invalid result identity')
    for row in result['rows']:
        validate_row_identity(row, receipt)
        index, rank = row['cache_index'], row['rank']
        if json.loads((output/f'sample{index:03d}.json').read_text()) != row:
            raise RuntimeError('sample file mismatch')
        check = json.loads((output/f'baseline_check_sample{index:03d}_rank{rank}.json').read_text())
        if (not check['passed'] or check['new_vs_original_changed_values'] != 0
                or check['original_vs_cache_changed_values'] != 0 or check['seed'] != row['seed']):
            raise RuntimeError('baseline reproduction mismatch')
        for name, history in row['trajectories'].items():
            for item in history:
                step = item['step']
                if json.loads((output/f'trajectory_sample{index:03d}_{name}_step{step}.json').read_text()) != item:
                    raise RuntimeError('trial trace mismatch')
    for rank in range(gpus):
        row = json.loads((output/f'rank{rank}.json').read_text())
        if row['rank'] != rank or row['world_size'] != gpus or row['cache_indices'] != assigned_indices(rank, gpus):
            raise RuntimeError('rank partition mismatch')
    print('CONFIRMATION_RESULT_VERIFIED GT_only_not_energy_or_generalization_success', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('hardware', 'preflight', 'prepare', 'run', 'verify'))
    parser.add_argument('--run-id', default='confirm24_4gpu_v1')
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
            raise FileExistsError('preserve partial run; use a fresh ID')
        run(output, args.gpus)


if __name__ == '__main__':
    main()
