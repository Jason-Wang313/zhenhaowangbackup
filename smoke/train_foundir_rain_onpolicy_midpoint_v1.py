#!/usr/bin/env python3
"""Bounded 200-update Rain semi-gradient branch, initialized from full step400 state."""
import argparse
from contextlib import nullcontext
from datetime import timedelta
import gc
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'third_party/flux2/src'))
from smoke import existing_flux_replay_contract_v1 as c
from smoke import existing_flux_replay_cluster_v1 as cluster
from smoke import rain_native50_continue_contract_v1 as parent_contract
from smoke import train_foundir_rain_native50_continue_v1 as parent
from smoke import evaluate_foundir_rain_midpoint_development_v1 as prior_eval
from smoke import rain_midpoint_readout_v1 as readout
from smoke import rain_onpolicy_midpoint_state_v1 as state_io
from lowlight_restoration import rain_onpolicy_midpoint as logic

CONFIG = 'configs/foundir_rain_onpolicy_midpoint_v1.json'
SOURCES = [CONFIG, 'lowlight_restoration/rain_onpolicy_midpoint.py',
           'smoke/rain_onpolicy_midpoint_state_v1.py',
           'smoke/train_foundir_rain_onpolicy_midpoint_v1.py',
           'smoke/run_foundir_rain_onpolicy_midpoint_v1_gpu.sh',
           'smoke/start_foundir_rain_onpolicy_midpoint_v1_tmux.sh', *prior_eval.SOURCES]


def config():
    cfg = c.read(ROOT / CONFIG)
    expected = dict(start_step=400, end_step=600, new_updates=200, global_batch=32,
                    learning_rate=1e-5, loss_weights={'gt_rgb_mse': 10., 'gt_lpips': .05},
                    schedule_steps=50, prefix_calls_without_grad=25, readout_call_index=25,
                    train_probe_steps=[400, 450, 500, 600], development_steps=[600],
                    training_slots=[12800, 19200], training_probe_slots=[12800, 12832],
                    development_slots=[64032, 64132], nodes=4, local_gpus=2, world_size=8,
                    gpu_family='H100_or_H200', minimum_memory_gib=75,
                    wall_cap_seconds=10800, minimum_free_disk_gib=100,
                    psnr_minimum=34.2, lpips_maximum=.08,
                    official_or_sealed_pixels_allowed=False, automatic_follow_on=False)
    if any(cfg.get(k) != v for k, v in expected.items()):
        raise ValueError('authorized midpoint training scope changed')
    if cfg['parent_run'] != prior_eval.config()['parent_run']:
        raise ValueError('parent run changed')
    return cfg


def output_path(run_id):
    c.run_path(ROOT, run_id)
    return c.path(ROOT, 'checkpoints/foundir_rain_onpolicy_midpoint_v1/' + run_id)


def make_receipt():
    cfg = config()
    source = c.path(ROOT, cfg['parent_run'])
    receipt = parent_contract.load_receipt(ROOT, source)
    readout.validate_rows(receipt['rows'])
    for row in receipt['rows'][12800:19200]:
        if row['usage'] != 'train_update' or row['split'] != 'train' or row['category'] != '10Rain':
            raise ValueError('only original training presentations may train')
    marker = source / 'steps/step_000400.json'
    if c.sha(marker) != cfg['parent_checkpoint_receipt_sha256']:
        raise ValueError('wrong step400 receipt')
    record = c.read(marker)
    if (record['sha256'] != cfg['parent_checkpoint_sha256'] or record['step'] != 400
            or record['receipt_hash'] != c.sha(source / 'receipt.json')):
        raise ValueError('wrong step400 checkpoint')
    if not c.path(source, record['file']).is_file():
        raise FileNotFoundError(record['file'])
    baseline = c.path(ROOT, cfg['baseline_run'])
    if (c.sha(baseline / 'result.json') != cfg['baseline_result_sha256']
            or c.sha(baseline / 'receipt.json') != cfg['baseline_receipt_sha256']):
        raise ValueError('matched fixed-prefix baseline changed')
    result = c.read(baseline / 'result.json')
    report = c.read(ROOT / cfg['baseline_report'])
    if (not result['matched_comparison_valid'] or result['summary']['count'] != 100
            or report['result_sha256'] != cfg['baseline_result_sha256']
            or result['checkpoint_step'] != 400 or result['fixed_call_index'] != 25):
        raise ValueError('baseline not validated')
    # Metadata only. Do not reread 6,400 cache tensors or re-generate teacher images.
    indices = [*range(12800, 19200), *range(64032, 64132)]
    cache_receipts = {str(i): c.sha(parent.cache_marker(source, i)) for i in indices}
    return {'config': cfg, 'parent_receipt_sha256': c.sha(source / 'receipt.json'),
            'checkpoint': record, 'cache_receipts': cache_receipts,
            'selection_digest': c.digest([receipt['rows'][i] for i in indices]),
            'sources': {**receipt['sources'], **{n: c.sha(ROOT / n) for n in SOURCES},
                        cfg['baseline_report']: c.sha(ROOT / cfg['baseline_report'])},
            'baseline_per_image': result['per_image_metrics'],
            'parent_optimizer_restored': True, 'objective_changed': True,
            'new_teacher_inference': False, 'pixels_read_in_preflight': False}


def load_receipt(output):
    bound = c.read(output / 'receipt.json')
    if bound['config'] != config():
        raise ValueError('bound config changed')
    for name, digest in bound['sources'].items():
        if c.sha(ROOT / name) != digest:
            raise ValueError('bound source changed: ' + name)
    source = c.path(ROOT, bound['config']['parent_run'])
    if c.sha(source / 'receipt.json') != bound['parent_receipt_sha256']:
        raise ValueError('parent receipt changed')
    receipt = parent_contract.load_receipt(ROOT, source)
    indices = [*range(12800, 19200), *range(64032, 64132)]
    if c.digest([receipt['rows'][i] for i in indices]) != bound['selection_digest']:
        raise ValueError('sampler or evaluation rows changed')
    return bound, receipt, source


def load_parent_state(bound, source):
    import torch
    cfg = bound['config']
    if c.sha(source / 'steps/step_000400.json') != cfg['parent_checkpoint_receipt_sha256']:
        raise ValueError('parent step400 receipt changed')
    record = bound['checkpoint']
    file = c.path(source, record['file'])
    if c.sha(file) != cfg['parent_checkpoint_sha256']:
        raise ValueError('parent state SHA changed')
    saved = torch.load(file, map_location='cpu', weights_only=True)
    state_io.validate_state(saved, record, bound['parent_receipt_sha256'])
    if saved['step'] != 400:
        raise ValueError('must start from step400, not the last step2000')
    return saved


def cached(bound, receipt, source, index, device, *, training=False):
    if training and index not in range(12800, 19200):
        raise ValueError('evaluation cannot enter training')
    if str(index) not in bound['cache_receipts']:
        raise ValueError('unbound pixel access')
    if c.sha(parent.cache_marker(source, index)) != bound['cache_receipts'][str(index)]:
        raise ValueError('cached presentation changed')
    data, _ = parent.cached(source, receipt, index, device)
    # Teacher and GT latent are NOT passed to the new objective or generator.
    return {'lq': data['lq'], 'gt': data['gt']}


def inference(model, ae, lq, prompt, seed):
    """No GT, scorer, teacher or candidate selection in this path."""
    import torch
    with torch.no_grad():
        inputs = logic.generated_inputs(model, ae, lq, prompt, seed)
        return logic.decode_endpoint(ae, logic.predict_endpoint(model, inputs), checkpointed=False)


def image_identity(output, bound, receipt, index, step):
    return {'receipt_sha256': c.sha(output / 'receipt.json'), 'index': index, 'step': step,
            'row': receipt['rows'][index], 'call_index': 25,
            'checkpoint_receipt_sha256': c.sha(output / 'steps' / f'step_{step:06d}.json'),
            'cache_receipt_sha256': bound['cache_receipts'][str(index)]}


def probe(output, bound, receipt, source, model, ae, prompt, metric, rank, device, step, *, baseline_only=False):
    import torch
    from flux2.sampling import batched_prc_img
    from smoke.evaluate_foundir_blur_only_flux2_full_development_ddp import restore_batch
    rng = state_io.base.rng_state(device)
    model.eval()
    try:
        if baseline_only and step != 400:
            raise ValueError('matched baseline must use parent400 weights')
        indices = list(range(64032 + rank, 64132, 8)) if baseline_only else logic.probe_indices(step, rank)
        for index in indices:
            marker = output / 'probes' / f'step_{step:06d}_image_{index}.json'
            ident = image_identity(output, bound, receipt, index, step)
            if marker.exists():
                c.validate_record(c.read(marker), ident, output)
                continue
            data = cached(bound, receipt, source, index, device)
            seed = (receipt['config']['probe_seed'] + index - 64000 if index >= 64032
                    else receipt['config']['seed'] + 1000003 * (index + 1))
            rgb = inference(model, ae, data['lq'], prompt, seed)
            image = parent.quantize(rgb)
            if not torch.isfinite(rgb).all():
                raise ValueError('nonfinite probe output')
            control = None
            if step == 400 and not baseline_only:
                # Real GPU check that the grad-capable path starts from the same image.
                with torch.inference_mode():
                    snap, _ = readout.run_prefix(model, lambda m: restore_batch(
                        m, ae, data['lq'], prompt['cfg_context'], prompt['cfg_ids'], batched_prc_img,
                        seeds=[seed], steps=50, reference_time=10))
                    reference = parent.quantize(readout.decode_endpoint(snap, ae, device))
                control = bool(torch.equal(image, reference))
                if not control:
                    raise ValueError('new path differs from validated fixed readout; do not train')
            metrics = parent.scores(metric, image, data['gt'])
            name, digest = parent.replay.save_tensors(output, f'probe_{step}_{index}', {'uint8': image})
            c.write_new(marker, {'identity': ident, 'index': index, 'metrics': metrics,
                                'tensor': name, 'files': {name: digest},
                                'training_fit_only': index < 19200,
                                'exact_readout_control': control, 'GT_used_for_output_selection': False})
            print('RAIN_ONPOLICY_PROBE ' + json.dumps({'step': step, 'index': index, 'metrics': metrics}), flush=True)
    finally:
        state_io.base.restore_rng(rng, device)


def paired_summary(rows, baseline):
    if not rows or len({r['index'] for r in rows}) != len(rows):
        raise ValueError('empty or duplicate results')
    if {r['index'] for r in rows} != set(baseline):
        raise ValueError('unmatched results')
    for r in rows:
        if set(r['metrics']) != {'psnr', 'lpips'} or not all(math.isfinite(v) for v in r['metrics'].values()):
            raise ValueError('nonfinite metrics')
    if any(set(v) != {'psnr', 'lpips'} or not all(math.isfinite(x) for x in v.values()) for v in baseline.values()):
        raise ValueError('invalid baseline metrics')
    means = {k: sum(r['metrics'][k] for r in rows) / len(rows) for k in ('psnr', 'lpips')}
    old = {k: sum(v[k] for v in baseline.values()) / len(rows) for k in means}
    return {'count': len(rows), 'means': means, 'step400_prefix_means': old,
            'delta_vs_step400_prefix': {k: means[k] - old[k] for k in means},
            'psnr_worse_count': sum(r['metrics']['psnr'] < baseline[r['index']]['psnr'] for r in rows),
            'lpips_worse_count': sum(r['metrics']['lpips'] > baseline[r['index']]['lpips'] for r in rows),
            'both_nonworse_count': sum(r['metrics']['psnr'] >= baseline[r['index']]['psnr']
                                      and r['metrics']['lpips'] <= baseline[r['index']]['lpips'] for r in rows),
            'joint_mean_target_pass': means['psnr'] >= 34.2 and means['lpips'] <= .08}


def milestone(output, bound, receipt, step):
    groups = {'training_fit_only': []}
    if step == 600:
        groups['development'] = []
    indices = sorted(i for rank in range(8) for i in logic.probe_indices(step, rank))
    for index in indices:
        row = c.read(output / 'probes' / f'step_{step:06d}_image_{index}.json')
        if (row['identity'] != image_identity(output, bound, receipt, index, step)
                or row['index'] != index or row['training_fit_only'] != (index < 19200)
                or row['GT_used_for_output_selection']):
            raise ValueError('probe identity/scope mismatch')
        if not c.path(output, row['tensor']).is_file():
            raise FileNotFoundError(row['tensor'])
        if step == 400 and row['exact_readout_control'] is not True:
            raise ValueError('missing baseline readout control')
        groups['training_fit_only' if index < 19200 else 'development'].append(row)
    baseline = {i: c.read(output / 'probes' / f'step_000400_image_{i}.json')['metrics']
                for i in range(12800, 12832)}
    summaries = {'training_fit_only': paired_summary(groups['training_fit_only'], baseline)}
    if step == 600:
        dev_baseline = {}
        for index in range(64032, 64132):
            row = c.read(output / 'probes' / f'step_000400_image_{index}.json')
            if (row['identity'] != image_identity(output, bound, receipt, index, 400)
                    or row['training_fit_only'] or row['GT_used_for_output_selection']
                    or not c.path(output, row['tensor']).is_file()):
                raise ValueError('invalid same-environment parent400 baseline')
            dev_baseline[index] = row['metrics']
        summaries['development'] = paired_summary(groups['development'], dev_baseline)
        historical = {r['index']: r['metrics']['fixed_prefix26'] for r in bound['baseline_per_image']}
        summaries['development']['historical_prefix400_means'] = {
            k: sum(v[k] for v in historical.values()) / 100 for k in ('psnr', 'lpips')}
        summaries['development']['matched_baseline_minus_historical'] = {
            k: summaries['development']['step400_prefix_means'][k]
            - summaries['development']['historical_prefix400_means'][k] for k in ('psnr', 'lpips')}
    result = {'step': step, 'new_updates': step - 400, 'groups': summaries,
              'receipt_sha256': c.sha(output / 'receipt.json'),
              'development_is_previously_used_not_blind_test': True}
    c.write_new(output / 'milestones' / f'step_{step:06d}.json', result)
    print('RAIN_ONPOLICY_MILESTONE ' + json.dumps(result, sort_keys=True), flush=True)
    return result


def train(output, bound, receipt, source, rank, local):
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from flux2.util import load_ae
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    from lowlight_restoration.rain_flow_adaptation import enable_native_checkpointing
    from lowlight_restoration.flux2_endpoint_precision import assert_fp32_optimizer
    device = torch.device('cuda', local)
    cfg = bound['config']
    receipt_hash = c.sha(output / 'receipt.json')
    model = parent.new_model(receipt, device)
    enable_native_checkpointing(model)
    ae = parent.hopper.configure_vae(load_ae('flux.2-klein-base-4b', device=device).eval().requires_grad_(False))
    prompt = parent.load_prompt(receipt, device)
    parameters = [p for p in model.parameters() if p.requires_grad]
    wrapped = DDP(model, device_ids=[local], broadcast_buffers=False, gradient_as_bucket_view=True)
    optimizer = torch.optim.AdamW(parameters, lr=cfg['learning_rate'], weight_decay=0., fused=False)
    metric = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).to(device).eval().requires_grad_(False)
    saved, _ = state_io.load_latest(output, receipt_hash)
    fresh = saved is None
    if fresh:
        saved = load_parent_state(bound, source)
    starts = [None] * 8
    dist.all_gather_object(starts, saved['step'])
    if len(set(starts)) != 1:
        raise ValueError('ranks saw different progress')
    state_io.base.restore(model, optimizer, saved, rank, device)
    start = saved['step']
    history = saved['history']
    del saved
    print(f'RAIN_ONPOLICY_RESUMED rank={rank} global_step={start} next_global_slot={start * 32} objective_changed=true', flush=True)

    def save(step):
        rngs = [None] * 8
        dist.all_gather_object(rngs, state_io.base.rng_state(device))
        if step in cfg['train_probe_steps']:
            hashes = [None] * 8
            dist.all_gather_object(hashes, state_io.base.parameter_hash(state_io.base.trainable_state(model)))
            if len(set(hashes)) != 1:
                raise ValueError('rank parameters differ')
        if rank == 0:
            state_io.save_step(output, model, optimizer, step=step, world=8, rank_rng=rngs,
                               receipt_hash=receipt_hash, history=history)
            print(f'RAIN_ONPOLICY_CHECKPOINT global_step={step} new_updates={step - 400}', flush=True)
        dist.barrier()

    def evaluate(step):
        gc.collect()
        torch.cuda.empty_cache()
        if step == 600:
            # Same-environment matched baseline, only AFTER all 200 updates. Keep
            # historical metrics too. No GT selection, no extra optimizer update.
            current = state_io.base.trainable_state(model)
            initial = load_parent_state(bound, source)
            values = {n: p for n, p in model.named_parameters() if p.requires_grad}
            if list(values) != initial['parameter_order']:
                raise ValueError('parent parameter order changed')
            with torch.no_grad():
                for name, param in values.items():
                    param.copy_(initial['model'][name].to(param.device))
            del initial
            try:
                probe(output, bound, receipt, source, model, ae, prompt, metric, rank, device, 400, baseline_only=True)
            finally:
                with torch.no_grad():
                    for name, param in values.items():
                        param.copy_(current[name].to(param.device))
            if state_io.base.parameter_hash(state_io.base.trainable_state(model)) != state_io.base.parameter_hash(current):
                raise ValueError('failed to restore step600 after frozen baseline evaluation')
            del current
        probe(output, bound, receipt, source, model, ae, prompt, metric, rank, device, step)
        dist.barrier()
        if rank == 0:
            milestone(output, bound, receipt, step)
        dist.barrier()

    if fresh:
        save(400)
    if start in cfg['train_probe_steps']:
        evaluate(start)  # Also finishes interrupted per-image evaluation before advancing.
    for step in range(start, cfg['end_step']):
        begun = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        totals = {'total': 0., 'gt_rgb_mse': 0., 'gt_lpips': 0.}
        prefix_seconds = 0.
        for micro in range(4):
            position = logic.training_position(step, micro, rank)
            batch = cached(bound, receipt, source, position, device, training=True)
            seed = receipt['config']['seed'] + 1000003 * (position + 1)
            model.eval()
            torch.cuda.synchronize(device)
            prefix_start = time.monotonic()
            inputs = logic.generated_inputs(model, ae, batch['lq'], prompt, seed)
            torch.cuda.synchronize(device)
            prefix_seconds += time.monotonic() - prefix_start
            model.train()
            with parent.hopper.offload_activations(), (wrapped.no_sync() if micro < 3 else nullcontext()):
                loss, terms = logic.objective(wrapped, ae, inputs, batch['gt'], metric, cfg['loss_weights'])
                if not torch.isfinite(loss):
                    raise ValueError('nonfinite training objective')
                (loss / 4).backward()
            metric.reset()
            for key, value in {'total': loss, **terms}.items():
                totals[key] += float(value.detach()) / 4
            del batch, inputs, loss, terms
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
        if not any(p.grad is not None and bool(p.grad.abs().sum() > 0) for p in parameters):
            raise ValueError('missing differentiable FLUX gradient')
        optimizer.step()
        assert_fp32_optimizer(optimizer)
        optimizer.zero_grad(set_to_none=True)
        if any(not torch.isfinite(p).all() for p in parameters):
            raise ValueError('nonfinite LoRA')
        vector = torch.tensor(list(totals.values()), dtype=torch.float64, device=device)
        dist.all_reduce(vector)
        vector /= 8
        stats = torch.tensor([float(norm), time.monotonic() - begun, prefix_seconds,
                              torch.cuda.max_memory_allocated(device) / 1024**3], device=device)
        dist.all_reduce(stats, op=dist.ReduceOp.MAX)
        log = {'step': step + 1, 'new_updates': step + 1 - 400,
               'objective': 'current_policy_prefix25_detached_last_call_GT_RGB_MSE_LPIPS',
               'losses': dict(zip(totals, vector.tolist())), 'max_rank_grad_norm': float(stats[0]),
               'max_rank_step_seconds_excluding_checkpoint': float(stats[1]),
               'max_rank_prefix_seconds': float(stats[2]), 'peak_allocated_gib': float(stats[3]),
               'next_global_slot': (step + 1) * 32, 'training_calls_per_presentation': 26}
        history.append(log)
        if rank == 0:
            print('RAIN_ONPOLICY_TRAIN ' + json.dumps(log, sort_keys=True), flush=True)
        save_start = time.monotonic()
        save(step + 1)
        if rank == 0:
            print(f'RAIN_ONPOLICY_SAVE_SECONDS step={step + 1} seconds={time.monotonic() - save_start:.3f}', flush=True)
        if step + 1 in cfg['train_probe_steps']:
            evaluate(step + 1)
    if rank == 0:
        final = state_io.base.trainable_state(model)
        marker = output / 'final_lora.json'
        if marker.exists():
            entry = c.read(marker)
            if entry['receipt_hash'] != receipt_hash or entry['parameter_hash'] != state_io.base.parameter_hash(final):
                raise ValueError('existing export differs')
            for name, digest in entry['files'].items():
                if c.sha(c.path(output, name)) != digest:
                    raise ValueError('corrupt export')
        else:
            name, digest = parent.replay.save_tensors(output, 'rain_onpolicy_midpoint_step600', final)
            c.write_new(marker, {'step': 600, 'tensor': name, 'files': {name: digest},
                                'receipt_hash': receipt_hash, 'parameter_hash': state_io.base.parameter_hash(final)})
    dist.barrier()


def summarize(output, bound, receipt):
    parent.check_runtimes(output)
    saved, record = state_io.load_latest(output, c.sha(output / 'receipt.json'))
    if saved is None or saved['step'] != 600:
        raise ValueError('200 new updates not complete')
    summaries = {str(s): milestone(output, bound, receipt, s) for s in bound['config']['train_probe_steps']}
    export = c.read(output / 'final_lora.json')
    if export['parameter_hash'] != saved['parameter_hash']:
        raise ValueError('final export differs from saved state')
    for name, digest in export['files'].items():
        if c.sha(c.path(output, name)) != digest:
            raise ValueError('final export corrupt')
    result = {'status': 'rain_onpolicy_midpoint_200_updates_complete', 'last_global_step': 600,
              'new_updates': 200, 'new_presentations': 6400, 'parent_step': 400,
              'receipt_sha256': c.sha(output / 'receipt.json'), 'latest_checkpoint': record,
              'final_lora': export, 'milestones': summaries, 'history_new_objective': saved['history'][400:],
              'development_gate_pass': summaries['600']['groups']['development']['joint_mean_target_pass'],
              'full_tensor_audit_performed': False, 'lpips_reexecuted_in_summary': False,
              'official_or_sealed_test_accessed': False, 'automatic_follow_on': False,
              'not_a_MoE_compatibility_qualification': True, 'prefix_derivatives_omitted': True}
    c.write_new(output / 'result.json', result)
    print('RAIN_ONPOLICY_RESULT_VERIFIED ' + json.dumps({k: v for k, v in result.items()
          if k not in ('history_new_objective', 'milestones')}, sort_keys=True), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('phase', choices=['preflight', 'prepare', 'hardware', 'runtime', 'run', 'summarize', 'publish', 'wait'])
    ap.add_argument('--event', choices=['boot', 'prepared', 'ready', 'done', 'verified', 'failed'])
    ap.add_argument('--run-id', default='step400_onpolicy200_4x2_v1')
    ap.add_argument('--launch-id')
    ap.add_argument('--node-rank', type=int, choices=range(4), default=0)
    args = ap.parse_args()
    cfg = config()
    output = output_path(args.run_id)
    if str(ROOT) != cfg['active_root']:
        raise ValueError('execute only in migrated GPU workspace')
    if args.phase in ('publish', 'wait'):
        if not args.launch_id or not args.event:
            raise ValueError('missing launch/event')
        if args.phase == 'publish':
            cluster.publish(output, args.launch_id, args.node_rank, args.event)
        else:
            deadline = time.monotonic() + cfg['wall_cap_seconds']
            while not cluster.check(output, args.launch_id, args.event):
                if time.monotonic() > deadline:
                    raise TimeoutError('four-node barrier timed out')
                time.sleep(2)
        return
    if args.phase in ('preflight', 'prepare'):
        import shutil
        if args.phase == 'prepare' and args.node_rank != 0:
            raise ValueError('node0 only')
        committed = len(list((output / 'steps').glob('step_*.json')))
        needed = 100 if committed == 0 else 10 + max(0, 201 - committed) * .4
        if shutil.disk_usage(ROOT).free < needed * 1024**3:
            raise ValueError(f'need {needed:.1f}GiB free; never delete history automatically')
        bound = make_receipt()
        if args.phase == 'prepare':
            c.write_new(output / 'receipt.json', bound)
        print('RAIN_ONPOLICY_PREFLIGHT_PASS pixels_read=false GPU_used=false cache_regeneration=false', flush=True)
        return
    if args.phase == 'hardware':
        print(json.dumps(parent.hopper.hardware(True)))
        return
    bound, receipt, source = load_receipt(output)
    if args.phase == 'runtime':
        prior_eval.gap.runtime(output, receipt, args.node_rank, True)
        return
    if args.phase == 'summarize':
        if args.node_rank != 0 or not cluster.check(output, args.launch_id, 'done'):
            raise ValueError('four-node done required')
        summarize(output, bound, receipt)
        return
    import torch
    import torch.distributed as dist
    rank, local = c.worker_rank(args.node_rank, int(os.environ['LOCAL_RANK']), int(os.environ['WORLD_SIZE']),
                                int(os.environ['LOCAL_WORLD_SIZE']), int(os.environ['RANK']))
    if int(os.environ['REPLAY_NODE_RANK']) != args.node_rank:
        raise ValueError('node rank mismatch')
    prior_eval.gap.runtime(output, receipt, args.node_rank, False)
    parent.check_runtimes(output)
    torch.cuda.set_device(local)
    device = torch.device('cuda', local)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    os.environ.update(KLEIN_4B_BASE_MODEL_PATH=str(c.path(ROOT, receipt['model']['flow_weights'])),
                      AE_MODEL_PATH=str(c.path(ROOT, receipt['model']['autoencoder_weights'])),
                      TORCH_HOME=str(ROOT / 'models/torch'), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    dist.init_process_group('nccl', device_id=device, timeout=timedelta(minutes=30))
    try:
        train(output, bound, receipt, source, rank, local)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
