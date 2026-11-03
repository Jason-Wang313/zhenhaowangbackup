#!/usr/bin/env python3
"""Same on-policy objective, full step600 continuation to at most global800."""
import argparse
from contextlib import contextmanager, nullcontext
from datetime import timedelta
import gc
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'third_party/flux2/src'))
from smoke import train_foundir_rain_onpolicy_midpoint_v1 as previous
from smoke import rain_onpolicy_continue_state_v1 as state_io

c = previous.c
cluster = previous.cluster
parent = previous.parent
logic = previous.logic  # The GPU-qualified prefix, prediction and loss are unchanged.
CONFIG = 'configs/foundir_rain_onpolicy_continue_v1.json'
SOURCES = [CONFIG, 'smoke/rain_onpolicy_continue_state_v1.py',
           'smoke/train_foundir_rain_onpolicy_continue_v1.py',
           'smoke/run_foundir_rain_onpolicy_continue_v1_gpu.sh',
           'smoke/start_foundir_rain_onpolicy_continue_v1_tmux.sh', *previous.SOURCES]


def config():
    cfg = c.read(ROOT / CONFIG)
    old = previous.config()
    unchanged = ['active_root', 'global_batch', 'learning_rate', 'loss_weights', 'schedule_steps',
                 'prefix_calls_without_grad', 'readout_call_index', 'training_probe_slots',
                 'development_slots', 'nodes', 'local_gpus', 'world_size', 'gpu_family',
                 'minimum_memory_gib', 'wall_cap_seconds', 'minimum_free_disk_gib',
                 'psnr_minimum', 'lpips_maximum', 'official_or_sealed_pixels_allowed', 'automatic_follow_on']
    if any(cfg.get(k) != old[k] for k in unchanged):
        raise ValueError('continuation must keep the qualified training recipe')
    expected = dict(start_step=600, end_step=800, new_updates=200, training_slots=[19200, 25600],
                    train_probe_steps=[700, 800], development_steps=[700, 800])
    if any(cfg.get(k) != v for k, v in expected.items()):
        raise ValueError('authorized continuation scope changed')
    return cfg


def output_path(run_id):
    c.run_path(ROOT, run_id)
    return c.path(ROOT, 'checkpoints/foundir_rain_onpolicy_continue_v1/' + run_id)


def training_position(step, micro, rank):
    if step not in range(600, 800) or micro not in range(4) or rank not in range(8):
        raise ValueError('outside authorized 200 updates / global32')
    return step * 32 + micro * 8 + rank


def probe_indices(step, rank):
    if step not in (600, 700, 800) or rank not in range(8):
        raise ValueError('invalid probe checkpoint or rank')
    return [*range(12800 + rank, 12832, 8), *range(64032 + rank, 64132, 8)]


def make_receipt():
    cfg = config()
    if c.sha(ROOT / cfg['parent_report']) != cfg['parent_report_sha256']:
        raise ValueError('published terminal report changed')
    report = c.read(ROOT / cfg['parent_report'])
    if (report['last_global_step'] != 600 or report['new_updates'] != 200
            or len(report['nodes']) != 4 or any(n['status'] != 0 for n in report['nodes'])
            or report['result_sha256'] != cfg['parent_result_sha256']
            or report['latest_checkpoint']['sha256'] != cfg['parent_checkpoint_sha256']):
        raise ValueError('step600 parent not verified complete')
    source = c.path(ROOT, cfg['parent_run'])
    if (c.sha(source / 'receipt.json') != cfg['parent_receipt_sha256']
            or c.sha(source / 'result.json') != cfg['parent_result_sha256']):
        raise ValueError('wrong parent run/result')
    prior, receipt, cache_source = previous.load_receipt(source)
    previous.readout.validate_rows(receipt['rows'])
    marker = source / 'steps/step_000600.json'
    record = c.read(marker)
    if record != report['latest_checkpoint'] or not c.path(source, record['file']).is_file():
        raise ValueError('wrong or missing step600 state')
    for row in receipt['rows'][19200:25600]:
        if row['usage'] != 'train_update' or row['split'] != 'train' or row['category'] != '10Rain':
            raise ValueError('nontraining data in continuation')
    indices = [*range(12800, 12832), *range(19200, 25600), *range(64032, 64132)]
    cache_receipts = {str(i): c.sha(parent.cache_marker(cache_source, i)) for i in indices}
    return {'config': cfg, 'checkpoint': record, 'parent_receipt_sha256': cfg['parent_receipt_sha256'],
            'parent_checkpoint_receipt_sha256': c.sha(marker),
            'cache_source': prior['config']['parent_run'], 'cache_receipts': cache_receipts,
            'selection_digest': c.digest([receipt['rows'][i] for i in indices]),
            'sources': {**prior['sources'], **{n: c.sha(ROOT / n) for n in SOURCES},
                        cfg['parent_report']: cfg['parent_report_sha256']},
            'historical_step600_groups': report['milestones']['600']['groups'],
            'parent_optimizer_restored': True, 'objective_changed': False,
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
    prior, receipt, cache_source = previous.load_receipt(source)
    if cache_source != c.path(ROOT, bound['cache_source']):
        raise ValueError('cache source changed')
    indices = [*range(12800, 12832), *range(19200, 25600), *range(64032, 64132)]
    if c.digest([receipt['rows'][i] for i in indices]) != bound['selection_digest']:
        raise ValueError('sampler or evaluation changed')
    return bound, receipt, source


def load_parent_state(bound, source):
    import torch
    if c.sha(source / 'steps/step_000600.json') != bound['parent_checkpoint_receipt_sha256']:
        raise ValueError('parent step600 receipt changed')
    record = bound['checkpoint']
    file = c.path(source, record['file'])
    if c.sha(file) != bound['config']['parent_checkpoint_sha256']:
        raise ValueError('parent step600 tensor changed')
    saved = torch.load(file, map_location='cpu', weights_only=True)
    state_io.validate_state(saved, record, bound['parent_receipt_sha256'])
    if saved['step'] != 600:
        raise ValueError('must restore full step600, not fresh optimizer')
    return saved


def cached(bound, receipt, source, index, device, *, training=False):
    if training and index not in range(19200, 25600):
        raise ValueError('evaluation or old presentation cannot train')
    # Reuse unchanged reader/validation; apply the NEW training range above.
    return previous.cached(bound, receipt, c.path(ROOT, bound['cache_source']), index, device, training=False)


image_identity = previous.image_identity


@contextmanager
def temporary_weights(model, values):
    """Change only LoRA values for frozen baseline evaluation; never touch Adam/RNG."""
    import torch
    current = state_io.base.trainable_state(model)
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    if set(params) != set(values) or any(params[n].shape != v.shape or params[n].dtype != v.dtype
                                        for n, v in values.items()):
        raise ValueError('baseline parameter schema changed')
    try:
        with torch.no_grad():
            for name, p in params.items():
                p.copy_(values[name].to(p.device))
        yield
    finally:
        with torch.no_grad():
            for name, p in params.items():
                p.copy_(current[name].to(p.device))
    if state_io.base.parameter_hash(state_io.base.trainable_state(model)) != state_io.base.parameter_hash(current):
        raise ValueError('failed to restore trained weights')


def probe(output, bound, receipt, source, model, ae, prompt, metric, rank, device, step):
    import torch
    rng = state_io.base.rng_state(device)
    model.eval()
    try:
        for index in probe_indices(step, rank):
            marker = output / 'probes' / f'step_{step:06d}_image_{index}.json'
            ident = image_identity(output, bound, receipt, index, step)
            if marker.exists():
                c.validate_record(c.read(marker), ident, output)
                continue
            data = cached(bound, receipt, source, index, device)
            seed = (receipt['config']['probe_seed'] + index - 64000 if index >= 64032
                    else receipt['config']['seed'] + 1000003 * (index + 1))
            rgb = previous.inference(model, ae, data['lq'], prompt, seed)
            if not torch.isfinite(rgb).all():
                raise ValueError('nonfinite probe')
            image = parent.quantize(rgb)
            metrics = parent.scores(metric, image, data['gt'])
            name, digest = parent.replay.save_tensors(output, f'probe_{step}_{index}', {'uint8': image})
            c.write_new(marker, {'identity': ident, 'index': index, 'metrics': metrics,
                                'tensor': name, 'files': {name: digest}, 'training_fit_only': index < 19200,
                                'GT_used_for_output_selection': False})
            print('RAIN_ONPOLICY_CONTINUE_PROBE ' + json.dumps({'step': step, 'index': index, 'metrics': metrics}), flush=True)
    finally:
        state_io.base.restore_rng(rng, device)


def milestone(output, bound, receipt, step):
    if step not in (700, 800):
        raise ValueError('only the authorized two checkpoints')
    groups = {'training_fit_only': [], 'development': []}
    baselines = {'training_fit_only': {}, 'development': {}}
    for index in sorted(i for rank in range(8) for i in probe_indices(step, rank)):
        key = 'training_fit_only' if index < 19200 else 'development'
        for value in (600, step):
            row = c.read(output / 'probes' / f'step_{value:06d}_image_{index}.json')
            if (row['identity'] != image_identity(output, bound, receipt, index, value)
                    or row['index'] != index or row['training_fit_only'] != (index < 19200)
                    or row['GT_used_for_output_selection'] or not c.path(output, row['tensor']).is_file()):
                raise ValueError('invalid probe identity/scope')
            if value == 600:
                baselines[key][index] = row['metrics']
            else:
                groups[key].append(row)
    summaries = {}
    for key, rows in groups.items():
        summary = previous.paired_summary(rows, baselines[key])
        # The reusable arithmetic helper labels its baseline400; relabel honestly.
        summary['step600_prefix_means'] = summary.pop('step400_prefix_means')
        summary['delta_vs_step600_prefix'] = summary.pop('delta_vs_step400_prefix')
        history = bound['historical_step600_groups'][key]['means']
        summary['historical_step600_means'] = history
        summary['matched_baseline_minus_historical'] = {k: summary['step600_prefix_means'][k] - history[k]
                                                       for k in ('psnr', 'lpips')}
        summaries[key] = summary
    result = {'step': step, 'new_updates': step - 600, 'total_onpolicy_updates': step - 400,
              'groups': summaries, 'receipt_sha256': c.sha(output / 'receipt.json'),
              'development_is_previously_used_not_blind_test': True}
    c.write_new(output / 'milestones' / f'step_{step:06d}.json', result)
    print('RAIN_ONPOLICY_CONTINUE_MILESTONE ' + json.dumps(result, sort_keys=True), flush=True)
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
    print(f'RAIN_ONPOLICY_CONTINUE_RESUMED rank={rank} global_step={start} next_global_slot={start * 32} objective_changed=false', flush=True)

    def save(step):
        rngs = [None] * 8
        dist.all_gather_object(rngs, state_io.base.rng_state(device))
        if step == 600 or step in cfg['train_probe_steps']:
            hashes = [None] * 8
            dist.all_gather_object(hashes, state_io.base.parameter_hash(state_io.base.trainable_state(model)))
            if len(set(hashes)) != 1:
                raise ValueError('rank parameters differ')
        if rank == 0:
            state_io.save_step(output, model, optimizer, step=step, world=8, rank_rng=rngs,
                               receipt_hash=receipt_hash, history=history)
            print(f'RAIN_ONPOLICY_CONTINUE_CHECKPOINT global_step={step} new_updates={step - 600}', flush=True)
        dist.barrier()

    def evaluate(step):
        gc.collect()
        torch.cuda.empty_cache()
        # Create same-environment frozen step600 controls once; resume validates
        # and skips existing images. Restore trained values even on exceptions.
        initial = load_parent_state(bound, source)
        with temporary_weights(model, initial['model']):
            probe(output, bound, receipt, source, model, ae, prompt, metric, rank, device, 600)
        del initial
        probe(output, bound, receipt, source, model, ae, prompt, metric, rank, device, step)
        dist.barrier()
        if rank == 0:
            milestone(output, bound, receipt, step)
        dist.barrier()

    if fresh:
        save(600)
    if start in cfg['train_probe_steps']:
        evaluate(start)  # Also finishes interrupted per-image evaluation before advancing.
    for step in range(start, cfg['end_step']):
        begun = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        totals = {'total': 0., 'gt_rgb_mse': 0., 'gt_lpips': 0.}
        prefix_seconds = 0.
        for micro in range(4):
            position = training_position(step, micro, rank)
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
        log = {'step': step + 1, 'new_updates': step + 1 - 600,
               'objective': 'current_policy_prefix25_detached_last_call_GT_RGB_MSE_LPIPS',
               'losses': dict(zip(totals, vector.tolist())), 'max_rank_grad_norm': float(stats[0]),
               'max_rank_step_seconds_excluding_checkpoint': float(stats[1]),
               'max_rank_prefix_seconds': float(stats[2]), 'peak_allocated_gib': float(stats[3]),
               'next_global_slot': (step + 1) * 32, 'training_calls_per_presentation': 26}
        history.append(log)
        if rank == 0:
            print('RAIN_ONPOLICY_CONTINUE_TRAIN ' + json.dumps(log, sort_keys=True), flush=True)
        save_start = time.monotonic()
        save(step + 1)
        if rank == 0:
            print(f'RAIN_ONPOLICY_CONTINUE_SAVE_SECONDS step={step + 1} seconds={time.monotonic() - save_start:.3f}', flush=True)
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
            name, digest = parent.replay.save_tensors(output, 'rain_onpolicy_continue_step800', final)
            c.write_new(marker, {'step': 800, 'tensor': name, 'files': {name: digest},
                                'receipt_hash': receipt_hash, 'parameter_hash': state_io.base.parameter_hash(final)})
    dist.barrier()



def summarize(output, bound, receipt):
    parent.check_runtimes(output)
    saved, record = state_io.load_latest(output, c.sha(output / 'receipt.json'))
    if saved is None or saved['step'] != 800:
        raise ValueError('200 new updates not complete')
    summaries = {str(s): milestone(output, bound, receipt, s) for s in bound['config']['train_probe_steps']}
    export = c.read(output / 'final_lora.json')
    if export['parameter_hash'] != saved['parameter_hash']:
        raise ValueError('final export differs from saved state')
    for name, digest in export['files'].items():
        if c.sha(c.path(output, name)) != digest:
            raise ValueError('final export corrupt')
    result = {'status': 'rain_onpolicy_continuation_complete', 'last_global_step': 800,
              'new_updates': 200, 'new_presentations': 6400, 'parent_step': 600,
              'receipt_sha256': c.sha(output / 'receipt.json'), 'latest_checkpoint': record,
              'final_lora': export, 'milestones': summaries, 'history_continuation': saved['history'][600:],
              'development_gate_pass': summaries['800']['groups']['development']['joint_mean_target_pass'],
              'qualifying_checkpoint_steps': [s for s in (700, 800)
                  if summaries[str(s)]['groups']['development']['joint_mean_target_pass']],
              'full_tensor_audit_performed': False, 'lpips_reexecuted_in_summary': False,
              'official_or_sealed_test_accessed': False, 'automatic_follow_on': False,
              'not_a_MoE_compatibility_qualification': True, 'prefix_derivatives_omitted': True}
    c.write_new(output / 'result.json', result)
    print('RAIN_ONPOLICY_CONTINUE_RESULT_VERIFIED ' + json.dumps({k: v for k, v in result.items()
          if k not in ('history_continuation', 'milestones')}, sort_keys=True), flush=True)



def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('phase', choices=['preflight', 'prepare', 'hardware', 'runtime', 'run', 'summarize', 'publish', 'wait'])
    ap.add_argument('--event', choices=['boot', 'prepared', 'ready', 'done', 'verified', 'failed'])
    ap.add_argument('--run-id', default='continue600_to800_4x2_v1')
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
        print('RAIN_ONPOLICY_CONTINUE_PREFLIGHT_PASS pixels_read=false GPU_used=false cache_regeneration=false', flush=True)
        return
    if args.phase == 'hardware':
        print(json.dumps(parent.hopper.hardware(True)))
        return
    bound, receipt, source = load_receipt(output)
    if args.phase == 'runtime':
        previous.prior_eval.gap.runtime(output, receipt, args.node_rank, True)
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
    previous.prior_eval.gap.runtime(output, receipt, args.node_rank, False)
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
