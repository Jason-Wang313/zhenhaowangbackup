#!/usr/bin/env python3
"""Frozen five-expert integration, not router training or official evaluation."""
import argparse
import gc
import json
import math
import os
from pathlib import Path
import sys
import time
from functools import lru_cache

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'third_party/flux2/src'))
from smoke import train_foundir_rain_onpolicy_continue_v1 as rain
from smoke import existing_flux_common_protocol_contract_v1 as four_gt
from smoke import diagnose_foundir_existing_flux_common_protocol_v1 as four_impl

c, parent, cluster = rain.c, rain.parent, rain.cluster
CONFIG = 'configs/foundir_five_expert_integration_v1.json'
ORDER = ('blur', 'noise', 'haze', 'lowlight', 'rain')
SOURCES = [CONFIG, 'lowlight_restoration/five_expert_flux.py',
           'lowlight_restoration/protocol_routed_flux.py',
           'smoke/qualify_foundir_five_expert_integration_v1.py',
           'smoke/run_foundir_five_expert_integration_v1_gpu.sh',
           'smoke/start_foundir_five_expert_integration_v1_tmux.sh',
           'configs/foundir_rain_flux2_source_diverse_pilot_v1.json',
           'smoke/qualify_foundir_protocol_routed_flux_v1.py',
           'smoke/protocol_routed_flux_contract_v1.py',
           'smoke/audit_foundir_existing_flux_experts_v1.py',
           'smoke/existing_flux_common_protocol_contract_v1.py',
           'smoke/diagnose_foundir_existing_flux_common_protocol_v1.py']


def config():
    cfg = c.read(ROOT / CONFIG)
    expected = dict(expert_order=list(ORDER), four_records_per_type=2,
                    rain_training_slots=[12800, 12808], rain_development_slots=[64032, 64132],
                    arms=['original', 'one_hot', 'near_one_hot'], near_one_hot_primary_weight=.9,
                    near_one_hot_other_weight=.025, soft_on_training_only=True, schedule_steps=50,
                    rain_model_calls=26, late_rain_policy='zero_and_renormalize_nonrain',
                    nodes=4, local_gpus=2, world_size=8, minimum_memory_gib=75,
                    wall_cap_seconds=3600, training=False, learned_router=False,
                    official_pixels_in_this_qualification=False, automatic_training=False)
    if any(cfg.get(k) != v for k, v in expected.items()):
        raise ValueError('qualification scope changed')
    return cfg


def output_path(run_id):
    c.run_path(ROOT, run_id)
    return c.path(ROOT, 'checkpoints/foundir_five_expert_integration_v1/' + run_id)


def items_for_rank(bound, rank):
    if rank not in range(8):
        raise ValueError('invalid rank')
    return bound['items'][rank::8]


def arms(item):
    return ['original', 'one_hot', 'near_one_hot'] if item['training_fit_only'] else ['original', 'one_hot']


def rain_prompt_text():
    # The inherited endpoint-era model dictionary stores a cache path, NOT text.
    # Native50 contract already checked this exact prompt against that cache lineage.
    return c.read(ROOT / 'configs/foundir_rain_flux2_source_diverse_pilot_v1.json')['model']['prompt']


def make_receipt():
    cfg = config()
    four_root, rain_root = (c.path(ROOT, cfg[k]) for k in ('four_parent_run', 'rain_parent_run'))
    four = c.load_receipt(ROOT, four_root, 8)
    prompt_info = parent.replay.prompt_info(four_root, four)
    rb, rr, source = rain.load_receipt(rain_root)
    rain.previous.readout.validate_rows(rr['rows'])
    report = c.read(ROOT / cfg['rain_report'])
    if (report['last_global_step'] != 800 or not report['development_gate_pass']
            or len(report['nodes']) != 4 or any(n['status'] != 0 for n in report['nodes'])
            or report['result_sha256'] != cfg['rain_result_sha256']
            or c.sha(rain_root / 'result.json') != cfg['rain_result_sha256']
            or c.sha(rain_root / 'receipt.json') != report['receipt_sha256']):
        raise ValueError('Rain parent not the completed passing step800')
    export = report['final_lora']
    file = c.path(rain_root, export['tensor'])
    if (export['step'] != 800 or export['files'] != {export['tensor']: cfg['rain_lora_sha256']}
            or c.sha(file) != cfg['rain_lora_sha256']):
        raise ValueError('wrong Rain export')
    rows = {t: four['rows'][t][:2] for t in ORDER[:4]}
    manifest = next(n for n in four['assets'] if n.endswith('foundir_five_type_pairs_v1.csv'))
    gt = four_gt.select_gt(c.path(ROOT, manifest), rows)
    experts = dict(four['experts'])
    experts['rain'] = {'checkpoint': {'path': file.relative_to(ROOT).as_posix(), 'sha256': cfg['rain_lora_sha256']},
                       'prompt': rain_prompt_text(), 'inference': {'frozen_alpha': 1.},
                       'contract': 'onpolicy_prefix25_endpoint26_not_full50'}
    for t in ORDER[:4]:
        ck = experts[t]['checkpoint']
        if c.sha(c.path(ROOT, ck['path'])) != ck['sha256']:
            raise ValueError('inherited expert changed: ' + t)
    # The prior receipts bind shared model/VAE hashes; check agreement, not a new whole-dataset audit.
    for key in ('flow_weights', 'autoencoder_weights'):
        if four['assets'].get(rr['model'][key]) != rr['model'][key + '_sha256']:
            raise ValueError('Rain and other experts do not share the same backbone')
    items = []
    for t in ORDER[:4]:
        for row in rows[t]:
            key = t + '_' + row['record_id']
            old = four_root / 'original' / (key + '.json')
            items.append({'key': key, 'task': t, 'row': row, 'seed': row['inference_seed'],
                          'training_fit_only': True, 'history_marker': old.relative_to(ROOT).as_posix(),
                          'history_sha256': c.sha(old)})
    for index in [*range(*cfg['rain_training_slots']), *range(*cfg['rain_development_slots'])]:
        row = rr['rows'][index]
        training = index < 64032
        if row['category'] != '10Rain' or (training and (row['split'] != 'train' or row['usage'] != 'train_update')):
            raise ValueError('Rain selection changed')
        seed = rr['config']['seed'] + 1000003 * (index + 1) if training else rr['config']['probe_seed'] + index - 64000
        old = rain_root / 'probes' / f'step_000800_image_{index}.json'
        record = c.read(old)
        if record['identity'] != rain.image_identity(rain_root, rb, rr, index, 800):
            raise ValueError('Rain historical probe identity changed')
        items.append({'key': f'rain_{index}', 'task': 'rain', 'index': index, 'row': row, 'seed': seed,
                      'training_fit_only': training, 'history_marker': old.relative_to(ROOT).as_posix(),
                      'history_sha256': c.sha(old)})
    if len(items) != 116 or len({i['key'] for i in items}) != 116:
        raise ValueError('expected 16 training + 100 development items')
    return {'config': cfg, 'items': items, 'experts': experts, 'gt': gt, 'metric': rr['metric'],
            'four_receipt_sha256': c.sha(four_root / 'receipt.json'),
            'rain_receipt_sha256': c.sha(rain_root / 'receipt.json'),
            'rain_prompt': rr['rain']['prompt_context_cache'], 'four_prompts': prompt_info,
            'model': rr['model'], 'memory_policy': rr['config']['memory_policy'],
            'sources': {**four['sources'], **rb['sources'],
                        **{n: c.sha(ROOT / n) for n in SOURCES + [cfg['rain_report']]}},
            'official_pixels_read': False, 'learned_router': False}


def load_receipt(output):
    b = c.read(output / 'receipt.json')
    if b['config'] != config():
        raise ValueError('bound config changed')
    for name, digest in b['sources'].items():
        if c.sha(ROOT / name) != digest:
            raise ValueError('bound source changed: ' + name)
    for which in ('four', 'rain'):
        if c.sha(c.path(ROOT, b['config'][which + '_parent_run']) / 'receipt.json') != b[which + '_receipt_sha256']:
            raise ValueError('parent receipt changed')
    return b


def fixture(item, arm, device):
    import torch
    if arm not in arms(item):
        raise ValueError('soft route is training-only')
    w = torch.zeros(1, 5, device=device)
    if arm == 'near_one_hot':
        w.fill_(.025)
    w[0, ORDER.index(item['task'])] = .9 if arm == 'near_one_hot' else 1.
    return w


def protocols(bound):
    from lowlight_restoration.protocol_routed_flux import ExpertProtocol, ProtocolRouter
    return ProtocolRouter([ExpertProtocol(t, bound['experts'][t]['prompt'],
        bound['experts'][t]['inference'].get('frozen_alpha', 1.), t not in ('haze', 'rain')) for t in ORDER])


def identity(output, item, arm, rank):
    return {'receipt_sha256': c.sha(output / 'receipt.json'), 'item': item, 'arm': arm, 'rank': rank,
            'runtime_sha256': c.sha(output / f'runtime_node{rank // 2}.json')}


def marker(output, item, arm):
    return output / arm / (item['key'] + '.json')


def saved(output, item, arm, rank):
    from safetensors.torch import load_file
    from smoke.qualify_foundir_protocol_routed_flux_v1 import check_tensors
    entry = c.read(marker(output, item, arm))
    c.validate_record(entry, identity(output, item, arm, rank), output)
    if entry['tensor'] not in entry['files']:
        raise ValueError('unbound output tensor')
    tensors = load_file(str(c.path(output, entry['tensor'])), device='cpu')
    check_tensors(tensors)
    if arm != 'original':
        from lowlight_restoration.five_expert_flux import plan
        bound = c.read(output / 'receipt.json')
        if entry['route'] != plan(protocols(bound), fixture(item, arm, 'cpu')):
            raise ValueError('saved route/protocol changed')
    return entry, tensors


@lru_cache(maxsize=1)
def rain_context(run):
    return rain.load_receipt(c.path(ROOT, run))


def data(bound, item, device, *, gt=False):
    if item['task'] != 'rain':
        return four_impl.load_gt(bound, item['row'], device) if gt else parent.replay.lq_tensor(item['row'], 512, device)
    # Loading a legacy cache may map its GT fields; only LQ is supplied to generation.
    # Never call an official-test reader, regenerate teacher, or use GT in routing.
    rb, rr, source = rain_context(bound['config']['rain_parent_run'])
    tensors = rain.cached(rb, rr, source, item['index'], device)
    return tensors['gt' if gt else 'lq']


def run(output, bound, rank, local):
    import torch
    from safetensors.torch import load_file
    from flux2.util import load_flow_model, load_ae
    from flux2.sampling import batched_prc_img
    from lowlight_restoration.lora import inject_lora, load_lora_checkpoint, LoRALinear
    from lowlight_restoration.flux2_endpoint_precision import promote_lora_to_fp32
    from lowlight_restoration.existing_flux_experts import FLOW_CONTRACT
    from lowlight_restoration.protocol_routed_flux import install_protocol_bank
    from lowlight_restoration.five_expert_flux import restore_five_expert
    from smoke.audit_foundir_existing_flux_experts_v1 import expected_modules
    from smoke.evaluate_foundir_blur_only_flux2_full_development_ddp import restore_batch
    from smoke.qualify_foundir_protocol_routed_flux_v1 import check_tensors, parity
    device = torch.device('cuda', local)
    items = items_for_rank(bound, rank)
    for item in items:
        for arm in arms(item):
            if marker(output, item, arm).exists():
                saved(output, item, arm, rank)
    if all(marker(output, i, a).exists() for i in items for a in arms(i)):
        return
    prompts = {}
    four_root = c.path(ROOT, bound['config']['four_parent_run'])
    for task in ORDER:
        spec = bound['rain_prompt'] if task == 'rain' else bound['four_prompts']['tensors'][task]
        root = ROOT if task == 'rain' else four_root
        file = c.path(root, spec['path'])
        if c.sha(file) != spec['sha256']:
            raise ValueError('prompt cache changed')
        prompts[task] = load_file(str(file), device=str(device))
    ae = parent.hopper.configure_vae(load_ae('flux.2-klein-base-4b', device=device).eval().requires_grad_(False))
    model = load_flow_model('flux.2-klein-base-4b', device=device).eval().requires_grad_(False)
    rules = protocols(bound)
    lora = c.read(ROOT / 'configs/foundir_blur_only_flux2_full_v1.json')['lora']
    def strip():
        for name in expected_modules():
            module = model.get_submodule(name)
            if not isinstance(module, LoRALinear):
                raise ValueError('unexpected original wrapper')
            owner, _, child = name.rpartition('.')
            setattr(model.get_submodule(owner), child, module.base)
    def commit(item, arm, raw, u8, route, begun):
        tensors = {'decoded': raw.cpu(), 'uint8': u8.cpu()}
        check_tensors(tensors)
        comparison = parity(saved(output, item, 'original', rank)[1], tensors) if arm == 'one_hot' else None
        name, digest = parent.replay.save_tensors(output, arm + '_' + item['key'], tensors)
        c.write_new(marker(output, item, arm), {'identity': identity(output, item, arm, rank),
            'tensor': name, 'files': {name: digest}, 'parity': comparison, 'route': route,
            'seconds': time.monotonic() - begun, 'peak_allocated_gib': torch.cuda.max_memory_allocated(device) / 1024**3})
        print('FIVE_EXPERT_IMAGE_SAVED ' + json.dumps({'rank': rank, 'key': item['key'], 'arm': arm, 'parity': comparison}), flush=True)
        if comparison is not None and not comparison['pass']:
            raise ValueError('single-selection parity failed; saved failure, no router training')
    for task in ORDER:
        pending = [i for i in items if i['task'] == task and not marker(output, i, 'original').exists()]
        if not pending:
            continue
        names = inject_lora(model, lora['target_patterns'], 32, 32., 0.)
        if set(names) != set(expected_modules()):
            raise ValueError('LoRA sites changed')
        if task == 'rain':
            promote_lora_to_fp32(model)
        spec = bound['experts'][task]['checkpoint']
        if c.sha(c.path(ROOT, spec['path'])) != spec['sha256']:
            raise ValueError('expert checkpoint changed')
        load_lora_checkpoint(model, c.path(ROOT, spec['path']))
        model.eval().requires_grad_(False)
        for item in pending:
            begun = time.monotonic(); lq = data(bound, item, device)
            if task == 'rain':
                raw = rain.previous.inference(model, ae, lq, prompts[task], item['seed'])
                u8 = parent.quantize(raw)
            else:
                prompt = prompts[task]
                raw = restore_batch(model, ae, lq, prompt['cfg_context'], prompt['cfg_ids'], batched_prc_img,
                                    seeds=[item['seed']], steps=50, reference_time=10)
                u8 = parent.replay.quantize(task, lq, raw, bound['experts'][task]['inference'].get('frozen_alpha', 1.))
            commit(item, 'original', raw, u8, {'independent_original_entry': True}, begun)
            del raw, u8, lq
        strip()
    # Tensor interface is shared. Rain's limited semantic domain is enforced by
    # the new trajectory adapter; passing FLOW_CONTRACT here is NOT full50 qualification.
    states = []
    for task in ORDER:
        spec = bound['experts'][task]['checkpoint']; file = c.path(ROOT, spec['path'])
        if c.sha(file) != spec['sha256']:
            raise ValueError('bank checkpoint changed')
        state = load_file(str(file), device='cpu')
        expected_dtype = torch.float32 if task == 'rain' else torch.bfloat16
        if any(v.dtype != expected_dtype for v in state.values()):
            raise ValueError('expert precision changed')
        states.append((task, state, 32., FLOW_CONTRACT))
    gates = install_protocol_bank(model, modules=expected_modules(), expert_states=states)
    del states
    model.eval().requires_grad_(False)
    for item in items:
        for arm in arms(item)[1:]:
            if marker(output, item, arm).exists():
                continue
            begun = time.monotonic(); lq = data(bound, item, device)
            raw, u8, route = restore_five_expert(model, gates, rules, fixture(item, arm, device), ae, lq, prompts, seed=item['seed'])
            commit(item, arm, raw, u8, route, begun)
            del raw, u8, lq


def score(output, bound, rank, local):
    import torch
    from safetensors.torch import load_file
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    device = torch.device('cuda', local)
    metric = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).to(device).eval().requires_grad_(False)
    for item in items_for_rank(bound, rank):
        target = output / 'metrics' / (item['key'] + '.json')
        binding = {a: c.sha(marker(output, item, a)) for a in arms(item)}
        ident = identity(output, item, 'metrics', rank)
        if target.exists():
            old = c.read(target)
            if old['identity'] != ident or old['binding'] != binding:
                raise ValueError('metric binding changed')
            continue
        images = {a: saved(output, item, a, rank)[1]['uint8'].to(device) for a in arms(item)}
        gt = data(bound, item, device, gt=True)
        values = {}
        with torch.inference_mode():
            for arm, image in images.items():
                if item['task'] == 'rain':
                    values[arm] = parent.scores(metric, image, gt)
                else:
                    mse = float((image.float() / 255. - gt).square().mean())
                    values[arm] = {'psnr': 100. if mse <= 1e-12 else -10 * math.log10(mse),
                                   'lpips': float(metric(image.float() / 255., gt))}
                    metric.reset()
        history_path = c.path(ROOT, item['history_marker'])
        if c.sha(history_path) != item['history_sha256']:
            raise ValueError('historical output record changed')
        history = c.read(history_path)
        hist_root = c.path(ROOT, bound['config']['rain_parent_run' if item['task'] == 'rain' else 'four_parent_run'])
        if history['tensor'] not in history['files'] or c.sha(c.path(hist_root, history['tensor'])) != history['files'][history['tensor']]:
            raise ValueError('historical output tensor changed')
        old_u8 = load_file(str(c.path(hist_root, history['tensor'])), device='cpu')['uint8']
        historical_equal = torch.equal(images['original'].cpu(), old_u8)
        c.write_new(target, {'identity': ident, 'binding': binding, 'metrics': values,
            'historical_uint8_equal': historical_equal, 'historical_metrics': history.get('metrics'),
            'training_fit_only': item['training_fit_only'], 'GT_used_for_route': False})


def summarize(output, bound):
    from smoke.qualify_foundir_protocol_routed_flux_v1 import parity
    records, comparisons = [], []
    for rank in range(8):
        for item in items_for_rank(bound, rank):
            tensors = {a: saved(output, item, a, rank)[1] for a in arms(item)}
            result = parity(tensors['original'], tensors['one_hot'])
            if not result['pass']:
                raise ValueError('independently recomputed single-selection parity failed')
            comparisons.append({'key': item['key'], **result})
            entry = c.read(output / 'metrics' / (item['key'] + '.json'))
            if (entry['identity'] != identity(output, item, 'metrics', rank)
                    or entry['binding'] != {a: c.sha(marker(output, item, a)) for a in arms(item)}
                    or entry['GT_used_for_route'] or set(entry['metrics']) != set(arms(item))
                    or entry['metrics']['original'] != entry['metrics']['one_hot']
                    or any(not math.isfinite(v) for m in entry['metrics'].values() for v in m.values())):
                raise ValueError('invalid metric identity/arithmetic')
            records.append(entry)
    expected = {i['key'] for i in bound['items']}
    for arm in ['original', 'one_hot', 'near_one_hot', 'metrics']:
        keys = expected if arm != 'near_one_hot' else {i['key'] for i in bound['items'] if i['training_fit_only']}
        if {p.stem for p in (output / arm).glob('*.json')} != keys:
            raise ValueError('missing/extra qualification records')
    groups = {}
    for task in ORDER:
        for training in (True, False):
            rows = [r for r in records if r['identity']['item']['task'] == task and r['training_fit_only'] == training]
            if not rows:
                continue
            means = {a: {m: sum(r['metrics'][a][m] for r in rows) / len(rows) for m in ('psnr', 'lpips')}
                     for a in rows[0]['metrics']}
            key = task + ('_training_fit_only' if training else '_development_used_before')
            groups[key] = {'count': len(rows), 'means': means,
                'historical_uint8_equal_count': sum(r['historical_uint8_equal'] for r in rows),
                'regressions_vs_original': {a: {m: sum((r['metrics'][a][m] < r['metrics']['original'][m]) if m == 'psnr'
                    else (r['metrics'][a][m] > r['metrics']['original'][m]) for r in rows) for m in ('psnr', 'lpips')}
                    for a in means if a != 'original'}}
            if task == 'rain' and not training:
                history = {m: sum(r['historical_metrics'][m] for r in rows) / len(rows) for m in ('psnr', 'lpips')}
                groups[key]['historical_step800_means'] = history
                groups[key]['current_original_minus_history'] = {m: means['original'][m] - history[m] for m in history}
    dev = groups['rain_development_used_before']
    rain_means = dev['means']['one_hot']
    gate = rain_means['psnr'] >= 34.2 and rain_means['lpips'] <= .08
    reproduced = dev['historical_uint8_equal_count'] == 100
    result = {'status': 'five_expert_integration_engineering_complete', 'receipt_sha256': c.sha(output / 'receipt.json'),
              'single_selection_parity_pass': True, 'rain_development_gate_pass': gate,
              'rain_historical_pixel_reproduction_pass': reproduced,
              'ready_for_router_design_review': gate and reproduced,
              'groups': groups, 'comparisons': comparisons, 'records': records,
              'soft_weights_are_manual_not_a_router': True, 'soft_quality_pass_claimed': False,
              'official_pixels_read': False, 'automatic_training': False}
    c.write_new(output / 'result.json', result)
    print('FIVE_EXPERT_RESULT_VERIFIED ' + json.dumps({'rain_gate': gate, 'rain_history_reproduced': reproduced, 'groups': groups}), flush=True)


def runtime(output, bound, node, idle):
    value = parent.hopper.hardware(idle)
    value['metrics'] = four_impl.metric_runtime({'metric': bound['metric']})
    value['memory_policy'] = bound['memory_policy']
    old = c.read(c.path(ROOT, bound['config']['rain_parent_run']) / f'runtime_node{node}.json')
    if parent.hopper.execution_signature(value) != parent.hopper.execution_signature(old):
        raise ValueError('runtime differs from step800; record a migration instead of silently switching')
    target = output / f'runtime_node{node}.json'
    if idle:
        c.write_new(target, value)
    elif value != c.read(target):
        raise ValueError('runtime changed within run')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('phase', choices=['preflight', 'prepare', 'hardware', 'runtime', 'run', 'summarize', 'publish', 'wait'])
    ap.add_argument('--event', choices=['boot', 'prepared', 'ready', 'done', 'verified', 'failed'])
    ap.add_argument('--run-id', default='rain800_bank_4x2_v1')
    ap.add_argument('--launch-id')
    ap.add_argument('--node-rank', type=int, choices=range(4), default=0)
    args = ap.parse_args(); cfg = config(); output = output_path(args.run_id)
    if str(ROOT) != cfg['active_root']:
        raise ValueError('destination workspace only; old source is not an execution root')
    if args.phase in ('publish', 'wait'):
        if not args.event or not args.launch_id:
            raise ValueError('launch and event required')
        if args.phase == 'publish':
            cluster.publish(output, args.launch_id, args.node_rank, args.event)
        else:
            deadline = time.monotonic() + cfg['wall_cap_seconds']
            while not cluster.check(output, args.launch_id, args.event):
                if time.monotonic() > deadline:
                    raise TimeoutError('four-node barrier timeout')
                time.sleep(2)
        return
    if args.phase in ('preflight', 'prepare'):
        import shutil
        if args.phase == 'prepare' and args.node_rank != 0:
            raise ValueError('node0 prepare only')
        if shutil.disk_usage(ROOT).free < cfg['minimum_free_disk_gib'] * 1024**3:
            raise ValueError('need 5GiB free; do not delete history')
        bound = make_receipt()
        if args.phase == 'prepare':
            c.write_new(output / 'receipt.json', bound)
        print('FIVE_EXPERT_PREFLIGHT_PASS pixels_read=false GPU_used=false', flush=True)
        return
    if args.phase == 'hardware':
        print(json.dumps(parent.hopper.hardware(True))); return
    bound = load_receipt(output)
    if args.phase == 'runtime':
        runtime(output, bound, args.node_rank, True); return
    if args.phase == 'summarize':
        if args.node_rank != 0 or not cluster.check(output, args.launch_id, 'done'):
            raise ValueError('four-node done required')
        summarize(output, bound); return
    import torch
    rank, local = c.worker_rank(args.node_rank, int(os.environ['LOCAL_RANK']), int(os.environ['WORLD_SIZE']),
                               int(os.environ['LOCAL_WORLD_SIZE']), int(os.environ['RANK']))
    if int(os.environ['REPLAY_NODE_RANK']) != args.node_rank:
        raise ValueError('node rank mismatch')
    runtime(output, bound, args.node_rank, False); parent.check_runtimes(output)
    torch.cuda.set_device(local)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    os.environ.update(KLEIN_4B_BASE_MODEL_PATH=str(c.path(ROOT, bound['model']['flow_weights'])),
                      AE_MODEL_PATH=str(c.path(ROOT, bound['model']['autoencoder_weights'])),
                      TORCH_HOME=str(ROOT / 'models/torch'), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    run(output, bound, rank, local)
    gc.collect(); torch.cuda.empty_cache()
    score(output, bound, rank, local)


if __name__ == '__main__':
    main()
