#!/usr/bin/env python3
"""CPU-only matched image/VLM CE training from existing frozen LQ features.

This does NOT render restored images, access GT pixels, update FLUX or prove
restoration gates. No development features are loaded until both arms finish.
"""
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from smoke import analyze_foundir_frozen_router_errors_v1 as analysis
from smoke import router_supervised_state_v1 as state
c = state.c
ORDER = analysis.ORDER
CONFIG = 'configs/foundir_router_supervised_repair_v1.json'
HARD_REPORT = 'reports/foundir_router_hard_counterfactual_terminal_2026-11-04.json'
RUN = 'fresh_ce400_matched_v1'
OUTPUT = 'checkpoints/foundir_router_supervised_repair_v1/' + RUN
REPORT = 'reports/foundir_router_supervised_repair_terminal_2026-11-04.json'
SOURCES = [CONFIG, 'lowlight_restoration/supervised_singleton_router.py',
           'smoke/router_supervised_state_v1.py', 'smoke/train_foundir_router_supervised_repair_v1.py',
           'smoke/run_foundir_router_supervised_repair_v1_cpu.sh',
           'smoke/start_foundir_router_supervised_repair_v1_tmux.sh',
           'smoke/analyze_foundir_frozen_router_errors_v1.py',
           'smoke/rain_native50_continue_state_v1.py', 'smoke/existing_flux_replay_contract_v1.py',
           'lowlight_restoration/frozen_expert_router.py']
require = analysis.require


def config():
    cfg = c.read(ROOT/CONFIG)
    expected = {'run_id': RUN, 'parent_run': analysis.RUN, 'arms': ['image_only', 'image_plus_vlm'],
                'steps_per_arm': 400, 'global_batch': 32, 'learning_rate': .001, 'weight_decay': 0.,
                'image_dim': 6144, 'vlm_dim': 3584, 'hidden': 256, 'train_per_task': 128,
                'development_per_task': 8, 'cpu_threads': 8, 'wall_cap_seconds': 1800,
                'deployment_policy': 'predicted_top1_singleton_only',
                'fresh_model_and_adam_per_arm': True, 'automatic_gpu_or_followup': False,
                'read_development_features_only_after_both_arms_finished': True, 'may_seed_formal': False}
    require(all(cfg.get(k) == v for k,v in expected.items()), 'fixed repair scope changed')
    return cfg


def validate_split(items):
    require(len(items) == 680 and len({i['key'] for i in items}) == 680, '640 train / 40 development required')
    for split, count in [('train', 128), ('validation', 8)]:
        rows = [i for i in items if i['split'] == split]
        require(Counter(i['task'] for i in rows) == Counter({t: count for t in ORDER}), 'per-task split changed')
        require(all(i['row']['split'] == split and int(i['row']['composition_size']) == 1 for i in rows),
                'only existing singleton feature records allowed')
    for field in ('source_group', 'gt_sha256'):
        train = {i['row'][field] for i in items if i['split'] == 'train'}
        dev = {i['row'][field] for i in items if i['split'] == 'validation'}
        require(train.isdisjoint(dev), 'train/development source or GT overlap')


def training_indices(items, step):
    require(type(step) is int and 0 <= step < 400, 'outside fixed training budget')
    require(len(items) == 640 and all(i['split'] == 'train' for i in items), 'training batch cannot contain development')
    groups = {t: [j for j,i in enumerate(items) if i['task'] == t] for t in ORDER}
    require(all(len(g) == 128 for g in groups.values()), '128 train records per task required')
    return [groups[ORDER[slot % 5]][(slot//5) % 128] for slot in range(step*32, (step+1)*32)]


def prepare():
    cfg = config(); parent = ROOT/'checkpoints/foundir_frozen_expert_router_v1'/analysis.RUN
    require(c.sha(parent/'receipt.json') == cfg['parent_receipt_sha256']
            and c.sha(parent/'result.json') == cfg['parent_result_sha256'], 'parent identity changed')
    require(c.sha(ROOT/HARD_REPORT) == cfg['hard_diagnostic_sha256'], 'hard diagnostic changed')
    require(analysis.diagnose(ROOT) == c.read(ROOT/analysis.OUTPUT), 'published error analysis differs from source metadata')
    bound = c.read(parent/'receipt.json'); items = bound['items']; validate_split(items)
    features = {}
    for item in items:
        file = parent/'features'/(item['key']+'.json'); record = c.read(file)
        require(record['identity'] == {'receipt_sha256': cfg['parent_receipt_sha256'], 'item': item, 'arm': 'features'},
                'feature metadata identity changed')
        require(record['GT_input'] is False and record['vlm_receipt_sha256'] == c.sha(parent/'vlm'/(item['key']+'.json')),
                'unverified or GT-conditioned feature cache')
        tensor = c.path(parent, record['tensor'])
        require(set(record['files']) == {record['tensor']} and tensor.is_file(), 'feature artifact missing')
        features[item['key']] = {'path': tensor.relative_to(ROOT).as_posix(),
                                'sha256': record['files'][record['tensor']], 'marker_sha256': c.sha(file)}
    # Bind inherited code identities without loading any frozen model/checkpoint.
    for name,digest in bound['sources'].items():
        require(c.sha(ROOT/name) == digest, 'old bound source changed: '+name)
    value = {'config': cfg, 'items': items, 'features': features,
             'parent_receipt_sha256': cfg['parent_receipt_sha256'],
             'sources': {**bound['sources'], **{name: c.sha(ROOT/name) for name in SOURCES}},
             'parent_breakdown_sha256': c.sha(ROOT/analysis.OUTPUT),
             'GT_pixels_read': False, 'training_labels_allowed': True, 'GPU_used': False}
    output = ROOT/OUTPUT
    if not (output/'receipt.json').exists():
        require(shutil.disk_usage(ROOT).free >= cfg['minimum_free_disk_gib']*1024**3, 'need 30GiB for immutable per-step states')
    c.write_new(output/'receipt.json', value)
    print('SUPERVISED_ROUTER_PREFLIGHT_PASS pixels_read=false GPU_used=false train=640 development=40', flush=True)


def load_bound():
    bound = c.read(ROOT/OUTPUT/'receipt.json')
    require(bound['config'] == config(), 'bound config changed')
    for name,digest in bound['sources'].items():
        require(c.sha(ROOT/name) == digest, 'bound source changed: '+name)
    require(c.sha(ROOT/analysis.OUTPUT) == bound['parent_breakdown_sha256'], 'parent breakdown changed')
    validate_split(bound['items'])
    return bound


def cpu_runtime(cfg):
    import torch
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(), 'CPU-only job; hide CUDA devices')
    torch.set_num_threads(cfg['cpu_threads'])
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    cpu_model = next((line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                      if line.startswith('model name')), 'unknown')
    value = {'torch': str(torch.__version__), 'python': platform.python_version(), 'machine': platform.machine(),
             'threads': torch.get_num_threads(), 'interop_threads': torch.get_num_interop_threads(),
             'cpu_model': cpu_model, 'cpu_capability': torch.backends.cpu.get_cpu_capability(),
             'torch_build_digest': c.digest(torch.__config__.show()),
             'deterministic_algorithms': True, 'device': 'cpu'}
    c.write_new(ROOT/OUTPUT/'runtime.json', value)
    return value


def load_features(bound, split):
    import torch
    from safetensors import safe_open
    require(split in ('train', 'validation'), 'invalid split')
    if split == 'validation':
        for arm in bound['config']['arms']:
            require((ROOT/OUTPUT/arm/'steps/step_000400.json').is_file(), 'development features locked until both arms finish')
    items = [i for i in bound['items'] if i['split'] == split]
    images = []; vlms = []
    for item in items:
        spec = bound['features'][item['key']]; file = c.path(ROOT, spec['path'])
        require(c.sha(file) == spec['sha256'], 'frozen feature tensor changed')
        with safe_open(str(file), framework='pt', device='cpu') as f:
            # Do not load the cached latent reference, any RGB tensor, or weights.
            image, vlm = f.get_tensor('image'), f.get_tensor('vlm')
        require(image.shape == (1, 6144) and vlm.shape == (1, 3584)
                and torch.isfinite(image).all() and torch.isfinite(vlm).all(), 'invalid pooled LQ features')
        images.append(image.float()); vlms.append(vlm.float())
    return items, torch.cat(images), torch.cat(vlms), torch.tensor([ORDER.index(i['task']) for i in items])


def make_model(cfg):
    import torch
    from lowlight_restoration.supervised_singleton_router import FrozenExpertRouter
    random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
    model = FrozenExpertRouter(cfg['image_dim'], cfg['vlm_dim'], cfg['hidden']).cpu().float()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'],
                                 foreach=False, fused=False)
    return model, optimizer


def train_arm(bound, dataset, arm):
    import torch
    from lowlight_restoration.supervised_singleton_router import train_batch
    cfg = bound['config']; items, image, vlm, labels = dataset
    output = ROOT/OUTPUT/arm
    receipt_hash = c.digest({'receipt': c.sha(ROOT/OUTPUT/'receipt.json'), 'runtime': c.sha(ROOT/OUTPUT/'runtime.json')})
    model, optimizer = make_model(cfg)
    initial = state.model_state(model)
    init_hash = state.io.parameter_hash(initial)
    c.write_new(ROOT/OUTPUT/'initialization.json', {'parameter_hash': init_hash, 'seed': cfg['seed'],
                'fresh_model_and_adam': True, 'qualification_weights_loaded': False})
    saved = state.load(output, receipt_hash, arm)
    if saved is None:
        history = []; start = 0
        state.save(output, model, optimizer, step=0, history=history, receipt_hash=receipt_hash, arm=arm)
    else:
        state.restore(model, optimizer, saved); history = saved['history']; start = saved['step']
    del saved
    for step in range(start, cfg['steps_per_arm']):
        begun = time.monotonic()
        indices = torch.tensor(training_indices(items, step))
        metrics = train_batch(model, optimizer, image[indices], vlm[indices], labels[indices], arm=arm)
        metrics.update(step=step+1, seconds_before_save=time.monotonic()-begun)
        history.append(metrics)
        state.save(output, model, optimizer, step=step+1, history=history, receipt_hash=receipt_hash, arm=arm)
        if (step+1) % 25 == 0:
            print('SUPERVISED_ROUTER_STEP '+json.dumps({'arm': arm, **metrics}), flush=True)
    if arm == 'image_only':
        require(all(torch.equal(p.detach(), initial[n]) for n,p in model.named_parameters() if n.startswith('vlm.')),
                'disabled VLM weights changed in matched control')
    else:
        require(any(h['gradient_norms']['vlm'] > 0 for h in history), 'VLM never received training gradients')
    require(not torch.cuda.is_initialized(), 'unexpected CUDA initialization')
    return model


def evaluate_features(model, dataset, arm, *, include_rows):
    import torch
    from lowlight_restoration.supervised_singleton_router import infer
    items, image, vlm, labels = dataset
    result = infer(model, image, vlm, arm=arm)
    predicted = result['hard_weights'].argmax(-1)
    per_task = {}
    for task in ORDER:
        mask = labels == ORDER.index(task)
        per_task[task] = {'count': int(mask.sum()), 'correct': int((predicted[mask] == labels[mask]).sum())}
    summary = {'count': len(items), 'correct': int((predicted == labels).sum()), 'per_task': per_task,
               'ce': float(torch.nn.functional.cross_entropy(result['logits'], labels)),
               'confusion': {t: dict(Counter(ORDER[int(predicted[j])] for j,i in enumerate(items) if i['task'] == t)) for t in ORDER}}
    if include_rows:
        summary['rows'] = [{'key': item['key'], 'task_for_reporting_only': item['task'], 'seed': item['seed'],
                            'crop_xy': item['row']['crop_xy'], 'predicted_anchor': ORDER[int(predicted[j])],
                            'correct': bool(predicted[j] == labels[j]),
                            **{k: v[j].tolist() for k,v in result.items()}}
                           for j,item in enumerate(items)]
    return summary


def train():
    bound = load_bound(); cfg = bound['config']; runtime = cpu_runtime(cfg)
    training = load_features(bound, 'train')
    models = {arm: train_arm(bound, training, arm) for arm in cfg['arms']}
    # No development features/labels reach an optimizer, early stop, or best-checkpoint choice.
    development = load_features(bound, 'validation')
    summaries = {arm: {'train': evaluate_features(model, training, arm, include_rows=False),
                       'development': evaluate_features(model, development, arm, include_rows=True),
                       'checkpoint': c.read(ROOT/OUTPUT/arm/'steps/step_000400.json')}
                 for arm,model in models.items()}
    old = {r['key']: r for r in c.read(ROOT/analysis.OUTPUT)['per_image']}
    transitions = {}
    for arm in cfg['arms']:
        rows = summaries[arm]['development']['rows']
        transitions[arm] = {name: [r['key'] for r in rows if old[r['key']]['step408']['correct'] == before and r['correct'] == after]
                           for name,before,after in [('kept_correct',True,True), ('became_correct',False,True),
                                                    ('became_wrong',True,False), ('stayed_wrong',False,False)]}
    value = {'status': 'matched_cpu_router_training_complete', 'run_id': RUN, 'steps_per_arm': 400,
             'receipt_sha256': c.sha(ROOT/OUTPUT/'receipt.json'), 'runtime': runtime,
             'initialization': c.read(ROOT/OUTPUT/'initialization.json'), 'arms': summaries,
             'transitions_vs_old_step408': transitions,
             'matched_vlm_minus_image_correct': summaries['image_plus_vlm']['development']['correct']-summaries['image_only']['development']['correct'],
             'GPU_used': False, 'GT_pixels_read': False, 'restoration_images_generated': 0,
             'restoration_PSNR_LPIPS_verified': False, 'frozen_experts_changed': False,
             'automatic_followup': False, 'deployment_policy': cfg['deployment_policy'],
             'interpretation': ['Only source-separated router classification on the existing 40 development images.',
                'Both arms use the same fresh initialization, batches, steps, optimizer and data.',
                'Image-only versus image+VLM is matched here; neither arm is an exact replay of old CUDA warmup.',
                'Top-1 is a singleton ability-preservation policy, not a mixed-degradation solution.',
                'No new PSNR/LPIPS or official-test gate is measured; do not infer success from accuracy.',
                'Forty development examples and one seed do not establish statistically robust VLM benefit.']}
    c.write_new(ROOT/REPORT, value)
    print('SUPERVISED_ROUTER_CPU_RESULT_VERIFIED '+json.dumps({arm: {
        'train_correct': summaries[arm]['train']['correct'],
        'development_correct': summaries[arm]['development']['correct']} for arm in cfg['arms']}), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('phase', choices=['prepare', 'train', 'verify'])
    args = ap.parse_args()
    require(str(ROOT) == config()['active_root'], 'destination workspace only; frozen old root is not executable')
    if args.phase == 'prepare': prepare()
    elif args.phase == 'train': train()
    else: verify()


def verify():
    bound = load_bound(); result = c.read(ROOT/REPORT)
    require(result['status'] == 'matched_cpu_router_training_complete' and result['steps_per_arm'] == 400
            and result['receipt_sha256'] == c.sha(ROOT/OUTPUT/'receipt.json')
            and result['runtime'] == c.read(ROOT/OUTPUT/'runtime.json'), 'result/run mismatch')
    require(result['GPU_used'] is False and result['GT_pixels_read'] is False
            and result['restoration_images_generated'] == 0 and result['restoration_PSNR_LPIPS_verified'] is False,
            'classification report must not claim restoration success')
    items = {i['key']: i for i in bound['items'] if i['split'] == 'validation'}
    require(set(result['arms']) == set(bound['config']['arms']), 'both matched arms required')
    require(result['initialization'] == c.read(ROOT/OUTPUT/'initialization.json'), 'initialization identity changed')
    receipt_hash = c.digest({'receipt': c.sha(ROOT/OUTPUT/'receipt.json'), 'runtime': c.sha(ROOT/OUTPUT/'runtime.json')})
    for arm, summary in result['arms'].items():
        folder = ROOT/OUTPUT/arm; marker = c.read(folder/'steps/step_000400.json')
        initial = c.read(folder/'steps/step_000000.json')
        require(initial['parameter_hash'] == result['initialization']['parameter_hash']
                and marker['receipt_hash'] == receipt_hash and marker['arm'] == arm and marker['step'] == 400,
                'matched initialization or final identity changed')
        require(summary['checkpoint'] == marker and c.sha(c.path(folder, marker['file'])) == marker['sha256'],
                'final checkpoint changed')
        rows = summary['development']['rows']
        require(len(rows) == 40 and {r['key'] for r in rows} == set(items), 'exact development population required')
        for row in rows:
            item = items[row['key']]; logits = row['logits']
            require(len(logits) == 5 and all(analysis.number(v) for v in logits), 'invalid logits')
            index = max(range(5), key=lambda j: logits[j]); predicted = ORDER[index]
            require(row['task_for_reporting_only'] == item['task'] and row['seed'] == item['seed']
                    and row['crop_xy'] == item['row']['crop_xy'] and row['predicted_anchor'] == predicted
                    and row['hard_weights'] == [float(j == index) for j in range(5)]
                    and row['correct'] == (predicted == item['task']), 'route/GT/input mismatch')
        require(summary['development']['correct'] == sum(r['correct'] for r in rows), 'correct count mismatch')
        for task in ORDER:
            selected = [r for r in rows if r['task_for_reporting_only'] == task]
            require(summary['development']['per_task'][task] == {'count': 8, 'correct': sum(r['correct'] for r in selected)},
                    'per-task count mismatch')
        require(summary['train']['count'] == 640, 'training population mismatch')
    require(result['matched_vlm_minus_image_correct'] == result['arms']['image_plus_vlm']['development']['correct']
            - result['arms']['image_only']['development']['correct'], 'matched difference mismatch')
    print('SUPERVISED_ROUTER_FINAL_VERIFIED classification_only=true GPU_used=false', flush=True)


if __name__ == '__main__':
    main()
