#!/usr/bin/env python3
"""Bounded five-frozen-expert router qualification; not automatic formal training."""
import argparse
from contextlib import nullcontext
from datetime import timedelta
import gc
import json
import math
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'third_party/flux2/src'))
from smoke import frozen_expert_router_mixed_driver_contract_v1 as contract
from smoke import frozen_expert_router_state_v1 as state_io
c, parent, ORDER = contract.c, contract.parent, contract.ORDER
native, cluster = parent.parent, parent.cluster


def pixels(item, side, device):
    import numpy as np
    import torch
    from PIL import Image
    row = item['row']; file = c.path(ROOT, row[side + '_path'])
    if side not in ('lq', 'gt') or c.sha(file) != row[side + '_sha256']:
        raise ValueError('image identity changed')
    with Image.open(file) as im:
        if im.size != (int(row['width']), int(row['height'])):
            raise ValueError('image dimensions changed')
        x, y = row['crop_xy']
        array = np.array(im.convert('RGB').crop((x, y, x+512, y+512)), copy=True)
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).float().div(127.5).sub(1).to(device)


def identity(output, item, arm, **extra):
    # Feature extraction is round-robin once; subsequent training may read from any rank.
    return {'receipt_sha256': c.sha(output / 'receipt.json'), 'item': item, 'arm': arm, **extra}


def cached(output, folder, item, *, extra=None):
    from safetensors.torch import load_file
    path = output / folder / (item['key'] + '.json')
    if not path.exists():
        return None
    record = c.read(path)
    if record['identity'] != identity(output, item, folder, **(extra or {})):
        raise ValueError('cached item identity changed')
    file = c.path(output, record['tensor'])
    if record['files'] != {record['tensor']: c.sha(file)}:
        raise ValueError('cached tensor changed')
    return record, load_file(str(file), device='cpu')


def commit(output, folder, item, tensors, *, extra=None, values=None):
    name, digest = native.replay.save_tensors(output, folder.replace('/', '_')+'_'+item['key'], tensors)
    record = {'identity': identity(output, item, folder, **(extra or {})), 'tensor': name,
              'files': {name: digest}, **(values or {})}
    c.write_new(output / folder / (item['key'] + '.json'), record)
    return record


def features(output, bound, rank, device):
    import torch
    from PIL import Image
    from smoke.train_qwen_multidegradation_lora_ddp import load_model
    from lowlight_restoration.qwen_lora import load_qwen_adapter
    from lowlight_restoration.qwen_five_class_gate import image_embedding_sequence
    from lowlight_restoration.frozen_expert_router import pooled_image_features
    from flux2.util import load_flow_model, load_ae
    assigned = bound['items'][rank::8]
    missing = [i for i in assigned if cached(output, 'vlm', i) is None]
    if missing:
        model, processor, _ = load_model(bound['qcfg'], variant='merger_trainable', device=device)
        load_qwen_adapter(model, c.path(ROOT, bound['qwen']['adapter']))
        model.eval().requires_grad_(False)
        for number, item in enumerate(missing):
            lq = pixels(item, 'lq', 'cpu')
            array = ((lq[0]+1)*127.5).round().clamp(0, 255).byte().permute(1, 2, 0).numpy()
            inputs = processor.image_processor(images=[Image.fromarray(array)], return_tensors='pt')
            with torch.no_grad():
                seq = image_embedding_sequence(model.get_image_features(
                    inputs.pixel_values.to(device), inputs.image_grid_thw.to(device)))
                if len(seq) != 1 or seq[0].ndim != 2 or seq[0].shape[-1] != 3584:
                    raise ValueError('unexpected visual merger tokens')
                value = seq[0].float().mean(0, keepdim=True)
            if not torch.isfinite(value).all():
                raise ValueError('nonfinite visual feature')
            commit(output, 'vlm', item, {'vlm': value}, values={'GT_input': False})
            if number % 10 == 0:
                print(f'FROZEN_ROUTER_VLM rank={rank} completed={number+1}/{len(missing)}', flush=True)
        del model, processor, inputs, seq, value
        gc.collect(); torch.cuda.empty_cache()
    missing = [i for i in assigned if cached(output, 'features', i) is None]
    if missing:
        model = load_flow_model('flux.2-klein-base-4b', device=device).eval().requires_grad_(False)
        ae = native.hopper.configure_vae(load_ae('flux.2-klein-base-4b', device=device).eval().requires_grad_(False))
        for number, item in enumerate(missing):
            lq = pixels(item, 'lq', device)
            with torch.no_grad():
                reference = ae.encode(lq).to(torch.bfloat16)
                image = pooled_image_features(model, reference)
            vlm = cached(output, 'vlm', item)[1]['vlm']
            if image.shape != (1, 6144) or not torch.isfinite(image).all():
                raise ValueError('invalid frozen FLUX input features')
            commit(output, 'features', item, {'image': image, 'vlm': vlm, 'reference': reference},
                   values={'GT_input': False, 'vlm_receipt_sha256': c.sha(output/'vlm'/(item['key']+'.json'))})
            if number % 10 == 0:
                print(f'FROZEN_ROUTER_FEATURE rank={rank} completed={number+1}/{len(missing)}', flush=True)
        del model, ae
        gc.collect(); torch.cuda.empty_cache()


def load_bank(bound, device):
    import torch
    from safetensors.torch import load_file
    from flux2.util import load_flow_model, load_ae
    from lowlight_restoration.protocol_routed_flux import install_protocol_bank
    from lowlight_restoration.existing_flux_experts import FLOW_CONTRACT
    from lowlight_restoration.frozen_expert_router import enable_routed_checkpointing
    from smoke.audit_foundir_existing_flux_experts_v1 import expected_modules
    b = bound['parent']; prompts = {}
    for task in ORDER:
        spec = b['rain_prompt'] if task == 'rain' else b['four_prompts']['tensors'][task]
        root = ROOT if task == 'rain' else c.path(ROOT, b['config']['four_parent_run'])
        file = c.path(root, spec['path'])
        if c.sha(file) != spec['sha256']:
            raise ValueError('prompt cache changed')
        prompts[task] = load_file(str(file), device=str(device))
    model = load_flow_model('flux.2-klein-base-4b', device=device).eval().requires_grad_(False)
    states = []
    for task in ORDER:
        spec = b['experts'][task]['checkpoint']; file = c.path(ROOT, spec['path'])
        if c.sha(file) != spec['sha256']:
            raise ValueError('expert checkpoint changed')
        values = load_file(str(file), device='cpu')
        if any(v.dtype != (torch.float32 if task == 'rain' else torch.bfloat16) for v in values.values()):
            raise ValueError('expert precision changed')
        states.append((task, values, 32., FLOW_CONTRACT))
    gates = install_protocol_bank(model, modules=expected_modules(), expert_states=states)
    model.eval().requires_grad_(False)
    enable_routed_checkpointing(model)
    ae = native.hopper.configure_vae(load_ae('flux.2-klein-base-4b', device=device).eval().requires_grad_(False))
    return model, gates, parent.protocols(b), ae, prompts


def parity_item(items, rank):
    task = ORDER[rank % 5]
    return [i for i in items if i['split'] == 'train' and i['task'] == task][rank // 5]


def parity(output, bound, bank, rank, device, feature_map):
    import torch
    from lowlight_restoration.frozen_expert_router import differentiable_restore
    from lowlight_restoration.five_expert_flux import restore_five_expert
    item = parity_item(bound['items'], rank)
    path = output / 'parity' / f'rank{rank}.json'
    ident = identity(output, item, 'autograd_forward_parity', rank=rank)
    if path.exists():
        old = c.read(path)
        if old['identity'] != ident or not old['pass']:
            raise ValueError('invalid previous parity')
        return
    model, gates, rules, ae, prompts = bank
    lq = pixels(item, 'lq', device)
    w = torch.zeros(1, 5, device=device); w[0, ORDER.index(item['task'])] = 1.
    original, u8, _ = restore_five_expert(model, gates, rules, w, ae, lq, prompts, seed=item['seed'])
    # Actual checkpoint-enabled grad path, not just an inference-only alternative.
    with native.hopper.offload_activations():
        raw, image, _ = differentiable_restore(model, gates, rules, w.requires_grad_(), ae, lq,
            feature_map[item['key']]['reference'].to(device), prompts, seed=item['seed'])
        difference = float((raw.detach()-original).abs().max())
        equal = torch.equal(((image.detach()+1)*127.5).round().byte(), u8)
        image.float().square().mean().backward()
    finite_gradient = w.grad is not None and bool(torch.isfinite(w.grad).all())
    gradient_norm = float(w.grad.norm()) if finite_gradient else None
    value = {'identity': ident, 'decoded_max_abs': difference, 'uint8_equal': equal,
             'gradient_finite': finite_gradient, 'gradient_norm': gradient_norm,
             'pass': difference <= 1e-6 and equal and finite_gradient and gradient_norm > 0}
    c.write_new(path, value)
    print('FROZEN_ROUTER_PARITY ' + json.dumps(value), flush=True)
    if not value['pass']:
        raise ValueError('autograd path parity/gradient failed; no quality training')


def evaluate(output, bound, bank, router, metric, rank, device, step, feature_map):
    import torch
    from lowlight_restoration.five_expert_flux import restore_five_expert
    model, gates, rules, ae, prompts = bank
    model_hash = state_io.io.parameter_hash(state_io.model_state(router))
    items = [i for i in bound['items'] if i['split'] == 'validation'][rank::8]
    for item in items:
        lq = pixels(item, 'lq', device)
        for arm in ('known_class_control', f'router_step{step}'):
            extra = {'router_hash': model_hash} if arm.startswith('router') else {}
            folder = 'evaluation/' + arm
            if cached(output, folder, item, extra=extra) is not None:
                continue
            with torch.no_grad():
                if arm == 'known_class_control':
                    weights = torch.zeros(1, 5, device=device); weights[0, ORDER.index(item['task'])] = 1.
                else:
                    f = feature_map[item['key']]
                    _, weights, _, _ = router(f['image'].to(device), f['vlm'].to(device), vlm_enabled=step > 400)
                raw, u8, route = restore_five_expert(model, gates, rules, weights, ae, lq, prompts, seed=item['seed'])
                # GT loaded only AFTER routing and generation; no per-image GT selection.
                gt = pixels(item, 'gt', device)
                target = (gt+1)*.5; image = u8.float()/255.
                mse = float((image-target).square().mean())
                values = {'psnr': 100. if mse <= 1e-12 else -10*math.log10(mse),
                          'lpips': float(metric(image, target))}
                metric.reset()
            commit(output, folder, item, {'uint8': u8}, extra=extra,
                   values={'metrics': values, 'route': route, 'known_class_control_not_deployed': arm == 'known_class_control'})
        print(f'FROZEN_ROUTER_EVAL rank={rank} step={step} key={item["key"]}', flush=True)


def summaries(output, bound, step):
    arms = ('known_class_control', 'router_step400') if step == 400 else ('known_class_control', 'router_step400', 'router_step408')
    groups = {}
    for task in ORDER:
        items = [i for i in bound['items'] if i['task'] == task and i['split'] == 'validation']
        values = {arm: [c.read(output/'evaluation'/arm/(i['key']+'.json')) for i in items] for arm in arms}
        groups[task] = {'count': len(items), 'means': {a: {k: sum(v['metrics'][k] for v in rows)/len(rows)
            for k in ('psnr', 'lpips')} for a, rows in values.items()},
            'router_anchor_correct': {a: sum(v['route']['anchor'] == task for v in rows) for a, rows in values.items()},
            'regressions_vs_known_class': {a: {k: sum((r['metrics'][k] < b['metrics'][k]) if k == 'psnr'
                else (r['metrics'][k] > b['metrics'][k]) for r, b in zip(rows, values['known_class_control']))
                for k in ('psnr', 'lpips')} for a, rows in values.items() if a != 'known_class_control'}}
    value = {'step': step, 'groups': groups, 'development_not_official_or_blind_test': True,
             'known_class_control_uses_label_not_deployable_router': True}
    c.write_new(output/'milestones'/f'step_{step}.json', value)
    print('FROZEN_ROUTER_MILESTONE ' + json.dumps(value), flush=True)
    return value


def train(output, bound, rank, local):
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    from lowlight_restoration.frozen_expert_router import FrozenExpertRouter, differentiable_restore, objective
    device = torch.device('cuda', local); cfg = bound['config']
    features(output, bound, rank, device)
    dist.barrier()
    feature_map = {}
    for item in bound['items']:
        record, values = cached(output, 'features', item)
        if (record['GT_input'] or record['vlm_receipt_sha256'] != c.sha(output/'vlm'/(item['key']+'.json'))
                or values['image'].shape != (1, 6144) or values['vlm'].shape != (1, 3584)
                or any(not torch.isfinite(v).all() for v in values.values())):
            raise ValueError('invalid frozen features')
        feature_map[item['key']] = values
    random.seed(cfg['seed']); torch.manual_seed(cfg['seed']); torch.cuda.manual_seed_all(cfg['seed'])
    router = FrozenExpertRouter().to(device)
    wrapped = DDP(router, device_ids=[local], broadcast_buffers=False, gradient_as_bucket_view=True)
    optimizer = torch.optim.AdamW(router.parameters(), lr=cfg['warmup_learning_rate'], weight_decay=0., fused=False)
    receipt_hash = c.sha(output/'receipt.json')
    saved = state_io.load(output, receipt_hash)
    start = 0 if saved is None else saved['step']; history = [] if saved is None else saved['history']
    if saved is not None:
        state_io.restore(router, optimizer, saved, rank, device)
    starts = [None]*8; dist.all_gather_object(starts, start)
    if len(set(starts)) != 1:
        raise ValueError('ranks saw different committed progress')
    del saved

    def save(step):
        rngs = [None]*8; dist.all_gather_object(rngs, state_io.io.rng_state(device))
        hashes = [None]*8
        dist.all_gather_object(hashes, state_io.io.parameter_hash(state_io.model_state(router)))
        if len(set(hashes)) != 1:
            raise ValueError('DDP router parameters differ')
        if rank == 0:
            state_io.save(output, router, optimizer, step=step, history=history, rank_rng=rngs, receipt_hash=receipt_hash)
            if step % 25 == 0 or step > 400:
                print(f'FROZEN_ROUTER_CHECKPOINT step={step}/408 next_slot={step*32}', flush=True)
        dist.barrier()

    if start == 0 and not (output/'steps/step_000000.json').exists():
        save(0)
    bank = None; metric = None; frozen_versions = None

    def prepare_quality():
        nonlocal bank, metric, frozen_versions
        if bank is not None:
            return
        # Construction consumes RNG. Preserve it so interrupted/resumed runs have
        # the same next-step state as uninterrupted execution.
        rng = state_io.io.rng_state(device)
        bank = load_bank(bound, device)
        metric = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).to(device).eval().requires_grad_(False)
        frozen_versions = [(p, p._version) for m in (bank[0], bank[3], metric) for p in m.parameters()]
        if any(p.requires_grad for p, _ in frozen_versions):
            raise ValueError('only router parameters may train')
        parity(output, bound, bank, rank, device, feature_map)
        dist.barrier()
        state_io.io.restore_rng(rng, device)

    def evaluate_step(step):
        rng = state_io.io.rng_state(device)
        evaluate(output, bound, bank, router, metric, rank, device, step, feature_map)
        dist.barrier()
        if rank == 0:
            summaries(output, bound, step)
        dist.barrier(); state_io.io.restore_rng(rng, device)

    if start >= 400:
        prepare_quality()
        if start in (400, 408):
            evaluate_step(start)
    for step in range(start, 408):
        if step == 400 and bank is None:
            prepare_quality(); evaluate_step(400)
        begun = time.monotonic(); quality = step >= 400
        for group in optimizer.param_groups:
            group['lr'] = cfg['quality_learning_rate' if quality else 'warmup_learning_rate']
        optimizer.zero_grad(set_to_none=True)
        totals = torch.zeros(15, device=device, dtype=torch.float64)
        for micro in range(4):
            slot = step*32 + micro*8 + rank
            item = contract.training_item(bound['items'], slot); f = feature_map[item['key']]
            label = torch.tensor([ORDER.index(item['task'])], device=device)
            with (wrapped.no_sync() if micro < 3 else nullcontext()):
                logits, weights, _, _ = wrapped(f['image'].to(device), f['vlm'].to(device), vlm_enabled=quality)
                if quality:
                    # CE branches from logits, not weights. This hook isolates
                    # evidence of an actual image-loss gradient through FLUX.
                    image_gradients = []
                    handle = weights.register_hook(lambda g: image_gradients.append(g.detach().float().norm()))
                    model, gates, rules, ae, prompts = bank
                    lq, gt = pixels(item, 'lq', device), pixels(item, 'gt', device)
                    with native.hopper.offload_activations():
                        raw, image, _ = differentiable_restore(model, gates, rules, weights, ae, lq,
                            f['reference'].to(device), prompts, seed=item['seed'])
                        loss, terms = objective(image, gt, metric, logits, label,
                            lpips_weight=.2 if item['task'] == 'blur' else .05)
                        if not torch.isfinite(loss):
                            raise ValueError('nonfinite image objective')
                        (loss/4).backward()
                    handle.remove()
                    if len(image_gradients) != 1 or not torch.isfinite(image_gradients[0]):
                        raise ValueError('missing/nonfinite image-loss route gradient')
                    totals[4] += image_gradients[0].double()*4/32
                    metric.reset()
                    totals[1] += float(terms['rgb_mse'].detach())/32
                    totals[2] += float(terms['lpips'].detach())/32
                    totals[3] += float(terms['route_ce'].detach())/32
                    del raw, image, loss, terms, lq, gt
                else:
                    loss = torch.nn.functional.cross_entropy(logits, label)
                    (loss/4).backward()
                    totals[3] += float(loss.detach())/32
                totals[0] += float(weights[0, label.item()].detach())/32
                totals[5:10] += weights[0].detach().double()/32
                totals[10+int(weights.detach().argmax())] += 1
        grads = {name: float(torch.nn.utils.clip_grad_norm_(module.parameters(), float('inf'), error_if_nonfinite=True))
                 for name, module in [('image', router.image), ('vlm', router.vlm)]}
        if quality and any(p.grad is not None or p._version != old for p, old in frozen_versions):
            raise ValueError('frozen backbone/expert/VAE/metric modified')
        optimizer.step()
        if any(not torch.isfinite(p).all() for p in router.parameters()):
            raise ValueError('nonfinite router update')
        dist.all_reduce(totals)
        torch.cuda.synchronize(device)
        record = {'step': step+1, 'phase': 'quality' if quality else 'image_ce',
                  'correct_class_weight': float(totals[0]), 'rgb_mse': float(totals[1]),
                  'lpips': float(totals[2]), 'ce': float(totals[3]), 'gradient_norms': grads,
                  'image_loss_route_gradient_norm_mean': float(totals[4]),
                  'expert_weight_means': dict(zip(ORDER, totals[5:10].tolist())),
                  'anchor_counts': dict(zip(ORDER, totals[10:15].tolist())),
                  'seconds_before_save': time.monotonic()-begun,
                  'peak_allocated_gib': torch.cuda.max_memory_allocated(device)/1024**3}
        # Rank0 timings are the canonical history; gradients are DDP synchronized.
        records = [record if rank == 0 else None]; dist.broadcast_object_list(records, src=0)
        history.append(records[0]); save(step+1)
        if rank == 0 and (quality or (step+1)%25 == 0):
            print('FROZEN_ROUTER_STEP '+json.dumps(records[0]), flush=True)
    if start != 408:
        prepare_quality(); evaluate_step(408)


def check_cluster_runtimes(output):
    from smoke.frozen_router_runtime_policy_v1 import check_runtimes
    return check_runtimes([c.read(output/f'runtime_node{n}.json') for n in range(4)])


def runtime(output, bound, node, idle):
    actual = native.hopper.hardware(idle)
    actual['metrics'] = parent.four_impl.metric_runtime({'metric': bound['parent']['metric']})
    actual['memory_policy'] = {'inherited': bound['parent']['memory_policy'], 'router_explicit_gate_checkpoint': True}
    # Cross-node driver releases may differ; exact per-node runtime is still
    # immutable within this run. CUDA/NCCL and image parity are tested on GPU.
    file = output/f'runtime_node{node}.json'
    if idle:
        c.write_new(file, actual)
    elif actual != c.read(file):
        raise ValueError('runtime changed within run')


def summarize(output, bound):
    state = state_io.load(output, c.sha(output/'receipt.json'))
    if state is None or state['step'] != 408:
        raise ValueError('incomplete optimizer progress')
    for rank in range(8):
        value = c.read(output/'parity'/f'rank{rank}.json')
        if value['identity'] != identity(output, parity_item(bound['items'], rank), 'autograd_forward_parity', rank=rank) or not value['pass']:
            raise ValueError('parity incomplete')
    # Lightweight terminal verification: final state, small markers and saved
    # per-image outputs, not all 409 historical checkpoint tensors.
    for step in (400, 408):
        checkpoint = c.read(output/'steps'/f'step_{step:06d}.json')
        for item in [i for i in bound['items'] if i['split'] == 'validation']:
            for arm in ('known_class_control', f'router_step{step}'):
                extra = {} if arm == 'known_class_control' else {'router_hash': checkpoint['parameter_hash']}
                row, _ = cached(output, 'evaluation/'+arm, item, extra=extra)
                if any(not math.isfinite(v) for v in row['metrics'].values()):
                    raise ValueError('nonfinite evaluation')
        summaries(output, bound, step)
    result = {'status': 'frozen_expert_router_engineering_complete', 'step': 408,
              'receipt_sha256': c.sha(output/'receipt.json'), 'final_checkpoint': c.read(output/'steps/step_000408.json'),
              'before_quality': c.read(output/'milestones/step_400.json'),
              'after_quality': c.read(output/'milestones/step_408.json'),
              'official_pixels_read': False, 'frozen_experts': True, 'scientific_success_claimed': False,
              'VLM_causal_contribution_proven': False, 'may_seed_formal': False, 'automatic_formal': False}
    c.write_new(output/'result.json', result)
    print('FROZEN_ROUTER_RESULT_VERIFIED '+json.dumps(result), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('phase', choices=['preflight', 'prepare', 'hardware', 'runtime', 'train', 'summarize', 'publish', 'wait'])
    ap.add_argument('--event', choices=['boot', 'prepared', 'ready', 'done', 'verified', 'failed'])
    ap.add_argument('--run-id', default='qualification_mixed_driver_4x2_v1')
    ap.add_argument('--launch-id')
    ap.add_argument('--node-rank', type=int, choices=range(4), default=0)
    args = ap.parse_args(); cfg = contract.config(); output = contract.output_path(args.run_id)
    if str(ROOT) != cfg['active_root']:
        raise ValueError('destination workspace only; do not launch in the migration source')
    if args.phase in ('publish', 'wait'):
        if not args.event or not args.launch_id:
            raise ValueError('launch/event required')
        if args.phase == 'publish':
            cluster.publish(output, args.launch_id, args.node_rank, args.event)
        else:
            deadline = time.monotonic()+cfg['wall_cap_seconds']
            while not cluster.check(output, args.launch_id, args.event):
                if time.monotonic() >= deadline:
                    raise TimeoutError('four-node barrier timeout')
                time.sleep(2)
        return
    if args.phase in ('preflight', 'prepare'):
        import shutil
        if args.phase == 'prepare' and args.node_rank != 0:
            raise ValueError('node0 prepares shared receipt')
        if (output/'receipt.json').exists():
            contract.load_receipt(output)
        else:
            if shutil.disk_usage(ROOT).free < cfg['minimum_free_disk_gib']*1024**3:
                raise ValueError('need 20GiB free; never delete history automatically')
            b = contract.make_receipt()
            if args.phase == 'prepare':
                c.write_new(output/'receipt.json', b)
        print('FROZEN_ROUTER_PREFLIGHT_PASS pixels_read=false GPU_used=false', flush=True); return
    if args.phase == 'hardware':
        print(json.dumps(native.hopper.hardware(True))); return
    b = contract.load_receipt(output)
    if args.phase == 'runtime':
        runtime(output, b, args.node_rank, True); return
    if args.phase == 'summarize':
        if args.node_rank != 0 or not cluster.check(output, args.launch_id, 'done'):
            raise ValueError('four-node done required')
        summarize(output, b); return
    import torch
    import torch.distributed as dist
    rank, local = c.worker_rank(args.node_rank, int(os.environ['LOCAL_RANK']), int(os.environ['WORLD_SIZE']),
        int(os.environ['LOCAL_WORLD_SIZE']), int(os.environ['RANK']))
    if int(os.environ['REPLAY_NODE_RANK']) != args.node_rank:
        raise ValueError('node-rank mismatch')
    runtime(output, b, args.node_rank, False); check_cluster_runtimes(output)
    torch.cuda.set_device(local)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    os.environ.update(KLEIN_4B_BASE_MODEL_PATH=str(c.path(ROOT, b['parent']['model']['flow_weights'])),
        AE_MODEL_PATH=str(c.path(ROOT, b['parent']['model']['autoencoder_weights'])),
        TORCH_HOME=str(ROOT/'models/torch'), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    dist.init_process_group('nccl', device_id=torch.device('cuda', local), timeout=timedelta(minutes=30))
    train(output, b, rank, local)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
