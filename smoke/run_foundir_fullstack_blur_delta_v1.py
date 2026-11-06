#!/usr/bin/env python3
"""Blur-only LQ-preserving delta-velocity pilot. Never reads official pixels."""
import argparse
import gc
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'third_party/flux2/src'))
from smoke.existing_flux_replay_contract_v1 import read, sha, write_new as write
from smoke import run_foundir_fullstack_frequency_repair_v1 as parent
from smoke import blur_delta_state_v1 as state
from lowlight_restoration import blur_delta_velocity_v1 as core

OUT = ROOT / 'checkpoints/foundir_blur_delta_velocity_v1'
SOURCES = ['smoke/run_foundir_fullstack_blur_delta_v1.py',
           'smoke/run_foundir_fullstack_blur_delta_v1_gpu.sh', 'smoke/blur_delta_state_v1.py',
           'lowlight_restoration/blur_delta_velocity_v1.py', 'lowlight_restoration/lora.py',
           'lowlight_restoration/flux2_endpoint_precision.py', 'lowlight_restoration/rain_flow_adaptation.py',
           'smoke/rain_hopper_memory_v1.py', 'smoke/rain_native50_continue_state_v1.py',
           'smoke/existing_flux_replay_contract_v1.py',
           'third_party/flux2/src/flux2/model.py', 'third_party/flux2/src/flux2/sampling.py',
           'third_party/flux2/src/flux2/autoencoder.py', 'third_party/flux2/src/flux2/util.py']
RECIPE = {'version': 1, 'task': 'blur', 'train': 128, 'development': 32, 'crop': 512,
          'qualification_steps': 2, 'train_steps': 200, 'global_batch': 8,
          'learning_rate': 1e-5, 'weight_decay': 0., 'clip_grad': 1., 'seed': 60419,
          'loss': '10*MSE_RGB01 + .1*LPIPS_pyiqa_default_continuous',
          'inference': 'LQ + decode(zLQ - (v_train(zLQ,t1)-v_frozen(zLQ,t1))) - decode(zLQ)',
          'native_CFG': 1., 'reference_time': 10, 'model_calls_per_image': 2,
          'vae': 'FP32_native_conv_query_chunk256', 'trainable': 'original_50_FLUX_LoRA_sites_FP32',
          'initialization': 'frozen_Blur_step10000_copy; fresh_Adam; pilot_not_seeded_by_qualification',
          'world': 2, 'node': 3, 'sample_order': 'epoch_shuffle_without_replacement; fixed_crops',
          'checkpoint': 'every_complete_step_Adam_all_rank_RNG_sampler; same_world_only',
          'evaluation': 'fixed_final_step200; all128train_and32development; no_checkpoint_search',
          'official_pixels': False, 'automatic_official': False, 'cumulative_GPU_hour_cap': None,
          'development_gate': 'PSNR>=25.9 and LPIPS<=.18; both better than old50step; worst_PSNR_loss<=1dB',
          'failure_stop': 'nonfinite/zero_gradient/hash_mismatch/OOM; no_automatic_retry'}
PATTERNS = ['double_blocks.*.img_attn.qkv', 'double_blocks.*.img_attn.proj',
            'single_blocks.*.linear1', 'single_blocks.*.linear2']
need = parent.need


def report(phase):
    return ROOT / f'reports/foundir_blur_delta_velocity_v1_{phase}_2026-11-06.json'


def prepare():
    begin = time.monotonic()
    b = parent.bound('blur')
    p = read(parent.PARENT)['original']['parent']
    prompt = p['four_prompts']['tensors']['blur']
    assets = {p['model'][k]: p['model'][k+'_sha256'] for k in ('flow_weights', 'autoencoder_weights')}
    assets[p['experts']['blur']['checkpoint']['path']] = p['experts']['blur']['checkpoint']['sha256']
    prompt_path = str(Path(p['config']['four_parent_run']) / prompt['path'])
    assets[prompt_path] = prompt['sha256']
    for name, digest in assets.items():
        need(sha(ROOT/name) == digest, 'asset changed: '+name)
    parent.verify_cache('blur', b)
    for item in b['rows']:
        for field in ('lq', 'gt'):
            a = parent.crop(item, field)
            need(a.shape == (512, 512, 3), 'crop mismatch')
    from smoke.official_vlm_v4_metric_identity_fix import checked_metric_models
    checked_metric_models('cpu', b['metric_identity'])
    need(shutil.disk_usage(ROOT).free > 110*1024**3, 'need 110GiB; do not delete prior assets')
    value = {'recipe': RECIPE, 'parent_receipt_sha256': sha(parent.out('blur')/'receipt.json'),
             'rows': b['rows'], 'assets': assets, 'metric_identity': b['metric_identity'],
             'model': p['model'], 'lora': p['experts']['blur']['checkpoint'], 'prompt': prompt_path,
             'sources': {name: sha(ROOT/name) for name in sorted(set(SOURCES) | set(b['sources']))},
             'historical_exposure_not_erased': True, 'development_is_not_independent_blind_test': True}
    write(OUT/'receipt.json', value)
    result = {'status': 'CPU_preflight_pass', 'seconds': time.monotonic()-begin, 'exit_code': 0,
              'GPU_hours': 0, 'CUDA_qualification': 'pending', 'official_pass': False,
              'receipt_sha256': sha(OUT/'receipt.json'), 'train': 128, 'development': 32,
              'cache_inputs_GT_free': True, 'recipe': RECIPE}
    if not report('preflight').exists():
        write(report('preflight'), result)
    print('BLUR_DELTA_CPU_PREFLIGHT_PASS '+json.dumps(result), flush=True)


def bound():
    b = read(OUT/'receipt.json')
    need(b['recipe'] == RECIPE and sha(parent.out('blur')/'receipt.json') == b['parent_receipt_sha256'],
         'recipe/parent changed')
    for name, digest in b['sources'].items():
        need(sha(ROOT/name) == digest, 'bound source changed: '+name)
    need(parent.bound('blur')['rows'] == b['rows'], 'data scope changed')
    return b


def setup(b):
    import torch
    for name, digest in b['assets'].items():
        need(sha(ROOT/name) == digest, 'asset changed: '+name)
    rank, device = parent.hardware()
    random.seed(RECIPE['seed']+rank)
    torch.manual_seed(RECIPE['seed']+rank)
    torch.cuda.manual_seed(RECIPE['seed']+rank)
    os.environ['KLEIN_4B_BASE_MODEL_PATH'] = str(ROOT/b['model']['flow_weights'])
    os.environ['AE_MODEL_PATH'] = str(ROOT/b['model']['autoencoder_weights'])
    return rank, device


def load_model(b, device):
    import torch
    from flux2.util import load_flow_model
    from lowlight_restoration.lora import inject_lora, load_lora_checkpoint
    from lowlight_restoration.flux2_endpoint_precision import promote_lora_to_fp32
    from lowlight_restoration.rain_flow_adaptation import enable_native_checkpointing
    model = load_flow_model('flux.2-klein-base-4b', device=device).eval()
    names = inject_lora(model, PATTERNS, 32, 32., 0.)
    parameters = promote_lora_to_fp32(model)
    need(len(names) == 50 and len(parameters) == 100 and sum(p.numel() for p in parameters) == 32440320,
         'LoRA schema changed')
    load_lora_checkpoint(model, ROOT/b['lora']['path'])
    enable_native_checkpointing(model)
    return model


def load_vae_prompt(b, device):
    from flux2.util import load_ae
    from safetensors.torch import load_file
    from smoke.rain_hopper_memory_v1 import configure_vae
    from smoke.expanded_vlm_h100_runtime_v4 import bounded_vae
    ae = bounded_vae(configure_vae(load_ae('flux.2-klein-base-4b', device=device).eval().requires_grad_(False)))
    return ae, load_file(str(ROOT/b['prompt']), device=str(device))


def rgb(item, field, device):
    import torch
    return torch.from_numpy(parent.crop(item, field)).permute(2, 0, 1).unsqueeze(0).to(device).float()/127.5-1


def cached(item, device):
    import torch
    from safetensors.torch import load_file
    marker = OUT/'cache'/(parent.key(item)+'.json')
    if not marker.exists():
        return None
    r = read(marker)
    need(r['item'] == item and r['receipt_sha256'] == sha(OUT/'receipt.json') and r['GT_input'] is False,
         'cache identity differs')
    file = OUT/r['tensor']
    need(file.resolve().parent == (OUT/'tensors').resolve() and sha(file) == r['sha256'], 'cache SHA/path changed')
    data = load_file(str(file), device=str(device))
    need(set(data) == {'lq', 'latent', 'reference_velocity', 'reference_decode'}, 'cache schema/GT contamination')
    need(all(t.dtype == torch.float32 and torch.isfinite(t).all() for t in data.values()), 'nonfinite/nonFP32 cache')
    need(data['lq'].shape == data['reference_decode'].shape == (1, 3, 512, 512)
         and data['latent'].shape == data['reference_velocity'].shape, 'cache geometry differs')
    return data


def cache_items(b, model, ae, prompt, items, rank, device):
    import torch
    from safetensors.torch import save_file
    from flux2.sampling import batched_prc_img
    for item in items[rank::2]:
        if cached(item, device) is not None:
            continue
        start = time.monotonic()
        lq = rgb(item, 'lq', device)
        with torch.no_grad():
            latent = ae.encode(lq).float()
            velocity = core.native_velocity(model, latent, prompt, batched_prc_img)
            decoded = ae.decode(latent).float()
        data = {'lq': lq, 'latent': latent, 'reference_velocity': velocity, 'reference_decode': decoded}
        need(all(torch.isfinite(t).all() for t in data.values()), 'nonfinite reference cache')
        directory = OUT/'tensors'
        directory.mkdir(parents=True, exist_ok=True)
        file = directory/(parent.key(item)+f'_{time.time_ns()}.safetensors')
        save_file({k: v.cpu().contiguous() for k, v in data.items()}, str(file))
        with file.open('rb') as handle:
            os.fsync(handle.fileno())
        write(OUT/'cache'/(parent.key(item)+'.json'), {'item': item, 'rank': rank, 'GT_input': False,
              'receipt_sha256': sha(OUT/'receipt.json'), 'tensor': str(file.relative_to(OUT)), 'sha256': sha(file)})
        print(f'BLUR_DELTA_CACHE rank={rank} key={parent.key(item)} seconds={time.monotonic()-start:.2f}', flush=True)


def prediction(model, ae, prompt, data):
    import torch
    from torch.utils.checkpoint import checkpoint
    from flux2.sampling import batched_prc_img
    velocity = core.native_velocity(model, data['latent'], prompt, batched_prc_img)
    latent = core.corrected_latent(data['latent'], velocity, data['reference_velocity'])
    # FP32 VAE in forward and recomputation; no BF16 decode path mismatch.
    decoded = checkpoint(ae.decode, latent, use_reentrant=False) if torch.is_grad_enabled() else ae.decode(latent)
    return core.corrected_rgb(data['lq'], decoded, data['reference_decode'])


def cache_phase():
    b = bound()
    rank, device = setup(b)
    model = load_model(b, device)
    ae, prompt = load_vae_prompt(b, device)
    cache_items(b, model, ae, prompt, b['rows'], rank, device)
    write(OUT/f'cache_done{rank}.json', {'receipt_sha256': sha(OUT/'receipt.json'),
          'keys': [parent.key(i) for i in b['rows'][rank::2]], 'rank': rank})


def optimizer_for(model):
    import torch
    return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                             lr=RECIPE['learning_rate'], weight_decay=0., fused=False)


def train(qualification=False):
    import torch
    import torch.distributed as dist
    from datetime import timedelta
    from lowlight_restoration.flux2_endpoint_precision import assert_fp32_optimizer
    from smoke.official_vlm_v4_metric_identity_fix import checked_metric_models
    begin = time.monotonic()
    b = bound()
    rank, device = setup(b)
    dist.init_process_group('nccl', timeout=timedelta(minutes=20))
    need(dist.get_world_size() == 2 and dist.get_rank() == rank, 'single-node two-rank training only')
    phase = 'qualification' if qualification else 'pilot'
    output = OUT/phase
    receipt_sha = sha(OUT/'receipt.json')
    if not qualification:
        need(read(report('qualification'))['engineering_pass'] is True, 'qualification not passed')
        need(read(report('qualification'))['receipt_sha256'] == receipt_sha, 'qualification binding mismatch')
        for local in range(2):
            need(read(OUT/f'cache_done{local}.json')['receipt_sha256'] == receipt_sha, 'full cache incomplete')
    model = load_model(b, device)
    ae, prompt = load_vae_prompt(b, device)
    metrics, _ = checked_metric_models(device, b['metric_identity'])
    metrics['lpips'].as_loss = True
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = optimizer_for(model)
    train_rows = [i for i in b['rows'] if i['role'] == 'train']
    if qualification:
        train_rows = train_rows[:8]
        cache_items(b, model, ae, prompt, train_rows, rank, device)
        dist.barrier()
    identity = []
    if not list((output/'steps').glob('step_*.json')):
        for item in train_rows[:8][rank::2]:
            data = cached(item, device)
            model.eval()
            with torch.no_grad():
                pred = prediction(model, ae, prompt, data)
            maxdiff = int((core.uint8_output(pred).int()-core.uint8_output(data['lq']).int()).abs().max())
            need(maxdiff == 0, 'initial delta does not reproduce LQ uint8')
            # Check checkpointed train-mode computation as well as eval-mode.
            model.train()
            pred_train = prediction(model, ae, prompt, data)
            need(torch.equal(core.uint8_output(pred), core.uint8_output(pred_train)), 'train/eval initial mismatch')
            identity.append({'key': parent.key(item), 'uint8_max_difference': maxdiff,
                             'float_max_difference': float((pred-data['lq']).abs().max()),
                             'train_eval_float_max_difference': float((pred-pred_train.detach()).abs().max())})
            del pred_train, pred, data
        write(output/f'identity_rank{rank}.json', {'receipt_sha256': receipt_sha, 'records': identity})
    saved = state.load(output, receipt_sha, 2)
    steps = RECIPE['qualification_steps' if qualification else 'train_steps']
    start, history = (0, []) if saved is None else (saved['step'], saved['history'])
    if saved is not None:
        state.restore(model, optimizer, saved, rank, device)
    del saved

    def save(step):
        rngs, hashes = [None]*2, [None]*2
        dist.all_gather_object(rngs, state.rng_state(device))
        digest = state.parameter_hash(state.trainable_state(model))
        dist.all_gather_object(hashes, digest)
        need(len(set(hashes)) == 1, 'rank parameters diverged')
        if rank == 0:
            record = state.save(output, model, optimizer, step=step, world=2, rank_rng=rngs,
                                receipt_sha=receipt_sha, history=history)
            print(f'BLUR_DELTA_SAVED phase={phase} step={step} sha={record["sha256"]}', flush=True)
        dist.barrier()

    if not list((output/'steps').glob('step_*.json')):
        save(0)
    for step in range(start, steps):
        begun = time.monotonic()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        totals = torch.zeros(3, device=device, dtype=torch.float64)
        for index in core.sample_indices(step, rank, count=len(train_rows)):
            item = train_rows[index]
            data = cached(item, device)
            need(data is not None and item['role'] == 'train', 'only training cache may update parameters')
            target = rgb(item, 'gt', device)
            pred = prediction(model, ae, prompt, data)
            loss, terms = core.objective(pred, target, metrics['lpips'])
            need(bool(torch.isfinite(loss)) and loss.requires_grad, 'invalid/detached loss')
            if qualification and step == 0:
                a = torch.autograd.grad(10*terms['mse'], pred, retain_graph=True)[0]
                z = torch.autograd.grad(.1*terms['lpips'], pred, retain_graph=True)[0]
                need(bool(torch.isfinite(a).all() and torch.isfinite(z).all())
                     and float(a.norm()) > 0 and float(z.norm()) > 0, 'loss component has no RGB gradient')
                print('BLUR_DELTA_LOSS_GRAD '+json.dumps({'rank': rank, 'key': parent.key(item),
                      'MSE_RGB_gradient_norm': float(a.norm()), 'LPIPS_RGB_gradient_norm': float(z.norm())}), flush=True)
                del a, z
            (loss/8).backward()
            totals += torch.tensor([float(loss.detach()), float(terms['mse'].detach()),
                                    float(terms['lpips'].detach())], device=device, dtype=torch.float64)/8
            del data, target, pred, loss, terms
        # Loss / global batch and SUM across ranks gives the exact global mean.
        for parameter in parameters:
            need(parameter.grad is not None, 'trainable parameter gradient missing')
            dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
        need(float(norm) > 0, 'zero parameter gradient')
        optimizer.step()
        assert_fp32_optimizer(optimizer)
        optimizer.zero_grad(set_to_none=True)
        need(all(torch.isfinite(p).all() for p in parameters), 'nonfinite parameters')
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        speed = torch.tensor([time.monotonic()-begun, torch.cuda.max_memory_allocated(device)/1024**3], device=device)
        dist.all_reduce(speed, op=dist.ReduceOp.MAX)
        log = {'step': step+1, 'loss': float(totals[0]), 'mse': float(totals[1]), 'lpips': float(totals[2]),
               'gradient_norm': float(norm), 'max_rank_seconds': float(speed[0]),
               'peak_allocated_GiB': float(speed[1]), 'next_global_slot': (step+1)*8}
        history.append(log)
        if rank == 0:
            print('BLUR_DELTA_TRAIN '+json.dumps(log), flush=True)
        save(step+1)
        if qualification and step == 0:
            # Actual GPU roundtrip: restore Adam, parameters and per-rank RNG before step2.
            value = state.load(output, receipt_sha, 2)
            previous = state.parameter_hash(state.trainable_state(model))
            state.restore(model, optimizer, value, rank, device)
            need(state.parameter_hash(state.trainable_state(model)) == previous, 'GPU checkpoint restore changed model')
            for key, entry in optimizer.state_dict()['state'].items():
                for name, tensor in entry.items():
                    need(torch.equal(tensor.cpu(), value['optimizer']['state'][key][name]), 'GPU restore changed Adam')
            del value
    write(output/f'train_done{rank}.json', {'receipt_sha256': receipt_sha, 'rank': rank, 'step': steps,
          'seconds': time.monotonic()-begin, 'peak_allocated_GiB': torch.cuda.max_memory_allocated(device)/1024**3,
          'parameter_hash': state.parameter_hash(state.trainable_state(model)), 'CUDA': torch.version.cuda,
          'torch': str(torch.__version__), 'GPU': torch.cuda.get_device_name(device), 'exit_code': 0})
    dist.barrier()
    print(f'BLUR_DELTA_TRAIN_END phase={phase} rank={rank} status=0', flush=True)
    dist.destroy_process_group()


def verify_training(phase):
    b = bound()
    receipt = sha(OUT/'receipt.json')
    output = OUT/phase
    steps = RECIPE['qualification_steps' if phase == 'qualification' else 'train_steps']
    markers = [read(output/f'train_done{r}.json') for r in range(2)]
    need(all(m['step'] == steps and m['exit_code'] == 0 and m['receipt_sha256'] == receipt for m in markers),
         'training ranks incomplete')
    value = state.load(output, receipt, 2)
    need(value['step'] == steps and all(m['parameter_hash'] == value['parameter_hash'] for m in markers),
         'rank/final checkpoint mismatch')
    initial = read(output/'steps/step_000000.json')
    need(initial['parameter_hash'] != value['parameter_hash'], 'parameters never changed')
    for step in range(steps+1):
        r = read(output/'steps'/f'step_{step:06d}.json')
        need(r['receipt_sha256'] == receipt and sha(output/r['file']) == r['sha256'], 'step history corrupt')
    identity = [record for rank in range(2) for record in read(output/f'identity_rank{rank}.json')['records']]
    need(len(identity) == 8 and len({r['key'] for r in identity}) == 8
         and all(r['uint8_max_difference'] == 0 for r in identity), 'initialization identity incomplete')
    result = {'engineering_pass': True, 'scientific_pass': False, 'official_pass': False,
              'receipt_sha256': receipt, 'identity': identity, 'ranks': markers, 'history': value['history'],
              'initial_parameter_hash': initial['parameter_hash'], 'final_parameter_hash': value['parameter_hash'],
              'training_data_only': True, 'GT_used_at_inference': False}
    if phase == 'qualification':
        write(report('qualification'), result)
    return result


def evaluate():
    import torch
    from smoke.official_vlm_v4_metric_identity_fix import checked_metric_models, producer
    b = bound()
    rank, device = setup(b)
    model = load_model(b, device)
    saved = state.load(OUT/'pilot', sha(OUT/'receipt.json'), 2)
    need(saved is not None and saved['step'] == RECIPE['train_steps'], 'fixed final200 required')
    optimizer = optimizer_for(model)
    state.restore(model, optimizer, saved, rank, device)
    final_hash = saved['parameter_hash']
    del saved, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    model.eval()
    ae, prompt = load_vae_prompt(b, device)
    metrics, _ = checked_metric_models(device, b['metric_identity'])
    for item in b['rows'][rank::2]:
        marker = OUT/'scores'/(parent.key(item)+'.json')
        if marker.exists():
            old = read(marker)
            need(old['parameter_hash'] == final_hash and old['item'] == item
                 and sha(OUT/old['output']['path']) == old['output']['sha256'], 'score identity changed')
            continue
        data = cached(item, device)
        with torch.no_grad():
            result = core.uint8_output(prediction(model, ae, prompt, data))
        array = result[0].permute(1, 2, 0).cpu().numpy()
        file = OUT/'png'/(parent.key(item)+'.png')
        producer.png_once(file, array)
        # Freeze candidate PNG before reading GT. No fallback or GT-based selection.
        target = (rgb(item, 'gt', device)+1)*.5
        _, old = parent.cached('blur', item, b['parent_receipt_sha256'])
        old_tensor = torch.from_numpy(old['baseline']).permute(2, 0, 1).unsqueeze(0).to(device).float()/255
        values = {}
        for name, tensor in [('lq', (data['lq']+1)*.5), ('old50step', old_tensor), ('candidate', result.float()/255)]:
            with torch.no_grad():
                values[name] = {key: float(metric(tensor, target).mean()) for key, metric in metrics.items()}
        need(all(math.isfinite(v) for scores in values.values() for v in scores.values()), 'nonfinite metrics')
        write(marker, {'item': item, 'parameter_hash': final_hash, 'receipt_sha256': sha(OUT/'receipt.json'),
                      'metrics': values, 'GT_input': False, 'training_fit_only': item['role'] == 'train',
                      'output': {'path': str(file.relative_to(OUT)), 'sha256': sha(file)}})
        print(f'BLUR_DELTA_SCORE rank={rank} key={parent.key(item)} metrics={values["candidate"]}', flush=True)
    write(OUT/f'evaluate_done{rank}.json', {'receipt_sha256': sha(OUT/'receipt.json'),
          'parameter_hash': final_hash, 'keys': [parent.key(i) for i in b['rows'][rank::2]], 'rank': rank})


def finish():
    start = time.monotonic()
    result = verify_training('pilot')
    b = bound()
    for rank in range(2):
        r = read(OUT/f'evaluate_done{rank}.json')
        need(r['receipt_sha256'] == sha(OUT/'receipt.json') and r['parameter_hash'] == result['final_parameter_hash']
             and r['keys'] == [parent.key(i) for i in b['rows'][rank::2]], 'evaluation shard incomplete')
    rows = []
    for item in b['rows']:
        row = read(OUT/'scores'/(parent.key(item)+'.json'))
        need(row['item'] == item and row['parameter_hash'] == result['final_parameter_hash']
             and row['receipt_sha256'] == sha(OUT/'receipt.json') and row['GT_input'] is False
             and sha(OUT/row['output']['path']) == row['output']['sha256'], 'score/output changed')
        rows.append(row)
    summary = {}
    for role in ('train', 'development'):
        subset = [r for r in rows if r['item']['role'] == role]
        means = {name: {metric: sum(r['metrics'][name][metric] for r in subset)/len(subset)
                      for metric in ('psnr', 'lpips', 'ssim')} for name in ('lq', 'old50step', 'candidate')}
        means['count'] = len(subset)
        means['worst_PSNR_loss_vs_old'] = max(r['metrics']['old50step']['psnr']-r['metrics']['candidate']['psnr'] for r in subset)
        summary[role] = means
    d = summary['development']
    gate = (d['candidate']['psnr'] >= 25.9 and d['candidate']['lpips'] <= .18
            and d['candidate']['psnr'] > d['old50step']['psnr']
            and d['candidate']['lpips'] < d['old50step']['lpips'] and d['worst_PSNR_loss_vs_old'] <= 1.)
    result.update(summary=summary, records=rows, development_gate=gate,
                  seconds=time.monotonic()-start, exit_code=0, GPU_hours=0, official_evaluated=False)
    write(report('result'), result)
    print('BLUR_DELTA_RESULT_VERIFIED '+json.dumps({'summary': summary, 'development_gate': gate}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=['prepare', 'cache', 'qualify', 'verify-qualification', 'train', 'evaluate', 'finish'])
    a = p.parse_args()
    if a.phase == 'prepare': prepare()
    elif a.phase == 'cache': cache_phase()
    elif a.phase == 'qualify': train(True)
    elif a.phase == 'verify-qualification':
        print('BLUR_DELTA_QUALIFICATION_VERIFIED '+json.dumps(verify_training('qualification')), flush=True)
    elif a.phase == 'train': train()
    elif a.phase == 'evaluate': evaluate()
    else: finish()


if __name__ == '__main__':
    main()
