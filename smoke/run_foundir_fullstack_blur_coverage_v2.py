#!/usr/bin/env python3
"""Fixed-compute Blur data-coverage control; training data only, official150 sole gate."""
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

OUT = ROOT / 'checkpoints/foundir_blur_delta_coverage_v2'
SOURCES = ['smoke/run_foundir_fullstack_blur_coverage_v2.py',
           'smoke/run_foundir_fullstack_blur_coverage_v2_gpu.sh', 'smoke/blur_delta_state_v1.py',
           'lowlight_restoration/blur_delta_velocity_v1.py', 'lowlight_restoration/lora.py',
           'lowlight_restoration/flux2_endpoint_precision.py', 'lowlight_restoration/rain_flow_adaptation.py',
           'smoke/rain_hopper_memory_v1.py', 'smoke/rain_native50_continue_state_v1.py',
           'smoke/existing_flux_replay_contract_v1.py',
           'third_party/flux2/src/flux2/model.py', 'third_party/flux2/src/flux2/sampling.py',
           'third_party/flux2/src/flux2/autoencoder.py', 'third_party/flux2/src/flux2/util.py']
RECIPE = {'version': 2, 'task': 'blur', 'train': 1024, 'development': 0, 'crop': 512,
          'qualification_steps': 2, 'train_steps': 200, 'global_batch': 8,
          'learning_rate': 1e-5, 'weight_decay': 0., 'clip_grad': 1., 'seed': 60419,
          'loss': '10*MSE_RGB01 + .1*LPIPS_pyiqa_default_continuous',
          'inference': 'LQ + decode(zLQ - (v_train(zLQ,t1)-v_frozen(zLQ,t1))) - decode(zLQ)',
          'native_CFG': 1., 'reference_time': 10, 'model_calls_per_image': 2,
          'vae': 'FP32_native_conv_query_chunk256', 'trainable': 'original_50_FLUX_LoRA_sites_FP32',
          'initialization': 'frozen_Blur_step10000_copy; fresh_Adam; not_seeded_by_prior200',
          'world': 2, 'node': 3, 'sample_order': 'epoch_shuffle_without_replacement; fixed_crops',
          'source_selection': '1024 distinct training-only1080p groups; SHA order; no severity/score filtering',
          'checkpoint': 'every_complete_step_Adam_all_rank_RNG_sampler; same_world_only',
          'evaluation': 'fixed_final_step200; complete_official150_only; no_checkpoint_search',
          'official_pixels_in_training': False, 'automatic_official_after_engineering_checks': True,
          'cumulative_GPU_hour_cap': None,
          'failure_stop': 'nonfinite/zero_gradient/hash_mismatch/OOM; no_automatic_retry'}
MANIFEST = ROOT/'manifests/foundir_blur_heavy_training_v1.csv'
MANIFEST_SHA = '072dcca8f80d09713fbc43e68ed1b9cb19cd20636ce4b3c70bdcc809e51b6212'
SELECTION_SEED = 'blur_delta_coverage_v2_20261106'
OLD_RECEIPT = ROOT/'checkpoints/foundir_blur_delta_velocity_v1/receipt.json'
OLD_QUALIFICATION = ROOT/'reports/foundir_blur_delta_velocity_v1_qualification_2026-11-06.json'

PATTERNS = ['double_blocks.*.img_attn.qkv', 'double_blocks.*.img_attn.proj',
            'single_blocks.*.linear1', 'single_blocks.*.linear2']
need = parent.need


def report(phase):
    return ROOT / f'reports/foundir_blur_delta_coverage_v2_{phase}_2026-11-06.json'


def select_rows(rows, blocked, protected, count=1024):
    import hashlib
    groups = {}
    for row in rows:
        if row['split'] != 'train' or row['category'] != '01Blur' or row['composition_size'] != '1':
            continue
        if (row['width'], row['height']) != ('1920', '1080') or row['source_group'] in blocked:
            continue
        if any(row[k+'_sha256'] in protected for k in ('lq','gt')):
            continue
        need(all(row[k+'_path'].startswith('data/foundir/MillionIRData/Train/01Blur/') for k in ('lq','gt')), 'nontraining path')
        groups.setdefault(row['source_group'], []).append(row)
    def digest(text): return hashlib.sha256((SELECTION_SEED+text).encode()).hexdigest()
    need(len(groups) >= count and count % 8 == 0, 'insufficient distinct training sources')
    result=[]
    for group in sorted(groups, key=digest)[:count]:
        row=min(groups[group], key=lambda row:digest(row['record_id']))
        rng=random.Random(int(digest('crop:'+row['record_id']),16))
        result.append({'role':'train','row':row,'crop_xy':[rng.randrange(1920-512+1),rng.randrange(1080-512+1)]})
    need(len({i['row']['gt_sha256'] for i in result})==count, 'duplicate exact GT across source groups')
    return result


def prepare():
    import csv
    from concurrent.futures import ThreadPoolExecutor
    from smoke import run_foundir_fullstack_blur_delta_v1 as previous
    start=time.monotonic(); old=previous.bound(); screen=read(parent.SCREEN); official=read(parent.PARENT)
    need(sha(MANIFEST)==MANIFEST_SHA, 'training manifest changed')
    need(sha(parent.PARENT)==screen['official_receipt_sha256'], 'source exclusion parent changed')
    protected={r[k] for r in official['rows'] for k in ('lq_sha256','gt_sha256')}
    rows=select_rows(list(csv.DictReader(MANIFEST.open())),set(screen['blocked_source_groups']),protected)
    def check(item):
        for side in ('lq','gt'):
            array=parent.crop(item,side)
            need(array.shape==(512,512,3), 'training crop dimensions changed')
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(check, rows))
    for name,digest in old['assets'].items(): need(sha(ROOT/name)==digest, 'original model/prompt changed')
    from smoke.official_vlm_v4_metric_identity_fix import checked_metric_models
    checked_metric_models('cpu', old['metric_identity'])
    qualification=read(OLD_QUALIFICATION)
    need(qualification['engineering_pass'] is True and qualification['receipt_sha256']==sha(OLD_RECEIPT), 'prior CUDA implementation not qualified')
    need(shutil.disk_usage(ROOT).free>120*1024**3, 'need120GiB; preserve all old assets')
    value={k:old[k] for k in ('assets','model','lora','prompt','metric_identity')}
    value.update(recipe=RECIPE, rows=rows, manifest_sha256=MANIFEST_SHA,
      source_screen_sha256=sha(parent.SCREEN), official_parent_receipt_sha256=sha(parent.PARENT),
      prior_receipt_sha256=sha(OLD_RECEIPT), prior_qualification_sha256=sha(OLD_QUALIFICATION),
      sources={n:sha(ROOT/n) for n in sorted(set(SOURCES)|set(old['sources']))},
      inherited_CUDA_qualification=True, new_data_CUDA_validation='first8 initialization plus every step finite gradient',
      historical_exposure_not_erased=True, semantic_disjointness_guaranteed=False)
    write(OUT/'receipt.json',value)
    result={'status':'CPU_preflight_pass','seconds':time.monotonic()-start,'exit_code':0,'GPU_hours':0,
      'train_sources':1024,'development_sources':0,'all2048_training_files_SHA_and_crop_checked':True,
      'official_GT_pixels_decoded':False,'receipt_sha256':sha(OUT/'receipt.json'), 'recipe':RECIPE,
      'scope_change':'data population1080p and source diversity only; same200steps and1600training examples',
      'inherited_prior_CUDA_engineering_pass':True,'new_training_started':False,'official_Blur_pass':False}
    write(report('preflight'),result)
    print('BLUR_COVERAGE_PREFLIGHT_PASS '+json.dumps(result),flush=True)


def bound():
    b=read(OUT/'receipt.json')
    need(b['recipe']==RECIPE and sha(MANIFEST)==b['manifest_sha256']==MANIFEST_SHA, 'recipe/manifest changed')
    need(sha(parent.SCREEN)==b['source_screen_sha256'] and sha(parent.PARENT)==b['official_parent_receipt_sha256'], 'source screen/official identity changed')
    need(sha(OLD_RECEIPT)==b['prior_receipt_sha256'] and sha(OLD_QUALIFICATION)==b['prior_qualification_sha256'], 'prior implementation qualification changed')
    for name,digest in b['sources'].items(): need(sha(ROOT/name)==digest, 'bound source changed: '+name)
    import csv
    official=read(parent.PARENT); screen=read(parent.SCREEN)
    protected={r[k] for r in official['rows'] for k in ('lq_sha256','gt_sha256')}
    expected=select_rows(list(csv.DictReader(MANIFEST.open())), set(screen['blocked_source_groups']), protected)
    need(b['rows']==expected, 'deterministic training population changed')
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
        need(b['inherited_CUDA_qualification'] is True and read(OLD_QUALIFICATION)['engineering_pass'] is True, 'prior CUDA implementation qualification missing')
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


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase', choices=['prepare','cache','train']);args=parser.parse_args()
    {'prepare':prepare,'cache':cache_phase,'train':train}[args.phase]()
