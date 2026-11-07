#!/usr/bin/env python3
"""Final Blur delta checkpoint, complete official150; no other evaluation set."""
import argparse
import gc
import io
import json
import math
import os
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'third_party/flux2/src'))
os.environ['FOUNDIR_OFFICIAL_NNODES'] = '4'  # Frozen parent identity only.
from smoke import train_foundir_blur_fullgrid_v1 as train
from smoke import official_vlm_v4_metric_identity_fix as fix
from smoke.existing_flux_replay_contract_v1 import read, sha, write_new as write
from lowlight_restoration import blur_delta_velocity_v1 as core
x, need = fix.x, train.need
OUT = ROOT/'checkpoints/foundir_blur_fullgrid_official_v1'
PARENT = ROOT/x.OUTPUT
REPORT = ROOT/'reports/foundir_blur_fullgrid_official_v1_result_2026-11-07.json'
SOURCES = ['smoke/evaluate_foundir_blur_fullgrid_official_v1.py',
           'smoke/run_foundir_fullstack_blur_fullgrid_official_v1.sh',
           'smoke/official_vlm_router_v4_contract.py', 'smoke/evaluate_foundir_official_vlm_router_v4.py']
RECIPE = {'population': 'complete_official_Blur150', 'selected_step': 9472, 'world': 4, 'nodes': [1, 3],
          'inference': train.RECIPE['inference'], 'model_calls_per_tile': 2,
          'tile': 512, 'overlap': 128, 'stitch': x.RECIPE['stitch'], 'padding': x.RECIPE['padding'],
          'metrics': x.RECIPE['metrics'], 'route': 'reuse_frozen_label_free_conditional_v4_decisions',
          'new_training': False, 'development_evaluation': False,
          'gate': {'psnr_min': 25.9, 'lpips_max': .18},
          'GT_input_or_selection': False, 'independent_blind_test': False,
          'official_used_for_iterative_development': True, 'cumulative_GPU_hours_cap': None}


def assigned(rows, rank):
    if rank not in range(4) or len(rows) != 150 or len({i['row']['record_id'] for i in rows}) != 150:
        raise ValueError('complete150 and four ranks required')
    return rows[rank::4]


def metric_summary(records):
    need(len(records) == 150 and len({r['item']['row']['record_id'] for r in records}) == 150,
         'must aggregate all150 unique records')
    result = {name: {k: math.fsum(r['metrics'][name][k] for r in records)/150
                    for k in ('psnr', 'lpips', 'ssim')} for name in ('lq', 'old50step', 'candidate')}
    c = result['candidate']
    return {'means': result, 'gate_checks': {'psnr': c['psnr'] >= 25.9, 'lpips': c['lpips'] <= .18},
            'official_Blur_pass': c['psnr'] >= 25.9 and c['lpips'] <= .18,
            'PSNR_delta_vs_old': c['psnr']-result['old50step']['psnr'],
            'LPIPS_relative_reduction_vs_old_percent': 100*(1-c['lpips']/result['old50step']['lpips']),
            'PSNR_worse_than_LQ': sum(r['metrics']['candidate']['psnr'] < r['metrics']['lq']['psnr'] for r in records),
            'joint_better_than_old': sum(r['metrics']['candidate']['psnr'] > r['metrics']['old50step']['psnr']
                                        and r['metrics']['candidate']['lpips'] < r['metrics']['old50step']['lpips'] for r in records)}


def plan():
    start = time.monotonic()
    b = train.bound(); parent = read(PARENT/'receipt.json')
    x.validate_rows(parent['rows'])
    need(sha(ROOT/x.MANIFEST) == x.MANIFEST_SHA, 'official manifest changed')
    need(b['metric_identity'] == parent['metric_identity'], 'metric identity mismatch')
    rows = []
    for row in parent['rows']:
        if row['category'] != '01Blur':
            continue
        decision_path = PARENT/'decisions'/(row['record_id']+'.json')
        decision = read(decision_path)
        need(decision['record_id'] == row['record_id'] and decision['lq_sha256'] == row['lq_sha256']
             and decision['GT_input'] is False and decision['class_label_input'] is False
             and decision['receipt_sha256'] == sha(PARENT/'receipt.json'), 'invalid cached label-free route')
        logits = decision['decisions']['conditional_v4']['logits']
        anchor = x.ORDER[max(range(5), key=lambda i: logits[i])]
        need(anchor == decision['decisions']['conditional_v4']['anchor'] == 'blur',
             'cannot route using official category; frozen route must select Blur itself')
        old_path = PARENT/'results'/(row['record_id']+'.json'); old = read(old_path)
        need(old['row'] == row and old['receipt_sha256'] == sha(PARENT/'receipt.json')
             and old['decision_sha256'] == sha(decision_path) and old['GT_used_for_deployed_selection'] is False,
             'old official record identity changed')
        rows.append({'row': row, 'decision_sha256': sha(decision_path),
                     'parent_record_sha256': sha(old_path), 'old_metrics': old['metrics']['blur']})
    [assigned(rows, rank) for rank in range(4)]
    value = {'recipe': RECIPE, 'rows': rows, 'train_receipt_sha256': sha(train.OUT/'receipt.json'),
             'parent_receipt_sha256': sha(PARENT/'receipt.json'), 'manifest_sha256': x.MANIFEST_SHA,
             'metric_identity': b['metric_identity'],
             'sources': {n: sha(ROOT/n) for n in sorted(set(SOURCES) | set(b['sources']))}}
    write(OUT/'plan.json', value)
    counts = [sum(len(x.tiles(i['row'])) for i in assigned(rows, r)) for r in range(4)]
    result = {'status': 'official150_metadata_plan_pass', 'records': 150, 'tiles_per_rank': counts,
              'total_model_calls': 2*sum(counts), 'GT_pixels_decoded': False, 'GPU_started': False,
              'seconds': time.monotonic()-start, 'exit_code': 0, 'GPU_hours': 0,
              'plan_sha256': sha(OUT/'plan.json'), 'independent_blind_test': False}
    if not (OUT/'plan_result.json').exists(): write(OUT/'plan_result.json', result)
    print('BLUR_FULLGRID_OFFICIAL_PLAN '+json.dumps(result), flush=True)


def plan_bound():
    value = read(OUT/'plan.json')
    need(value['recipe'] == RECIPE and value['train_receipt_sha256'] == sha(train.OUT/'receipt.json')
         and value['parent_receipt_sha256'] == sha(PARENT/'receipt.json')
         and sha(ROOT/x.MANIFEST) == value['manifest_sha256'], 'official plan identity changed')
    train.bound()
    for n, digest in value['sources'].items(): need(sha(ROOT/n) == digest, 'source changed: '+n)
    for item in value['rows']:
        rid = item['row']['record_id']
        need(sha(PARENT/'decisions'/(rid+'.json')) == item['decision_sha256']
             and sha(PARENT/'results'/(rid+'.json')) == item['parent_record_sha256'], 'parent decision/result changed')
    return value


def prepare():
    from safetensors.torch import save_file, load_file
    start = time.monotonic(); v = plan_bound(); b = train.bound()
    queue = ROOT/'checkpoints/foundir_fullstack_executor_3x2_h100_v2'
    ends = [read(queue/'jobs'/f'round20_blur_fullgrid_train_v1_node{n}'/f'node{n}.ended.json') for n in (1,3)]
    need(all(e['status'] == 0 for e in ends), 'both training nodes must exit0')
    result = read(train.REPORT)
    need(result['status'] == 'fixed1024_all15_views_four_epochs_training_complete', 'training CPU finish required')
    saved, final = train.state.latest(train.OUT, sha(train.OUT/'receipt.json'))
    need(saved is not None and saved['step'] == 9472, 'only fixed final9472 allowed')
    for rank in range(4):
        w = read(train.OUT/f'worker{rank}.json')
        need(w['step'] == 9472 and w['exit_code'] == 0 and w['parameter_hash'] == saved['parameter_hash'], 'rank incomplete')
    file = OUT/'weights'/('blur_delta9472_'+saved['parameter_hash']+'.safetensors')
    file.parent.mkdir(parents=True, exist_ok=True)
    if not file.exists():
        temporary = file.with_name(file.name+f'.pending.{time.time_ns()}')
        save_file(saved['model'], str(temporary))
        with temporary.open('rb') as handle: os.fsync(handle.fileno())
        os.link(temporary, file); temporary.unlink()
    need(train.parent.state.parameter_hash(load_file(str(file))) == saved['parameter_hash'], 'export changed parameters')
    for item in v['rows']:
        for side in ('lq', 'gt'):
            row = item['row']; need(sha(ROOT/row[side+'_path']) == row[side+'_sha256'], 'official image hash changed')
    fix.checked_metric_models('cpu', v['metric_identity'])
    for name, digest in b['assets'].items(): need(sha(ROOT/name) == digest, 'original asset changed')
    value = {'plan_sha256': sha(OUT/'plan.json'), 'final_checkpoint_receipt': final,
             'parameter_hash': saved['parameter_hash'], 'weight': {'path': str(file.relative_to(ROOT)), 'sha256': sha(file)},
             'model_assets': b['assets'], 'metric_identity': v['metric_identity']}
    write(OUT/'receipt.json', value)
    report = {'status': 'destination_CPU_preflight_pass', 'records': 150, 'final_step': 9472,
              'final_parameter_hash': saved['parameter_hash'], 'final_SHA_Adam_RNG_sampler_checked': True,
              'step1792_to9472_receipts_contiguous': True, 'all_large_checkpoints_rehashed': False,
              'GT_pixels_decoded': False, 'GPU_started': False, 'seconds': time.monotonic()-start,
              'exit_code': 0, 'GPU_hours': 0, 'completed_training_jobs': ends}
    write(ROOT/'reports/foundir_blur_fullgrid_official_v1_preflight_2026-11-07.json', report)
    print('BLUR_FULLGRID_OFFICIAL_PREFLIGHT_PASS '+json.dumps(report), flush=True)


def bound():
    v = plan_bound(); b = read(OUT/'receipt.json')
    need(b['plan_sha256'] == sha(OUT/'plan.json') and sha(ROOT/b['weight']['path']) == b['weight']['sha256'],
         'final evaluation binding changed')
    return v, b


def png(path, digest):
    from PIL import Image
    import numpy as np
    data = path.read_bytes()
    import hashlib
    need(hashlib.sha256(data).hexdigest() == digest, 'PNG changed')
    with Image.open(io.BytesIO(data)) as im: return np.array(im.convert('RGB'), copy=True)


def check_record(record, item, rank):
    need(record['item'] == item and record['rank'] == rank and record['receipt_sha256'] == sha(OUT/'receipt.json')
         and record['GT_input_or_selection'] is False, 'record identity changed')
    need(sha(OUT/record['output']['path']) == record['output']['sha256'], 'full PNG changed')
    need(record['metrics']['old50step'] == item['old_metrics'], 'baseline metrics changed')
    need(set(record['metrics']) == {'lq', 'old50step', 'candidate'} and all(set(m)=={'psnr','lpips','ssim'}
         and all(math.isfinite(a) for a in m.values()) for m in record['metrics'].values()), 'invalid metric schema')


def run():
    import torch
    from lowlight_restoration.lora import load_lora_checkpoint
    from flux2.sampling import batched_prc_img
    start = time.monotonic(); v, b = bound(); training = train.bound()
    local, device = train.setup(training)
    node = int(os.environ['FOUNDIR_FULLSTACK_NODE']); need(node in (1, 3), 'two evaluation nodes required')
    rank = 2*((node-1)//2)+local
    runtime = {'rank': rank, 'node': node, 'torch': str(torch.__version__), 'cuda': torch.version.cuda,
               'GPU': torch.cuda.get_device_name(device), 'receipt_sha256': sha(OUT/'receipt.json'),
               'VAE': train.RECIPE['vae'], 'TF32': False}
    write(OUT/f'runtime{rank}.json', runtime)
    reference = train.load_model(training, device).eval().requires_grad_(False)
    model = train.load_model(training, device).eval()
    load_lora_checkpoint(model, ROOT/b['weight']['path']); model.requires_grad_(False)
    ae, prompt = train.load_vae_prompt(training, device)
    metrics, _ = fix.checked_metric_models(device, b['metric_identity'])
    for number, item in enumerate(assigned(v['rows'], rank)):
        row = item['row']; path = OUT/'records'/(row['record_id']+'.json')
        if path.exists(): check_record(read(path), item, rank); continue
        begun = time.monotonic(); lq = png(ROOT/row['lq_path'], row['lq_sha256'])
        need(lq.shape == (int(row['height']), int(row['width']), 3), 'official LQ dimensions changed')
        images = {}
        for ti, tile in enumerate(x.tiles(row)):
            identity = {'receipt_sha256': sha(OUT/'receipt.json'), 'record_id': row['record_id'],
                        'lq_sha256': row['lq_sha256'], 'tile': tile, 'rank': rank,
                        'runtime_sha256': sha(OUT/f'runtime{rank}.json')}
            marker = OUT/'tiles'/(tile['key']+'.json')
            if marker.exists():
                saved = read(marker)
                need(saved['identity'] == identity and saved['GT_input'] is False and saved['model_calls'] == 2,
                     'tile identity changed')
                image = png(OUT/saved['png']['path'], saved['png']['sha256'])
            else:
                crop, _, _ = x.tile_array(lq, tile); input_tensor = fix.producer.tensor(crop, device)
                with torch.no_grad():
                    latent = ae.encode(input_tensor).float()
                    ref_velocity = core.native_velocity(reference, latent, prompt, batched_prc_img)
                    ref_decode = ae.decode(latent).float()
                    data = {'lq': input_tensor, 'latent': latent, 'reference_velocity': ref_velocity,
                            'reference_decode': ref_decode}
                    predicted = train.prediction(model, ae, prompt, data)
                    need(bool(torch.isfinite(predicted).all()), 'nonfinite official output')
                    image = core.uint8_output(predicted)[0].permute(1, 2, 0).cpu().numpy()
                file = OUT/'tile_png'/(tile['key']+'.png'); fix.producer.png_once(file, image)
                write(marker, {'identity': identity, 'GT_input': False, 'model_calls': 2,
                      'png': {'path': str(file.relative_to(OUT)), 'sha256': sha(file)}})
                del input_tensor, latent, ref_velocity, ref_decode, data, predicted
            images[tile['key']] = image
            if (ti+1) % 5 == 0 or ti+1 == len(x.tiles(row)):
                print(f'BLUR_FULLGRID_OFFICIAL_TILE rank={rank} id={row["record_id"]} tile={ti+1}/{len(x.tiles(row))}', flush=True)
        full = x.stitch(row, images); file = OUT/'restored'/(row['record_id']+'.png')
        fix.producer.png_once(file, full)
        # The sole output is committed before official GT is decoded. No fallback.
        gt = png(ROOT/row['gt_path'], row['gt_sha256']); need(gt.shape == full.shape, 'GT/output dimensions differ')
        def tensor(a): return torch.from_numpy(a.copy()).permute(2,0,1).unsqueeze(0).to(device).float()/255
        target = tensor(gt); values = {'old50step': item['old_metrics']}
        for name, array in [('lq', lq), ('candidate', full)]:
            image = tensor(array)
            with torch.no_grad(): values[name] = {k: float(m(image, target).mean()) for k,m in metrics.items()}
            del image
        del target
        record = {'item': item, 'rank': rank, 'receipt_sha256': sha(OUT/'receipt.json'),
                  'runtime_sha256': sha(OUT/f'runtime{rank}.json'), 'metrics': values,
                  'output': {'path': str(file.relative_to(OUT)), 'sha256': sha(file)},
                  'GT_input_or_selection': False, 'seconds': time.monotonic()-begun}
        check_record(record, item, rank); write(path, record)
        print(f'BLUR_FULLGRID_OFFICIAL_RECORD rank={rank} complete={number+1}/{len(assigned(v["rows"], rank))} id={row["record_id"]} metrics={values["candidate"]}', flush=True)
    write(OUT/f'worker{rank}.json', {'receipt_sha256': sha(OUT/'receipt.json'), 'rank': rank,
          'records': [i['row']['record_id'] for i in assigned(v['rows'], rank)],
          'seconds': time.monotonic()-start, 'peak_allocated_GiB': torch.cuda.max_memory_allocated(device)/1024**3})
    print(f'BLUR_FULLGRID_OFFICIAL_END rank={rank} status=0', flush=True)


def finish():
    start = time.monotonic(); v, b = bound(); records = []; signatures = []
    for rank in range(4):
        marker = read(OUT/f'worker{rank}.json'); runtime = read(OUT/f'runtime{rank}.json')
        need(marker['receipt_sha256'] == sha(OUT/'receipt.json') and marker['rank'] == rank
             and marker['records'] == [i['row']['record_id'] for i in assigned(v['rows'], rank)], 'worker incomplete')
        signatures.append({k:d for k,d in runtime.items() if k not in ('rank','node')})
        for item in assigned(v['rows'], rank):
            row = item['row']; record = read(OUT/'records'/(row['record_id']+'.json'))
            check_record(record, item, rank)
            need(record['runtime_sha256'] == sha(OUT/f'runtime{rank}.json'), 'record runtime mismatch')
            for tile in x.tiles(row):
                t = read(OUT/'tiles'/(tile['key']+'.json'))
                expected = {'receipt_sha256': sha(OUT/'receipt.json'), 'record_id': row['record_id'],
                            'lq_sha256': row['lq_sha256'], 'tile': tile, 'rank': rank,
                            'runtime_sha256': sha(OUT/f'runtime{rank}.json')}
                need(t['identity'] == expected and t['GT_input'] is False and t['model_calls'] == 2
                     and sha(OUT/t['png']['path']) == t['png']['sha256'], 'tile integrity mismatch')
            records.append(record)
    need(all(s == signatures[0] for s in signatures), 'numerical runtimes differ')
    result = {'status': 'complete_official_Blur150', 'records': records, **metric_summary(records),
              'recipe': RECIPE, 'receipt_sha256': sha(OUT/'receipt.json'), 'weight': b['weight'],
              'GT_used_in_training_or_selection': False, 'independent_blind_test': False,
              'other_categories_evaluated': False, 'development_population_evaluated': False,
              'seconds': time.monotonic()-start, 'exit_code': 0, 'GPU_hours': 0}
    write(REPORT, result)
    print('BLUR_FULLGRID_OFFICIAL_RESULT_VERIFIED '+json.dumps({k:a for k,a in result.items() if k not in ('records','recipe')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('phase', choices=['plan','prepare','run','finish'])
    globals()[parser.parse_args().phase]()
