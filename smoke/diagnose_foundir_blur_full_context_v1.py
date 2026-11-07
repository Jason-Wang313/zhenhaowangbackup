#!/usr/bin/env python3
"""Frozen1280 on first8 training full frames; compare identical trained-crop pixels."""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'third_party/flux2/src'))
from smoke import evaluate_foundir_blur_exposure_official_2x2_v2 as official
from smoke.blur_context_geometry_v1 import crop, pixel_statistics
from smoke.existing_flux_replay_contract_v1 import read, sha, write_new as write

train, x, core, need = official.train.parent, official.x, official.core, official.need
OUT = ROOT / 'checkpoints/foundir_blur_full_context_v1'
FIT = ROOT / 'checkpoints/foundir_blur_exposure_fit_v1'
FIT_RESULT = ROOT / 'reports/foundir_blur_exposure_fit_v1_result_2026-11-07.json'
REPORT = ROOT / 'reports/foundir_blur_full_context_v1_result_2026-11-07.json'
SOURCES = ['smoke/diagnose_foundir_blur_full_context_v1.py', 'smoke/blur_context_geometry_v1.py',
           'smoke/run_foundir_fullstack_blur_full_context_v1.sh']
ARMS = ('full_lq', 'full_prediction', 'crop_lq', 'crop_standalone', 'crop_recomputed', 'crop_from_full',
        'center384_lq', 'center384_standalone', 'center384_recomputed', 'center384_from_full')
RECIPE = {'sources': 'fixed_first8_of_same1024_training_sources', 'checkpoint_step': 1280,
          'node': 3, 'world': 2, 'tile': 512, 'overlap': 128, 'center_margin': 64,
          'stitch': x.RECIPE['stitch'], 'padding': x.RECIPE['padding'],
          'inference': train.RECIPE['inference'], 'training': False, 'new_quality_gate': False,
          'GT_input_or_selection': False, 'arms': list(ARMS), 'scope': 'training_fit_only',
          'original_crop_uncached_control': True}


def prepare():
    start = time.monotonic()
    plan, weights = official.bound()
    training = train.bound()
    fit = read(FIT / 'receipt.json')
    result = read(FIT_RESULT)
    need(result['status'] == 'training_fit_diagnostic_complete', 'training64 must complete first')
    items = training['rows'][:8]
    need(items == fit['items'][:8] and len(items) == 8, 'same first8 required')
    need(fit['parameter_hash'] == weights['parameter_hash'] and fit['weight'] == weights['weight'],
         'frozen final1280 must match training64')
    bindings = {}
    for item in items:
        row = item['row']
        need(item['role'] == 'train' and row['split'] == 'train'
             and row['category'] == '01Blur', 'training Blur only')
        lq = official.png(ROOT / row['lq_path'], row['lq_sha256'])
        need(lq.shape == (1080, 1920, 3) and len(x.tiles(row)) == 15, 'fixed full-frame geometry')
        crop(lq, item)
        need(sha(ROOT / row['gt_path']) == row['gt_sha256'], 'GT file changed')
        key = train.parent.key(item)
        path = FIT / 'records' / (key + '.json')
        r = read(path)
        need(r['item'] == item and r['parameter_hash'] == weights['parameter_hash']
             and r['GT_input_or_selection'] is False, 'fixed crop record changed')
        need(sha(FIT / r['output']['path']) == r['output']['sha256'], 'standalone PNG changed')
        bindings[key] = {'record_sha256': sha(path), 'output': r['output']}
    receipt = {'recipe': RECIPE, 'items': items, 'crop_bindings': bindings,
               'weight': weights['weight'], 'parameter_hash': weights['parameter_hash'],
               'training_receipt_sha256': sha(train.OUT / 'receipt.json'),
               'fit_receipt_sha256': sha(FIT / 'receipt.json'), 'fit_result_sha256': sha(FIT_RESULT),
               'official_receipt_sha256': sha(official.OUT / 'receipt.json'),
               'sources': {n: sha(ROOT / n) for n in sorted(set(plan['sources']) | set(SOURCES))}}
    write(OUT / 'receipt.json', receipt)
    preflight = {'status': 'destination_CPU_preflight_pass', 'training_sources': 8,
                 'full_frames': 8, 'tiles': 120, 'uncached_control_crops': 8, 'model_calls': 256,
                 'original_crop_outputs_reused': 8, 'new_training': False,
                 'official_pixels_read': False, 'GT_pixels_decoded': False,
                 'new_quality_gate': False, 'seconds': time.monotonic() - start,
                 'receipt_sha256': sha(OUT / 'receipt.json'), 'exit_code': 0, 'GPU_hours': 0}
    write(ROOT / 'reports/foundir_blur_full_context_v1_preflight_2026-11-07.json', preflight)
    print('BLUR_FULL_CONTEXT_PREFLIGHT_PASS ' + json.dumps(preflight), flush=True)


def bound():
    b = read(OUT / 'receipt.json')
    need(b['recipe'] == RECIPE and b['items'] == train.bound()['rows'][:8], 'scope changed')
    for path, field in ((train.OUT / 'receipt.json', 'training_receipt_sha256'),
                        (FIT / 'receipt.json', 'fit_receipt_sha256'), (FIT_RESULT, 'fit_result_sha256'),
                        (official.OUT / 'receipt.json', 'official_receipt_sha256')):
        need(sha(path) == b[field], 'parent binding changed')
    for n, digest in b['sources'].items():
        need(sha(ROOT / n) == digest, 'source changed: ' + n)
    need(sha(ROOT / b['weight']['path']) == b['weight']['sha256'], 'weight changed')
    for key, v in b['crop_bindings'].items():
        need(sha(FIT / 'records' / (key + '.json')) == v['record_sha256'], 'crop result changed')
        need(sha(FIT / v['output']['path']) == v['output']['sha256'], 'crop output changed')
    return b


def check_record(r, item, rank):
    need(r['item'] == item and r['rank'] == rank and r['receipt_sha256'] == sha(OUT / 'receipt.json')
         and r['GT_input_or_selection'] is False, 'record identity changed')
    need(set(r['metrics_training_fit_only']) == set(ARMS), 'missing metric arm')
    for arm in r['metrics_training_fit_only'].values():
        need(set(arm) == {'psnr', 'lpips', 'ssim'} and all(math.isfinite(v) for v in arm.values()),
             'invalid metrics')
    need(sha(OUT / r['output']['path']) == r['output']['sha256'], 'full output changed')
    control = r['uncached_crop']
    need(control['crop_xy'] == item['crop_xy'] and control['GT_input'] is False and control['model_calls'] == 2
         and sha(OUT / control['output']['path']) == control['output']['sha256'], 'uncached control changed')


def run():
    import numpy as np
    import torch
    from flux2.sampling import batched_prc_img
    from lowlight_restoration.lora import load_lora_checkpoint, lora_state_dict
    start = time.monotonic()
    b, training = bound(), train.bound()
    rank, device = train.setup(training)
    need(int(os.environ['FOUNDIR_FULLSTACK_NODE']) == 3, 'physical node3 required')
    reference = train.load_model(training, device).eval().requires_grad_(False)
    model = train.load_model(training, device).eval()
    load_lora_checkpoint(model, ROOT / b['weight']['path'])
    model.requires_grad_(False)
    initial = train.state.parameter_hash(lora_state_dict(model))
    reference_initial = train.state.parameter_hash(lora_state_dict(reference))
    need(initial == b['parameter_hash'], 'wrong loaded final checkpoint')
    ae, prompt = train.load_vae_prompt(training, device)
    metrics, identity = official.fix.checked_metric_models(device, training['metric_identity'])
    runtime = {'rank': rank, 'GPU': torch.cuda.get_device_name(device), 'torch': str(torch.__version__),
               'cuda': torch.version.cuda, 'parameter_hash': initial, 'reference_parameter_hash': reference_initial,
               'receipt_sha256': sha(OUT / 'receipt.json')}
    write(OUT / f'runtime{rank}.json', runtime)

    def tensor(array):
        return torch.from_numpy(array.copy()).permute(2, 0, 1).unsqueeze(0).to(device).float() / 255

    def score(array, target):
        with torch.no_grad():
            value = tensor(array)
            return {k: float(m(value, target).mean()) for k, m in metrics.items()}

    def infer(array):
        input_tensor = official.fix.producer.tensor(array, device)
        with torch.no_grad():
            latent = ae.encode(input_tensor).float()
            ref_velocity = core.native_velocity(reference, latent, prompt, batched_prc_img)
            ref_decode = ae.decode(latent).float()
            data = {'lq': input_tensor, 'latent': latent, 'reference_velocity': ref_velocity,
                    'reference_decode': ref_decode}
            prediction = train.prediction(model, ae, prompt, data)
            need(bool(torch.isfinite(prediction).all()), 'nonfinite output')
            return core.uint8_output(prediction)[0].permute(1, 2, 0).cpu().numpy()

    for item in b['items'][rank::2]:
        row, key = item['row'], train.parent.key(item)
        lq = official.png(ROOT / row['lq_path'], row['lq_sha256'])
        tiles = {}
        for tile in x.tiles(row):
            marker = OUT / 'tiles' / (tile['key'] + '.json')
            identity = {'item': item, 'tile': tile, 'rank': rank,
                        'receipt_sha256': sha(OUT / 'receipt.json'),
                        'runtime_sha256': sha(OUT / f'runtime{rank}.json')}
            if marker.exists():
                saved = read(marker)
                need(saved['identity'] == identity and saved['GT_input'] is False, 'tile changed')
                image = official.png(OUT / saved['png']['path'], saved['png']['sha256'])
            else:
                tile_rgb, _, _ = x.tile_array(lq, tile)
                image = infer(tile_rgb)
                path = OUT / 'tile_png' / (tile['key'] + '.png')
                official.fix.producer.png_once(path, image)
                write(marker, {'identity': identity, 'GT_input': False, 'model_calls': 2,
                               'png': {'path': str(path.relative_to(OUT)), 'sha256': sha(path)}})
            tiles[tile['key']] = image
        full = x.stitch(row, tiles)
        path = OUT / 'full' / (key + '.png')
        official.fix.producer.png_once(path, full)
        recomputed = infer(crop(lq, item))
        control_path = OUT / 'uncached_crop' / (key + '.png')
        official.fix.producer.png_once(control_path, recomputed)
        # Every generated output is fixed before reading GT; no alternate selection.
        gt = official.png(ROOT / row['gt_path'], row['gt_sha256'])
        crop_record = read(FIT / 'records' / (key + '.json'))
        standalone = official.png(FIT / crop_record['output']['path'], crop_record['output']['sha256'])
        target = tensor(gt)
        values = {'full_lq': score(lq, target), 'full_prediction': score(full, target)}
        del target
        # Keep the historical training-crop target arithmetic exactly for paired scoring.
        crop_target = (train.rgb(item, 'gt', device) + 1) * .5
        need(np.array_equal(core.uint8_output(crop_target * 2 - 1)[0].permute(1, 2, 0).cpu().numpy(),
                            crop(gt, item)), 'trained crop target pixels differ')
        values.update(crop_lq=score(crop(lq, item), crop_target),
                      crop_standalone=score(standalone, crop_target),
                      crop_recomputed=score(recomputed, crop_target),
                      crop_from_full=score(crop(full, item), crop_target))
        for old, new in (('lq', 'crop_lq'), ('candidate', 'crop_standalone')):
            for metric in ('psnr', 'lpips', 'ssim'):
                need(abs(values[new][metric] - crop_record['metrics_training_fit_only'][old][metric]) < 1e-6,
                     'same crop rescoring differs')
        target = crop_target[:, :, 64:-64, 64:-64]
        values.update(center384_lq=score(crop(lq, item, 64), target),
                      center384_standalone=score(standalone[64:-64, 64:-64], target),
                      center384_recomputed=score(recomputed[64:-64, 64:-64], target),
                      center384_from_full=score(crop(full, item, 64), target))
        statistics = pixel_statistics(lq, gt, full, recomputed, item)
        statistics['uncached_vs_cached_crop_output_RMS_RGB01'] = float(np.sqrt(np.mean(
            ((recomputed.astype(np.float64) - standalone.astype(np.float64)) / 255) ** 2)))
        statistics['uncached_vs_cached_crop_max_abs_gray'] = int(np.abs(
            recomputed.astype(np.int16) - standalone.astype(np.int16)).max())
        r = {'item': item, 'rank': rank, 'receipt_sha256': sha(OUT / 'receipt.json'),
             'runtime_sha256': sha(OUT / f'runtime{rank}.json'), 'parameter_hash': initial,
             'metrics_training_fit_only': values, 'pixel_statistics': statistics,
             'output': {'path': str(path.relative_to(OUT)), 'sha256': sha(path)},
             'uncached_crop': {'crop_xy': item['crop_xy'], 'GT_input': False, 'model_calls': 2,
                               'output': {'path': str(control_path.relative_to(OUT)), 'sha256': sha(control_path)}},
             'original_crop_record_sha256': sha(FIT / 'records' / (key + '.json')),
             'GT_input_or_selection': False}
        check_record(r, item, rank)
        write(OUT / 'records' / (key + '.json'), r)
        print('BLUR_FULL_CONTEXT_RECORD ' + json.dumps({'key': key, 'rank': rank, 'metrics': values}), flush=True)
        del crop_target, target
    final = train.state.parameter_hash(lora_state_dict(model))
    reference_final = train.state.parameter_hash(lora_state_dict(reference))
    need(initial == final and reference_initial == reference_final, 'diagnosis changed weights')
    write(OUT / f'worker{rank}.json', {'rank': rank, 'keys': [train.parent.key(i) for i in b['items'][rank::2]],
          'receipt_sha256': sha(OUT / 'receipt.json'), 'parameter_hash_before': initial,
          'parameter_hash_after': final, 'reference_hash_before': reference_initial,
          'reference_hash_after': reference_final, 'seconds': time.monotonic() - start,
          'peak_allocated_GiB': torch.cuda.max_memory_allocated(device) / 1024**3})
    print(f'BLUR_FULL_CONTEXT_END rank={rank} status=0', flush=True)


def finish():
    start, b, records = time.monotonic(), bound(), []
    runtimes = [read(OUT / f'runtime{rank}.json') for rank in (0, 1)]
    need({k: v for k, v in runtimes[0].items() if k != 'rank'} ==
         {k: v for k, v in runtimes[1].items() if k != 'rank'}, 'rank runtime/reference weights differ')
    for rank in (0, 1):
        worker = read(OUT / f'worker{rank}.json')
        need(worker['rank'] == rank and worker['keys'] == [train.parent.key(i) for i in b['items'][rank::2]]
             and worker['receipt_sha256'] == sha(OUT / 'receipt.json')
             and worker['parameter_hash_before'] == worker['parameter_hash_after'] == b['parameter_hash']
             and worker['reference_hash_before'] == worker['reference_hash_after'], 'worker incomplete')
        for item in b['items'][rank::2]:
            key = train.parent.key(item)
            r = read(OUT / 'records' / (key + '.json'))
            check_record(r, item, rank)
            need(r['runtime_sha256'] == sha(OUT / f'runtime{rank}.json') and r['parameter_hash'] == b['parameter_hash'],
                 'runtime/weights differ')
            for tile in x.tiles(item['row']):
                t = read(OUT / 'tiles' / (tile['key'] + '.json'))
                expected = {'item': item, 'tile': tile, 'rank': rank,
                            'receipt_sha256': sha(OUT / 'receipt.json'),
                            'runtime_sha256': sha(OUT / f'runtime{rank}.json')}
                need(t['identity'] == expected and t['GT_input'] is False and t['model_calls'] == 2
                     and sha(OUT / t['png']['path']) == t['png']['sha256'], 'tile invalid')
            records.append(r)
    need(len(records) == len({r['item']['row']['record_id'] for r in records}) == 8, 'scope incomplete')
    summary = {name: {k: math.fsum(r['metrics_training_fit_only'][name][k] for r in records) / 8
                      for k in ('psnr', 'lpips', 'ssim')} for name in ARMS}
    value = {'status': 'training8_full_context_diagnostic_complete', 'records': records, 'summary': summary,
             'pixel_statistics_means': {k: math.fsum(r['pixel_statistics'][k] for r in records) / 8
                                        for k in records[0]['pixel_statistics']},
             'weights_unchanged': True, 'receipt_sha256': sha(OUT / 'receipt.json'),
             'new_training': False, 'new_quality_gate': False, 'official_Blur_pass': False,
             'scope': 'training_fit_only', 'seconds': time.monotonic() - start, 'exit_code': 0, 'GPU_hours': 0}
    write(REPORT, value)
    print('BLUR_FULL_CONTEXT_VERIFIED ' + json.dumps({k: v for k, v in value.items() if k != 'records'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['prepare', 'run', 'finish'])
    globals()[parser.parse_args().phase]()
