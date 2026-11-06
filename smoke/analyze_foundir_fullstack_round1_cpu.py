#!/usr/bin/env python3
"""Read-only official600 error decomposition; no fitting, gradients or new outputs."""
import concurrent.futures
import hashlib
import io
import json
import math
from pathlib import Path
import time

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / 'checkpoints/foundir_official_vlm_router_v4/official_vlm_v4_frozen4800_v1_4x2'
OUTPUT = ROOT / 'checkpoints/foundir_fullstack_round1_cpu_audit'
REPORT = ROOT / 'reports/foundir_fullstack_round1_cpu_audit_2026-11-06.json'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + '\n'
    if path.exists():
        if path.read_text() != data:
            raise ValueError(f'existing result differs: {path}')
        return
    temporary = path.with_name(path.name + f'.tmp.{time.time_ns()}')
    temporary.write_text(data)
    temporary.replace(path)


def pixels(path, expected_sha):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_sha:
        raise ValueError(f'pixel hash changed: {path}')
    with Image.open(io.BytesIO(data)) as image:
        return np.asarray(image.convert('RGB'), dtype=np.float32) / 255.


def origins(length):
    if length <= 512:
        return [0]
    result = list(range(0, length - 512 + 1, 384))
    if result[-1] != length - 512:
        result.append(length - 512)
    return result


def analyze(row, identity):
    target = OUTPUT / 'records' / (row['record_id'] + '.json')
    if target.exists():
        result = read(target)
        if result['identity'] != identity:
            raise ValueError('audit identity changed')
        return result
    saved = read(PARENT / 'results' / (row['record_id'] + '.json'))
    labels = {'01Blur':'blur', '05Noise':'noise', '08Haze':'haze', '10Rain':'rain', '14Lowlight':'lowlight'}
    anchor = labels[row['category']]
    spec = saved['outputs'][anchor]
    lq = pixels(ROOT / row['lq_path'], row['lq_sha256'])
    gt = pixels(ROOT / row['gt_path'], row['gt_sha256'])
    pred = pixels(PARENT / spec['path'], spec['sha256'])
    if lq.shape != gt.shape or pred.shape != gt.shape:
        raise ValueError('unaligned official full-resolution image')
    h, w, _ = lq.shape
    input_error = np.mean((lq - gt)**2, axis=2, dtype=np.float64)
    output_error = np.mean((pred - gt)**2, axis=2, dtype=np.float64)
    input_mse, output_mse = float(input_error.mean()), float(output_error.mean())
    psnr = lambda mse: 10 * math.log10(1 / (mse + 1e-8))
    seam = np.zeros((h, w), dtype=bool)
    for y in origins(h):
        for edge in (y, y + 512):
            if 4 <= edge < h - 4:
                seam[edge-4:edge+4, :] = True
    for x in origins(w):
        for edge in (x, x + 512):
            if 4 <= edge < w - 4:
                seam[:, edge-4:edge+4] = True
    regions = {}
    for name, mask in (('tile_edge_bands', seam), ('interior', ~seam)):
        if mask.any():
            a, b = float(input_error[mask].mean()), float(output_error[mask].mean())
            regions[name] = {'input_mse': a, 'output_mse': b, 'output_over_input_mse': b / max(a, 1e-12)}
    result = {'identity': identity, 'record_id': row['record_id'], 'category': row['category'],
              'height': h, 'width': w, 'input_psnr_diagnostic': psnr(input_mse),
              'known_class_psnr_diagnostic': psnr(output_mse),
              'saved_known_class_psnr': saved['metrics'][anchor]['psnr'],
              'known_class_psnr_gain_db': psnr(output_mse)-psnr(input_mse),
              'known_class_lpips': saved['metrics'][anchor]['lpips'],
              'regions': regions,
              'mean_rgb_error': (pred-gt).mean(axis=(0,1), dtype=np.float64).tolist(),
              'GT_used_only_for_diagnostics': True}
    atomic(target, result)
    return result


def main():
    start = time.monotonic()
    terminal = ROOT / 'reports/foundir_official_vlm_router_v4_terminal_4x2_2026-11-05.json'
    parent = read(PARENT / 'receipt.json')
    identity = {'source_sha256': sha(Path(__file__)), 'terminal_sha256': sha(terminal),
                'receipt_sha256': sha(PARENT/'receipt.json'), 'official_records': 600}
    rows = parent['rows']
    if len(rows) != 600 or any(not r['lq_path'].startswith('downloads/foundir_official/') for r in rows):
        raise ValueError('official600 only')
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        for item in pool.map(lambda row: analyze(row, identity), rows):
            results.append(item)
            if len(results) % 50 == 0:
                print(f'FULLSTACK_CPU_AUDIT completed={len(results)}/600', flush=True)
    categories = {}
    for category in sorted({r['category'] for r in rows}):
        group = [r for r in results if r['category'] == category]
        mean = lambda key: sum(r[key] for r in group) / len(group)
        categories[category] = {'count': len(group), 'input_psnr_diagnostic': mean('input_psnr_diagnostic'),
            'known_class_psnr': mean('saved_known_class_psnr'), 'mean_gain_db': mean('known_class_psnr_gain_db'),
            'psnr_worse_than_input': sum(r['known_class_psnr_gain_db'] < 0 for r in group),
            'max_psnr_formula_difference': max(abs(r['saved_known_class_psnr']-r['known_class_psnr_diagnostic']) for r in group),
            'worst_gain_records': sorted(group, key=lambda r:r['known_class_psnr_gain_db'])[:5]}
    report = {'status':'completed', 'identity':identity, 'categories':categories, 'seconds':time.monotonic()-start,
              'gpus':0, 'training':False, 'parameter_fitting':False, 'new_restorations':False,
              'scope':'official600 saved-output error decomposition',
              'note':'NumPy RGB MSE diagnostic with pyiqa epsilon; final acceptance uses original pyiqa scores.'}
    atomic(REPORT, report)
    print('FULLSTACK_CPU_AUDIT_END status=0 '+json.dumps({k:{n:v for n,v in c.items() if n!='worst_gain_records'} for k,c in categories.items()}), flush=True)


if __name__ == '__main__':
    main()
