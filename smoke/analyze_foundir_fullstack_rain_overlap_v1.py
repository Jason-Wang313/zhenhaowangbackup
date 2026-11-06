#!/usr/bin/env python3
"""Read-only Rain overlap/low-frequency audit. No fitted corrections or GPU."""
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ['FOUNDIR_OFFICIAL_NNODES'] = '4'  # Historical parent identity only.
from smoke import official_vlm_router_v4_contract as contract

PARENT = ROOT / contract.OUTPUT
OUT = ROOT / 'checkpoints/foundir_fullstack_rain_overlap_audit_v1'
REPORT = ROOT / 'reports/foundir_fullstack_rain_overlap_audit_v1_2026-11-06.json'
RECEIPT_SHA = '98eb74a6534eb234e75a164a2460f43b0a2f6beaeb08a686e1812f85a7615157'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def write_once(path, data):
    text = json.dumps(data, sort_keys=True, indent=2) + '\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        assert path.read_text() == text, f'conflicting artifact: {path}'
    else:
        temp = path.with_name(path.name + f'.tmp.{os.getpid()}')
        with temp.open('x') as f:
            f.write(text); f.flush(); os.fsync(f.fileno())
        os.replace(temp, path)


def png(path, expected):
    import numpy as np
    from PIL import Image
    data = path.read_bytes()
    assert hashlib.sha256(data).hexdigest() == expected
    with Image.open(io.BytesIO(data)) as im:
        return np.array(im.convert('RGB'), copy=True)


def block_error(error, block=32):
    """Exact orthogonal block-mean energy, including partial edge blocks."""
    import numpy as np
    h, w, channels = error.shape
    ys, xs = np.arange(0, h, block), np.arange(0, w, block)
    counts = np.minimum(block, h-ys)[:, None] * np.minimum(block, w-xs)[None, :]
    sums = np.add.reduceat(np.add.reduceat(error.astype(np.float64), ys, axis=0), xs, axis=1)
    low = float(np.sum(sums*sums/counts[..., None]) / (h*w*channels))
    mse = float(np.mean(error.astype(np.float64)**2))
    assert 0 <= low <= mse + 1e-12
    return {'mse': mse, 'block32_mean_energy': low,
            'block32_energy_fraction': low/max(mse, 1e-30),
            'global_RGB_mean_energy_fraction': float(np.mean(error.mean((0, 1), dtype=np.float64)**2))/max(mse, 1e-30)}


def neighbors(tiles):
    """Unique horizontal/vertical adjacent grid edges, excluding diagonals."""
    by_xy = {(t['x'], t['y']): t for t in tiles}
    xs, ys = sorted({t['x'] for t in tiles}), sorted({t['y'] for t in tiles})
    assert len(by_xy) == len(xs)*len(ys)
    pairs = []
    for y in ys:
        pairs += [(by_xy[x1, y], by_xy[x2, y]) for x1, x2 in zip(xs, xs[1:])]
    for x in xs:
        pairs += [(by_xy[x, y1], by_xy[x, y2]) for y1, y2 in zip(ys, ys[1:])]
    return pairs


def overlap_stats(a, b, ta, tb, height, width):
    import numpy as np
    x0, y0 = max(ta['x'], tb['x']), max(ta['y'], tb['y'])
    x1 = min(ta['x']+a.shape[1], tb['x']+b.shape[1], width)
    y1 = min(ta['y']+a.shape[0], tb['y']+b.shape[0], height)
    assert x1 > x0 and y1 > y0
    aa = a[y0-ta['y']:y1-ta['y'], x0-ta['x']:x1-ta['x']].astype(np.float64)/255
    bb = b[y0-tb['y']:y1-tb['y'], x0-tb['x']:x1-tb['x']].astype(np.float64)/255
    diff = aa-bb
    mean = diff.mean((0, 1))
    return {'a': ta['key'], 'b': tb['key'], 'pixels': (y1-y0)*(x1-x0),
            'mean_RGB_difference': mean.tolist(), 'difference_MSE': float(np.mean(diff**2)),
            'constant_RGB_difference_energy': float(np.mean(mean**2))}


def analyze(row, identity):
    import numpy as np
    from safetensors.numpy import load
    output = OUT / 'records' / (row['record_id']+'.json')
    if output.exists():
        value = read(output); assert value['identity'] == identity
        return value
    saved = read(PARENT / 'results' / (row['record_id']+'.json'))
    assert saved['row'] == row and saved['receipt_sha256'] == RECEIPT_SHA
    assert saved['decisions']['conditional_v4']['anchor'] == 'rain'
    tiles = contract.tiles(row); predictions, tile_hashes = {}, {}
    for tile in tiles:
        path = PARENT / 'restored_tiles' / (tile['key']+'__rain.json')
        record = read(path); ident = record['identity']; item = ident['item']
        assert ident['receipt_sha256'] == RECEIPT_SHA and ident['arm'] == 'restored_tiles'
        assert item == {'key': tile['key']+'__rain', 'record_id': row['record_id'],
                        'lq_sha256': row['lq_sha256'], 'tile': tile, 'anchor': 'rain'}
        assert ident['runtime_sha256'] == saved['runtime_sha256']
        assert record['GT_input'] is False and record['route']['anchor'] == 'rain'
        assert record['route']['model_calls'] == 26 and record['route']['readout'] == 'endpoint_at_call25'
        assert set(record['files']) == {record['tensor']}
        blob = (PARENT / record['tensor']).read_bytes()
        assert hashlib.sha256(blob).hexdigest() == record['files'][record['tensor']]
        tensors = load(blob); assert set(tensors) == {'uint8'}
        arr = tensors['uint8']; assert arr.dtype == np.uint8 and arr.shape == (1, 3, 512, 512)
        predictions[tile['key']] = arr[0].transpose(1, 2, 0)
        tile_hashes[tile['key']] = sha(path)
    spec = saved['outputs']['rain']; final = png(PARENT/spec['path'], spec['sha256'])
    assert np.array_equal(contract.stitch(row, predictions), final), 'parent stitch does not reproduce'
    h, w = int(row['height']), int(row['width'])
    assert final.shape == (h, w, 3)
    pairs = [overlap_stats(predictions[a['key']], predictions[b['key']], a, b, h, w)
             for a, b in neighbors(tiles)]
    # GT is used only after all output-only pair statistics are fixed.
    gt = png(ROOT/row['gt_path'], row['gt_sha256']); assert gt.shape == final.shape
    error = final.astype(np.float64)/255-gt.astype(np.float64)/255
    energies = block_error(error)
    psnr = 10*math.log10(1/(energies['mse']+1e-8))
    assert abs(psnr-saved['metrics']['rain']['psnr']) < 1e-4
    value = {'identity': identity, 'record_id': row['record_id'], 'tiles': len(tiles),
             'tile_metadata_hashes': tile_hashes, 'stitch_pixel_exact': True,
             'pairs': pairs, 'error': energies, 'psnr': psnr,
             'GT_used_only_for_diagnostics': True}
    write_once(output, value)
    return value


def main():
    start = time.monotonic()
    assert sha(PARENT/'receipt.json') == RECEIPT_SHA
    identity = {'receipt_sha256': RECEIPT_SHA, 'source_sha256': sha(Path(__file__)),
                'contract_sha256': sha(ROOT/'smoke/official_vlm_router_v4_contract.py')}
    rows = [r for r in read(PARENT/'receipt.json')['rows'] if r['category'] == '10Rain']
    assert len(rows) == 100
    results = []
    for i, row in enumerate(rows):
        results.append(analyze(row, identity))
        if (i+1)%10 == 0:
            print(f'RAIN_OVERLAP_PROGRESS records={i+1}/100 seconds={time.monotonic()-start:.1f}', flush=True)
    pairs = [p for r in results for p in r['pairs']]
    total = sum(p['pixels'] for p in pairs)
    difference = math.fsum(p['pixels']*p['difference_MSE'] for p in pairs)/total
    constant = math.fsum(p['pixels']*p['constant_RGB_difference_energy'] for p in pairs)/total
    mean = lambda xs: math.fsum(xs)/len(xs)
    report = {'status': 'complete', 'identity': identity, 'records': len(results),
              'tiles': sum(r['tiles'] for r in results), 'pairs': len(pairs),
              'pixel_exact_stitches': sum(r['stitch_pixel_exact'] for r in results),
              'overlap_RGB_RMS_255': 255*math.sqrt(difference),
              'overlap_constant_RGB_RMS_255': 255*math.sqrt(constant),
              'constant_RGB_fraction_of_overlap_disagreement': constant/difference,
              'mean_image_error': {k: mean([r['error'][k] for r in results]) for k in results[0]['error']},
              'records_sha256': {r['record_id']: sha(OUT/'records'/(r['record_id']+'.json')) for r in results},
              'seconds': time.monotonic()-start, 'gpus': 0, 'exit_code': 0,
              'training': False, 'parameter_fitting': False, 'new_restorations': False,
              'limitations': ['Overlap differences include crop context, relative position and different per-tile seeds; not an isolated context effect.',
                             'Block-mean energy is diagnostic, not an achievable correction.',
                             'Exact restitch proves aggregation consistency, not optimality or scientific success.']}
    write_once(REPORT, report)
    print('RAIN_OVERLAP_COMPLETE '+json.dumps({k:v for k,v in report.items() if k!='records_sha256'}), flush=True)


if __name__ == '__main__':
    main()
