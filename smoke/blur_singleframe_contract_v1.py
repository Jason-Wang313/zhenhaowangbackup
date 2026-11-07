"""One deterministic training frame; all deployment contexts plus its original crop."""
import math
from smoke.official_vlm_router_v4_contract import tiles


def views(rows):
    if len(rows) != 1024:
        raise ValueError('original1024 population required')
    item = rows[0]
    row = item['row']
    if (item['role'] != 'train' or row['split'] != 'train' or row['category'] != '01Blur'
            or row['record_id'] != 'foundir_train_0034314'
            or (int(row['width']), int(row['height'])) != (1920, 1080)):
        raise ValueError('fixed first training frame required')
    for side in ('lq', 'gt'):
        if row[side+'_path'] != f'data/foundir/MillionIRData/Train/01Blur/{side.upper()}/0034315.png':
            raise ValueError('training-only fixed image path required')
    grid = tiles(row)
    result = [{**item, 'crop_xy': [t['x'], t['y']], 'view': i, 'kind': 'deployment_grid',
               'tile_key': t['key']} for i, t in enumerate(grid)]
    result.append({**item, 'view': 15, 'kind': 'original_crop', 'tile_key': None})
    if len(result) != 16 or len({tuple(i['crop_xy']) for i in result}) != 16:
        raise ValueError('16 distinct contexts required')
    return result


def key(item):
    x, y = item['crop_xy']
    return f"{item['row']['record_id']}__v{item['view']:02d}_x{x}_y{y}"


def summarize(records, items, step, parameter_hash, receipt_sha):
    from smoke.blur_singleframe_state_v1 import SNAPSHOTS
    indexed = {r['index']: r for r in records}
    if step not in SNAPSHOTS or len(records) != 16 or set(indexed) != set(range(16)):
        raise ValueError('incomplete or unplanned snapshot')
    for index, item in enumerate(items):
        r = indexed[index]
        if (r['item'] != item or r['rank'] != index % 2 or r['step'] != step
                or r['parameter_hash'] != parameter_hash or r['receipt_sha256'] != receipt_sha
                or r['scope'] != 'training_fit_only'):
            raise ValueError('snapshot identity differs')
    means = {}
    for arm in ('lq', 'candidate'):
        means[arm] = {}
        for metric in ('psnr', 'lpips', 'ssim'):
            values = [r['metrics'][arm][metric] for r in records]
            if not all(math.isfinite(v) for v in values):
                raise ValueError('nonfinite metric')
            means[arm][metric] = math.fsum(values)/16
    return {'step': step, 'parameter_hash': parameter_hash, 'receipt_sha256': receipt_sha,
            'scope': 'training_fit_only', 'quality_gate': False, 'means': means,
            'both_improve_count': sum(r['metrics']['candidate']['psnr'] > r['metrics']['lq']['psnr']
                                      and r['metrics']['candidate']['lpips'] < r['metrics']['lq']['lpips']
                                      for r in records)}
