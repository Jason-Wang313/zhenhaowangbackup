"""Fixed training-only population and complete snapshot aggregation; no quality gate."""
import math

STEPS = 100
SNAPSHOTS = (0, 1, 25, 50, 100)


def select_items(parent_rows):
    if len(parent_rows) != 1024:
        raise ValueError('original1024 population required')
    items = parent_rows[:8]
    for item in items:
        row = item['row']
        if item['role'] != 'train' or row['split'] != 'train' or row['category'] != '01Blur':
            raise ValueError('training Blur only')
        for side in ('lq', 'gt'):
            if not row[side+'_path'].startswith('data/foundir/MillionIRData/Train/01Blur/'):
                raise ValueError('nontraining path')
    for key in ('record_id', 'source_group', 'gt_sha256'):
        if len({i['row'][key] for i in items}) != 8:
            raise ValueError('duplicate fixed training source')
    return items


def summarize(records, items, step, parameter_hash, receipt_sha):
    if step not in SNAPSHOTS or len(records) != 8:
        raise ValueError('incomplete or unplanned snapshot')
    indexed = {r['index']: r for r in records}
    if set(indexed) != set(range(8)):
        raise ValueError('duplicate/missing image')
    for index, item in enumerate(items):
        r = indexed[index]
        if (r['item'] != item or r['rank'] != index % 2 or r['step'] != step
                or r['parameter_hash'] != parameter_hash or r['receipt_sha256'] != receipt_sha):
            raise ValueError('snapshot identity differs')
    means = {}
    for arm in ('lq', 'candidate'):
        means[arm] = {}
        for metric in ('psnr', 'lpips', 'ssim'):
            values = [r['metrics'][arm][metric] for r in records]
            if not all(math.isfinite(v) for v in values):
                raise ValueError('nonfinite metric')
            means[arm][metric] = math.fsum(values)/8
    return {'step': step, 'parameter_hash': parameter_hash, 'receipt_sha256': receipt_sha,
            'scope': 'training_fit_only', 'quality_gate': False, 'means': means,
            'PSNR_delta_vs_LQ': means['candidate']['psnr']-means['lq']['psnr'],
            'LPIPS_delta_vs_LQ': means['candidate']['lpips']-means['lq']['lpips'],
            'PSNR_worse_count': sum(r['metrics']['candidate']['psnr'] < r['metrics']['lq']['psnr'] for r in records),
            'LPIPS_worse_count': sum(r['metrics']['candidate']['lpips'] > r['metrics']['lq']['lpips'] for r in records),
            'both_improve_count': sum(r['metrics']['candidate']['psnr'] > r['metrics']['lq']['psnr']
                                      and r['metrics']['candidate']['lpips'] < r['metrics']['lq']['lpips'] for r in records)}
