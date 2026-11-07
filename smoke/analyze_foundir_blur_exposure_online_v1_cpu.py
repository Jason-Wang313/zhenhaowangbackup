#!/usr/bin/env python3
"""Read existing complete-epoch logs; no images, model loading, or GPU work."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import time

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / 'reports/foundir_blur_coverage_v2_train_ended_2026-11-06.json'
OUT = ROOT / 'checkpoints/foundir_blur_exposure_2x2_v2'
LOG = ROOT / ('checkpoints/foundir_fullstack_executor_3x2_h100_v2/jobs/'
              'round14b_blur_exposure_train_v2_node1/node1.log')


def analyze(through):
    start = time.monotonic()
    if through not in range(256, 1281, 128):
        raise ValueError('require a complete epoch boundary: 256,384,...,1280')
    receipt = json.loads((OUT / 'receipt.json').read_text())
    assert len(receipt['rows']) == 1024 and receipt['recipe']['global_batch'] == 8
    parent = json.loads(PARENT.read_text())
    history = parent['training_logs']
    assert [v['step'] for v in history] == list(range(1, 201))
    raw = LOG.read_text(errors='strict')
    history += [json.loads(s) for s in re.findall(r'BLUR_EXPOSURE_TRAIN (\{[^\n]*?\})', raw)
                if json.loads(s)['step'] <= through]
    assert [v['step'] for v in history] == list(range(1, through + 1)), 'incomplete/duplicate history'
    for entry in history:
        step = entry['step']
        assert entry['next_global_slot'] == step * 8
        assert all(math.isfinite(entry[k]) for k in ('loss', 'mse', 'lpips', 'gradient_norm'))
        assert abs(entry['loss'] - (10 * entry['mse'] + .1 * entry['lpips'])) < 1e-6
        if step > 200:
            assert entry['world'] == 4 and entry['global_batch'] == 8
            marker = json.loads((OUT / 'steps' / f'step_{step:06d}.json').read_text())
            assert marker['step'] == step and marker['next_global_slot'] == step * 8
    epochs = []
    for index in range(through // 128):
        rows = history[index * 128:(index + 1) * 128]
        epochs.append({'epoch': index + 1, 'steps': [rows[0]['step'], rows[-1]['step']],
                       'samples': 1024,
                       'online_means': {k: statistics.fmean(r[k] for r in rows)
                                        for k in ('loss', 'mse', 'lpips', 'gradient_norm')},
                       'batches_gradient_norm_above_clip1': sum(r['gradient_norm'] > 1 for r in rows)})
    first, last = epochs[0]['online_means'], epochs[-1]['online_means']
    return {'scope': 'training_fit_only_online_logs_not_fixed_checkpoint_evaluation',
            'through_step': through, 'epochs': epochs,
            'last_vs_first_relative_change_percent':
                {k: 100 * (last[k] / first[k] - 1) for k in ('loss', 'mse', 'lpips')},
            'limitations': ['Each batch is measured before its update; weights change throughout an epoch.',
                            'Same full1024 source set and fixed crops, but each epoch reshuffles order.',
                            'Epoch2 includes the declared world2-to4 migration at step200.',
                            'These are continuous training metrics, not official quantized full-image metrics.',
                            'Do not derive mean image PSNR from mean MSE; no official-quality claim.'],
            'parent_report_sha256': hashlib.sha256(PARENT.read_bytes()).hexdigest(),
            'receipt_sha256': hashlib.sha256((OUT / 'receipt.json').read_bytes()).hexdigest(),
            'selected_history_sha256': hashlib.sha256(json.dumps(history, sort_keys=True).encode()).hexdigest(),
            'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'new_GPU_work': False, 'image_pixels_read': False, 'official_result_produced': False,
            'seconds': time.monotonic() - start, 'exit_code': 0}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--through-step', required=True, type=int)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = analyze(args.through_step)
    data = json.dumps(result, ensure_ascii=False, indent=2) + '\n'
    if args.output:
        with args.output.open('x') as handle:
            handle.write(data)
    print(data, end='')
