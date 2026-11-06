#!/usr/bin/env python3
"""Fixed first8 training crops: LQ-only alignment and explicitly GT-only diagnostic."""
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from smoke import run_foundir_fullstack_frequency_repair_v1 as experiment
from lowlight_restoration import blur_local_alignment_v1 as alignment

OUT = ROOT / 'checkpoints/foundir_blur_local_alignment_v1'
REPORT = ROOT / 'reports/foundir_blur_local_alignment_v1_CPU_2026-11-06.json'
RECIPE = {'selection': 'first8_training_rows_in_frozen_frequency_receipt_no_score_selection',
          'arms': ['baseline', 'LQ_reference', 'GT_reference_diagnostic_only'],
          'match_sigma_pixels': alignment.MATCH_SIGMA_PIXELS,
          'max_flow_component_pixels': alignment.MAX_COMPONENT_PIXELS,
          'method': 'OpenCV_DIS_MEDIUM_backward_flow_bilinear_border_replicate',
          'score': 'full512_no_inner_crop_original_pyiqa_identity',
          'parameter_search': False, 'official_or_development_pixels_read': False,
          'GT_reference_arm_not_deployable': True, 'training': False, 'new_FLUX_calls': 0}


def main():
    import cv2
    import torch
    from smoke import official_vlm_v4_metric_identity_fix as fix
    if REPORT.exists():
        raise ValueError('preserve completed diagnostic')
    start = time.monotonic()
    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    bound = experiment.bound('blur')
    source = experiment.out('blur') / 'receipt.json'
    receipt_sha = experiment.sha(source)
    items = bound['rows'][:8]
    experiment.need(all(i['role'] == 'train' for i in items), 'training-only scope')
    value = {'recipe': RECIPE, 'items': items, 'parent_receipt_sha256': receipt_sha,
             'sources': {name: experiment.sha(ROOT / name) for name in
                         ['lowlight_restoration/blur_local_alignment_v1.py',
                          'smoke/diagnose_foundir_blur_local_alignment_v1_cpu.py']},
             'cv2_version': cv2.__version__, 'metric_identity': bound['metric_identity']}
    experiment.write(OUT / 'receipt.json', value)
    models, _ = fix.checked_metric_models('cpu', bound['metric_identity'])
    rows = []
    for item in items:
        _, arrays = experiment.cached('blur', item, receipt_sha)
        baseline = arrays['baseline']
        lq = experiment.crop(item, 'lq')
        proposed, lq_flow = alignment.align(lq, baseline)
        # LQ policy output is frozen before GT is loaded; GT control is separate.
        lq_path = OUT / 'images' / (experiment.key(item) + '_LQ_reference.png')
        fix.producer.png_once(lq_path, proposed)
        gt = experiment.crop(item, 'gt')
        oracle, gt_flow = alignment.align(gt, baseline)
        gt_path = OUT / 'images' / (experiment.key(item) + '_GT_reference_diagnostic.png')
        fix.producer.png_once(gt_path, oracle)
        target = torch.from_numpy(gt.copy()).permute(2, 0, 1).unsqueeze(0).float() / 255
        metrics = {}
        for arm, array in [('baseline', baseline), ('LQ_reference', proposed),
                           ('GT_reference_diagnostic_only', oracle)]:
            pred = torch.from_numpy(array.copy()).permute(2, 0, 1).unsqueeze(0).float() / 255
            with torch.no_grad():
                metrics[arm] = {name: float(model(pred, target).item()) for name, model in models.items()}
        experiment.need(all(math.isfinite(v) for m in metrics.values() for v in m.values()), 'nonfinite metrics')
        record = {'item': item, 'metrics': metrics, 'LQ_flow': lq_flow, 'GT_diagnostic_flow': gt_flow,
                  'outputs': {str(p.relative_to(OUT)): experiment.sha(p) for p in (lq_path, gt_path)},
                  'training_fit_only': True, 'LQ_policy_GT_input': False,
                  'GT_diagnostic_is_not_a_deployment_result': True}
        experiment.write(OUT / 'records' / (experiment.key(item) + '.json'), record)
        rows.append(record)
        print('BLUR_LOCAL_ALIGNMENT_RECORD ' + item['row']['record_id'] + ' ' + json.dumps(metrics), flush=True)
    summary = {arm: {metric: math.fsum(r['metrics'][arm][metric] for r in rows) / len(rows)
                     for metric in ('psnr', 'lpips', 'ssim')} for arm in RECIPE['arms']}
    result = {'status': 'fixed_training_only_geometry_diagnostic_complete', 'recipe': RECIPE,
              'receipt_sha256': experiment.sha(OUT / 'receipt.json'), 'summary': summary, 'rows': rows,
              'training_fit_only': True, 'new_training': False, 'new_GPU': False, 'GPU_hours': 0,
              'official_success_claim': False, 'seconds': time.monotonic() - start, 'exit_code': 0}
    experiment.write(REPORT, result)
    print('BLUR_LOCAL_ALIGNMENT_COMPLETE ' + json.dumps({'summary': summary, 'seconds': result['seconds']}), flush=True)


if __name__ == '__main__':
    main()
