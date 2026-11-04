#!/usr/bin/env python3
"""Read-only CPU evaluation of saved V3 step800/1600 on its training features.

No optimizer updates, architecture changes, holdout features, or new calibration
selection. Reuse existing calibration predictions. Writes separate audit outputs.
"""
import argparse
from collections import Counter
import math
import os
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from smoke import repair_foundir_vlm_preserve_router_v3 as v3

c, require, ORDER = v3.c, v3.require, v3.ORDER
RUN = 'fit800_1600_v1'
OUTPUT = 'checkpoints/foundir_vlm_preserve_fit_audit_v1/' + RUN
REPORT = 'reports/foundir_vlm_preserve_fit_audit_2026-11-04.json'
STEPS = (800, 1600)
SOURCES = ['smoke/audit_foundir_vlm_preserve_fit_v1.py', 'smoke/run_foundir_vlm_preserve_fit_audit_v1_cpu.sh']


def validate_summary(summary, items):
    rows = summary['rows']
    require(summary['count'] == len(rows) == len(items) and len({r['key'] for r in rows}) == len(items)
            and {r['key'] for r in rows} == set(items), 'evaluation population mismatch')
    correct, counts, losses = Counter(), Counter(), []
    for r in rows:
        i = items[r['key']]
        z, im, delta = r['logits'], r['image_logits'], r['vlm_delta']
        require(all(len(v) == 5 and all(math.isfinite(x) for x in v) for v in (z, im, delta)), 'invalid five scores')
        require(all(math.isclose(a, b+d, rel_tol=2e-6, abs_tol=2e-6) for a,b,d in zip(z, im, delta)), 'score components mismatch')
        pred = ORDER[max(range(5), key=lambda k: z[k])]
        require((r['task_for_reporting_only'], r['seed'], r['crop_xy'], r['predicted_anchor']) ==
                (i['task'], i['seed'], i['row']['crop_xy'], pred), 'prediction identity mismatch')
        counts[i['task']] += 1
        correct[i['task']] += int(pred == i['task'])
        mx = max(z)
        losses.append(mx + math.log(sum(math.exp(x-mx) for x in z)) - z[ORDER.index(i['task'])])
    require(summary['correct'] == sum(correct.values()) and abs(summary['ce'] - sum(losses)/len(rows)) < 2e-5,
            'summary metrics mismatch')
    require(summary['per_task'] == {t: {'count': counts[t], 'correct': correct[t]} for t in ORDER}, 'class counts mismatch')


def compact(summary):
    return {k: v for k,v in summary.items() if k != 'rows'}


def diagnostics(base, corrected):
    transitions = v3.transitions(base, corrected)
    rows = corrected['rows']
    penalties, protected_margins = [], []
    for r in rows:
        target = ORDER.index(r['task_for_reporting_only'])
        im, z = r['image_logits'], r['logits']
        if max(range(5), key=lambda k: im[k]) == target:
            image_margin = im[target] - max(im[k] for k in range(5) if k != target)
            margin = z[target] - max(z[k] for k in range(5) if k != target)
            protected_margins.append(margin)
            penalties.append(max(0., min(max(image_margin, 0.), v3.MARGIN_CAP) - margin))
    def ratio(num, den):
        return num / den if den else None
    per_task = {}
    for t, tr in transitions['per_task'].items():
        per_task[t] = {**tr, 'harm_rate_among_image_correct': ratio(tr['harm'], tr['harm']+tr['retained_correct']),
                       'fix_rate_among_image_wrong': ratio(tr['fix'], tr['fix']+tr['still_wrong'])}
    return {'transitions': transitions, 'per_task_rates': per_task,
            'harm_rate_among_image_correct': ratio(transitions['harm'], base['correct']),
            'fix_rate_among_image_wrong': ratio(transitions['fix'], base['count']-base['correct']),
            'protected_count': len(penalties), 'active_penalty_count': sum(p > 0 for p in penalties),
            'preserve_penalty_full_population_mean': sum(penalties)/len(rows),
            'minimum_protected_final_margin': min(protected_margins) if protected_margins else None}


def history_summary(history, step):
    require(len(history) == step, 'history step mismatch')
    tail = history[-100:]
    fields = ('loss', 'ce', 'preserve_penalty', 'protected_training_rows',
              'harmful_training_overrides', 'vlm_grad_norm_before_clip')
    require(all(all(k in x and isinstance(x[k], (int, float)) and math.isfinite(x[k]) for k in fields) for x in tail),
            'training loss records incomplete or nonfinite')
    return {'steps': len(tail), 'first_step': step-len(tail)+1, 'last_step': step,
            'mean': {k: sum(x[k] for x in tail)/len(tail) for k in fields},
            'note': 'Historical pre-update minibatch statistics, not current full-training-set evaluation.'}


def context():
    v3.verify()
    b, report = v3.receipt(), c.read(ROOT / v3.REPORT)
    out = ROOT / v3.OUTPUT
    identity = c.digest({'receipt': c.sha(out / 'receipt.json'), 'runtime': report['runtime'],
                         'image': report['frozen_image_hash']})
    markers = {str(k): c.read(out / f'steps/step_{k:06d}.json') for k in STEPS}
    for k, m in markers.items():
        require(m['step'] == int(k) and m['identity'] == identity and
                c.sha(c.path(out, m['file'])) == m['sha256'], 'fixed checkpoint mismatch')
    return b, report, identity, markers


def build_report(b, old, markers, output):
    fit_items = {i['key']: i for i in b['items'] if i['role'] == 'fit'}
    cal_items = {i['key']: i for i in b['items'] if i['role'] == 'calibration'}
    require(len(fit_items) == 2560 and len(cal_items) == 400, 'fixed populations required')
    base = c.read(output / 'fit_image_only.json')
    validate_summary(base, fit_items)
    calbase = old['calibration']['image_only']
    validate_summary(calbase, cal_items)
    arms = {}
    for step in STEPS:
        fit = c.read(output / f'fit_{step}.json')
        cal = c.read(ROOT / v3.OUTPUT / f'calibration_{step}.json')
        require(c.sha(ROOT / v3.OUTPUT / f'calibration_{step}.json') == old['selection']['calibration_sha256'][str(step)],
                'historical calibration changed')
        validate_summary(fit, fit_items)
        validate_summary(cal, cal_items)
        hist = c.read(output / f'history_{step}.json')
        # The tail is verified against the saved checkpoint during audit(), not a new training log.
        require(hist['step'] == step and len(hist['tail']) == 100, 'wrong history tail')
        full_shape_tail = [{}] * (step - 100) + hist['tail']
        fd, cd = diagnostics(base, fit), diagnostics(calbase, cal)
        arms[str(step)] = {'checkpoint': markers[str(step)], 'fit': compact(fit), 'calibration': compact(cal),
                          'fit_diagnostics': fd, 'calibration_diagnostics': cd,
                          'historical_last100': history_summary(full_shape_tail, step),
                          'calibration_minus_fit_harm_rate': None if fd['harm_rate_among_image_correct'] is None or
                          cd['harm_rate_among_image_correct'] is None else
                          cd['harm_rate_among_image_correct'] - fd['harm_rate_among_image_correct']}
    return {'status': 'vlm_preserve_training_fit_audit_complete', 'run_id': RUN,
            'receipt_sha256': c.sha(output / 'receipt.json'), 'parent_result_sha256': c.sha(ROOT / v3.REPORT),
            'runtime': old['runtime'], 'fit_image_only': compact(base), 'calibration_image_only': compact(calbase),
            'checkpoints': arms,
            'artifact_sha256': {name: c.sha(output / name) for name in
                               ['fit_image_only.json', 'fit_800.json', 'fit_1600.json', 'history_800.json', 'history_1600.json']},
            'training_performed': False, 'architecture_changed': False, 'GPU_used': False,
            'calibration_features_loaded': False, 'old_holdout_features_loaded': False,
            'reserved_confirmation_features_loaded': False, 'restoration_metrics_measured': False,
            'automatic_next_experiment': False,
            'limitations': ['Training-fit diagnostic, not independent generalization confirmation.',
                           'Calibration reused without selecting another checkpoint or tuning thresholds.',
                           'Low training harm with higher calibration harm supports a generalization gap, not a unique root cause.',
                           'Giving the correction head image logits remains a conditional proposal, not an implemented or proven fix.']}


def audit():
    import torch
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(), 'CPU only')
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    b, old, identity, markers = context()
    runtime = {'torch': str(torch.__version__), 'build': c.digest(torch.__config__.show()),
               'python': platform.python_version(), 'threads': 8, 'interop': 1, 'deterministic': True,
               'cpu': next(l for l in Path('/proc/cpuinfo').read_text().splitlines() if l.startswith('model name'))}
    require(runtime == old['runtime'], 'use original qualified CPU runtime; do not silently migrate predictions')
    output = ROOT / OUTPUT
    receipt = {'run_id': RUN, 'steps': list(STEPS), 'parent_receipt_sha256': c.sha(ROOT / v3.OUTPUT / 'receipt.json'),
               'parent_result_sha256': c.sha(ROOT / v3.REPORT), 'runtime': runtime, 'checkpoints': markers,
               'sources': {n: c.sha(ROOT/n) for n in SOURCES}}
    c.write_new(output / 'receipt.json', receipt)
    if (ROOT / REPORT).exists():
        verify()
        print('VLM_FIT_AUDIT_ALREADY_COMPLETE no_feature_reread=true', flush=True)
        return
    data = v3.load_role(b, 'fit')  # The ONLY feature-loading call in this script.
    m, unused_optimizer = v3.parent.model(b)
    del unused_optimizer
    m.eval().requires_grad_(False)
    image_hash = v3.state.io.parameter_hash(dict(m.image.named_parameters()))
    require(image_hash == old['frozen_image_hash'], 'wrong frozen image branch')
    c.write_new(output / 'fit_image_only.json', v3.parent.evaluate(m, data, False))
    history800 = None
    for step in STEPS:
        checkpoint = v3.state.load_step(ROOT / v3.OUTPUT, step, identity)
        if step == 800:
            history800 = checkpoint['history']
        else:
            require(checkpoint['history'][:800] == history800, '800/1600 histories are not from same trajectory')
        m.vlm.load_state_dict(checkpoint['vlm'], strict=True)
        before = v3.state.io.parameter_hash(m.vlm.state_dict())
        require(before == markers[str(step)]['parameter_hash'], 'loaded VLM differs from checkpoint')
        c.write_new(output / f'fit_{step}.json', v3.parent.evaluate(m, data, True))
        c.write_new(output / f'history_{step}.json', {'step': step, 'tail': checkpoint['history'][-100:]})
        require(v3.state.io.parameter_hash(m.vlm.state_dict()) == before and
                v3.state.io.parameter_hash(dict(m.image.named_parameters())) == image_hash and
                all(p.grad is None for p in m.parameters()), 'evaluation modified model')
        print(f'VLM_FIT_AUDIT_EVALUATED step={step} fit_records=2560 no_updates=true', flush=True)
    c.write_new(ROOT / REPORT, build_report(b, old, markers, output))
    verify()


def verify():
    b, old, _, markers = context()
    output = ROOT / OUTPUT
    receipt = c.read(output / 'receipt.json')
    require(receipt['run_id'] == RUN and receipt['steps'] == list(STEPS) and receipt['checkpoints'] == markers and
            receipt['parent_receipt_sha256'] == c.sha(ROOT / v3.OUTPUT / 'receipt.json') and
            receipt['parent_result_sha256'] == c.sha(ROOT / v3.REPORT) and receipt['runtime'] == old['runtime'],
            'audit parent identity changed')
    require(receipt['sources'] == {n: c.sha(ROOT/n) for n in SOURCES}, 'audit code changed')
    require(c.read(ROOT / REPORT) == build_report(b, old, markers, output), 'audit result differs from saved per-record evaluation')
    print('VLM_FIT_AUDIT_RESULT_VERIFIED no_training=true no_GPU=true', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('phase', choices=['audit', 'verify'])
    args = ap.parse_args()
    require(str(ROOT) == v3.parent.s.analysis.ACTIVE_ROOT, 'destination workspace only')
    {'audit': audit, 'verify': verify}[args.phase]()


if __name__ == '__main__':
    main()
