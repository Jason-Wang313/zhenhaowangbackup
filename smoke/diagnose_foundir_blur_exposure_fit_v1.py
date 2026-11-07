#!/usr/bin/env python3
"""Same historical training64 inputs, frozen final1280; no optimizer or new gate."""
import argparse
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'third_party/flux2/src'))
from smoke import diagnose_foundir_blur_coverage_fit_v2 as fit
from smoke import evaluate_foundir_blur_exposure_official_2x2_v2 as official
from smoke.existing_flux_replay_contract_v1 import read, sha, write_new as write

OLD_OUT, OLD_REPORT = fit.OUT, fit.REPORT
fit.OUT = ROOT / 'checkpoints/foundir_blur_exposure_fit_v1'
fit.REPORT = ROOT / 'reports/foundir_blur_exposure_fit_v1_result_2026-11-07.json'
fit.official = official
fit.SOURCES = ['smoke/diagnose_foundir_blur_exposure_fit_v1.py',
               'smoke/run_foundir_fullstack_blur_exposure_fit_v1.sh',
               'smoke/diagnose_foundir_blur_coverage_fit_v2.py']
COMPARISON = ROOT / 'reports/foundir_blur_exposure_fit_v1_comparison_2026-11-07.json'


def prepare():
    start = time.monotonic()
    assert official.RECIPE['selected_step'] == 1280
    old = read(OLD_OUT / 'receipt.json')
    assert old['items'] == fit.train.bound()['rows'][:64], 'historical training64 changed'
    fit.prepare()
    receipt = fit.bound()
    final = read(official.train.REPORT)['final_checkpoint']
    assert final['step'] == 1280 and receipt['parameter_hash'] == final['parameter_hash']
    assert receipt['items'] == old['items'] and receipt['weight'] == read(official.REPORT)['weight']
    write(ROOT / 'reports/foundir_blur_exposure_fit_v1_preflight_2026-11-07.json',
          {'status': 'destination_CPU_preflight_pass', 'fixed_training_records': 64,
           'same_historical_training64': True, 'frozen_step': 1280,
           'parameter_hash': receipt['parameter_hash'], 'weight': receipt['weight'],
           'receipt_sha256': sha(fit.OUT / 'receipt.json'),
           'old_diagnostic_sha256': sha(OLD_REPORT), 'new_training': False,
           'new_quality_gate': False, 'seconds': time.monotonic() - start,
           'GPU_hours': 0, 'exit_code': 0})


def run():
    fit.run()


def finish():
    fit.finish()
    old, new = read(OLD_REPORT), read(fit.REPORT)
    a = {r['item']['row']['record_id']: r for r in old['records']}
    b = {r['item']['row']['record_id']: r for r in new['records']}
    assert len(a) == len(b) == 64 and a.keys() == b.keys()
    paired = []
    for rid in sorted(a):
        assert a[rid]['item'] == b[rid]['item']
        assert a[rid]['metrics_training_fit_only']['lq'] == b[rid]['metrics_training_fit_only']['lq']
        x = a[rid]['metrics_training_fit_only']['candidate']
        y = b[rid]['metrics_training_fit_only']['candidate']
        paired.append({'record_id': rid, 'step200': x, 'step1280': y,
                       'PSNR_delta': y['psnr'] - x['psnr'], 'LPIPS_delta': y['lpips'] - x['lpips']})
    write(COMPARISON, {'scope': 'training_fit_only_same64_frozen_checkpoint_comparison',
          'step200_summary': old['summary'], 'step1280_summary': new['summary'],
          'PSNR_delta': math.fsum(r['PSNR_delta'] for r in paired) / 64,
          'LPIPS_delta': math.fsum(r['LPIPS_delta'] for r in paired) / 64,
          'joint_better_than_step200': sum(r['PSNR_delta'] > 0 and r['LPIPS_delta'] < 0 for r in paired),
          'records': paired, 'parent_result_sha256': {'step200': sha(OLD_REPORT), 'step1280': sha(fit.REPORT)},
          'new_quality_gate': False, 'new_training': False, 'official_Blur_pass': False})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['prepare', 'run', 'finish'])
    phase = parser.parse_args().phase
    globals()[phase]()
    print('BLUR_EXPOSURE_FIT_PHASE_END phase=' + phase + ' status=0', flush=True)
