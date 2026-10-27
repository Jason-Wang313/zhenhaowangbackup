"""User-frozen ENERGY-ONLY gain gate, relative to a matched FLUX baseline.

No restoration model, dataset access, threshold fitting or absolute Restormer
target is hidden here. PSNR is averaged in dB; LPIPS uses the same arithmetic
per-record averaging as the frozen reference evaluation.
"""
from __future__ import annotations

import math

MINIMUM_PSNR_GAIN_DB = 0.30
REFERENCE_LPIPS = 0.050362067760033824
REFERENCE_LPIPS_DROP = 0.005
MINIMUM_LPIPS_REDUCTION_FRACTION = REFERENCE_LPIPS_DROP / REFERENCE_LPIPS


def compare_paired(baseline_rows, energy_rows, *, baseline_protocol, energy_protocol):
    """Score exactly paired records, not unrelated checkpoint averages.

Each row needs record_id, source_group, crop_xy, input_uint8_sha256, psnr and
lpips. Protocols must contain the same frozen data/crop/quantization/metric
contract. Input hashes and source identities must match, not just record IDs.
The caller must still enforce protected-data and GT-free inference rules.
    """
    required_protocol = ('manifest_sha256', 'crop_size', 'crop_seed',
                         'quantization', 'lpips_definition', 'baseline_checkpoint_sha256',
                         'baseline_alpha', 'inference_steps', 'inference_seed_namespace')
    if not isinstance(baseline_protocol, dict) or not isinstance(energy_protocol, dict):
        raise ValueError('explicit paired evaluation protocols required')
    if baseline_protocol != energy_protocol or any(k not in baseline_protocol for k in required_protocol):
        raise ValueError('baseline and energy evaluation protocols must match exactly')

    def index(rows):
        result = {}
        for row in rows:
            for key in ('record_id', 'source_group', 'crop_xy', 'input_uint8_sha256', 'psnr', 'lpips'):
                if key not in row:
                    raise ValueError('missing record field: ' + key)
            rid = row['record_id']
            if not isinstance(rid, str) or not rid or rid in result:
                raise ValueError('empty or duplicate record ID')
            if not isinstance(row['source_group'], str) or not row['source_group']:
                raise ValueError('missing source identity')
            digest = row['input_uint8_sha256']
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
                raise ValueError('invalid input-pixel identity')
            xy = row['crop_xy']
            if not isinstance(xy, (list, tuple)) or len(xy) != 2 or any(type(v) is not int or v < 0 for v in xy):
                raise ValueError('invalid crop identity')
            for key in ('psnr', 'lpips'):
                value = row[key]
                if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
                    raise ValueError('nonfinite/non-numeric metric')
            if row['lpips'] < 0:
                raise ValueError('LPIPS must be nonnegative')
            result[rid] = row
        if not result:
            raise ValueError('empty evaluation')
        return result

    baseline, energy = index(baseline_rows), index(energy_rows)
    if baseline.keys() != energy.keys():
        raise ValueError('record sets differ')
    for rid in baseline:
        for key in ('source_group', 'input_uint8_sha256'):
            if baseline[rid][key] != energy[rid][key]:
                raise ValueError('paired identity changed: ' + key)
        if tuple(baseline[rid]['crop_xy']) != tuple(energy[rid]['crop_xy']):
            raise ValueError('paired crop changed')
    def average(rows, metric):
        return math.fsum(rows[rid][metric] for rid in sorted(rows)) / len(rows)
    bp, bl = average(baseline, 'psnr'), average(baseline, 'lpips')
    ep, el = average(energy, 'psnr'), average(energy, 'lpips')
    pt = bp + MINIMUM_PSNR_GAIN_DB
    lt = bl * (1.0 - MINIMUM_LPIPS_REDUCTION_FRACTION)
    return dict(records=len(baseline), baseline_psnr=bp, baseline_lpips=bl,
                energy_psnr=ep, energy_lpips=el, psnr_gain_db=ep-bp,
                lpips_relative_reduction=(bl-el)/bl if bl > 0 else None,
                psnr_target=pt, lpips_target=lt,
                psnr_pass=ep >= pt, lpips_pass=el <= lt,
                joint_gate_pass=ep >= pt and el <= lt,
                baseline_gain_counted_as_energy=False)
