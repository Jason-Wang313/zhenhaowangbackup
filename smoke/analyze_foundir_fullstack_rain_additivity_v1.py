#!/usr/bin/env python3
"""Check additive bright-rain premise on existing official pairs; no fitting."""
import hashlib
import io
import json
import math
from pathlib import Path
import time

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT/'checkpoints/foundir_official_vlm_router_v4/official_vlm_v4_frozen4800_v1_4x2'
REPORT = ROOT/'reports/foundir_fullstack_rain_additivity_v1_2026-11-06.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def png(path, expected):
    data = path.read_bytes()
    assert hashlib.sha256(data).hexdigest() == expected
    with Image.open(io.BytesIO(data)) as im:
        return np.array(im.convert('RGB'), dtype=np.int16)


def main():
    start = time.monotonic()
    receipt = json.loads((PARENT/'receipt.json').read_text())
    rows = [r for r in receipt['rows'] if r['category']=='10Rain']
    assert len(rows)==100
    results=[]
    for index,row in enumerate(rows):
        old = json.loads((PARENT/'results'/(row['record_id']+'.json')).read_text())
        assert old['row']==row and old['decisions']['conditional_v4']['anchor']=='rain'
        lq = png(ROOT/row['lq_path'],row['lq_sha256'])
        gt = png(ROOT/row['gt_path'],row['gt_sha256'])
        spec=old['outputs']['rain'];pred=png(PARENT/spec['path'],spec['sha256'])
        assert lq.shape==gt.shape==pred.shape
        delta=lq-gt;err=(pred-gt).astype(np.float64)**2;outside=pred>lq
        results.append({'record_id':row['record_id'],'components':int(delta.size),
            'negative_LQ_minus_GT_components':int((delta<0).sum()),
            'below_GT_by_more_than_one_gray_components':int((delta < -1).sum()),
            'minimum_LQ_minus_GT_255':int(delta.min()),
            'equal_LQ_GT_components':int((delta==0).sum()),
            'prediction_brighter_than_LQ_components':int(outside.sum()),
            'fraction_error_energy_where_prediction_exceeds_LQ':float(err[outside].sum()/max(err.sum(),1e-30)),
            'pixel_hashes':{'lq':row['lq_sha256'],'gt':row['gt_sha256'],'prediction':spec['sha256']}})
        if (index+1)%10==0:print(f'RAIN_ADDITIVITY records={index+1}/100 seconds={time.monotonic()-start:.1f}',flush=True)
    count=sum(r['components'] for r in results)
    report={'status':'complete','records':results,'record_count':100,'source_sha256':sha(Path(__file__)),
        'parent_receipt_sha256':sha(PARENT/'receipt.json'),
        'negative_LQ_minus_GT_component_fraction':sum(r['negative_LQ_minus_GT_components'] for r in results)/count,
        'LQ_below_GT_by_more_than_one_gray_fraction':sum(r['below_GT_by_more_than_one_gray_components'] for r in results)/count,
        'equal_LQ_GT_component_fraction':sum(r['equal_LQ_GT_components'] for r in results)/count,
        'prediction_brighter_than_LQ_component_fraction':sum(r['prediction_brighter_than_LQ_components'] for r in results)/count,
        'mean_error_energy_fraction_where_prediction_exceeds_LQ':math.fsum(r['fraction_error_energy_where_prediction_exceeds_LQ'] for r in results)/100,
        'minimum_LQ_minus_GT_255':min(r['minimum_LQ_minus_GT_255'] for r in results),
        'seconds':time.monotonic()-start,'gpus':0,'exit_code':0,'training':False,'parameter_fitting':False,
        'new_restorations':False,'GT_used_only_for_diagnostics':True,'independent_blind_test':False,
        'limitation':'Empirical premise check on reused official benchmark; not proof of a universal rain formation model. Projection, if tested, must use LQ and prediction only.'}
    with REPORT.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print('RAIN_ADDITIVITY_COMPLETE '+json.dumps({k:v for k,v in report.items() if k!='records'}),flush=True)


if __name__=='__main__':main()
