#!/usr/bin/env python3
"""Describe already fixed training tiles by metadata-defined view exposure; no fitting."""
import json
import math
from pathlib import Path
import sys
import time
import numpy as np
from PIL import Image
import io
import hashlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from smoke.existing_flux_replay_contract_v1 import read, sha, write_new

CURRENT = ROOT/'checkpoints/foundir_blur_multiview_context_v1'
PRIOR = ROOT/'checkpoints/foundir_blur_full_context_v1'
TRAIN = ROOT/'checkpoints/foundir_blur_multiview_v1'
RESULT = ROOT/'reports/foundir_blur_multiview_context_v1_result_2026-11-07.json'


def image(path, expected):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected: raise ValueError('image SHA changed')
    with Image.open(io.BytesIO(data)) as im: return np.array(im.convert('RGB'), copy=True)


def main():
    begun=time.monotonic(); b=read(CURRENT/'receipt.json'); prior=read(PRIOR/'receipt.json')
    result=read(RESULT); training=read(TRAIN/'receipt.json')
    assert result['status']=='training8_full_context_diagnostic_complete' and result['weights_unchanged']
    assert b['items']==prior['items'] and len(b['items'])==8
    assert b['prior_receipt_sha256']==sha(PRIOR/'receipt.json')
    assert b['official_receipt_sha256']==sha(ROOT/'checkpoints/foundir_blur_multiview_official_v1/receipt.json')
    records=[]
    for item in b['items']:
        row=item['row']; rid=row['record_id']
        views={tuple(v['crop_xy']) for v in training['views'] if v['row']==row}
        assert len(views)==4
        lq,gt=[image(ROOT/row[k+'_path'],row[k+'_sha256']) for k in ('lq','gt')]
        assert lq.shape==gt.shape==(1080,1920,3)
        for y in (0,384,568):
            for x in (0,384,768,1152,1408):
                key=f'{rid}__x{x}_y{y}'; target=gt[y:y+512,x:x+512].astype(np.float64)/255
                images={'lq':lq[y:y+512,x:x+512].astype(np.float64)/255}
                for name,out,receipt in (('step1280',PRIOR,prior),('step1792',CURRENT,b)):
                    marker=read(out/'tiles'/(key+'.json'))
                    assert marker['identity']['item']==item and marker['identity']['tile']['key']==key
                    assert marker['identity']['receipt_sha256']==sha(out/'receipt.json') and not marker['GT_input']
                    spec=marker['png']; images[name]=image(out/spec['path'],spec['sha256']).astype(np.float64)/255
                metrics={name:{'mse':float(np.mean((array-target)**2))} for name,array in images.items()}
                for values in metrics.values():
                    assert values['mse']>0
                    values['psnr']=-10*math.log10(values['mse'])
                records.append({'record_id':rid,'x':x,'y':y,'trained_grid_view':(x,y) in views,'metrics':metrics,
                                'PSNR_change_1792_minus1280':metrics['step1792']['psnr']-metrics['step1280']['psnr']})
    assert len(records)==120 and sum(r['trained_grid_view'] for r in records)==32
    groups={}
    for name,rows in [('all',records),('trained32',[r for r in records if r['trained_grid_view']]),
                      ('other88',[r for r in records if not r['trained_grid_view']])]:
        means={arm:{metric:math.fsum(r['metrics'][arm][metric] for r in rows)/len(rows)
                    for metric in ('mse','psnr')} for arm in ('lq','step1280','step1792')}
        groups[name]={'count':len(rows),'means':means,
                      'PSNR_change_1792_minus1280':means['step1792']['psnr']-means['step1280']['psnr'],
                      'PSNR_gain1792_over_LQ':means['step1792']['psnr']-means['lq']['psnr'],
                      'new_PSNR_below_LQ_count':sum(r['metrics']['step1792']['psnr']<r['metrics']['lq']['psnr'] for r in rows),
                      'new_PSNR_below1280_count':sum(r['PSNR_change_1792_minus1280']<0 for r in rows)}
    value={'status':'fixed_training120_tile_exposure_description_complete','groups':groups,'records':records,
           'scope':'training_fit_only','not_independent_sources':True,'new_predictions_generated':False,
           'GT_used_only_after_outputs_fixed':True,'parameters_fitted':False,'GT_output_selection':False,
           'LPIPS_recomputed':False,'new_quality_gate':False,
           'note':'Overlapping tiles are correlated;32/88 groups differ in content/difficulty, not randomized treatment arms.',
           'result_sha256':sha(RESULT),'training_receipt_sha256':sha(TRAIN/'receipt.json'),
           'source_sha256':sha(Path(__file__)),'seconds':time.monotonic()-begun,'GPU_hours':0,'exit_code':0}
    write_new(ROOT/'reports/foundir_blur_multiview_tile_fit_v1_CPU_2026-11-07.json',value)
    print('BLUR_MULTIVIEW_TILE_FIT_VERIFIED '+json.dumps({k:v for k,v in value.items() if k!='records'}),flush=True)


if __name__=='__main__':main()
