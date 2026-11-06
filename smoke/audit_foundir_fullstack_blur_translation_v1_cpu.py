#!/usr/bin/env python3
"""Read-only global-translation diagnostic; never publish shifted restorations."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import time

import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
PARENT=ROOT/'checkpoints/foundir_official_vlm_router_v4/official_vlm_v4_frozen4800_v1_4x2'
OUT=ROOT/'checkpoints/foundir_fullstack_blur_translation_audit_v1'
REPORT=ROOT/'reports/foundir_fullstack_blur_translation_audit_v1_2026-11-06.json'


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def require(ok,text):
    if not ok:raise ValueError(text)
def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():require(read(p)==v,'existing immutable result differs');return
    t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n');t.replace(p)
def pixels(p,expected):
    data=p.read_bytes();require(hashlib.sha256(data).hexdigest()==expected,'PNG identity changed')
    with Image.open(io.BytesIO(data)) as im:return np.array(im.convert('RGB'),dtype=np.float32)/255


def spectrum(image,window=True):
    gray=image@np.array([.2126,.7152,.0722],np.float32);gray=gray-gray.mean()
    if window:gray=gray*np.outer(np.hanning(gray.shape[0]),np.hanning(gray.shape[1]))
    return np.fft.rfft2(gray)


def translation(reference,moving,shape):
    cross=reference*np.conj(moving);power=np.abs(cross)
    normalized=np.divide(cross,power,out=np.zeros_like(cross),where=power>1e-12)
    response=np.fft.irfft2(normalized,s=shape)
    y,x=np.unravel_index(int(response.argmax()),shape)
    if y>shape[0]//2:y-=shape[0]
    if x>shape[1]//2:x-=shape[1]
    return [int(y),int(x)]


def common_psnr(moving,reference,shift=(0,0),margin=16):
    y,x=shift;h,w=reference.shape[:2]
    require(abs(y)<=8 and abs(x)<=8 and min(h,w)>2*margin,'outside fixed diagnostic window')
    error=moving[margin-y:h-margin-y,margin-x:w-margin-x]-reference[margin:h-margin,margin:w-margin]
    mse=float(np.square(error,dtype=np.float64).mean())
    return -10*math.log10(mse+1e-8)


def analyze(row,identity):
    dest=OUT/'records'/(row['record_id']+'.json')
    if dest.exists():r=read(dest);require(r['identity']==identity,'binding changed');return r
    old=read(PARENT/'results'/(row['record_id']+'.json'));require(old['row']==row,'parent row changed')
    spec=old['outputs']['blur'];lq=pixels(ROOT/row['lq_path'],row['lq_sha256']);gt=pixels(ROOT/row['gt_path'],row['gt_sha256']);pred=pixels(PARENT/spec['path'],spec['sha256'])
    require(lq.shape==gt.shape==pred.shape,'shape mismatch');fl,fg,fp=[spectrum(v) for v in (lq,gt,pred)];shape=gt.shape[:2]
    shifts={'prediction_to_GT_diagnostic':translation(fg,fp,shape),'prediction_to_LQ_without_GT':translation(fl,fp,shape),'LQ_to_GT_diagnostic':translation(fg,fl,shape)}
    baseline=common_psnr(pred,gt);scores={}
    for name,shift in shifts.items():
        moving=lq if name=='LQ_to_GT_diagnostic' else pred
        if max(abs(v) for v in shift)>8:scores[name]=None
        else:scores[name]={'common_interior_psnr':common_psnr(moving,gt,shift),'delta_vs_unshifted_same_image':common_psnr(moving,gt,shift)-common_psnr(moving,gt)}
    result={'identity':identity,'record_id':row['record_id'],'parent_psnr':old['metrics']['blur']['psnr'],
            'common_interior_baseline_psnr':baseline,'estimated_integer_shifts':shifts,'diagnostic_scores':scores,
            'GT_used_only_for_analysis':True,'new_output_saved':False}
    write(dest,result);return result


def main():
    start=time.monotonic();require(not REPORT.exists(),'preserve completed audit');parent=read(PARENT/'receipt.json');rows=[r for r in parent['rows'] if r['category']=='01Blur'];require(len(rows)==150,'Blur150 required')
    identity={'source_sha256':sha(Path(__file__)),'parent_receipt_sha256':sha(PARENT/'receipt.json'),
              'method':'one_Hann_luma_phase_correlation_integer_peak_no_search','max_shift_pixels':8,'common_interior_margin':16,
              'not_official_metrics':True,'no_output_policy_selected':True}
    write(OUT/'receipt.json',identity);records=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        for row in pool.map(lambda row:analyze(row,identity),rows):
            records.append(row)
            if len(records)%25==0:print(f'BLUR_TRANSLATION_AUDIT {len(records)}/150',flush=True)
    summary={}
    for name in records[0]['diagnostic_scores']:
        valid=[r['diagnostic_scores'][name] for r in records if r['diagnostic_scores'][name] is not None];gains=[r['delta_vs_unshifted_same_image'] for r in valid]
        summary[name]={'within_fixed8px_count':len(valid),'outside_fixed8px_count':150-len(valid),
                       'nonzero_shift_count':sum(r['estimated_integer_shifts'][name]!=[0,0] for r in records),
                       'mean_delta_valid_only':sum(gains)/len(gains) if gains else None,
                       'median_delta_valid_only':statistics.median(gains) if gains else None,
                       'min_delta_valid_only':min(gains) if gains else None,'max_delta_valid_only':max(gains) if gains else None}
    result={'status':'read_only_diagnostic_complete','identity':identity,'records':records,'summary':summary,
            'seconds':time.monotonic()-start,'exit_code':0,'GPU_hours':0,'new_training':False,'new_restorations':False,
            'official_scores_or_gates_replaced':False,'GT_used_for_deployable_selection':False,
            'limitations':'Integer global translation diagnostic only; not local warp/subpixel alignment, not proof against all geometry failures.'}
    write(REPORT,result);print('BLUR_TRANSLATION_AUDIT_END '+json.dumps({'summary':summary,'seconds':result['seconds']}),flush=True)


if __name__=='__main__':main()
