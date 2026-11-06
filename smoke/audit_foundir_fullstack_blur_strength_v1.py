#!/usr/bin/env python3
"""Fixed post-quantization residual-strength ablation, official600, one node."""
import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ['FOUNDIR_OFFICIAL_NNODES'] = '4'  # Frozen parent identity only.
from smoke import official_vlm_v4_metric_identity_fix as fix
c, require = fix.c, fix.require
PARENT = ROOT/fix.x.OUTPUT
PARENT_REPORT = ROOT/fix.x.REPORT
OUT = ROOT/'checkpoints/foundir_fullstack_blur_strength_v1'
REPORT = ROOT/'reports/foundir_fullstack_blur_strength_v1_result_2026-11-06.json'
SOURCES = ['smoke/audit_foundir_fullstack_blur_strength_v1.py',
           'smoke/run_foundir_fullstack_blur_strength_v1.sh']
SCALES = {'half':(1,2), 'three_quarters':(3,4), 'five_quarters':(5,4), 'three_halves':(3,2)}
THRESHOLDS = {'01Blur':(25.9,.18),'05Noise':(38.9,.10),'08Haze':(22.1,.19),
              '10Rain':(34.2,.08),'14Lowlight':(18.9,.28)}


def residual_image(lq, pred, numerator, denominator):
    """No GT input. Integer numerator, round-to-even, clip, output uint8."""
    import numpy as np
    require(lq.dtype==pred.dtype==np.uint8 and lq.shape==pred.shape, 'uint8 aligned images required')
    require(type(numerator) is int and type(denominator) is int and 0<=numerator<=8 and 1<=denominator<=8,
            'bounded rational strength required')
    value = (denominator-numerator)*lq.astype(np.int32)+numerator*pred.astype(np.int32)
    return np.rint(value.astype(np.float64)/denominator).clip(0,255).astype(np.uint8)


def manifest():
    parent=fix.verify(); report=c.read(PARENT_REPORT); original=c.read(parent/'receipt.json')
    rows=original['rows']; deployed={r['record_id']:r for r in report['arms']['conditional_v4']['rows']}
    require(len(rows)==len(deployed)==600,'official600 required')
    plans=[];parent_receipt_sha=c.sha(parent/'receipt.json')
    for row in rows:
        record=parent/'results'/(row['record_id']+'.json'); saved=c.read(record); selected=deployed[row['record_id']]
        require(saved['row']==row and saved['receipt_sha256']==parent_receipt_sha, 'parent row changed')
        require(saved['decisions']['conditional_v4']['anchor']==selected['anchor'], 'deployed route changed')
        require(all(saved['metrics'][selected['anchor']][m]==selected[m] for m in ('psnr','lpips','ssim')), 'parent metric changed')
        plans.append({'row':row,'deployed':selected,'result_sha256':c.sha(record),
                      'output':saved['outputs'][selected['anchor']]})
    return {'receipt_sha256':c.sha(parent/'receipt.json'),'parent_report_sha256':c.sha(PARENT_REPORT),
            'metricfix_sha256':c.sha(parent/'metric_identity_fix.json'),
            'sources':{s:c.sha(ROOT/s) for s in SOURCES},'rows':plans,'world':2,'nodes':[0],
            'scales':{k:list(v) for k,v in SCALES.items()},'metric_identity':original['metric_identity'],
            'apply_when':'existing conditional_v4 predicted anchor == blur',
            'unchanged_other_routes':'reuse exact parent scores and output identities',
            'transform':'round_even(clip(LQ + scale * (saved_uint8_output - LQ), 0, 255))',
            'not_equivalent_to_raw_alpha':True,'GT_used_for_selection':False,'training':False}


def prepare():
    value=manifest();c.write_new(OUT/'receipt.json',value)
    print('BLUR_STRENGTH_PREPARED records=600 affected='+str(sum(r['deployed']['anchor']=='blur' for r in value['rows'])),flush=True)


def bound():
    value=c.read(OUT/'receipt.json');require(value==manifest(),'ablation source or parent changed');return value


def load_png(path, expected):
    import numpy as np
    from PIL import Image
    data=path.read_bytes();require(hashlib.sha256(data).hexdigest()==expected,'PNG hash changed')
    with Image.open(io.BytesIO(data)) as im:return np.array(im.convert('RGB'),copy=True)


def save_png(path, array):
    from PIL import Image
    buffer=io.BytesIO();Image.fromarray(array).save(buffer,format='PNG');data=buffer.getvalue()
    digest=hashlib.sha256(data).hexdigest();path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():require(c.sha(path)==digest,'existing PNG differs; preserve and stop')
    else:
        tmp=path.with_name(path.name+f'.tmp.{os.getpid()}.{time.time_ns()}')
        with tmp.open('xb') as stream:stream.write(data);stream.flush();os.fsync(stream.fileno())
        os.replace(tmp,path)
    return {'path':str(path.relative_to(OUT)), 'sha256':digest}


def check_record(record, item, receipt, rank):
    require(record['receipt_sha256']==receipt and record['item']==item and record['rank']==rank,'record identity changed')
    require(set(record['arms'])==set(SCALES),'arm incomplete')
    for arm in record['arms'].values():
        require(set(arm['metrics'])=={'psnr','lpips','ssim'} and all(math.isfinite(v) for v in arm['metrics'].values()),'invalid metrics')
        if not arm['reused_parent']:require(c.sha(OUT/arm['output']['path'])==arm['output']['sha256'],'candidate PNG changed')


def run(node):
    require(node==0,'single-node task requires new node0')
    import numpy as np
    import torch
    value=bound();receipt=c.sha(OUT/'receipt.json');rank=int(os.environ['LOCAL_RANK'])
    require(rank in (0,1) and int(os.environ['LOCAL_WORLD_SIZE'])==2,'single node two workers')
    assigned=value['rows'][rank::2];marker=OUT/f'worker{rank}.json'
    if marker.exists():
        completed=c.read(marker);require(completed['receipt_sha256']==receipt and completed['records']==[i['row']['record_id'] for i in assigned],'worker changed')
        for item in assigned:check_record(c.read(OUT/'records'/(item['row']['record_id']+'.json')),item,receipt,rank)
        print(f'BLUR_STRENGTH_WORKER_REUSED rank={rank}',flush=True);return
    torch.cuda.set_device(rank);device=torch.device('cuda',rank)
    models,_=fix.checked_metric_models(device,value['metric_identity']);start=time.monotonic()
    def tensor(array):return torch.from_numpy(array).permute(2,0,1).unsqueeze(0).to(device).float()/255
    for index,item in enumerate(assigned):
        row=item['row'];target=OUT/'records'/(row['record_id']+'.json')
        if target.exists():check_record(c.read(target),item,receipt,rank);continue
        arms={}
        if item['deployed']['anchor']=='blur':
            lq=load_png(ROOT/row['lq_path'],row['lq_sha256']);pred=load_png(PARENT/item['output']['path'],item['output']['sha256'])
            require(lq.shape==pred.shape==(int(row['height']),int(row['width']),3),'image dimensions changed')
            # Generate every fixed candidate before loading GT; GT only enters metric calls.
            candidates={name:residual_image(lq,pred,*scale) for name,scale in SCALES.items()}
            gt=load_png(ROOT/row['gt_path'],row['gt_sha256']);gt_t=tensor(gt)
            for name,array in candidates.items():
                spec=save_png(OUT/'candidates'/(row['record_id']+'__'+name+'.png'),array)
                image=tensor(array)
                with torch.no_grad():metrics={m:float(model(image,gt_t).item()) for m,model in models.items()}
                arms[name]={'metrics':metrics,'output':spec,'reused_parent':False};del image
            del gt_t,candidates
        else:
            arms={name:{'metrics':{m:item['deployed'][m] for m in ('psnr','lpips','ssim')},'output':item['output'],'reused_parent':True} for name in SCALES}
        record={'receipt_sha256':receipt,'item':item,'rank':rank,'arms':arms}
        check_record(record,item,receipt,rank);c.write_new(target,record)
        if index%10==0:print(f'BLUR_STRENGTH_PROGRESS rank={rank} records={index+1}/{len(assigned)}',flush=True)
    c.write_new(marker,{'receipt_sha256':receipt,'rank':rank,'records':[i['row']['record_id'] for i in assigned],
                       'seconds':time.monotonic()-start,'peak_allocated_gib':torch.cuda.max_memory_allocated()/1024**3})
    print(f'BLUR_STRENGTH_WORKER_COMPLETE rank={rank}',flush=True)


def summarize(records):
    require(len(records)==600 and len({r['item']['row']['record_id'] for r in records})==600,'600 unique records required')
    result={}
    for name in SCALES:
        per_category={};checks={}
        for category,(psnr,lpips) in THRESHOLDS.items():
            rows=[r['arms'][name]['metrics'] for r in records if r['item']['row']['category']==category]
            require(len(rows)=={'01Blur':150,'05Noise':100,'08Haze':200,'10Rain':100,'14Lowlight':50}[category],'category count changed')
            metrics={m:math.fsum(r[m] for r in rows)/len(rows) for m in ('psnr','lpips','ssim')}
            per_category[category]=metrics;checks[category]={'psnr':metrics['psnr']>=psnr,'lpips':metrics['lpips']<=lpips}
        result[name]={'per_category':per_category,'checks':checks,'all_ten_pass':all(all(v.values()) for v in checks.values())}
    return result


def finish():
    value=bound();receipt=c.sha(OUT/'receipt.json');records=[]
    for rank in range(2):
        assigned=value['rows'][rank::2];marker=c.read(OUT/f'worker{rank}.json')
        require(marker['receipt_sha256']==receipt and marker['rank']==rank and marker['records']==[i['row']['record_id'] for i in assigned],'worker incomplete')
        for item in assigned:
            record=c.read(OUT/'records'/(item['row']['record_id']+'.json'));check_record(record,item,receipt,rank);records.append(record)
    arms=summarize(records)
    report={'status':'complete','records':600,'affected_records':sum(r['item']['deployed']['anchor']=='blur' for r in records),
            'receipt_sha256':receipt,'arms':arms,'parent':c.read(PARENT_REPORT)['arms']['conditional_v4']['per_category'],
            'GT_used_for_selection':False,'training':False,'new_FLUX_generation':False,
            'scope':'fixed post-uint8 output ablation, not raw-alpha equivalence or new learned expert',
            'independent_blind_test':False,'selected_for_deployment':None}
    c.write_new(REPORT,report);print('BLUR_STRENGTH_VERIFIED '+json.dumps(arms),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','run','finish']);p.add_argument('--node',type=int,default=0)
    args=p.parse_args();require(str(ROOT)==fix.producer.ACTIVE,'destination workspace only')
    if args.phase=='run':run(args.node)
    else:globals()[args.phase]()
