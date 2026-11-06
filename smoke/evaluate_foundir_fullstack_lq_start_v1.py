#!/usr/bin/env python3
"""One fixed LQ-start Blur counterfactual; frozen official routing, all600 summary."""
import argparse
import hashlib
import io
import math
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'third_party/flux2/src'))
os.environ['FOUNDIR_OFFICIAL_NNODES']='4'  # Immutable parent only.
from smoke import official_vlm_v4_metric_identity_fix as fix
c,x,require=fix.c,fix.x,fix.require
producer=fix.producer
from lowlight_restoration.flux_lq_start_v1 import select_suffix,restore_lq_start
OUT=ROOT/'checkpoints/foundir_fullstack_lq_start_v1'
REPORT=ROOT/'reports/foundir_fullstack_lq_start_v1_result_2026-11-06.json'
PREFLIGHT=ROOT/'reports/foundir_fullstack_lq_start_v1_preflight_2026-11-06.json'
WORLD=6
SOURCES=['smoke/evaluate_foundir_fullstack_lq_start_v1.py','smoke/run_foundir_fullstack_lq_start_v1.sh',
         'lowlight_restoration/flux_lq_start_v1.py']


def planned_rows(parent,report):
    deployed={r['record_id']:r for r in report['arms']['conditional_v4']['rows']}
    x.validate_rows(parent['rows']);require(len(deployed)==600,'600 routes required')
    result=[];oldsha=c.sha(ROOT/x.OUTPUT/'receipt.json')
    for row in parent['rows']:
        path=ROOT/x.OUTPUT/'results'/(row['record_id']+'.json');old=c.read(path);d=deployed[row['record_id']]
        require(old['row']==row and old['receipt_sha256']==oldsha,'old row changed')
        require(old['decisions']['conditional_v4']['anchor']==d['anchor'],'old route changed')
        require(all(old['metrics'][d['anchor']][m]==d[m] for m in ('psnr','lpips','ssim')),'old metrics changed')
        result.append({'row':row,'deployed':d,'parent_record_sha256':c.sha(path),'parent_output':old['outputs'][d['anchor']]})
    return result


def used_assets(parent):
    p=parent['original']['parent'];assets={}
    for key in ('flow_weights','autoencoder_weights'):assets[p['model'][key]]=p['model'][key+'_sha256']
    for task in x.ORDER:
        s=p['experts'][task]['checkpoint'];assets[s['path']]=s['sha256']
        s=p['rain_prompt'] if task=='rain' else p['four_prompts']['tensors'][task]
        path=s['path'] if task=='rain' else str(Path(p['config']['four_parent_run'])/s['path'])
        assets[path]=s['sha256']
    return assets


def manifest():
    fix.verify();parent=c.read(ROOT/x.OUTPUT/'receipt.json');report=c.read(ROOT/x.REPORT)
    rows=planned_rows(parent,report)
    return {'parent_receipt_sha256':c.sha(ROOT/x.OUTPUT/'receipt.json'),'parent_report_sha256':c.sha(ROOT/x.REPORT),
            'metricfix_sha256':c.sha(ROOT/x.OUTPUT/'metric_identity_fix.json'),
            'sources':{p:c.sha(ROOT/p) for p in SOURCES},'assets':used_assets(parent),'rows':rows,'world':WORLD,
            'nodes':[0,1,2],'local_gpus':2,'maximum_start_time':.5,'schedule':'suffix of original shifted 50-step schedule; first t<=.5',
            'initial_latent':'(1-t)*encoded_LQ+t*same_seed_noise, native BF16 arithmetic',
            'reference_time':10,'guidance':1.,'output_alpha':.75,'double_uint8':True,
            'apply_when':'frozen conditional_v4 anchor == blur','training':False,'new_route':False,'GT_input':False,
            'metric_identity':parent['metric_identity']}


def bound():
    value=c.read(OUT/'receipt.json');require(value==manifest(),'LQ-start binding changed');return value


def prepare():
    from flux2.sampling import get_schedule
    start=time.monotonic();v=manifest()
    for path,expected in v['assets'].items():require(c.sha(ROOT/path)==expected,'asset changed: '+path)
    index,suffix=select_suffix(get_schedule(50,1024))
    counts=[sum(len(x.tiles(i['row'])) for i in v['rows'][r::WORLD] if i['deployed']['anchor']=='blur') for r in range(WORLD)]
    c.write_new(OUT/'receipt.json',v)
    report={'status':'CPU_preflight_passed','receipt_sha256':c.sha(OUT/'receipt.json'),'records':600,
            'regenerated_records':sum(i['deployed']['anchor']=='blur' for i in v['rows']),
            'tiles_per_rank':counts,'tiles':sum(counts),'start_index':index,'start_time':suffix[0],
            'calls_per_tile':len(suffix)-1,'total_native_CFG_calls':sum(counts)*(len(suffix)-1),
            'world':WORLD,'nodes':3,'seconds':time.monotonic()-start,
            'multi_node_reason':'Full-resolution FLUX regeneration over thousands of tiles; materially heavier than cached metric scoring.',
            'GPU_started':False,'official_GT_decoded':False,'training':False}
    c.write_new(PREFLIGHT,report);print('LQ_START_PREPARED '+str(report),flush=True)


def load_png(path,expected):
    import numpy as np
    from PIL import Image
    data=path.read_bytes();require(hashlib.sha256(data).hexdigest()==expected,'PNG changed')
    with Image.open(io.BytesIO(data)) as image:return np.array(image.convert('RGB'),copy=True)


def record_check(record,item,receipt,rank):
    require(record['item']==item and record['receipt_sha256']==receipt and record['rank']==rank,'result identity changed')
    require(set(record['metrics'])=={'psnr','lpips','ssim'} and all(math.isfinite(v) for v in record['metrics'].values()),'invalid metrics')
    require(record['reused_parent']==(item['deployed']['anchor']!='blur'),'wrong modified population')
    if not record['reused_parent']:require(c.sha(OUT/record['output']['path'])==record['output']['sha256'],'rendered PNG changed')


def tile_read(path,identity):
    if not path.exists():return None
    record=c.read(path);require(record['identity']==identity and record['meta']['GT_input'] is False,'tile identity changed')
    require(record['meta']['maximum_start_time']==.5,'start policy changed')
    return load_png(OUT/record['png']['path'],record['png']['sha256'])


def run(node):
    import numpy as np
    import torch
    from lowlight_restoration.protocol_routed_flux import RouteBoundFlux
    from smoke import train_foundir_frozen_expert_router_mixed_driver_v1 as impl
    from smoke.expanded_vlm_h100_runtime_v4 import bounded_vae,POLICY
    require(node in range(3),'new node0..2 required');local=int(os.environ['LOCAL_RANK']);rank=2*node+local
    require(local in (0,1) and int(os.environ['LOCAL_WORLD_SIZE'])==2,'two local workers required')
    v=bound();receipt=c.sha(OUT/'receipt.json');assigned=v['rows'][rank::WORLD]
    marker=OUT/f'worker{rank}.json'
    if marker.exists():
        old=c.read(marker);require(old['receipt_sha256']==receipt and old['records']==[i['row']['record_id'] for i in assigned],'completed worker changed')
        for item in assigned:record_check(c.read(OUT/'records'/(item['row']['record_id']+'.json')),item,receipt,rank)
        print(f'LQ_START_WORKER_REUSED rank={rank}',flush=True);return
    torch.cuda.set_device(local);device=torch.device('cuda',local)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    runtime={'torch':str(torch.__version__),'cuda':torch.version.cuda,'device':torch.cuda.get_device_name(local),
             'vae_policy':POLICY,'tf32':False,'rank':rank,'node':node,'receipt_sha256':receipt}
    c.write_new(OUT/f'runtime_rank{rank}.json',runtime)
    runtime_sha=c.sha(OUT/f'runtime_rank{rank}.json');start=time.monotonic()
    parent=c.read(ROOT/x.OUTPUT/'receipt.json');p=parent['original']['parent']['model']
    for key in ('flow_weights','autoencoder_weights'):
        require(c.sha(ROOT/p[key])==p[key+'_sha256'],'base model changed before load')
    os.environ.update(KLEIN_4B_BASE_MODEL_PATH=str(ROOT/p['flow_weights']),AE_MODEL_PATH=str(ROOT/p['autoencoder_weights']))
    bank=impl.load_bank(parent['original'],device);model,gates,rules,ae,prompts=bank;ae=bounded_vae(ae)
    weights=torch.tensor([[1.,0.,0.,0.,0.]],device=device)
    require(rules.anchor(weights).name=='blur' and rules.anchor(weights).alpha==.75,'Blur protocol changed')
    selected=RouteBoundFlux(model,gates,weights);metrics,_=fix.checked_metric_models(device,v['metric_identity'])
    for number,item in enumerate(assigned):
        row=item['row'];target=OUT/'records'/(row['record_id']+'.json')
        if target.exists():record_check(c.read(target),item,receipt,rank);continue
        if item['deployed']['anchor']!='blur':
            record={'receipt_sha256':receipt,'item':item,'rank':rank,'metrics':{k:item['deployed'][k] for k in metrics},
                    'output':item['parent_output'],'reused_parent':True}
        else:
            begin=time.monotonic();array=load_png(ROOT/row['lq_path'],row['lq_sha256']);images={};tile_list=x.tiles(row)
            require(array.shape==(int(row['height']),int(row['width']),3),'LQ size changed')
            for ti,tile in enumerate(tile_list):
                identity={'receipt_sha256':receipt,'runtime_sha256':runtime_sha,'record_id':row['record_id'],
                          'lq_sha256':row['lq_sha256'],'tile':tile,'rank':rank}
                receipt_path=OUT/'tiles'/(tile['key']+'.json');image=tile_read(receipt_path,identity)
                if image is None:
                    crop,_,_=x.tile_array(array,tile);lq=producer.tensor(crop,device)
                    with torch.no_grad():
                        raw,meta=restore_lq_start(selected,ae,lq,prompts['blur'],seed=tile['seed'])
                        u8=rules.output(lq,raw,weights)
                    expected=c.read(PREFLIGHT)
                    require(meta['start_index']==expected['start_index'] and meta['model_calls']==expected['calls_per_tile'],'unexpected suffix plan')
                    image=u8[0].permute(1,2,0).cpu().numpy();path=OUT/'tile_png'/(tile['key']+'.png')
                    producer.png_once(path,image)
                    c.write_new(receipt_path,{'identity':identity,'meta':meta,'png':{'path':str(path.relative_to(OUT)),'sha256':c.sha(path)}})
                    del raw,u8,lq
                images[tile['key']]=image
                if (ti+1)%5==0 or ti+1==len(tile_list):print(f'LQ_START_TILE rank={rank} id={row["record_id"]} tile={ti+1}/{len(tile_list)} seconds={time.monotonic()-begin:.1f}',flush=True)
            full=x.stitch(row,images);path=OUT/'restored'/(row['record_id']+'.png');producer.png_once(path,full)
            # Only after the sole fixed output is published is GT read for scoring.
            gt=load_png(ROOT/row['gt_path'],row['gt_sha256']);require(gt.shape==full.shape,'GT dimensions changed')
            pred=torch.from_numpy(full.copy()).permute(2,0,1).unsqueeze(0).to(device).float()/255
            target_gt=torch.from_numpy(gt.copy()).permute(2,0,1).unsqueeze(0).to(device).float()/255
            with torch.no_grad():scores={k:float(m(pred,target_gt).item()) for k,m in metrics.items()}
            del pred,target_gt
            record={'receipt_sha256':receipt,'item':item,'rank':rank,'metrics':scores,
                    'output':{'path':str(path.relative_to(OUT)),'sha256':c.sha(path)},'reused_parent':False,'seconds':time.monotonic()-begin}
        record_check(record,item,receipt,rank);c.write_new(target,record)
        print(f'LQ_START_RECORD rank={rank} records={number+1}/{len(assigned)} id={row["record_id"]} reused={record["reused_parent"]}',flush=True)
    c.write_new(marker,{'receipt_sha256':receipt,'rank':rank,'records':[i['row']['record_id'] for i in assigned],
                       'seconds':time.monotonic()-start,'peak_allocated_gib':torch.cuda.max_memory_allocated()/1024**3})
    print(f'LQ_START_WORKER_COMPLETE rank={rank}',flush=True)


def finish():
    v=bound();receipt=c.sha(OUT/'receipt.json');records=[];signatures=[]
    for rank in range(WORLD):
        assigned=v['rows'][rank::WORLD];marker=c.read(OUT/f'worker{rank}.json');runtime=c.read(OUT/f'runtime_rank{rank}.json')
        signatures.append({k:w for k,w in runtime.items() if k not in ('rank','node')})
        require(marker['receipt_sha256']==receipt and marker['rank']==rank and marker['records']==[i['row']['record_id'] for i in assigned],'worker incomplete')
        for item in assigned:
            row=item['row'];record=c.read(OUT/'records'/(row['record_id']+'.json'));record_check(record,item,receipt,rank);records.append(record)
            if not record['reused_parent']:
                for tile in x.tiles(row):
                    identity={'receipt_sha256':receipt,'runtime_sha256':c.sha(OUT/f'runtime_rank{rank}.json'),'record_id':row['record_id'],'lq_sha256':row['lq_sha256'],'tile':tile,'rank':rank}
                    require(tile_read(OUT/'tiles'/(tile['key']+'.json'),identity) is not None,'missing tile')
    require(all(s==signatures[0] for s in signatures),'worker numerical runtimes differ')
    require(len(records)==600 and len({r['item']['row']['record_id'] for r in records})==600,'incomplete600')
    per_category={cat:{m:math.fsum(r['metrics'][m] for r in records if r['item']['row']['category']==cat)/count
                       for m in ('psnr','lpips','ssim')} for cat,count in x.COUNTS.items()}
    checks={task:{'psnr':per_category[x.CATEGORIES[task]]['psnr']>=goal[0],
                  'lpips':per_category[x.CATEGORIES[task]]['lpips']<=goal[1]} for task,goal in x.TARGETS.items()}
    result={'status':'complete','records':600,'new_blur_records':150,'reused_unchanged_records':450,
            'receipt_sha256':receipt,'per_category':per_category,'checks':checks,'all_ten_pass':all(all(v.values()) for v in checks.values()),
            'parent':c.read(ROOT/x.REPORT)['arms']['conditional_v4']['per_category'],'training':False,
            'GT_input':False,'independent_blind_test':False,'scope':'One fixed LQ-start inference counterfactual, not trained anchored flow.'}
    c.write_new(REPORT,result);print('LQ_START_RESULT_VERIFIED '+str(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','run','finish']);p.add_argument('--node',type=int,default=0)
    a=p.parse_args();require(str(ROOT)==producer.ACTIVE,'destination workspace only')
    if a.phase=='run':run(a.node)
    else:globals()[a.phase]()
