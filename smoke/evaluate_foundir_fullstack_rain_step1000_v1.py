#!/usr/bin/env python3
"""Fixed Rain step1000 official evaluation; retain accepted Haze and other outputs."""
import argparse
import copy
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
from lowlight_restoration.five_expert_flux import restore_five_expert
OUT=ROOT/'checkpoints/foundir_fullstack_rain_step1000_v1'
REPORT=ROOT/'reports/foundir_fullstack_rain_step1000_v1_result_2026-11-06.json'
PREFLIGHT=ROOT/'reports/foundir_fullstack_rain_step1000_v1_preflight_2026-11-06.json'
WORLD=6
SOURCES=['smoke/evaluate_foundir_fullstack_rain_step1000_v1.py','smoke/run_foundir_fullstack_rain_step1000_v1.sh',
         'lowlight_restoration/five_expert_flux.py', 'smoke/expanded_vlm_h100_runtime_v4.py',
         'smoke/train_foundir_frozen_expert_router_mixed_driver_v1.py']
TRAIN=ROOT/'checkpoints/foundir_fullstack_rain_continue_3x2_v1'
TRAIN_REPORT=ROOT/'reports/foundir_fullstack_rain_continue_v1_result_2026-11-06.json'
ACCEPTED=ROOT/'checkpoints/foundir_fullstack_haze_consensus_v1/accepted_followup_manifest.json'
OVERLAP=ROOT/'reports/foundir_fullstack_rain_overlap_exposure_verified_2026-11-06.json'
TRAIN_JOB=ROOT/'checkpoints/foundir_fullstack_executor_3x2_h100_v2/jobs/round4_rain_continue_3x2_v1'


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


def accepted_rows(parent,report,accepted):
    rows=planned_rows(parent,report)
    require(len(accepted['records'])==600,'accepted600 required')
    prior={r['record_id']:r for r in accepted['records']}
    require(len(prior)==600 and set(prior)=={i['row']['record_id'] for i in rows},'accepted IDs differ')
    for item in rows:
        old=prior[item['row']['record_id']]
        require(old['anchor']==item['deployed']['anchor'],'accepted route changed')
        require(set(old['metrics'])=={'psnr','lpips','ssim'} and all(math.isfinite(v) for v in old['metrics'].values()),'accepted metrics invalid')
        if old['anchor']!='haze':
            require(all(old['metrics'][k]==item['deployed'][k] for k in old['metrics']),'non-Haze metrics changed')
            output=item['parent_output'];path=Path(output['path'])
            if not path.is_absolute():path=ROOT/x.OUTPUT/path
            require(path==ROOT/old['path'] and output['sha256']==old['sha256'],'non-Haze accepted output changed')
        item['accepted']=old
    affected=[i for i in rows if i['deployed']['anchor']=='rain']
    require(len(affected)==100 and all(i['row']['category']=='10Rain' for i in affected),'frozen routing must match Rain100')
    require(sum(i['row']['category']=='10Rain' for i in rows)==100,'all Rain100 must be included')
    return rows


def evaluation_bank(original,spec):
    value=copy.deepcopy(original)
    value['parent']['experts']['rain']['checkpoint']=dict(spec)
    return value


def plan():
    fix.verify();parent=c.read(ROOT/x.OUTPUT/'receipt.json');report=c.read(ROOT/x.REPORT)
    accepted=c.read(ACCEPTED)
    require(accepted['checks']['08Haze']=={'psnr':True,'lpips':True},'accepted Haze missing')
    require(accepted['GT_used_for_per_image_selection'] is False,'GT selection forbidden')
    haze_result=ROOT/'reports/foundir_fullstack_haze_consensus_v1_result_2026-11-06.json'
    require(c.sha(haze_result)==accepted['result_sha256'],'accepted Haze report changed')
    rows=accepted_rows(parent,report,accepted)
    return {'parent_receipt_sha256':c.sha(ROOT/x.OUTPUT/'receipt.json'),'parent_report_sha256':c.sha(ROOT/x.REPORT),
            'metricfix_sha256':c.sha(ROOT/x.OUTPUT/'metric_identity_fix.json'),
            'accepted_manifest_sha256':c.sha(ACCEPTED),'overlap_audit_sha256':c.sha(OVERLAP),
            'sources':{p:c.sha(ROOT/p) for p in SOURCES},'rows':rows,'world':WORLD,
            'nodes':[0,1,2],'local_gpus':2,'fixed_step':1000,'schedule_steps':50,'model_calls':26,
            'readout':'endpoint_at_call25','reference_time':10,'guidance':1.,'output_alpha':1.,
            'quantization':'raw float -> round/clamp uint8 once, then original full-image stitch',
            'apply_when':'frozen conditional_v4 anchor == rain','training':False,'new_route':False,'GT_input':False,
            'retain_Haze_consensus':True,'metric_identity':parent['metric_identity'],
            'official_is_adaptive_benchmark':True,'known_legacy_GT_source_overlap_records':30}


def manifest():
    v=plan();final=c.read(TRAIN/'final_lora.json');result=c.read(TRAIN_REPORT)
    require(final['step']==result['last_step']==1000 and result['new_updates']==200,'fixed1000 required')
    require(result['final_lora']==final and result['receipt_sha256']==c.sha(TRAIN/'receipt.json'),'training final mismatch')
    require(result['checkpoint']['parameter_hash']==final['parameter_hash'],'training parameter hash mismatch')
    for node in range(3):require(c.read(TRAIN_JOB/f'node{node}.ended.json')['status']==0,'training nodes not all done')
    require(c.read(TRAIN_JOB/'cpu_finish.json')['status']==0,'CPU finish required')
    name=final['tensor'];require(name in final['files'],'unbound final tensor')
    spec={'path':str((TRAIN/name).relative_to(ROOT)),'sha256':final['files'][name]}
    v.update(rain_checkpoint=spec,training_result_sha256=c.sha(TRAIN_REPORT),final_export_sha256=c.sha(TRAIN/'final_lora.json'))
    parent=c.read(ROOT/x.OUTPUT/'receipt.json');assets=used_assets(parent)
    del assets[parent['original']['parent']['experts']['rain']['checkpoint']['path']]
    assets[spec['path']]=spec['sha256'];v['assets']=assets
    return v


def bound():
    value=c.read(OUT/'receipt.json');require(value==manifest(),'Rain100 evaluation binding changed');return value


def prepare():
    start=time.monotonic();v=manifest()
    for path,expected in v['assets'].items():require(c.sha(ROOT/path)==expected,'asset changed: '+path)
    for item in v['rows']:
        old=item['accepted'];require(c.sha(ROOT/old['path'])==old['sha256'],'accepted output changed')
    counts=[sum(len(x.tiles(i['row'])) for i in v['rows'][r::WORLD] if i['deployed']['anchor']=='rain') for r in range(WORLD)]
    require(sum(counts)==4680,'Rain tile coverage changed')
    c.write_new(OUT/'receipt.json',v)
    report={'status':'CPU_preflight_passed','receipt_sha256':c.sha(OUT/'receipt.json'),'records':600,
            'regenerated_records':100,'reused_records':500,'tiles_per_rank':counts,'tiles':sum(counts),
            'calls_per_tile':26,'total_native_CFG_calls':26*sum(counts),'world':WORLD,'nodes':3,
            'seconds':time.monotonic()-start,'GPU_started':False,'official_GT_decoded':False,'training':False,
            'fixed_step':1000,'retains_accepted_Haze':True,'independent_blind_test':False}
    c.write_new(PREFLIGHT,report);print('RAIN1000_PREPARED '+str(report),flush=True)


def load_png(path,expected):
    import numpy as np
    from PIL import Image
    data=path.read_bytes();require(hashlib.sha256(data).hexdigest()==expected,'PNG changed')
    with Image.open(io.BytesIO(data)) as image:return np.array(image.convert('RGB'),copy=True)


def record_check(record,item,receipt,rank):
    require(record['item']==item and record['receipt_sha256']==receipt and record['rank']==rank,'result identity changed')
    require(set(record['metrics'])=={'psnr','lpips','ssim'} and all(math.isfinite(v) for v in record['metrics'].values()),'invalid metrics')
    require(record['reused_parent']==(item['deployed']['anchor']!='rain'),'wrong modified population')
    if record['reused_parent']:
        require(record['metrics']==item['accepted']['metrics'] and record['output']=={'path':item['accepted']['path'],'sha256':item['accepted']['sha256']},'accepted result changed')
        path=ROOT/record['output']['path']
    else:path=OUT/record['output']['path']
    require(c.sha(path)==record['output']['sha256'],'output PNG changed')


def tile_read(path,identity,decode=True):
    if not path.exists():return None
    record=c.read(path);require(record['identity']==identity and record['meta']['GT_input'] is False,'tile identity changed')
    require(record['meta']['model_calls']==26 and record['meta']['anchor']=='rain' and record['meta']['readout']=='endpoint_at_call25','Rain readout changed')
    file=OUT/record['png']['path']
    if not decode:
        require(c.sha(file)==record['png']['sha256'],'tile PNG changed')
        return True
    return load_png(file,record['png']['sha256'])


def run(node):
    import numpy as np
    import torch
    from smoke import train_foundir_frozen_expert_router_mixed_driver_v1 as impl
    from smoke.expanded_vlm_h100_runtime_v4 import bounded_vae,POLICY
    require(node in range(3),'new node0..2 required');local=int(os.environ['LOCAL_RANK']);rank=2*node+local
    require(local in (0,1) and int(os.environ['LOCAL_WORLD_SIZE'])==2,'two local workers required')
    v=bound();receipt=c.sha(OUT/'receipt.json');assigned=v['rows'][rank::WORLD]
    marker=OUT/f'worker{rank}.json'
    if marker.exists():
        old=c.read(marker);require(old['receipt_sha256']==receipt and old['records']==[i['row']['record_id'] for i in assigned],'completed worker changed')
        for item in assigned:record_check(c.read(OUT/'records'/(item['row']['record_id']+'.json')),item,receipt,rank)
        print(f'RAIN1000_WORKER_REUSED rank={rank}',flush=True);return
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
    bank=impl.load_bank(evaluation_bank(parent['original'],v['rain_checkpoint']),device);model,gates,rules,ae,prompts=bank;ae=bounded_vae(ae)
    weights=torch.tensor([[0.,0.,0.,0.,1.]],device=device)
    require(rules.anchor(weights).name=='rain' and rules.anchor(weights).alpha==1.,'Rain protocol changed')
    metrics,_=fix.checked_metric_models(device,v['metric_identity'])
    for number,item in enumerate(assigned):
        row=item['row'];target=OUT/'records'/(row['record_id']+'.json')
        if target.exists():record_check(c.read(target),item,receipt,rank);continue
        if item['deployed']['anchor']!='rain':
            record={'receipt_sha256':receipt,'item':item,'rank':rank,'metrics':item['accepted']['metrics'],
                    'output':{'path':item['accepted']['path'],'sha256':item['accepted']['sha256']},'reused_parent':True}
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
                        raw,u8,meta=restore_five_expert(model,gates,rules,weights,ae,lq,prompts,seed=tile['seed'])
                    meta['GT_input']=False
                    require(meta['model_calls']==26 and meta['anchor']=='rain' and meta['readout']=='endpoint_at_call25','unexpected Rain readout')
                    image=u8[0].permute(1,2,0).cpu().numpy();path=OUT/'tile_png'/(tile['key']+'.png')
                    producer.png_once(path,image)
                    c.write_new(receipt_path,{'identity':identity,'meta':meta,'png':{'path':str(path.relative_to(OUT)),'sha256':c.sha(path)}})
                    del raw,u8,lq
                images[tile['key']]=image
                if (ti+1)%5==0 or ti+1==len(tile_list):print(f'RAIN1000_TILE rank={rank} id={row["record_id"]} tile={ti+1}/{len(tile_list)} seconds={time.monotonic()-begin:.1f}',flush=True)
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
        print(f'RAIN1000_RECORD rank={rank} records={number+1}/{len(assigned)} id={row["record_id"]} reused={record["reused_parent"]}',flush=True)
    c.write_new(marker,{'receipt_sha256':receipt,'rank':rank,'records':[i['row']['record_id'] for i in assigned],
                       'seconds':time.monotonic()-start,'peak_allocated_gib':torch.cuda.max_memory_allocated()/1024**3})
    print(f'RAIN1000_WORKER_COMPLETE rank={rank}',flush=True)


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
                    require(tile_read(OUT/'tiles'/(tile['key']+'.json'),identity,decode=False) is True,'missing tile')
    require(all(s==signatures[0] for s in signatures),'worker numerical runtimes differ')
    require(len(records)==600 and len({r['item']['row']['record_id'] for r in records})==600,'incomplete600')
    per_category={cat:{m:math.fsum(r['metrics'][m] for r in records if r['item']['row']['category']==cat)/count
                       for m in ('psnr','lpips','ssim')} for cat,count in x.COUNTS.items()}
    checks={task:{'psnr':per_category[x.CATEGORIES[task]]['psnr']>=goal[0],
                  'lpips':per_category[x.CATEGORIES[task]]['lpips']<=goal[1]} for task,goal in x.TARGETS.items()}
    result={'status':'complete','records':600,'new_rain_records':100,'reused_unchanged_records':500,
            'receipt_sha256':receipt,'per_category':per_category,'checks':checks,'all_ten_pass':all(all(v.values()) for v in checks.values()),
            'parent':c.read(ACCEPTED)['per_category'],'training':False,'fixed_step':1000,
            'GT_input':False,'independent_blind_test':False,'known_legacy_GT_overlap_records':30,
            'scope':'Fixed step1000 Rain100 generation; accepted Haze and other500 outputs preserved.'}
    overlap=set(c.read(OVERLAP)['official_records_sharing_these_GT_files'])
    rain=[r for r in records if r['item']['row']['category']=='10Rain']
    result['descriptive_Rain_subgroups']={}
    for label,group in [('known_GT_source_overlap30',[r for r in rain if r['item']['row']['record_id'] in overlap]),
                        ('other70_not_claimed_independent',[r for r in rain if r['item']['row']['record_id'] not in overlap])]:
        require(len(group)==(30 if label.startswith('known') else 70),'overlap subgroup changed')
        result['descriptive_Rain_subgroups'][label]={'count':len(group),'means':{k:math.fsum(r['metrics'][k] for r in group)/len(group) for k in ('psnr','lpips','ssim')}}
    result['Rain_changes']={k:per_category['10Rain'][k]-result['parent']['10Rain'][k] for k in ('psnr','lpips','ssim')}
    result['Rain_single_image_improved']={k:sum((r['metrics'][k]>r['item']['accepted']['metrics'][k]) if k!='lpips' else (r['metrics'][k]<r['item']['accepted']['metrics'][k]) for r in rain) for k in ('psnr','lpips')}
    result['Rain_worst_PSNR_change']=min(r['metrics']['psnr']-r['item']['accepted']['metrics']['psnr'] for r in rain)
    result['no_per_image_GT_selection']=True
    result['used_GPU_hours']=sum(c.read(OUT/f'worker{rank}.json')['seconds'] for rank in range(WORLD))/3600
    c.write_new(REPORT,result);print('RAIN1000_RESULT_VERIFIED '+str(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['plan','prepare','run','finish']);p.add_argument('--node',type=int,default=0)
    a=p.parse_args();require(str(ROOT)==producer.ACTIVE,'destination workspace only')
    if a.phase=='run':run(a.node)
    elif a.phase=='plan':
        value=plan();counts=[sum(len(x.tiles(i['row'])) for i in value['rows'][r::WORLD] if i['deployed']['anchor']=='rain') for r in range(WORLD)]
        print('RAIN1000_PLAN records=600 rain=100 tiles_per_rank='+str(counts)+' final_weight_binding=pending_step1000')
    else:globals()[a.phase]()
