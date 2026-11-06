#!/usr/bin/env python3
"""Frozen Noise/Lowlight router coverage and crop aggregation audit. No fitting."""
import hashlib
import json
import os
from collections import Counter,defaultdict
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));os.environ['FOUNDIR_OFFICIAL_NNODES']='4'
from smoke import evaluate_foundir_official_vlm_router_v4 as producer
c,x,require=producer.c,producer.x,producer.require
OFFICIAL=ROOT/x.OUTPUT
TRAIN=ROOT/'checkpoints/foundir_expanded_vlm_router_v4/expanded_conditional_v4'
OUT=ROOT/'checkpoints/foundir_fullstack_noise_router_audit_v1'
REPORT=ROOT/'reports/foundir_fullstack_noise_router_audit_v1_2026-11-06.json'


def summarize(logits):
    import torch
    z=logits.float();mean=z.mean(0);prob=z.softmax(-1).mean(0)
    vote=Counter(x.ORDER[i] for i in z.argmax(-1).tolist())
    return {'views':len(z),'crop_votes':dict(vote),'mean_logits':mean.tolist(),
            'mean_logit_route':x.ORDER[int(mean.argmax())],
            'mean_probability_route':x.ORDER[int(prob.argmax())],
            'mean_probabilities':prob.tolist(),
            'noise_minus_lowlight_margin_min':float((z[:,1]-z[:,3]).min()),
            'noise_minus_lowlight_margin_max':float((z[:,1]-z[:,3]).max())}


def load_feature(spec):
    from safetensors.torch import load
    data=(ROOT/spec['path']).read_bytes();require(hashlib.sha256(data).hexdigest()==spec['sha256'],'feature hash changed')
    v=load(data);require(v['image'].shape==(1,6144) and v['vlm'].shape==(1,3584),'feature shape changed')
    return v['image'].float(),v['vlm'].float()


def main():
    import torch
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    start=time.monotonic();require(not REPORT.exists(),'preserve completed audit')
    parent=c.read(OFFICIAL/'receipt.json');model=producer.load_router(parent)
    training=c.read(TRAIN/'receipt.json');index=c.read(TRAIN/'feature_index.json')
    report=c.read(ROOT/'reports/foundir_expanded_vlm_router_v4_training_2026-11-05.json')
    require(c.sha(TRAIN/'receipt.json')==report['receipt_sha256'] and c.sha(TRAIN/'feature_index.json')==report['feature_index_sha256'],'training binding changed')
    identity={'source_sha256':c.sha(Path(__file__)),'official_receipt_sha256':c.sha(OFFICIAL/'receipt.json'),
              'training_receipt_sha256':c.sha(TRAIN/'receipt.json'),'feature_index_sha256':c.sha(TRAIN/'feature_index.json'),
              'selected_checkpoint':report['selected_checkpoint'],'alternative':'one_fixed_mean_probability_no_temperature_or_threshold_sweep'}
    c.write_new(OUT/'receipt.json',identity)
    training_rows=[];items=[i for i in training['items'] if i['task'] in ('noise','lowlight')]
    for offset in range(0,len(items),64):
        batch=items[offset:offset+64];xs=[];vs=[]
        for item in batch:
            im,vlm=load_feature(index[item['key']]);xs.append(im);vs.append(vlm)
        with torch.no_grad():z,_,im,_=model(torch.cat(xs),torch.cat(vs))
        for item,logits,image in zip(batch,z,im):
            training_rows.append({'key':item['key'],'role':item['role'],'task':item['task'],
                                  'source_group':item['row']['source_group'],'logits':logits.tolist(),
                                  'image_route':x.ORDER[int(image.argmax())],'route':x.ORDER[int(logits.argmax())]})
        if offset%640==0:print(f'NOISE_ROUTER_AUDIT cached_training={min(offset+64,len(items))}/{len(items)}',flush=True)
    train_summary={}
    for role in ('fit','calibration','confirmation'):
        train_summary[role]={}
        for task in ('noise','lowlight'):
            group=[r for r in training_rows if r['role']==role and r['task']==task];by_source=defaultdict(list)
            for row in group:by_source[row['source_group']].append(row)
            summary=[{'source_group':key,**summarize(torch.tensor([v['logits'] for v in values]))} for key,values in by_source.items()]
            train_summary[role][task]={'crops':len(group),'sources':len(by_source),
                'crop_route_counts':dict(Counter(r['route'] for r in group)),
                'image_only_crop_route_counts':dict(Counter(r['image_route'] for r in group)),
                'source_mean_logit_route_counts':dict(Counter(r['mean_logit_route'] for r in summary)),
                'source_mean_probability_route_counts':dict(Counter(r['mean_probability_route'] for r in summary))}
    c.write_new(OUT/'training_rows.json',training_rows)
    print('NOISE_ROUTER_AUDIT training_summary='+json.dumps(train_summary),flush=True)
    officials=[]
    for row in parent['rows']:
        if row['category'] not in ('05Noise','14Lowlight'):continue
        rid=row['record_id'];decision=c.read(OFFICIAL/'decisions'/(rid+'.json'));zs=[];ims=[]
        marker=OUT/'records'/(rid+'.json')
        if marker.exists():officials.append(c.read(marker));continue
        for tile in x.tiles(row):
            file=OFFICIAL/'features'/(tile['key']+'.json')
            require(c.sha(file)==decision['feature_receipts'][tile['key']],'feature receipt changed')
            rec=c.read(file);require(rec['GT_input'] is False and rec['identity']['receipt_sha256']==identity['official_receipt_sha256'],'feature producer changed')
            spec={'path':str((OFFICIAL/rec['tensor']).relative_to(ROOT)),'sha256':rec['files'][rec['tensor']]}
            image,vlm=load_feature(spec)
            with torch.no_grad():z,_,im,_=model(image,vlm)
            zs.append(z[0]);ims.append(im[0])
        result=summarize(torch.stack(zs));image_summary=summarize(torch.stack(ims))
        expected=decision['decisions']['conditional_v4'];difference=max(abs(a-b) for a,b in zip(result['mean_logits'],expected['logits']))
        require(result['mean_logit_route']==expected['anchor'] and difference<1e-4,'frozen CPU router did not reproduce old decision')
        result.update(record_id=rid,category=row['category'],GT_file_sha256=row['gt_sha256'],
                      original_replay_max_logit_difference=difference,image_only=image_summary)
        c.write_new(marker,result);officials.append(result)
        if len(officials)%10==0:print(f'NOISE_ROUTER_AUDIT official={len(officials)}/150',flush=True)
    summary={}
    for cat in ('05Noise','14Lowlight'):
        group=[r for r in officials if r['category']==cat]
        summary[cat]={'records':len(group),'mean_logit_route_counts':dict(Counter(r['mean_logit_route'] for r in group)),
                      'mean_probability_route_counts':dict(Counter(r['mean_probability_route'] for r in group)),
                      'changed_record_ids':[r['record_id'] for r in group if r['mean_logit_route']!=r['mean_probability_route']]}
    result={'status':'frozen_router_CPU_audit_complete','identity':identity,'training_summary':train_summary,
            'official_summary':summary,'official_records':officials,'training_rows_sha256':c.sha(OUT/'training_rows.json'),
            'seconds':time.monotonic()-start,'exit_code':0,'GPU_hours':0,'training':False,'pixels_decoded':0,
            'GT_or_labels_used_in_prediction':False,'official_labels_used_for_descriptive_counts_only':True,
            'new_routes_deployed':False,'new_restorations':False,'official_is_adaptive_benchmark':True,
            'other_three_official_categories_evaluated_for_alternative':False}
    c.write_new(REPORT,result);print('NOISE_ROUTER_AUDIT_END '+json.dumps({'summary':summary,'seconds':result['seconds']}),flush=True)


if __name__=='__main__':main()
