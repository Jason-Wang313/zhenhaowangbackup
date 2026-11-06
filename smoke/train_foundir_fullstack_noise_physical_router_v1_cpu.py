#!/usr/bin/env python3
"""One fixed400-step CPU-only binary route pilot. No expert updates or GPU launch."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time
from collections import Counter,defaultdict

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from PIL import Image
from lowlight_restoration.noise_lowlight_physical_router_v1 import features,FEATURE_NAMES,BinaryCorrection
OUT=ROOT/'checkpoints/foundir_fullstack_noise_physical_router_v1'
TRAIN=ROOT/'checkpoints/foundir_expanded_vlm_router_v4/expanded_conditional_v4'
AUDIT=ROOT/'checkpoints/foundir_fullstack_noise_router_audit_v1'
SCREEN=ROOT/'checkpoints/foundir_fullstack_noise_blur_sources_v1'
REPORT=ROOT/'reports/foundir_fullstack_noise_physical_router_v1_2026-11-06.json'
SOURCES=['lowlight_restoration/noise_lowlight_physical_router_v1.py','smoke/train_foundir_fullstack_noise_physical_router_v1_cpu.py']
RECIPE={'steps':400,'batch':64,'seed':2026110609,'lr':.001,'weight_decay':.01,'clip':1.,
        'delta_squared_regularizer':.01,'train_labels':'Noise_vs_Lowlight','other_routes':'unchanged',
        'selection':'fixed_step400_no_checkpoint_or_hyperparameter_search','GPU':False,'official_used_for_training':False,
        'maximum_CPU_wall_seconds':1800}


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def require(ok,message):
    if not ok:raise ValueError(message)
def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():require(read(p)==v,'immutable artifact differs');return
    temp=p.with_name(p.name+'.tmp');temp.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n');temp.replace(p)


def prepare():
    b=read(TRAIN/'receipt.json');screen=read(SCREEN/'plan.json');audit=read(AUDIT/'receipt.json')
    require(audit['training_receipt_sha256']==sha(TRAIN/'receipt.json'),'parent receipt changed')
    protected=[read(p) for p in sorted((SCREEN/'protected_index').glob('*.json'))]
    require(len(protected)==screen['official_unique_GT_files'],'protected source index incomplete')
    phashes=[int(p['phash64'],16) for p in protected]
    scores={r['key']:r for r in read(AUDIT/'training_rows.json')};groups=defaultdict(list);excluded=[]
    for item in b['items']:
        if item['task'] not in ('noise','lowlight'):continue
        row=item['row'];key=(item['role'],item['task'],row['source_group'])
        if any((int(row['gt_phash'],16)^p).bit_count()<=4 for p in phashes):excluded.append(item['key']);continue
        groups[key].append(item)
    records=[]
    for (role,task,source),items in sorted(groups.items()):
        require(len({i['row']['lq_sha256'] for i in items})==1,'multiple LQ images within group')
        row=items[0]['row']
        require(row['lq_path'].startswith('data/foundir/MillionIRData/Train/'),'nontraining-collection pixels forbidden')
        z=np.mean([scores[i['key']]['logits'] for i in items],axis=0).tolist()
        records.append({'key':f'{role}_{task}_{source}','role':role,'task':task,'source_group':source,
                        'lq_path':row['lq_path'],'lq_sha256':row['lq_sha256'],'base_logits':z,'views':len(items)})
    sets={role:{r['source_group'] for r in records if r['role']==role} for role in ('fit','calibration','confirmation')}
    require(not (sets['fit']&sets['calibration'] or sets['fit']&sets['confirmation'] or sets['calibration']&sets['confirmation']),'role source overlap')
    for role in sets:
        for task in ('noise','lowlight'):require(any(r['role']==role and r['task']==task for r in records),'empty role/class')
    value={'recipe':RECIPE,'sources':{s:sha(ROOT/s) for s in SOURCES},'parent_receipt_sha256':sha(TRAIN/'receipt.json'),
           'source_screen_sha256':sha(SCREEN/'plan.json'),'audit_training_rows_sha256':sha(AUDIT/'training_rows.json'),
           'records':records,'excluded_crop_keys':excluded,'feature_names':FEATURE_NAMES,
           'legacy_base_router_source_overlap_not_erased':True,'official_images_in_training':False}
    write(OUT/'receipt.json',value)
    print('PHYSICAL_ROUTER_PREPARED '+json.dumps({'counts':dict(Counter(r['role']+'_'+r['task'] for r in records)),'excluded_crops':len(excluded)}),flush=True)


def bound():
    b=read(OUT/'receipt.json');require(b['recipe']==RECIPE,'recipe changed')
    for s,h in b['sources'].items():require(sha(ROOT/s)==h,'source changed: '+s)
    require(sha(TRAIN/'receipt.json')==b['parent_receipt_sha256'] and sha(SCREEN/'plan.json')==b['source_screen_sha256'],'parent changed')
    return b


def extract(record):
    p=OUT/'features'/(record['key']+'.json')
    if p.exists():v=read(p);require(v['lq_sha256']==record['lq_sha256'],'LQ changed');return v
    data=(ROOT/record['lq_path']).read_bytes();require(hashlib.sha256(data).hexdigest()==record['lq_sha256'],'LQ file hash changed')
    with Image.open(io.BytesIO(data)) as image:values=features(image).tolist()
    value={'lq_sha256':record['lq_sha256'],'feature_names':FEATURE_NAMES,'values':values,'GT_input':False}
    write(p,value);return value


def evaluate(model,records,physical):
    z=torch.tensor([r['base_logits'] for r in records]);labels=torch.tensor([1 if r['task']=='noise' else 3 for r in records])
    with torch.no_grad():_,new,_=model(z,physical)
    old=z.argmax(-1);result={}
    for task,label in [('noise',1),('lowlight',3)]:
        mask=labels==label
        result[task]={'count':int(mask.sum()),'baseline_correct':int(((old==labels)&mask).sum()),'candidate_correct':int(((new==labels)&mask).sum()),
                      'fixed':int(((old!=labels)&(new==labels)&mask).sum()),'regressed':int(((old==labels)&(new!=labels)&mask).sum())}
    return {'summary':result,'rows':[{'key':r['key'],'task':r['task'],'baseline_route':int(o),'candidate_route':int(n)} for r,o,n in zip(records,old,new)]}


def run():
    require(not REPORT.exists(),'completed pilot must not rerun');start=time.monotonic();b=bound()
    records=b['records'];values={}
    with ThreadPoolExecutor(max_workers=2) as pool:
        for i,(record,value) in enumerate(zip(records,pool.map(extract,records))):
            require(time.monotonic()-start<RECIPE['maximum_CPU_wall_seconds'],'CPU time cap; preserve features')
            values[record['key']]=value['values']
            if (i+1)%100==0:print(f'PHYSICAL_ROUTER_FEATURES {i+1}/{len(records)}',flush=True)
    fit=[r for r in records if r['role']=='fit'];x=torch.tensor([values[r['key']] for r in fit]);z=torch.tensor([r['base_logits'] for r in fit]);y=torch.tensor([float(r['task']=='noise') for r in fit])
    torch.manual_seed(RECIPE['seed']);model=BinaryCorrection(x.mean(0),x.std(0,unbiased=False).clamp_min(1e-6));optimizer=torch.optim.AdamW(model.parameters(),lr=RECIPE['lr'],weight_decay=RECIPE['weight_decay'])
    groups=[torch.where(y==value)[0] for value in (0,1)];history=[]
    for step in range(400):
        require(time.monotonic()-start<RECIPE['maximum_CPU_wall_seconds'],'CPU time cap; preserve features')
        ids=torch.cat([group[torch.randint(len(group),(32,))] for group in groups]);margin,_,delta=model(z[ids],x[ids])
        loss=torch.nn.functional.binary_cross_entropy_with_logits(margin,y[ids])+.01*delta.square().mean()
        require(torch.isfinite(loss),'nonfinite CPU loss');optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
        history.append(float(loss.detach()))
    state=OUT/'fixed_step400.pt';require(not state.exists(),'preserve prior model');torch.save({'model':model.state_dict(),'recipe':RECIPE,'receipt_sha256':sha(OUT/'receipt.json')},state)
    results={}
    for role in ('fit','calibration','confirmation'):
        subset=[r for r in records if r['role']==role];results[role]=evaluate(model,subset,torch.tensor([values[r['key']] for r in subset]))
    value={'status':'fixed400_CPU_pilot_complete_not_deployed','receipt_sha256':sha(OUT/'receipt.json'),'checkpoint_sha256':sha(state),
           'recipe':RECIPE,'history':history,'classification':results,'seconds':time.monotonic()-start,'exit_code':0,'GPU_hours':0,
           'expert_weights_updated':False,'official_route_or_restoration_evaluated':False,'automatic_deployment':False,
           'legacy_base_router_source_overlap_not_erased':True,'source_screen_is_not_proof_of_semantic_disjointness':True}
    write(REPORT,value);print('PHYSICAL_ROUTER_END '+json.dumps({'summary':{k:v['summary'] for k,v in results.items()},'seconds':value['seconds']}),flush=True)


if __name__=='__main__':
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','run']);globals()[p.parse_args().phase]()
