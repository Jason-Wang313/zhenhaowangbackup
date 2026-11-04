#!/usr/bin/env python3
"""One bounded repair: frozen image router + regularized VLM correction.

Expanded, source-separated training/calibration/holdout; never tune on old40.
Feature extraction uses the original frozen FLUX/Qwen encoders, not restoration.
"""
import argparse
from collections import Counter
import csv
import math
import os
from pathlib import Path
import platform
import random
import re
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from smoke import analyze_foundir_router_generalization_v1 as audit
s=audit.s
c,require,ORDER=s.c,s.require,s.ORDER
RUN='expanded_frozen_image_v2'
OUTPUT='checkpoints/foundir_frozen_image_router_v2/'+RUN
REPORT='reports/foundir_frozen_image_router_v2_terminal_2026-11-04.json'
SEED=2026110402
STEPS=1600
CHECKS=(400,800,1200,1600)
QUOTAS={'fit':512,'calibration':80,'holdout':80}
EXPECTED_ITEMS_SHA='333117f97705c42acefedbbee698098935da143c21a1bb6f176cf409f3fc4ceb'
SOURCES=['smoke/repair_foundir_frozen_image_router_v2.py','smoke/frozen_image_router_state_v2.py',
         'smoke/run_foundir_frozen_image_router_v2.sh','smoke/start_foundir_frozen_image_router_v2_tmux.sh']


def select(rows,original,quotas=None):
    quotas=QUOTAS if quotas is None else quotas
    rows=list(rows); forbidden={field:{r[field] for r in rows if r['split']!='train'}
                               for field in ('source_group','gt_sha256','lq_sha256')}
    old_train=[i for i in original if i['split']=='train']
    for field in forbidden:
        forbidden[field].update(i['row'][field] for i in original if i['split']!='train')
    reserved={field:{i['row'][field] for i in old_train} for field in forbidden}
    require(all(not (reserved[f]&forbidden[f]) for f in forbidden),'old training overlaps protected pool')
    category_to_task={v:k for k,v in audit.CATEGORIES.items()}
    pools={t:{} for t in ORDER}
    def digest(*parts): return c.stable(SEED,*parts)
    for row in rows:
        if row['split']!='train' or row['category'] not in category_to_task: continue
        if int(row['composition_size'])!=1 or min(int(row['width']),int(row['height']))<512: continue
        if any(row[f] in forbidden[f] for f in forbidden): continue
        task=category_to_task[row['category']]; source=row['source_group']; prev=pools[task].get(source)
        if prev is None or digest('record',row['record_id'])<digest('record',prev['record_id']): pools[task][source]=row
    owners={f:{v:'fit' for v in reserved[f]} for f in forbidden}
    selected=[{**i,'role':'fit','reuse_original_key':i['key']} for i in old_train]
    used={(i['task'],i['row']['source_group']) for i in old_train}
    # Allocate scarce classes first using metadata only. All source/hash ownership
    # is global across classes; no feature, quality or error influences selection.
    for task in sorted(ORDER,key=lambda t:(len(pools[t]),ORDER.index(t))):
        for role in ('fit','calibration','holdout'):
            count=sum(i['role']==role and i['task']==task for i in selected)
            require(count<=quotas[role],'quota below retained training set')
            for source,row in sorted(pools[task].items(),key=lambda kv:digest('source',kv[0])):
                if count==quotas[role]: break
                if (task,source) in used: continue
                if any(row[f] in owners[f] and owners[f][row[f]]!=role for f in owners): continue
                if role!='fit' and any(row[f] in reserved[f] for f in reserved): continue
                r=dict(row); raw=bytes.fromhex(digest('crop',r['record_id']))
                require(re.fullmatch(r'[A-Za-z0-9_-]+',r['record_id']) is not None,'unsafe record key')
                r['crop_xy']=[int.from_bytes(raw[:8],'big')%(int(r['width'])-511),
                              int.from_bytes(raw[8:16],'big')%(int(r['height'])-511)]
                for side in ('lq_path','gt_path'):
                    require(r[side].startswith('data/foundir/MillionIRData/Train/'+r['category']+'/')
                            and '..' not in Path(r[side]).parts,'outside allowed training collection')
                selected.append({'key':'train_'+r['record_id'],'task':task,'split':'train','row':r,
                                 'seed':int.from_bytes(raw[16:20],'big'),'role':role,'reuse_original_key':None})
                for f in owners: owners[f][r[f]]=role
                used.add((task,source)); count+=1
            require(count==quotas[role],'insufficient independent sources: '+role+'/'+task)
    selected.sort(key=lambda i:(('fit','calibration','holdout').index(i['role']),ORDER.index(i['task']),i['key']))
    validate_selection(selected,original,quotas)
    return selected


def validate_selection(items,original,quotas=None):
    quotas=QUOTAS if quotas is None else quotas
    require(len({i['key'] for i in items})==len(items),'duplicate selected record')
    old={i['key']:i for i in original if i['split']=='train'}
    reused={i['reuse_original_key']:i for i in items if i['reuse_original_key'] is not None}
    require(set(reused)==set(old),'retain exactly original training records')
    for key,i in reused.items():
        require(i['role']=='fit' and {k:v for k,v in i.items() if k not in ('role','reuse_original_key')}==old[key],
                'reused input changed or leaked')
    for role,n in quotas.items():
        group=[i for i in items if i['role']==role]
        require(Counter(i['task'] for i in group)==Counter({t:n for t in ORDER}),'unbalanced role')
        require(all(len({i['row']['source_group'] for i in group if i['task']==t})==n for t in ORDER),'duplicate class source')
    for field in ('source_group','gt_sha256','lq_sha256'):
        groups={r:{i['row'][field] for i in items if i['role']==r} for r in quotas}
        protected={i['row'][field] for i in original if i['split']!='train'}
        require(all(not (groups[a]&groups[b]) for a in quotas for b in quotas if a!=b),'role identity overlap')
        require(all(not (g&protected) for g in groups.values()),'old development leaked')


def prepare():
    s.verify(); bound=s.load_bound(); original=c.read(ROOT/'checkpoints/foundir_frozen_expert_router_v1'/s.analysis.RUN/'receipt.json')
    parent_path=ROOT/'checkpoints/foundir_frozen_expert_router_v1'/s.analysis.RUN/'receipt.json'
    require(c.sha(parent_path)==bound['parent_receipt_sha256'],'original frozen producer receipt changed')
    for name,expected in original['asset_stats'].items():
        actual=c.path(ROOT,name).stat()
        require([actual.st_size,actual.st_mtime_ns]==expected,'frozen model/manifest metadata changed: '+name)
    manifest=ROOT/original['config']['manifest']
    require(c.sha(manifest)==original['config']['manifest_sha256'],'manifest changed')
    with manifest.open(newline='') as f: items=select(csv.DictReader(f),bound['items'])
    require(c.digest(items)==EXPECTED_ITEMS_SHA,'selection differs from verified manifest plan')
    for i in items:
        require(c.path(ROOT,i['row']['lq_path']).is_file(),'missing LQ input')
    sources={**bound['sources'],**{n:c.sha(ROOT/n) for n in SOURCES},
             'smoke/analyze_foundir_router_generalization_v1.py':c.sha(ROOT/'smoke/analyze_foundir_router_generalization_v1.py')}
    value={'run_id':RUN,'items':items,'original':original,'reuse':{k:v for k,v in bound['features'].items()
           if k in {i['key'] for i in items if i['reuse_original_key']}},'sources':sources,
           'cpu_parent_report_sha256':c.sha(ROOT/s.REPORT),'cpu_parent_receipt_sha256':c.sha(ROOT/s.OUTPUT/'receipt.json'),
           'frozen_producer_receipt_sha256':bound['parent_receipt_sha256'],
           'image_checkpoint':c.read(ROOT/s.REPORT)['arms']['image_only']['checkpoint'],
           'recipe':{'fit':2560,'calibration':400,'holdout':400,'steps':STEPS,'batch':32,'seed':SEED,
                     'lr':.001,'weight_decay':.01,'label_smoothing':.05,'checks':list(CHECKS),
                     'frozen_image':True,'training':'VLM_MLP_only','old40_used':False},
           'GPU_feature_cap_seconds_per_node':3600,'CPU_train_cap_seconds':3600,'automatic_restoration':False}
    c.write_new(ROOT/OUTPUT/'receipt.json',value)
    print('FROZEN_IMAGE_V2_PREPARED fit=2560 calibration=400 holdout=400 reuse=640 new_features=2720',flush=True)


def receipt():
    b=c.read(ROOT/OUTPUT/'receipt.json'); require(b['run_id']==RUN,'wrong run')
    require(c.digest(b['items'])==EXPECTED_ITEMS_SHA,'frozen selected inputs changed')
    for n,h in b['sources'].items(): require(c.sha(ROOT/n)==h,'bound code changed: '+n)
    require(c.sha(ROOT/s.REPORT)==b['cpu_parent_report_sha256'] and c.sha(ROOT/s.OUTPUT/'receipt.json')==b['cpu_parent_receipt_sha256'],
            'parent changed')
    parent=ROOT/'checkpoints/foundir_frozen_expert_router_v1'/s.analysis.RUN/'receipt.json'
    require(c.sha(parent)==b['frozen_producer_receipt_sha256'] and c.read(parent)==b['original'],'frozen producer changed')
    for name,expected in b['original']['asset_stats'].items():
        actual=c.path(ROOT,name).stat()
        require([actual.st_size,actual.st_mtime_ns]==expected,'frozen model/manifest metadata changed: '+name)
    validate_selection(b['items'],s.load_bound()['items'])
    return b


def hardware(node,idle):
    import torch
    from smoke import train_foundir_frozen_expert_router_mixed_driver_v1 as impl
    from smoke.frozen_router_runtime_policy_v1 import execution_signature
    b=receipt(); actual=impl.native.hopper.hardware(idle)
    old=b['original']; actual['metrics']=impl.parent.four_impl.metric_runtime({'metric':old['parent']['metric']})
    actual['memory_policy']={'inherited':old['parent']['memory_policy'],'router_explicit_gate_checkpoint':True}
    parent=ROOT/'checkpoints/foundir_frozen_expert_router_v1'/s.analysis.RUN
    require(all(execution_signature(actual)==execution_signature(c.read(parent/f'runtime_node{n}.json')) for n in range(4)),
            'feature runtime incompatible')
    path=ROOT/OUTPUT/f'runtime_node{node}.json'
    if idle: c.write_new(path,actual)
    else: require(c.read(path)==actual,'runtime changed on resume')
    return b,impl


def extract(node):
    import torch
    local=int(os.environ['LOCAL_RANK']); require(int(os.environ['WORLD_SIZE'])==2 and local in (0,1),'two workers per node')
    b,impl=hardware(node,False); torch.cuda.set_device(local); device=torch.device('cuda',local)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False; torch.backends.cudnn.benchmark=False
    old=b['original']; os.environ.update(KLEIN_4B_BASE_MODEL_PATH=str(c.path(ROOT,old['parent']['model']['flow_weights'])),
                                         AE_MODEL_PATH=str(c.path(ROOT,old['parent']['model']['autoencoder_weights'])))
    new=[i for i in b['items'] if i['reuse_original_key'] is None]; rank=node*2+local
    impl.features(ROOT/OUTPUT,{**old,'items':new},rank,device)
    # Never train while held-out features are being extracted; no GT pixels read here.
    completion={'receipt_sha256':c.sha(ROOT/OUTPUT/'receipt.json'),
                 'rank':rank,'count':len(new[rank::8]),'runtime_sha256':c.sha(ROOT/OUTPUT/f'runtime_node{node}.json'),
                 'peak_allocated_gib':torch.cuda.max_memory_allocated(device)/1024**3}
    marker=ROOT/OUTPUT/f'worker{rank}_complete.json'
    if marker.exists():
        previous=c.read(marker)
        require({k:v for k,v in previous.items() if k!='peak_allocated_gib'}==
                {k:v for k,v in completion.items() if k!='peak_allocated_gib'},'completed shard identity changed')
    else: c.write_new(marker,completion)
    print(f'FROZEN_IMAGE_V2_FEATURE_WORKER_PASS rank={rank} records={len(new[rank::8])}',flush=True)


def feature_index(b):
    output=ROOT/OUTPUT; new=[i for i in b['items'] if not i['reuse_original_key']]; result={}
    for rank in range(8):
        worker=c.read(output/f'worker{rank}_complete.json')
        require({k:v for k,v in worker.items() if k!='peak_allocated_gib'}=={'receipt_sha256':c.sha(output/'receipt.json'),
                'rank':rank,'count':len(new[rank::8]),'runtime_sha256':c.sha(output/f'runtime_node{rank//2}.json')},'worker incomplete')
        require(audit.quality.analysis.number(worker['peak_allocated_gib']) and worker['peak_allocated_gib']>=0,'invalid memory report')
    for i in b['items']:
        if i['reuse_original_key']:
            spec=b['reuse'][i['key']]; parent=ROOT/'checkpoints/foundir_frozen_expert_router_v1'/s.analysis.RUN
            require(c.sha(parent/'features'/(i['key']+'.json'))==spec['marker_sha256'],'reused feature marker changed')
            result[i['key']]={'path':spec['path'],'sha256':spec['sha256'],'producer':'original_640_cache'}
        else:
            row=c.read(output/'features'/(i['key']+'.json'))
            require(row['identity']=={'receipt_sha256':c.sha(output/'receipt.json'),'item':i,'arm':'features'}
                    and row['GT_input'] is False and row['vlm_receipt_sha256']==c.sha(output/'vlm'/(i['key']+'.json')),
                    'new feature identity mismatch')
            require(set(row['files'])=={row['tensor']},'invalid feature artifacts')
            file=c.path(output,row['tensor']); result[i['key']]={'path':file.relative_to(ROOT).as_posix(),
                'sha256':row['files'][row['tensor']],'producer':'expanded_v2'}
    return result


def load_role(b,index,role):
    import torch
    from safetensors import safe_open
    require(role in QUOTAS,'invalid role')
    if role=='holdout': require((ROOT/OUTPUT/'selection.json').is_file(),'holdout locked until selection committed')
    items=[i for i in b['items'] if i['role']==role]; xs=[]; vs=[]
    for i in items:
        spec=index[i['key']]; file=c.path(ROOT,spec['path']); require(c.sha(file)==spec['sha256'],'feature hash changed')
        with safe_open(str(file),framework='pt',device='cpu') as f: x,v=f.get_tensor('image'),f.get_tensor('vlm')
        require(x.shape==(1,6144) and v.shape==(1,3584) and torch.isfinite(x).all() and torch.isfinite(v).all(),'invalid features')
        xs.append(x.float()); vs.append(v.float())
    return items,torch.cat(xs),torch.cat(vs),torch.tensor([ORDER.index(i['task']) for i in items])


def batch_indices(items,step):
    require(0<=step<STEPS and len(items)==2560 and all(i['role']=='fit' for i in items),'fit-only training')
    groups={t:[j for j,i in enumerate(items) if i['task']==t] for t in ORDER}
    require(all(len(v)==512 for v in groups.values()),'fit quotas changed')
    return [groups[ORDER[slot%5]][(slot//5)%512] for slot in range(step*32,(step+1)*32)]


def model(b):
    import torch
    from lowlight_restoration.frozen_expert_router import FrozenExpertRouter
    random.seed(SEED); torch.manual_seed(SEED)
    result=FrozenExpertRouter().cpu().float(); spec=b['image_checkpoint']
    file=c.path(ROOT/s.OUTPUT/'image_only',spec['file']); require(c.sha(file)==spec['sha256'],'image checkpoint changed')
    old=torch.load(file,map_location='cpu',weights_only=True)
    require(s.state.io.parameter_hash(old['model'])==spec['parameter_hash'] and old['step']==400,'wrong frozen image model')
    result.image.load_state_dict({n.removeprefix('image.'):v for n,v in old['model'].items() if n.startswith('image.')},strict=True)
    result.image.requires_grad_(False); result.eval()
    opt=torch.optim.AdamW(result.vlm.parameters(),lr=.001,weight_decay=.01,foreach=False,fused=False)
    return result,opt


def update(m,opt,x,v,y):
    import torch
    opt.zero_grad(set_to_none=True); logits,_,_,_=m(x,v)
    loss=torch.nn.functional.cross_entropy(logits,y,label_smoothing=.05)
    require(bool(torch.isfinite(loss)),'nonfinite loss'); loss.backward()
    norm=float(torch.nn.utils.clip_grad_norm_(m.vlm.parameters(),1.,error_if_nonfinite=True)); opt.step()
    require(all(p.grad is None for p in m.image.parameters()),'frozen image received gradients')
    require(all(torch.isfinite(p).all() for p in m.parameters()),'nonfinite model')
    return {'loss':float(loss.detach()),'vlm_grad_norm_before_clip':norm}


def evaluate(m,data,enabled):
    import torch
    items,x,v,y=data
    with torch.no_grad(): logits,probs,im,delta=m(x,v,vlm_enabled=enabled)
    if not enabled: delta=torch.zeros_like(delta)
    pred=logits.argmax(-1)
    return {'count':len(items),'correct':int((pred==y).sum()),'ce':float(torch.nn.functional.cross_entropy(logits,y)),
            'per_task':{t:{'count':int((y==j).sum()),'correct':int(((pred==y)&(y==j)).sum())} for j,t in enumerate(ORDER)},
            'rows':[{'key':i['key'],'task_for_reporting_only':i['task'],'predicted_anchor':ORDER[int(pred[j])],
                      'logits':logits[j].tolist(),'image_logits':im[j].tolist(),'vlm_delta':delta[j].tolist(),
                      'confidence':float(probs[j].max()),'seed':i['seed'],'crop_xy':i['row']['crop_xy']}
                     for j,i in enumerate(items)]}


def train():
    import torch
    import shutil
    from smoke import frozen_image_router_state_v2 as state
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized(),'CPU-only training')
    torch.set_num_threads(8); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    b=receipt(); out=ROOT/OUTPUT
    runtime={'torch':str(torch.__version__),'build':c.digest(torch.__config__.show()),'python':platform.python_version(),
             'cpu':next(l for l in Path('/proc/cpuinfo').read_text().splitlines() if l.startswith('model name')),
             'threads':8,'interop':1,'deterministic':True}
    c.write_new(out/'cpu_runtime.json',runtime)
    index=feature_index(b); c.write_new(out/'feature_index.json',index)
    if (ROOT/REPORT).exists():
        require(c.read(ROOT/REPORT)['receipt_sha256']==c.sha(out/'receipt.json'),'existing result mismatch')
        print('FROZEN_IMAGE_V2_ALREADY_COMPLETE no_training_or_holdout_rerun=true',flush=True); return
    if not (out/'steps/step_000000.json').exists(): require(shutil.disk_usage(out).free>=30*1024**3,'need 30GiB for per-step saves')
    fit=load_role(b,index,'fit'); cal=load_role(b,index,'calibration'); m,opt=model(b)
    frozen_hash=s.state.io.parameter_hash(dict(m.image.named_parameters()))
    identity=c.digest({'receipt':c.sha(out/'receipt.json'),'runtime':runtime,'features':index,'image':frozen_hash})
    saved=state.load(out,identity)
    start=0; history=[]
    if saved:
        state.restore(m,opt,saved); start=saved['step']; history=saved['history']
    else: state.save(out,m,opt,0,history,identity)
    base=evaluate(m,cal,False); c.write_new(out/'calibration_image_only.json',base)
    # Resume may have committed a milestone before its evaluation JSON.
    if start in CHECKS and not (out/f'calibration_{start}.json').exists(): c.write_new(out/f'calibration_{start}.json',evaluate(m,cal,True))
    for step in range(start,STEPS):
        ids=torch.tensor(batch_indices(fit[0],step)); stats=update(m,opt,fit[1][ids],fit[2][ids],fit[3][ids]); history.append(stats)
        state.save(out,m,opt,step+1,history,identity)
        if (step+1)%100==0: print(f'FROZEN_IMAGE_V2_STEP step={step+1} loss={stats["loss"]}',flush=True)
        if step+1 in CHECKS: c.write_new(out/f'calibration_{step+1}.json',evaluate(m,cal,True))
    require(s.state.io.parameter_hash(dict(m.image.named_parameters()))==frozen_hash,'image branch changed')
    calibrations={step:c.read(out/f'calibration_{step}.json') for step in CHECKS}
    chosen=min(CHECKS,key=lambda step:(calibrations[step]['ce'],step))
    selection={'step':chosen,'criterion':'minimum_calibration_CE_then_earliest','calibration_sha256':
               {str(step):c.sha(out/f'calibration_{step}.json') for step in CHECKS},'receipt_sha256':c.sha(out/'receipt.json')}
    c.write_new(out/'selection.json',selection)  # Immutable BEFORE holdout features/labels are loaded.
    selected=state.load_step(out,chosen,identity); m.vlm.load_state_dict(selected['vlm'],strict=True)
    hold=load_role(b,index,'holdout'); baseline=evaluate(m,hold,False); corrected=evaluate(m,hold,True)
    gate=(corrected['correct']>baseline['correct'] and corrected['ce']<=baseline['ce']
          and all(corrected['per_task'][t]['correct']>=baseline['per_task'][t]['correct'] for t in ORDER))
    value={'status':'expanded_frozen_image_router_training_complete','run_id':RUN,'receipt_sha256':c.sha(out/'receipt.json'),
           'runtime':runtime,'selected_step':chosen,'selection_sha256':c.sha(out/'selection.json'),
           'selected_checkpoint':c.read(out/f'steps/step_{chosen:06d}.json'),'frozen_image_hash':frozen_hash,
           'calibration':{'image_only':base,'corrected':calibrations[chosen]},
           'holdout':{'image_only':baseline,'corrected':corrected},'classification_screen_pass':gate,
           'old40_pixels_or_features_accessed':False,'restoration_metrics_measured':False,'automatic_followup':False,'GPU_training':False,
           'limitations':['Frozen image was trained on original640; corrected head sees expanded2560: this is not a data-matched VLM ablation.',
                          'Source-separated for router; frozen FLUX/Qwen may have seen these sources.',
                          'Classification screen is not a PSNR/LPIPS gate or official restoration result.']}
    c.write_new(ROOT/REPORT,value)
    print(f'FROZEN_IMAGE_V2_TRAIN_COMPLETE selected={chosen} holdout={corrected["correct"]}/400 classification_screen={gate}',flush=True)


def verify():
    b=receipt(); out=ROOT/OUTPUT; r=c.read(ROOT/REPORT)
    require(r['status']=='expanded_frozen_image_router_training_complete' and r['receipt_sha256']==c.sha(out/'receipt.json')
            and r['runtime']==c.read(out/'cpu_runtime.json') and r['selection_sha256']==c.sha(out/'selection.json'),'terminal identity changed')
    selected=c.read(out/'selection.json'); require(selected['step']==r['selected_step'],'selected checkpoint changed')
    for step in CHECKS: require(c.sha(out/f'calibration_{step}.json')==selected['calibration_sha256'][str(step)],'calibration changed')
    require(r['selected_step']==min(CHECKS,key=lambda step:(c.read(out/f'calibration_{step}.json')['ce'],step)),'selection rule changed')
    marker=c.read(out/f'steps/step_{r["selected_step"]:06d}.json')
    require(marker==r['selected_checkpoint'] and c.sha(c.path(out,marker['file']))==marker['sha256'],'selected model changed')
    for role in ('calibration','holdout'):
        items={i['key']:i for i in b['items'] if i['role']==role}
        for arm in ('image_only','corrected'):
            summary=r[role][arm]; rows=summary['rows']; correct=Counter(); losses=[]
            require(summary['count']==len(rows)==400 and {x['key'] for x in rows}==set(items),'evaluation population changed')
            for row in rows:
                item=items[row['key']]; truth=ORDER.index(item['task']); logits=row['logits']; k=audit.argmax(logits)
                require(row['task_for_reporting_only']==item['task'] and row['seed']==item['seed']
                        and row['crop_xy']==item['row']['crop_xy'] and row['predicted_anchor']==ORDER[k],'evaluation input/prediction changed')
                correct[item['task']]+=int(k==truth); mx=max(logits)
                losses.append(mx+math.log(sum(math.exp(v-mx) for v in logits))-logits[truth])
            require(summary['correct']==sum(correct.values()) and abs(summary['ce']-sum(losses)/400)<2e-5,'evaluation aggregate changed')
            require(summary['per_task']=={t:{'count':80,'correct':correct[t]} for t in ORDER},'per-class count changed')
    base=r['holdout']['image_only']; fixed=r['holdout']['corrected']
    require(r['classification_screen_pass']==(fixed['correct']>base['correct'] and fixed['ce']<=base['ce']
            and all(fixed['per_task'][t]['correct']>=base['per_task'][t]['correct'] for t in ORDER)),'screen rule changed')
    require(r['restoration_metrics_measured'] is False and r['old40_pixels_or_features_accessed'] is False,'invalid success claims')
    print('FROZEN_IMAGE_V2_RESULT_VERIFIED classification_only=true',flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('phase',choices=['prepare','hardware','extract','train','verify'])
    ap.add_argument('--node',type=int,choices=range(4),default=0); args=ap.parse_args()
    require(str(ROOT)==s.analysis.ACTIVE_ROOT,'destination workspace only')
    if args.phase=='prepare': prepare()
    elif args.phase=='hardware': hardware(args.node,True)
    elif args.phase=='extract': extract(args.node)
    elif args.phase=='train': train()
    else: verify()


if __name__=='__main__': main()
