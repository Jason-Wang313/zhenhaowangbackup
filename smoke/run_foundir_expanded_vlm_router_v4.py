#!/usr/bin/env python3
"""One bounded expansion: features -> CPU training -> fixed 100-image restoration.

Four nodes/two H200 per node for independent GPU shards, no rendezvous. No extra
classification qualification gate. Selection is committed before confirmation.
"""
import argparse
from collections import Counter
import csv
import math
import os
from pathlib import Path
import platform
import random
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from smoke import expanded_vlm_router_contract_v4 as x
from smoke import expanded_vlm_router_state_v4 as state
from smoke import audit_foundir_vlm_preserve_fit_v1 as audit
c,require,ORDER=x.c,x.require,x.ORDER
EXPECTED_ITEMS='568ffd35a6cad695a95f1d86f90a2826f606e596a10f4b5b11f58351e64d6432'


def prepare():
    x.old.verify()
    old=x.old.receipt(); producer=x.old.parent.receipt()
    manifest=c.path(ROOT,producer['original']['config']['manifest'])
    require(c.sha(manifest)==producer['original']['config']['manifest_sha256'],'manifest changed')
    with manifest.open(newline='') as f:
        items=x.expand(csv.DictReader(f),producer['items'],old['future_confirmation']['items'])
    require(c.digest(items)==EXPECTED_ITEMS,'expansion differs from real metadata plan')
    for i in items:
        require(c.path(ROOT,i['row']['lq_path']).is_file(),'missing LQ image')
        if i['role']=='confirmation': require(c.path(ROOT,i['row']['gt_path']).is_file(),'missing confirmation GT')
    fit,groups=x.schedule(items)
    exposures=Counter(j for step in range(x.STEPS) for j in x.batch_indices(groups,step))
    counts={t:{'sources':len(groups[t]),'crops':sum(i['task']==t for i in fit),
               'exposure_min':min(exposures[j] for j,i in enumerate(fit) if i['task']==t),
               'exposure_max':max(exposures[j] for j,i in enumerate(fit) if i['task']==t)} for t in ORDER}
    out=ROOT/x.OUTPUT
    require((out/'receipt.json').exists() or shutil.disk_usage(ROOT).free>=150*1024**3,'need 150GiB for immutable state and caches')
    b={'run_id':x.RUN,'recipe':x.RECIPE,'items':items,'items_sha256':EXPECTED_ITEMS,'counts':counts,
       'unique_fit_sources':len({i['row']['source_group'] for i in fit}),
       'reuse':old['features'],'image_checkpoint':old['image_checkpoint'],'original':producer['original'],
       'parent_receipt_sha256':c.sha(ROOT/x.old.OUTPUT/'receipt.json'),
       'parent_result_sha256':c.sha(ROOT/x.old.REPORT),
       'sources':{**old['sources'],**{n:c.sha(ROOT/n) for n in x.SOURCES},
                  'smoke/audit_foundir_vlm_preserve_fit_v1.py':c.sha(ROOT/'smoke/audit_foundir_vlm_preserve_fit_v1.py')},
       'calibration':400,'confirmation':100,'automatic_budget_extension':False}
    c.write_new(out/'receipt.json',b)
    print(f'EXPANDED_V4_PREPARED fit={len(fit)} new_features={sum(not i["reuse_key"] for i in items)} counts={counts}',flush=True)


def bound():
    b=c.read(ROOT/x.OUTPUT/'receipt.json'); old=x.old.receipt()
    require(b['run_id']==x.RUN and b['recipe']==x.RECIPE and c.digest(b['items'])==b['items_sha256']==EXPECTED_ITEMS,'fixed scope changed')
    require(c.sha(ROOT/x.old.OUTPUT/'receipt.json')==b['parent_receipt_sha256'] and c.sha(ROOT/x.old.REPORT)==b['parent_result_sha256'],
            'parent identity changed')
    require(b['image_checkpoint']==old['image_checkpoint'] and b['reuse']==old['features'],'frozen image/features changed')
    for n,h in b['sources'].items(): require(c.sha(ROOT/n)==h,'bound source changed: '+n)
    return b


def hardware(node,phase,idle):
    import torch
    from smoke import train_foundir_frozen_expert_router_mixed_driver_v1 as impl
    from smoke.frozen_router_runtime_policy_v1 import execution_signature
    b=bound(); actual=impl.native.hopper.hardware(idle)
    require('H200' in actual['gpu_name'] and all(torch.cuda.get_device_properties(k).total_memory>=139*1024**3 for k in range(2)),
            'observed ~137GiB peak: two H200 >=139GiB required; H10080GB unqualified')
    original=b['original']
    actual['metrics']=impl.parent.four_impl.metric_runtime({'metric':original['parent']['metric']})
    actual['memory_policy']={'inherited':original['parent']['memory_policy'],'router_explicit_gate_checkpoint':True}
    prior=ROOT/'checkpoints/foundir_frozen_expert_router_v1'/x.old.parent.s.analysis.RUN
    require(all(execution_signature(actual)==execution_signature(c.read(prior/f'runtime_node{k}.json')) for k in range(4)),
            'GPU numerical runtime incompatible')
    path=ROOT/x.OUTPUT/f'{phase}_runtime_node{node}.json'
    if idle: c.write_new(path,actual)
    else: require(c.read(path)==actual,'node runtime changed; explicit migration required')
    return b,impl


def device_setup(node,phase):
    import torch
    local=int(os.environ['LOCAL_RANK'])
    require(int(os.environ['WORLD_SIZE'])==2 and int(os.environ['LOCAL_WORLD_SIZE'])==2 and local in (0,1),'two independent local workers')
    b,impl=hardware(node,phase,False)
    torch.cuda.set_device(local); device=torch.device('cuda',local)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False; torch.backends.cudnn.benchmark=False
    original=b['original']
    os.environ.update(KLEIN_4B_BASE_MODEL_PATH=str(c.path(ROOT,original['parent']['model']['flow_weights'])),
                      AE_MODEL_PATH=str(c.path(ROOT,original['parent']['model']['autoencoder_weights'])))
    return b,impl,node*2+local,device


def completion(phase,node,rank,count,extra=None):
    import torch
    out=ROOT/x.OUTPUT
    value={'receipt_sha256':c.sha(out/'receipt.json'),'rank':rank,'count':count,
           'runtime_sha256':c.sha(out/f'{phase}_runtime_node{node}.json'),**(extra or {})}
    file=out/f'{phase}_worker{rank}.json'
    if file.exists():
        require({k:v for k,v in c.read(file).items() if k!='peak_allocated_gib'}==value,'completed worker identity changed')
    else: c.write_new(file,{**value,'peak_allocated_gib':torch.cuda.max_memory_allocated()/1024**3})
    print(f'EXPANDED_V4_WORKER_COMPLETE phase={phase} rank={rank} count={count}',flush=True)


def features(node):
    b,impl,rank,device=device_setup(node,'features')
    fresh=[i for i in b['items'] if i['reuse_key'] is None]
    # Existing extractor binds each new item/receipt and atomically saves VLM+FLUX features.
    impl.features(ROOT/x.OUTPUT,{**b['original'],'items':fresh},rank,device)
    completion('features',node,rank,len(fresh[rank::8]))


def require_workers(phase,counts,extra=None):
    out=ROOT/x.OUTPUT
    for rank in range(8):
        row=c.read(out/f'{phase}_worker{rank}.json')
        require({k:v for k,v in row.items() if k!='peak_allocated_gib'}==
                {'receipt_sha256':c.sha(out/'receipt.json'),'rank':rank,'count':counts[rank],
                 'runtime_sha256':c.sha(out/f'{phase}_runtime_node{rank//2}.json'),**(extra or {})},'worker missing/mismatched')
        require(isinstance(row['peak_allocated_gib'],(int,float)) and math.isfinite(row['peak_allocated_gib']) and row['peak_allocated_gib']>=0,
                'invalid memory evidence')


def feature_index(b):
    out=ROOT/x.OUTPUT; fresh=[i for i in b['items'] if not i['reuse_key']]
    require_workers('features',[len(fresh[k::8]) for k in range(8)])
    index={}
    for i in b['items']:
        if i['reuse_key']:
            index[i['key']]=b['reuse'][i['reuse_key']]
        else:
            row=c.read(out/'features'/(i['key']+'.json'))
            require(row['identity']=={'receipt_sha256':c.sha(out/'receipt.json'),'item':i,'arm':'features'} and
                    row['GT_input'] is False and row['vlm_receipt_sha256']==c.sha(out/'vlm'/(i['key']+'.json')),'cache input/producer mismatch')
            require(set(row['files'])=={row['tensor']},'unexpected artifact set')
            file=c.path(out,row['tensor'])
            index[i['key']]={'path':file.relative_to(ROOT).as_posix(),'sha256':row['files'][row['tensor']]}
    return index


def load_role(b,index,role):
    import torch
    from safetensors import safe_open
    require(role in ('fit','calibration','confirmation'),'invalid population')
    if role=='confirmation': require((ROOT/x.OUTPUT/'selection.json').is_file(),'confirmation locked until fixed checkpoint selection')
    items=[i for i in b['items'] if i['role']==role]; xs=[]; vs=[]
    for i in items:
        spec=index[i['key']]; file=c.path(ROOT,spec['path'])
        require(c.sha(file)==spec['sha256'],'feature changed')
        with safe_open(str(file),framework='pt',device='cpu') as f: im,v=f.get_tensor('image'),f.get_tensor('vlm')
        require(im.shape==(1,6144) and v.shape==(1,3584) and torch.isfinite(im).all() and torch.isfinite(v).all(),'invalid feature dimensions/values')
        xs.append(im.float()); vs.append(v.float())
    return items,torch.cat(xs),torch.cat(vs),torch.tensor([ORDER.index(i['task']) for i in items])


def model(b):
    import torch
    from lowlight_restoration.conditional_vlm_router_v4 import ConditionalVLMRouter
    reference,unused=x.old.parent.model(b)
    del unused
    random.seed(x.SEED); torch.manual_seed(x.SEED)
    m=ConditionalVLMRouter().cpu().float()
    m.image.load_state_dict(reference.image.state_dict(),strict=True); m.image.requires_grad_(False); m.eval()
    return m,torch.optim.AdamW(m.vlm.parameters(),lr=.001,weight_decay=.01,foreach=False,fused=False)


def cpu_setup():
    import torch
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized(),'CPU-only stage')
    torch.set_num_threads(8); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    runtime={'torch':str(torch.__version__),'build':c.digest(torch.__config__.show()),'python':platform.python_version(),
             'cpu':next(l for l in Path('/proc/cpuinfo').read_text().splitlines() if l.startswith('model name')),
             'threads':8,'interop':1,'deterministic':True}
    require(runtime==c.read(ROOT/x.old.REPORT)['runtime'],'use qualified parent CPU environment; no silent runtime migration')
    return runtime


def train():
    import torch
    runtime=cpu_setup(); b=bound(); out=ROOT/x.OUTPUT
    c.write_new(out/'cpu_runtime.json',runtime)
    if (ROOT/x.TRAIN_REPORT).exists(): verify_training(); print('EXPANDED_V4_TRAIN_ALREADY_COMPLETE',flush=True); return
    index=feature_index(b); c.write_new(out/'feature_index.json',index)
    if not (out/'steps/step_000000.json').exists(): require(shutil.disk_usage(out).free>=120*1024**3,'need 120GiB free for 6401 states')
    fit,cal=load_role(b,index,'fit'),load_role(b,index,'calibration')
    ordered,groups=x.schedule(b['items']); require(ordered==fit[0],'sampling index changed')
    m,opt=model(b); frozen=state.io.parameter_hash(dict(m.image.named_parameters()))
    identity=c.digest({'receipt':c.sha(out/'receipt.json'),'runtime':runtime,'features':c.sha(out/'feature_index.json'),'image':frozen})
    saved=state.load(out,identity); start=0; history=[]
    if saved: state.restore(m,opt,saved); start=saved['step']; history=saved['history']
    else: state.save(out,m,opt,0,history,identity)
    base=x.old.parent.evaluate(m,cal,False); c.write_new(out/'calibration_image_only.json',base)
    if start in x.CHECKS and not (out/f'calibration_{start}.json').exists():
        c.write_new(out/f'calibration_{start}.json',x.old.parent.evaluate(m,cal,True))
    for step in range(start,x.STEPS):
        ids=torch.tensor(x.batch_indices(groups,step))
        stats=x.old.update(m,opt,fit[1][ids],fit[2][ids],fit[3][ids]); history.append(stats)
        state.save(out,m,opt,step+1,history,identity)
        if (step+1)%100==0: print(f'EXPANDED_V4_STEP step={step+1} loss={stats["loss"]}',flush=True)
        if step+1 in x.CHECKS: c.write_new(out/f'calibration_{step+1}.json',x.old.parent.evaluate(m,cal,True))
    require(state.io.parameter_hash(dict(m.image.named_parameters()))==frozen,'frozen image changed')
    summaries={k:c.read(out/f'calibration_{k}.json') for k in x.CHECKS}
    chosen=min(x.CHECKS,key=lambda k:(summaries[k]['ce'],k))
    selection={'step':chosen,'calibration_sha256':{str(k):c.sha(out/f'calibration_{k}.json') for k in x.CHECKS},
               'criterion':x.RECIPE['selection'],'receipt_sha256':c.sha(out/'receipt.json')}
    c.write_new(out/'selection.json',selection)  # BEFORE loading confirmation tensors.
    m.vlm.load_state_dict(state.load_step(out,chosen,identity)['vlm'],strict=True)
    confirmation=load_role(b,index,'confirmation')
    predictions={'image_only':x.old.parent.evaluate(m,confirmation,False),
                 'conditional_v4':x.old.parent.evaluate(m,confirmation,True)}
    legacy,unused=x.old.parent.model(b); del unused
    prior=c.read(ROOT/x.old.REPORT)
    prior_identity=c.digest({'receipt':c.sha(ROOT/x.old.OUTPUT/'receipt.json'),'runtime':prior['runtime'],'image':prior['frozen_image_hash']})
    require(state.io.parameter_hash(dict(legacy.image.named_parameters()))==frozen,'legacy baseline image differs')
    legacy.vlm.load_state_dict(x.old.state.load_step(ROOT/x.old.OUTPUT,prior['selection']['step'],prior_identity)['vlm'],strict=True)
    predictions['previous_v3']=x.old.parent.evaluate(legacy,confirmation,True)
    c.write_new(out/'predictions.json',predictions)
    plan=x.render_plan(b['items'],predictions); c.write_new(out/'render_plan.json',plan)
    value={'status':'expanded_vlm_v4_training_complete','run_id':x.RUN,'receipt_sha256':c.sha(out/'receipt.json'),
           'runtime':runtime,'identity':identity,'frozen_image_hash':frozen,'selection':selection,
           'selected_checkpoint':c.read(out/f'steps/step_{chosen:06d}.json'),
           'feature_index_sha256':c.sha(out/'feature_index.json'),'predictions_sha256':c.sha(out/'predictions.json'),
           'render_plan_sha256':c.sha(out/'render_plan.json'),'calibration':{str(k):audit.compact(v) for k,v in summaries.items()},
           'confirmation_classification':{arm:audit.compact(v) for arm,v in predictions.items()},
           'unique_renders':plan['unique_renders'],'restoration_measured':False,'classification_blocks_restoration':False,
           'counts':b['counts'],'ready_for_prespecified_restoration':True}
    c.write_new(ROOT/x.TRAIN_REPORT,value)
    verify_training()
    print(f'EXPANDED_V4_TRAIN_COMPLETE selected={chosen} unique_renders={plan["unique_renders"]}',flush=True)


def verify_training():
    b=bound(); out=ROOT/x.OUTPUT; report=c.read(ROOT/x.TRAIN_REPORT)
    require(report['status']=='expanded_vlm_v4_training_complete' and report['run_id']==x.RUN and
            report['receipt_sha256']==c.sha(out/'receipt.json'),'training identity mismatch')
    require(report['runtime']==c.read(out/'cpu_runtime.json') and report['feature_index_sha256']==c.sha(out/'feature_index.json'),
            'runtime or feature index changed')
    expected_identity=c.digest({'receipt':c.sha(out/'receipt.json'),'runtime':report['runtime'],
                                'features':report['feature_index_sha256'],'image':report['frozen_image_hash']})
    require(report['identity']==expected_identity,'state identity changed')
    selection=c.read(out/'selection.json'); summaries={k:c.read(out/f'calibration_{k}.json') for k in x.CHECKS}
    require(selection==report['selection'] and selection=={'step':min(x.CHECKS,key=lambda k:(summaries[k]['ce'],k)),
            'calibration_sha256':{str(k):c.sha(out/f'calibration_{k}.json') for k in x.CHECKS},
            'criterion':x.RECIPE['selection'],'receipt_sha256':c.sha(out/'receipt.json')},'selection changed')
    items={i['key']:i for i in b['items'] if i['role']=='calibration'}
    for summary in summaries.values(): audit.validate_summary(summary,items)
    require(report['calibration']=={str(k):audit.compact(v) for k,v in summaries.items()},'calibration report differs')
    require([p.name for p in sorted((out/'steps').glob('step_*.json'))]==[f'step_{k:06d}.json' for k in range(x.STEPS+1)],'checkpoint step gap')
    for step in {0,selection['step'],x.STEPS}:
        row=c.read(out/f'steps/step_{step:06d}.json')
        require(row['identity']==expected_identity and row['step']==step and c.sha(c.path(out,row['file']))==row['sha256'],'checkpoint hash/progress mismatch')
    require(report['selected_checkpoint']==c.read(out/f'steps/step_{selection["step"]:06d}.json'),'selected checkpoint changed')
    predictions=c.read(out/'predictions.json'); confirm={i['key']:i for i in b['items'] if i['role']=='confirmation'}
    for summary in predictions.values(): audit.validate_summary(summary,confirm)
    require(report['confirmation_classification']=={k:audit.compact(v) for k,v in predictions.items()},'classification report differs')
    require(c.sha(out/'predictions.json')==report['predictions_sha256'] and c.sha(out/'render_plan.json')==report['render_plan_sha256'] and
            c.read(out/'render_plan.json')==x.render_plan(b['items'],predictions),'render plan changed')
    require(report['classification_blocks_restoration'] is False and report['restoration_measured'] is False and
            report['ready_for_prespecified_restoration'] is True,'wrong scientific claim/gate')
    print('EXPANDED_V4_TRAIN_VERIFIED no_classification_stop_gate=true',flush=True)


def read_render(plan,node):
    out=ROOT/x.OUTPUT; file=out/'renders'/(plan['render_id']+'.json')
    if not file.exists(): return None
    row=c.read(file)
    require(row['identity']=={'plan':plan,'receipt_sha256':c.sha(out/'receipt.json'),
            'render_plan_sha256':c.sha(out/'render_plan.json'),'runtime_sha256':c.sha(out/f'render_runtime_node{node}.json')},'render identity changed')
    require(row['GT_used_for_deployed_selection'] is False and row['route']['anchor']==plan['anchor'] and
            row['route']['weights']==[plan['weights']],'wrong deployed route')
    all_arms=c.read(out/'render_plan.json')['arms']
    require(row['used_by_arms']==[arm for arm,ids in all_arms.items() if plan['render_id'] in ids] and
            row['reference_plan_uses_class_label_not_image_GT']==(plan['render_id'] in all_arms['known_class_reference']),
            'reference/deployed attribution changed')
    require(row['files']=={row['tensor']:c.sha(c.path(out,row['tensor']))},'image tensor changed')
    require(all(isinstance(v,(int,float)) and math.isfinite(v) for v in row['metrics'].values()) and set(row['metrics'])=={'psnr','lpips'},
            'invalid metrics')
    return row


def render(node):
    import torch
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    from lowlight_restoration.five_expert_flux import restore_five_expert
    verify_training(); b,impl,rank,device=device_setup(node,'render')
    out=ROOT/x.OUTPUT; plan=c.read(out/'render_plan.json')
    routes=[plan['routes'][k] for k in sorted(plan['routes'])]; assigned=routes[rank::8]
    pending=[p for p in assigned if read_render(p,node) is None]
    if pending:
        model,gates,rules,ae,prompts=impl.load_bank(b['original'],device)
        metric=LearnedPerceptualImagePatchSimilarity(net_type='alex',normalize=True).to(device).eval().requires_grad_(False)
        items={i['key']:i for i in b['items'] if i['role']=='confirmation'}
        for p in pending:
            start=time.monotonic(); item=items[p['key']]
            with torch.no_grad():
                lq=impl.pixels(item,'lq',device); weights=torch.tensor([p['weights']],device=device)
                raw,u8,route=restore_five_expert(model,gates,rules,weights,ae,lq,prompts,seed=p['seed'])
                gt=impl.pixels(item,'gt',device)  # Only after immutable deployed selection/generation.
                image,target=u8.float()/255.,(gt+1)*.5
                mse=float((image-target).square().mean())
                metrics={'psnr':100. if mse<=1e-12 else -10*math.log10(mse),'lpips':float(metric(image,target))}; metric.reset()
            tensor,sha=impl.native.replay.save_tensors(out,p['render_id'],{'uint8':u8})
            row={'identity':{'plan':p,'receipt_sha256':c.sha(out/'receipt.json'),'render_plan_sha256':c.sha(out/'render_plan.json'),
                             'runtime_sha256':c.sha(out/f'render_runtime_node{node}.json')},
                 'tensor':tensor,'files':{tensor:sha},'route':route,'metrics':metrics,'rank':rank,
                 'GT_used_for_deployed_selection':False,
                 'used_by_arms':[arm for arm,ids in plan['arms'].items() if p['render_id'] in ids],
                 'reference_plan_uses_class_label_not_image_GT':p['render_id'] in plan['arms']['known_class_reference'],
                 'seconds':time.monotonic()-start}
            c.write_new(out/'renders'/(p['render_id']+'.json'),row); read_render(p,node)
            print(f'EXPANDED_V4_RENDER rank={rank} id={p["render_id"]} metrics={metrics}',flush=True)
            del lq,gt,raw,u8,image,target
    completion('render',node,rank,len(assigned),{'render_plan_sha256':c.sha(out/'render_plan.json')})


def compute_costs(out):
    result={}
    for key,gpus in [('prepare',0),('train',0)]+[(f'{phase}_node{node}',2) for phase in ('features','render') for node in range(4)]:
        starts=sorted((out/'attempts').glob(key+'_*.started.json'))
        ends=sorted((out/'attempts').glob(key+'_*.ended.json'))
        require(starts and {p.name.removesuffix('.started.json') for p in starts}==
                {p.name.removesuffix('.ended.json') for p in ends},'missing/interrupted compute accounting: '+key)
        rows=[c.read(p) for p in ends]
        require(any(r['status']==0 for r in rows) and all(type(r['seconds']) is int and r['seconds']>=0 and
                r['gpus']==gpus and r['used_gpu_seconds']==r['seconds']*gpus for r in rows),'invalid/noncompleted attempt: '+key)
        result[key]={'attempts':len(rows),'failed_attempts':sum(r['status']!=0 for r in rows),
                     'seconds':sum(r['seconds'] for r in rows),'gpus':gpus,'used_gpu_seconds':sum(r['used_gpu_seconds'] for r in rows)}
    return {'phases':result,'used_gpu_hours':sum(r['used_gpu_seconds'] for r in result.values())/3600,
            'finish_cpu_attempt_reported_separately':True,'whole_machine_allocation_hours_known':False}


def final_report():
    verify_training(); b=bound(); out=ROOT/x.OUTPUT; plan=c.read(out/'render_plan.json')
    routes=[plan['routes'][k] for k in sorted(plan['routes'])]
    require_workers('render',[len(routes[k::8]) for k in range(8)],{'render_plan_sha256':c.sha(out/'render_plan.json')})
    metrics={}; artifacts={}
    prompts={t:b['original']['parent']['experts'][t]['prompt'] for t in ORDER}
    for index,p in enumerate(routes):
        row=read_render(p,(index%8)//2); require(row is not None and row['rank']==index%8,'render missing/wrong shard')
        x.old.parent.audit.quality.analysis.check_route(row['route'],p['anchor'],control=True,prompts=prompts)
        metrics[p['render_id']]=row['metrics']; artifacts[p['render_id']]={'record_sha256':c.sha(out/'renders'/(p['render_id']+'.json')),
                                                                      'tensor_sha256':row['files'][row['tensor']]}
    return {'status':'expanded_vlm_v4_restoration_complete','run_id':x.RUN,'receipt_sha256':c.sha(out/'receipt.json'),
            'training_report_sha256':c.sha(ROOT/x.TRAIN_REPORT),'render_plan_sha256':c.sha(out/'render_plan.json'),
            'selected_checkpoint':c.read(ROOT/x.TRAIN_REPORT)['selected_checkpoint'],
            **x.aggregate(plan,b['items'],metrics),'artifacts':artifacts,'compute_costs':compute_costs(out),
            'confirmation_records':100,'classification_blocked_restoration':False,'automatic_followup':False,
            'limitations':['Only five singleton types; no double/triple claim.',
                           '100 training-collection sources separated from router training; frozen FLUX/Qwen may have seen them.',
                           'Larger data plus changed correction head is not an isolated causal VLM ablation.',
                           'Numerical SoW thresholds retained as reference on new100, not an official-test pass.',
                           'All per-class regressions reported; mean improvement does not imply no harm.']}


def finish():
    value=final_report(); c.write_new(ROOT/x.REPORT,value)
    require(c.read(ROOT/x.REPORT)==value,'terminal report differs')
    print('EXPANDED_V4_RESULT_VERIFIED '+str({k:value[k] for k in ('joint_mean_improvement_over_image_and_previous_vlm','all_numeric_targets_on_new100_pass')}),flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('phase',choices=('prepare','hardware','features','train','verify_train','render','finish'))
    ap.add_argument('--node',type=int,choices=range(4),default=0)
    ap.add_argument('--stage',choices=('features','render'),default='features')
    args=ap.parse_args(); require(str(ROOT)==x.old.parent.s.analysis.ACTIVE_ROOT,'destination workspace only')
    if args.phase=='hardware': hardware(args.node,args.stage,True)
    elif args.phase in ('features','render'): globals()[args.phase](args.node)
    else: {'prepare':prepare,'train':train,'verify_train':verify_training,'finish':finish}[args.phase]()


if __name__=='__main__': main()
