#!/usr/bin/env python3
"""Bounded frozen-FLUX calibration: prepare -> GPU cache -> CPU fit -> GPU score -> finish."""
import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'third_party/flux2/src'))
os.environ['FOUNDIR_OFFICIAL_NNODES']='4'  # Immutable parent identity only, not execution topology.
from smoke.existing_flux_replay_contract_v1 import read,sha,write_new as write
from lowlight_restoration import flux_frequency_repair_v1 as spectral
SCREEN=ROOT/'checkpoints/foundir_fullstack_noise_blur_sources_v1/plan.json'
PARENT=ROOT/'checkpoints/foundir_official_vlm_router_v4/official_vlm_v4_frozen4800_v1_4x2/receipt.json'
ALPHAS={'noise':.30,'blur':.75}
CATEGORIES={'noise':'05Noise','blur':'01Blur'}
SOURCES=['smoke/run_foundir_fullstack_frequency_repair_v1.py','smoke/run_foundir_fullstack_frequency_repair_v1_gpu.sh',
         'lowlight_restoration/flux_frequency_repair_v1.py','lowlight_restoration/five_expert_flux.py',
         'lowlight_restoration/protocol_routed_flux.py','lowlight_restoration/existing_flux_experts.py',
         'smoke/train_foundir_frozen_expert_router_mixed_driver_v1.py','smoke/expanded_vlm_h100_runtime_v4.py',
         'smoke/official_vlm_v4_metric_identity_fix.py']
RECIPE={'version':1,'train_sources':128,'development_sources':32,'crop':512,'bands':list(spectral.BANDS),
        'ridge':spectral.RIDGE,'coefficients_bounds':[0.,1.],'parameters_per_task':6,
        'coefficient_sharing':'one scalar per band shared across RGB and all images of the task',
        'fit':'closed_form_train_only_normal_equations_then_box_projection',
        'frozen_experts':True,'new_RGB_network':False,'FLUX_calls':50,'reference_time':10,'guidance':1.,
        'official_pixels_accessed':False,'automatic_official_evaluation':False,
        'validation_selection':False,'checkpoint_search':False,
        'gate':'train and development mean PSNR strictly improves, LPIPS does not worsen; development worst PSNR loss <=1dB',
        'cache_wall_cap_seconds':2700,'score_wall_cap_seconds':900,'automatic_retry':False}


def need(ok,message):
    if not ok:raise ValueError(message)


def out(task):return ROOT/('checkpoints/foundir_fullstack_frequency_repair_v1/'+task)
def report(task,phase):return ROOT/f'reports/foundir_fullstack_{task}_frequency_repair_v1_{phase}_2026-11-06.json'
def key(item):return item['role']+'_'+item['row']['record_id']


def rows_for(plan,task):
    rows=[]
    for role,split in [('train','train'),('development','validation')]:
        for chosen in plan['plans'][task][split]:rows.append({**chosen,'role':role})
    validate_rows(rows,task,set(plan['blocked_source_groups']))
    return rows


def validate_rows(rows,task,blocked):
    need(task in ALPHAS and len(rows)==160 and len({key(i) for i in rows})==160,'bad task/count/keys')
    groups={};hashes={}
    for role,split,count in [('train','train',128),('development','validation',32)]:
        subset=[i for i in rows if i['role']==role];need(len(subset)==count,'wrong role size')
        groups[role]={i['row']['source_group'] for i in subset};hashes[role]={i['row']['gt_sha256'] for i in subset}
        need(len(groups[role])==count and not groups[role]&blocked,'duplicate/protected source')
        for item in subset:
            r=item['row'];x,y=item['crop_xy'];need(r['split']==split and r['category']==CATEGORIES[task] and r['composition_size']=='1','wrong data scope')
            need(all(r[k].startswith('data/foundir/MillionIRData/Train/'+CATEGORIES[task]+'/') for k in ('lq_path','gt_path')),'protected/nontraining collection')
            need(0<=x<=int(r['width'])-512 and 0<=y<=int(r['height'])-512,'crop out of bounds')
    need(not groups['train']&groups['development'] and not hashes['train']&hashes['development'],'train/development overlap')


def prepare(task,nodes):
    start=time.monotonic();from PIL import Image
    import numpy as np
    from smoke import official_vlm_v4_metric_identity_fix as fix
    plan=read(SCREEN);parent=read(PARENT);rows=rows_for(plan,task)
    spectral.assigned(rows,nodes[0],0,nodes)
    need(sha(ROOT/'manifests/foundir_five_type_pairs_v1.csv')==plan['manifest_sha256'],'training manifest changed')
    need(sha(PARENT)==plan['official_receipt_sha256'],'screening parent changed')
    protected={r[k] for r in parent['rows'] for k in ('lq_sha256','gt_sha256')}
    assets={};p=parent['original']['parent']
    for k in ('flow_weights','autoencoder_weights'):assets[p['model'][k]]=p['model'][k+'_sha256']
    for t in ('blur','noise','haze','lowlight','rain'):
        s=p['experts'][t]['checkpoint'];assets[s['path']]=s['sha256']
        s=p['rain_prompt'] if t=='rain' else p['four_prompts']['tensors'][t]
        path=s['path'] if t=='rain' else str(Path(p['config']['four_parent_run'])/s['path'])
        assets[path]=s['sha256']
    for path,digest in assets.items():need(sha(ROOT/path)==digest,'model/prompt hash changed: '+path)
    for item in rows:
        for field in ('lq','gt'):
            r=item['row'];need(r[field+'_sha256'] not in protected,'protected exact image')
            data=(ROOT/r[field+'_path']).read_bytes();need(hashlib.sha256(data).hexdigest()==r[field+'_sha256'],'data file hash mismatch')
            with Image.open(io.BytesIO(data)) as im:
                need(im.size==(int(r['width']),int(r['height'])),'data dimension mismatch')
                x,y=item['crop_xy'];a=np.asarray(im.convert('RGB').crop((x,y,x+512,y+512)))
                need(a.shape==(512,512,3) and a.dtype==np.uint8,'crop decode failed')
        print('FREQUENCY_PREFLIGHT '+task+' '+key(item),flush=True) if len(rows)<2 else None
    fix.checked_metric_models('cpu',parent['metric_identity'])
    need(shutil.disk_usage(ROOT).free>20*1024**3,'need 20GiB without deleting assets')
    names=sorted(set(parent['sources'])|set(SOURCES))
    value={'task':task,'recipe':RECIPE,'rows':rows,'nodes':nodes,'world':2*len(nodes),'parent_alpha':ALPHAS[task],
           'sources':{name:sha(ROOT/name) for name in names},'assets':assets,
           'screen_sha256':sha(SCREEN),'parent_receipt_sha256':sha(PARENT),'metric_identity':parent['metric_identity'],
           'legacy_checkpoint_exposure_not_erased':True,'development_previously_used_in_history':True,
           'source_screen_not_proof_of_semantic_disjointness':True}
    write(out(task)/'receipt.json',value)
    value={'status':'destination_CPU_preflight_pass','receipt_sha256':sha(out(task)/'receipt.json'),
           'task':task,'train':128,'development':32,'all320_image_files_SHA_and_crop_decode_checked':True,
           'model_prompt_hashes_checked':len(assets),'metric_identity_checked':True,'nodes':nodes,
           'GPU_started':False,'CUDA_qualification':'pending','official_pixels_decoded':False,
           'seconds':time.monotonic()-start,'GPU_hours':0,'exit_code':0}
    write(report(task,'preflight'),value);print('FREQUENCY_PREFLIGHT_PASS '+json.dumps(value),flush=True)


def bound(task):
    b=read(out(task)/'receipt.json');need(b['task']==task and b['recipe']==RECIPE and b['parent_alpha']==ALPHAS[task],'recipe changed')
    need(sha(SCREEN)==b['screen_sha256'] and sha(PARENT)==b['parent_receipt_sha256'],'parent binding changed')
    for name,digest in b['sources'].items():need(sha(ROOT/name)==digest,'source changed: '+name)
    validate_rows(b['rows'],task,set(read(SCREEN)['blocked_source_groups']))
    return b


def crop(item,field):
    from PIL import Image
    import numpy as np
    r=item['row'];data=(ROOT/r[field+'_path']).read_bytes()
    need(hashlib.sha256(data).hexdigest()==r[field+'_sha256'],'source image changed')
    x,y=item['crop_xy']
    with Image.open(io.BytesIO(data)) as im:return np.asarray(im.convert('RGB').crop((x,y,x+512,y+512))).copy()


def png_load(path,digest):
    from PIL import Image
    import numpy as np
    data=path.read_bytes();need(hashlib.sha256(data).hexdigest()==digest,'cached PNG changed')
    with Image.open(io.BytesIO(data)) as im:return np.array(im.convert('RGB'),copy=True)


def cached(task,item,receipt):
    root=out(task);p=root/'cache'/(key(item)+'.json')
    if not p.exists():return None
    r=read(p);need(r['item']==item and r['receipt_sha256']==receipt and r['GT_input'] is False,'cache identity mismatch')
    data={name:png_load(root/spec['path'],spec['sha256']) for name,spec in r['images'].items()}
    need(set(data)=={'raw','baseline'} and all(v.shape==(512,512,3) for v in data.values()),'cache shape/schema')
    return r,data


def hardware():
    import torch
    local=int(os.environ['LOCAL_RANK'])
    need(local in (0,1) and int(os.environ['LOCAL_WORLD_SIZE'])==2 and torch.cuda.device_count()==2,'two visible local GPUs required')
    torch.cuda.set_device(local);device=torch.device('cuda',local)
    need(any(n in torch.cuda.get_device_name(local) for n in ('H100','H200')) and torch.cuda.get_device_properties(local).total_memory>=75*1024**3,'75GiB Hopper GPU required')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    return local,device


def cache(task,node):
    import torch
    from lowlight_restoration.five_expert_flux import restore_five_expert
    from lowlight_restoration.scalar_residual import quantized_scalar_residual
    from smoke import train_foundir_frozen_expert_router_mixed_driver_v1 as impl
    from smoke import official_vlm_v4_metric_identity_fix as fix
    from smoke.expanded_vlm_h100_runtime_v4 import bounded_vae,POLICY
    begin=time.monotonic();b=bound(task);local,device=hardware()
    rank,items=spectral.assigned(b['rows'],node,local,b['nodes']);root=out(task);digest=sha(root/'receipt.json')
    runtime={'receipt_sha256':digest,'rank':rank,'node':node,'torch':str(torch.__version__),
             'cuda':torch.version.cuda,'device':torch.cuda.get_device_name(local),'vae_policy':POLICY,'tf32':False}
    write(root/f'cache_runtime{rank}.json',runtime)
    for path,h in b['assets'].items():need(sha(ROOT/path)==h,'asset changed before GPU load')
    p=read(PARENT)['original']['parent']['model']
    os.environ.update(KLEIN_4B_BASE_MODEL_PATH=str(ROOT/p['flow_weights']),AE_MODEL_PATH=str(ROOT/p['autoencoder_weights']))
    missing=[i for i in items if cached(task,i,digest) is None]
    if missing:
        model,gates,rules,ae,prompts=impl.load_bank(read(PARENT)['original'],device);ae=bounded_vae(ae)
        weights=torch.zeros(1,5,device=device);weights[0,('blur','noise','haze','lowlight','rain').index(task)]=1
        for item in missing:
            need(time.monotonic()-begin<RECIPE['cache_wall_cap_seconds'],'cache cap reached; preserve outputs')
            lq=fix.producer.tensor(crop(item,'lq'),device)
            with torch.no_grad():
                raw,base,meta=restore_five_expert(model,gates,rules,weights,ae,lq,prompts,seed=item['inference_seed'])
                raw8=quantized_scalar_residual(lq,raw,1.)[0]
            need(meta['anchor']==task and meta['model_calls']==50 and meta['alpha']==ALPHAS[task],'frozen native protocol mismatch')
            images={}
            for name,tensor in [('raw',raw8),('baseline',base)]:
                path=root/'png'/(key(item)+'_'+name+'.png')
                fix.producer.png_once(path,tensor[0].permute(1,2,0).cpu().numpy())
                images[name]={'path':str(path.relative_to(root)),'sha256':sha(path)}
            write(root/'cache'/(key(item)+'.json'),dict(item=item,receipt_sha256=digest,rank=rank,
                  runtime_sha256=sha(root/f'cache_runtime{rank}.json'),images=images,meta=meta,GT_input=False))
            print(f'FREQUENCY_CACHE task={task} rank={rank} key={key(item)} seconds={time.monotonic()-begin:.1f}',flush=True)
            del lq,raw,raw8,base
    for item in items:need(cached(task,item,digest) is not None,'incomplete cache')
    marker=root/f'cache_worker{rank}.json'
    if not marker.exists():write(marker,dict(receipt_sha256=digest,rank=rank,keys=[key(i) for i in items],seconds=time.monotonic()-begin,
                                          peak_allocated_GiB=torch.cuda.max_memory_allocated()/1024**3))
    print(f'FREQUENCY_CACHE_END task={task} rank={rank} status=0',flush=True)


def verify_cache(task,b):
    digest=sha(out(task)/'receipt.json');signatures=[]
    for node in b['nodes']:
        for local in (0,1):
            rank,items=spectral.assigned(b['rows'],node,local,b['nodes'])
            marker=read(out(task)/f'cache_worker{rank}.json');runtime=read(out(task)/f'cache_runtime{rank}.json')
            need(marker['receipt_sha256']==digest and marker['keys']==[key(i) for i in items] and marker['rank']==rank,'cache worker incomplete')
            signatures.append({k:v for k,v in runtime.items() if k not in ('rank','node','device')})
            for item in items:
                value=cached(task,item,digest);need(value is not None,'cache missing')
                need(value[0]['rank']==rank and value[0]['runtime_sha256']==sha(out(task)/f'cache_runtime{rank}.json'),'runtime binding changed')
    need(all(s==signatures[0] for s in signatures),'numerical cache runtime mismatch')


def fit(task):
    start=time.monotonic();b=bound(task);verify_cache(task,b);root=out(task);digest=sha(root/'receipt.json')
    target=root/'coefficients.json'
    if target.exists():need(read(target)['receipt_sha256']==digest,'coefficient identity changed');return
    stats=[];markers={}
    for item in b['rows']:
        if item['role']!='train':continue
        _,arrays=cached(task,item,digest)
        stats.append(spectral.statistics(crop(item,'lq'),arrays['raw'],crop(item,'gt')))
        markers[key(item)]=sha(root/'cache'/(key(item)+'.json'))
    need(len(stats)==128,'only128 training sources may fit')
    weights=spectral.fit(stats,ALPHAS[task]).tolist()
    value={'receipt_sha256':digest,'task':task,'weights':weights,'train_records':128,'training_cache_markers':markers,
           'train_statistics':stats,'GT_used_in_fit':'training_only','validation_or_official_feedback':False,
           'seconds':time.monotonic()-start,'exit_code':0,'GPU_hours':0}
    write(target,value);write(report(task,'fit'),value)
    print('FREQUENCY_FIT_END '+str({'task':task,'weights':weights,'seconds':value['seconds']}),flush=True)


def score(task,node):
    import numpy as np
    import torch
    from smoke import official_vlm_v4_metric_identity_fix as fix
    begin=time.monotonic();b=bound(task);local,device=hardware();root=out(task);digest=sha(root/'receipt.json')
    co=read(root/'coefficients.json');need(co['receipt_sha256']==digest and co['train_records']==128,'fit not complete')
    coefficient_sha=sha(root/'coefficients.json');metrics,_=fix.checked_metric_models(device,b['metric_identity'])
    # All scoring is deliberately one node/two local GPUs; no cross-node rendezvous.
    for item in b['rows'][local::2]:
        need(time.monotonic()-begin<RECIPE['score_wall_cap_seconds'],'score cap reached')
        marker=root/'scores'/(key(item)+'.json')
        if marker.exists():
            prior=read(marker);need(prior['coefficient_sha256']==coefficient_sha and prior['receipt_sha256']==digest and prior['item']==item,'score identity changed');continue
        _,arrays=cached(task,item,digest);lq=crop(item,'lq')
        candidate=spectral.apply(lq,arrays['raw'],co['weights'])
        constant=spectral.apply(lq,arrays['raw'],[ALPHAS[task]]*6)
        path=root/'candidate'/(key(item)+'.png');fix.producer.png_once(path,candidate)
        # All predictions are fixed/published before GT is loaded for metric calculation.
        gt=crop(item,'gt');target=torch.from_numpy(gt.copy()).permute(2,0,1).unsqueeze(0).to(device).float()/255
        result={}
        for name,image in [('lq',lq),('baseline',arrays['baseline']),('constant_band_control',constant),('candidate',candidate)]:
            pred=torch.from_numpy(image.copy()).permute(2,0,1).unsqueeze(0).to(device).float()/255
            with torch.no_grad():result[name]={k:float(m(pred,target).item()) for k,m in metrics.items()}
        need(all(math.isfinite(v) for vals in result.values() for v in vals.values()),'nonfinite scores')
        write(marker,dict(receipt_sha256=digest,coefficient_sha256=coefficient_sha,item=item,metrics=result,
                          cache_marker_sha256=sha(root/'cache'/(key(item)+'.json')),
                          output={'path':str(path.relative_to(root)),'sha256':sha(path)},
                          train_fit_only=item['role']=='train',GT_input=False,
                          constant_control_max_uint8_difference=int(np.abs(constant.astype(int)-arrays['baseline'].astype(int)).max())))
        print(f'FREQUENCY_SCORE task={task} local={local} key={key(item)}',flush=True)
    marker=root/f'score_worker{local}.json'
    if not marker.exists():write(marker,dict(receipt_sha256=digest,coefficient_sha256=coefficient_sha,node=node,local=local,
          keys=[key(i) for i in b['rows'][local::2]],seconds=time.monotonic()-begin))
    print(f'FREQUENCY_SCORE_END task={task} local={local} status=0',flush=True)


def gates(summary):
    return all(summary[role]['candidate']['psnr']>summary[role]['baseline']['psnr']
               and summary[role]['candidate']['lpips']<=summary[role]['baseline']['lpips']
               for role in ('train','development')) and summary['development']['worst_psnr_loss']<=1.


def finish(task):
    begin=time.monotonic();b=bound(task);verify_cache(task,b);root=out(task);digest=sha(root/'receipt.json')
    co=read(root/'coefficients.json');co_sha=sha(root/'coefficients.json');rows=[]
    for local in (0,1):
        marker=read(root/f'score_worker{local}.json')
        need(marker['receipt_sha256']==digest and marker['coefficient_sha256']==co_sha and marker['keys']==[key(i) for i in b['rows'][local::2]],'score worker incomplete')
    for item in b['rows']:
        r=read(root/'scores'/(key(item)+'.json'))
        need(r['receipt_sha256']==digest and r['item']==item and r['coefficient_sha256']==co_sha and r['GT_input'] is False,'score mismatch')
        need(sha(root/'cache'/(key(item)+'.json'))==r['cache_marker_sha256'] and sha(root/r['output']['path'])==r['output']['sha256'],'output/cache changed')
        need(all(math.isfinite(v) for scores in r['metrics'].values() for v in scores.values()),'nonfinite result');rows.append(r)
    summaries={}
    for role in ('train','development'):
        selected=[r for r in rows if r['item']['role']==role]
        summary={arm:{m:math.fsum(r['metrics'][arm][m] for r in selected)/len(selected) for m in ('psnr','lpips','ssim')}
                 for arm in ('lq','baseline','constant_band_control','candidate')}
        summary['count']=len(selected)
        summary['worst_psnr_loss']=max(r['metrics']['baseline']['psnr']-r['metrics']['candidate']['psnr'] for r in selected)
        summary['constant_control_max_uint8_difference']=max(r['constant_control_max_uint8_difference'] for r in selected)
        summaries[role]=summary
    result=dict(status='bounded_candidate_complete',task=task,receipt_sha256=digest,coefficient_sha256=co_sha,
                weights=co['weights'],summary=summaries,admit_official_evaluation=gates(summaries),
                official_gate_claim=False,official_evaluated=False,automatic_deployment=False,
                new_FLUX_training=False,fit_parameters=6,development_is_not_blind_test=True,
                seconds=time.monotonic()-begin,exit_code=0)
    target=report(task,'result')
    if target.exists():need(read(target)['coefficient_sha256']==co_sha,'prior result changed');return
    write(target,result);print('FREQUENCY_RESULT_VERIFIED '+json.dumps(result),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','cache','fit','score','finish','verify']);p.add_argument('--task',choices=ALPHAS,required=True)
    p.add_argument('--nodes',type=int,nargs='+',default=[0,1,2]);p.add_argument('--node',type=int,default=0);a=p.parse_args()
    if a.phase=='prepare':prepare(a.task,a.nodes)
    elif a.phase in ('cache','score'):globals()[a.phase](a.task,a.node)
    elif a.phase=='verify':bound(a.task);print('FREQUENCY_BINDING_VERIFIED')
    else:globals()[a.phase](a.task)

if __name__=='__main__':main()
