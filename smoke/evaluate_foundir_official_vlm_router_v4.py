#!/usr/bin/env python3
"""Frozen V4 official SOW600, original-size, three paired arms. No training."""
import argparse
import csv
import gc
import importlib.metadata
import os
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'third_party/flux2/src'))
from smoke import official_vlm_router_v4_contract as x
from smoke import run_foundir_expanded_vlm_router_v4 as prior
c,require=x.c,x.require
ACTIVE='/inspire/hdd/project/multimodal-discrete-diffusion/liuxiaohong-25080/zhenhaowang'


def metric_models(device):
    import pyiqa
    from smoke.rain_native50_continue_state_v1 import parameter_hash
    require(importlib.metadata.version('pyiqa')=='0.1.16','official pyiqa 0.1.16 required; do not silently replace')
    models={k:pyiqa.create_metric(k,device=device).eval().requires_grad_(False) for k in ('psnr','lpips','ssim')}
    identity={'version':'0.1.16','defaults':True,'state_hashes':{k:parameter_hash({n:t.detach().cpu() for n,t in m.state_dict().items()}) for k,m in models.items()}}
    return models,identity


def load_router(b):
    import torch
    from lowlight_restoration.conditional_vlm_router_v4 import ConditionalVLMRouter
    from smoke.rain_native50_continue_state_v1 import parameter_hash
    spec=b['image_checkpoint']; base=ROOT/prior.x.old.parent.s.OUTPUT/'image_only'
    file=c.path(base,spec['file']); require(c.sha(file)==spec['sha256'],'image head changed')
    saved=torch.load(file,map_location='cpu',weights_only=True)
    require(saved['step']==400 and parameter_hash(saved['model'])==spec['parameter_hash'],'wrong image head')
    model=ConditionalVLMRouter().cpu().float()
    model.image.load_state_dict({k.removeprefix('image.'):v for k,v in saved['model'].items() if k.startswith('image.')},strict=True)
    report=c.read(ROOT/prior.x.TRAIN_REPORT)
    require(c.sha(ROOT/prior.x.TRAIN_REPORT)==x.TRAIN_REPORT_SHA,'selected training report changed')
    selected=report['selected_checkpoint']; require(selected['step']==4800 and selected['sha256']==x.SELECTED_SHA,'fixed step4800 required')
    value=prior.state.load_step(ROOT/prior.x.OUTPUT,4800,report['identity'])
    require(value['parameter_hash']==selected['parameter_hash'],'wrong selected model')
    model.vlm.load_state_dict(value['vlm'],strict=True)
    model.eval().requires_grad_(False)
    return model


def prepare():
    # No old development/calibration pixel or feature is loaded. Parent metadata
    # and selected parameter states are consulted only to bind the frozen model.
    b=prior.bound(); router=load_router(b); del router
    file=ROOT/x.MANIFEST; require(c.sha(file)==x.MANIFEST_SHA,'wrong official manifest')
    with file.open(newline='') as f: rows=list(csv.DictReader(f))
    x.validate_rows(rows)
    for row in rows:
        for side in ('lq','gt'):
            require(c.path(ROOT,row[side+'_path']).is_file(),'missing official '+side+' asset: '+row['record_id'])
    official=c.read(ROOT/'configs/foundir_official_sow600_released_full_pipeline_v1.json')['metrics']
    require(c.sha(c.path(ROOT,official['official_metric_script']))==official['official_metric_script_sha256'], 'official metric script identity changed')
    metrics,metric_identity=metric_models('cpu'); del metrics
    # Bind source/weight hashes, never infer a model from a file name.
    sources={**b['sources'],**{p:c.sha(ROOT/p) for p in x.SOURCES},
             'configs/foundir_official_sow600_released_full_pipeline_v1.json':c.sha(ROOT/'configs/foundir_official_sow600_released_full_pipeline_v1.json')}
    sources[official['official_metric_script']]=official['official_metric_script_sha256']
    p=b['original']['parent']; assets={}
    for task in x.ORDER:
        spec=p['experts'][task]['checkpoint']; assets[spec['path']]=spec['sha256']
    for spec in p['four_prompts']['tensors'].values():
        assets[str(Path(p['config']['four_parent_run'])/spec['path'])]=spec['sha256']
    assets[p['rain_prompt']['path']]=p['rain_prompt']['sha256']
    q=b['original']['qwen']; assets[q['adapter']]=q['adapter_sha256']
    assets.update({path:h for path,h in b['original']['assets'].items()
                   if path!=b['original']['config']['manifest']})
    # Large base-model hashes are inherited from the original verified producer.
    for key in ('flow_weights','autoencoder_weights'):
        path=p['model'][key]; require(c.path(ROOT,path).is_file(),'missing base model: '+path)
        hashkey=key+'_sha256'
        if hashkey in p['model']: assets[path]=p['model'][hashkey]
        else: assets[path]=c.sha(c.path(ROOT,path))
    for path,h in assets.items(): require(c.sha(c.path(ROOT,path))==h,'frozen asset changed: '+path)
    require(shutil.disk_usage(ROOT).free>=100*1024**3,'need 100GiB free for all native tiles and full-size outputs')
    receipt={'recipe':x.RECIPE,'rows':rows,'manifest_sha256':x.MANIFEST_SHA,'assets':assets,'sources':sources,
             'parent_receipt_sha256':c.sha(ROOT/prior.x.OUTPUT/'receipt.json'),
             'selected_training_report_sha256':x.TRAIN_REPORT_SHA,'selected_checkpoint_sha256':x.SELECTED_SHA,
             'metric_identity':metric_identity,'original':b['original'],'image_checkpoint':b['image_checkpoint']}
    c.write_new(ROOT/x.OUTPUT/'receipt.json',receipt)
    counts=[sum(len(x.tiles(r)) for r in rows[rank::x.WORLD]) for rank in range(x.WORLD)]
    report={'status':'official_vlm_v4_metadata_and_model_preflight_pass','recipe':x.RECIPE,
            'receipt_sha256':c.sha(ROOT/x.OUTPUT/'receipt.json'),'records':600,'tile_counts_by_rank':counts,
            'minimum_restoration_tiles':sum(counts),'maximum_restoration_tiles':3*sum(counts),
            'official_pixels_decoded':False,'GPU_started':False,'H100_full_image_metrics_qualified':False}
    c.write_new(ROOT/x.PREFLIGHT,report)
    print('OFFICIAL_VLM_V4_PREPARED '+str(report),flush=True)


def bound():
    b=c.read(ROOT/x.OUTPUT/'receipt.json')
    require(b['recipe']==x.RECIPE and b['manifest_sha256']==x.MANIFEST_SHA,'official contract changed')
    require(c.sha(ROOT/x.MANIFEST)==x.MANIFEST_SHA,'manifest changed')
    with (ROOT/x.MANIFEST).open(newline='') as f: require(list(csv.DictReader(f))==b['rows'],'manifest population changed')
    x.validate_rows(b['rows'])
    require(c.sha(ROOT/prior.x.OUTPUT/'receipt.json')==b['parent_receipt_sha256'],'parent receipt changed')
    require(c.sha(ROOT/prior.x.TRAIN_REPORT)==x.TRAIN_REPORT_SHA,'selected report changed')
    require(b['selected_checkpoint_sha256']==x.SELECTED_SHA and b['selected_training_report_sha256']==x.TRAIN_REPORT_SHA,'selected identity changed')
    for path,h in b['sources'].items(): require(c.sha(ROOT/path)==h,'bound code changed: '+path)
    return b


def hardware(node, idle):
    import torch
    from smoke import train_foundir_frozen_expert_router_mixed_driver_v1 as impl
    from smoke.frozen_router_runtime_policy_v1 import execution_signature
    b=bound(); actual=impl.native.hopper.hardware(idle)
    actual['metrics']=impl.parent.four_impl.metric_runtime({'metric':b['original']['parent']['metric']})
    actual['memory_policy']={'inherited':b['original']['parent']['memory_policy'],'router_explicit_gate_checkpoint':True}
    baseline=ROOT/'checkpoints/foundir_frozen_expert_router_v1'/prior.x.old.parent.s.analysis.RUN
    require(all(execution_signature(actual)==execution_signature(c.read(baseline/f'runtime_node{k}.json')) for k in range(4)), 'unqualified numerical runtime')
    from smoke.expanded_vlm_h100_runtime_v4 import POLICY
    actual.update(world=x.WORLD,official_pyiqa=b['metric_identity'],vae_convolution=POLICY)
    file=ROOT/x.OUTPUT/f'runtime_node{node}.json'
    if idle: c.write_new(file,actual)
    else: require(c.read(file)==actual,'runtime changed within run; report instead of silent migration')
    return b,impl


def input_image(row, side):
    import numpy as np
    from PIL import Image
    require(side in ('lq','gt'),'unknown image side')
    file=c.path(ROOT,row[side+'_path']); require(c.sha(file)==row[side+'_sha256'],'official pixel file changed')
    with Image.open(file) as im:
        require(im.size==(int(row['width']),int(row['height'])),'wrong official dimensions')
        return np.array(im.convert('RGB'),copy=True)


def tensor(array, device):
    import torch
    return torch.from_numpy(array.copy()).permute(2,0,1).unsqueeze(0).float().div(127.5).sub(1).to(device)


def item(row,tile):
    return {'key':tile['key'],'record_id':row['record_id'],'lq_sha256':row['lq_sha256'],'tile':tile}


def cache(impl,folder,i,node):
    return impl.cached(ROOT/x.OUTPUT,folder,i,extra={'runtime_sha256':c.sha(ROOT/x.OUTPUT/f'runtime_node{node}.json')})


def save(impl,folder,i,values,node,meta=None):
    return impl.commit(ROOT/x.OUTPUT,folder,i,values,
                       extra={'runtime_sha256':c.sha(ROOT/x.OUTPUT/f'runtime_node{node}.json')},values=meta)


def extract(rows,b,impl,device,node):
    import torch
    from PIL import Image
    from smoke.train_qwen_multidegradation_lora_ddp import load_model
    from lowlight_restoration.qwen_lora import load_qwen_adapter
    from lowlight_restoration.qwen_five_class_gate import image_embedding_sequence
    from lowlight_restoration.frozen_expert_router import pooled_image_features
    from flux2.util import load_flow_model,load_ae
    from smoke.expanded_vlm_h100_runtime_v4 import bounded_vae
    if any(cache(impl,'vlm',item(r,t),node) is None for r in rows for t in x.tiles(r)):
        model,processor,_=load_model(b['original']['qcfg'],variant='merger_trainable',device=device)
        load_qwen_adapter(model,c.path(ROOT,b['original']['qwen']['adapter'])); model.eval().requires_grad_(False)
        for row in rows:
            missing=[t for t in x.tiles(row) if cache(impl,'vlm',item(row,t),node) is None]
            if not missing: continue
            array=input_image(row,'lq')
            for tile in missing:
                crop,_,_=x.tile_array(array,tile)
                inputs=processor.image_processor(images=[Image.fromarray(crop)],return_tensors='pt')
                with torch.no_grad():
                    seq=image_embedding_sequence(model.get_image_features(inputs.pixel_values.to(device),inputs.image_grid_thw.to(device)))
                    require(len(seq)==1 and seq[0].ndim==2 and seq[0].shape[-1]==3584,'unexpected Qwen tokens')
                    value=seq[0].float().mean(0,keepdim=True)
                require(torch.isfinite(value).all(),'nonfinite VLM feature')
                save(impl,'vlm',item(row,tile),{'vlm':value},node,{'GT_input':False})
            print('OFFICIAL_VLM_V4_FEATURE_VLM '+row['record_id'],flush=True)
        del model,processor,inputs,seq,value; gc.collect(); torch.cuda.empty_cache()
    if any(cache(impl,'features',item(r,t),node) is None for r in rows for t in x.tiles(r)):
        model=load_flow_model('flux.2-klein-base-4b',device=device).eval().requires_grad_(False)
        ae=bounded_vae(impl.native.hopper.configure_vae(load_ae('flux.2-klein-base-4b',device=device).eval().requires_grad_(False)))
        for row in rows:
            missing=[t for t in x.tiles(row) if cache(impl,'features',item(row,t),node) is None]
            if not missing: continue
            array=input_image(row,'lq')
            for tile in missing:
                crop,_,_=x.tile_array(array,tile)
                with torch.no_grad(): value=pooled_image_features(model,ae.encode(tensor(crop,device)).to(torch.bfloat16))
                require(value.shape==(1,6144) and torch.isfinite(value).all(),'invalid FLUX image feature')
                vlm=cache(impl,'vlm',item(row,tile),node)[1]['vlm']
                save(impl,'features',item(row,tile),{'image':value,'vlm':vlm},node,{'GT_input':False})
            print('OFFICIAL_VLM_V4_FEATURE_IMAGE '+row['record_id'],flush=True)
        del model,ae,value; gc.collect(); torch.cuda.empty_cache()


def make_decision(row,b,impl,router,node):
    import torch
    image=[];vlm=[]; hashes={}
    with torch.no_grad():
        for tile in x.tiles(row):
            v=cache(impl,'features',item(row,tile),node); require(v is not None,'missing feature')
            z,_,im,_=router(v[1]['image'],v[1]['vlm'])
            image.append(im[0].tolist());vlm.append(z[0].tolist())
            hashes[tile['key']]=c.sha(ROOT/x.OUTPUT/'features'/(tile['key']+'.json'))
    decision={'receipt_sha256':c.sha(ROOT/x.OUTPUT/'receipt.json'),'record_id':row['record_id'],
              'lq_sha256':row['lq_sha256'],'feature_receipts':hashes,'decisions':x.decisions(image,vlm),
              'GT_input':False,'class_label_input':False}
    c.write_new(ROOT/x.OUTPUT/'decisions'/(row['record_id']+'.json'),decision)
    return decision


def png_once(path,array):
    from PIL import Image
    import io
    import tempfile
    data=io.BytesIO();Image.fromarray(array).save(data,format='PNG'); content=data.getvalue()
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        require(path.read_bytes()==content,'existing official PNG differs');return
    fd,name=tempfile.mkstemp(prefix='.pending-',dir=path.parent)
    with os.fdopen(fd,'wb') as f:f.write(content);f.flush();os.fsync(f.fileno())
    try:os.link(name,path)
    except FileExistsError:require(path.read_bytes()==content,'racing PNG differs')
    else:os.unlink(name)
    fd=os.open(path.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)


def read_result(row,node):
    out=ROOT/x.OUTPUT; file=out/'results'/(row['record_id']+'.json')
    if not file.exists():return None
    result=c.read(file); decfile=out/'decisions'/(row['record_id']+'.json'); dec=c.read(decfile)
    require(result['receipt_sha256']==c.sha(out/'receipt.json') and result['row']==row and
            result['runtime_sha256']==c.sha(out/f'runtime_node{node}.json') and
            result['decision_sha256']==c.sha(decfile) and result['decisions']==dec['decisions'],'result identity changed')
    require(dec['GT_input'] is False and dec['class_label_input'] is False,'label leakage')
    require(set(result['outputs'])==set(x.anchors(row,result['decisions']).values())==set(result['metrics']),'missing/extra rendered anchor')
    for anchor,spec in result['outputs'].items():
        require(spec['path']==f"restored/{row['record_id']}__{anchor}.png" and c.sha(c.path(out,spec['path']))==spec['sha256'],'PNG changed')
    return result


def setup(node):
    import torch
    local=int(os.environ['LOCAL_RANK'])
    require(int(os.environ['WORLD_SIZE'])==int(os.environ['LOCAL_WORLD_SIZE'])==2 and local in (0,1),'two independent local workers required')
    b,impl=hardware(node,False);rank=node*2+local;torch.cuda.set_device(local);device=torch.device('cuda',local)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    for path,h in b['assets'].items(): require(c.sha(c.path(ROOT,path))==h,'asset changed before GPU load: '+path)
    model_config=b['original']['parent']['model']
    os.environ.update(KLEIN_4B_BASE_MODEL_PATH=str(c.path(ROOT,model_config['flow_weights'])),AE_MODEL_PATH=str(c.path(ROOT,model_config['autoencoder_weights'])))
    return b,impl,rank,device


def route(node):
    import torch
    b,impl,rank,device=setup(node); assigned=b['rows'][rank::x.WORLD];out=ROOT/x.OUTPUT
    extract(assigned,b,impl,device,node)
    router=load_router(b)
    for row in assigned:make_decision(row,b,impl,router,node)
    marker={'receipt_sha256':c.sha(out/'receipt.json'),'runtime_sha256':c.sha(out/f'runtime_node{node}.json'),
            'rank':rank,'records':[r['record_id'] for r in assigned]}
    file=out/f'route_worker{rank}.json'
    if file.exists():require({k:v for k,v in c.read(file).items() if k!='peak_allocated_gib'}==marker,'route worker identity changed')
    else:c.write_new(file,{**marker,'peak_allocated_gib':torch.cuda.max_memory_allocated()/1024**3})
    print(f'OFFICIAL_VLM_V4_ROUTE_COMPLETE rank={rank} count={len(assigned)}',flush=True)


def rendering_plan(b):
    out=ROOT/x.OUTPUT; decisions={}; workload=[]
    for rank in range(x.WORLD):
        rows=b['rows'][rank::x.WORLD]; marker=c.read(out/f'route_worker{rank}.json')
        require({k:v for k,v in marker.items() if k!='peak_allocated_gib'}==
                {'receipt_sha256':c.sha(out/'receipt.json'),'runtime_sha256':c.sha(out/f'runtime_node{rank//2}.json'),
                 'rank':rank,'records':[r['record_id'] for r in rows]},'incomplete route worker')
        count=0
        for row in rows:
            file=out/'decisions'/(row['record_id']+'.json'); d=c.read(file)
            require(d['receipt_sha256']==c.sha(out/'receipt.json') and d['record_id']==row['record_id'] and
                    d['lq_sha256']==row['lq_sha256'] and d['GT_input'] is False and d['class_label_input'] is False,'invalid route identity')
            expected={}
            for t in x.tiles(row):
                featurefile=out/'features'/(t['key']+'.json'); feature=c.read(featurefile)
                require(feature['identity']=={'receipt_sha256':c.sha(out/'receipt.json'),'item':item(row,t),'arm':'features',
                        'runtime_sha256':c.sha(out/f'runtime_node{rank//2}.json')},'feature input/runtime changed')
                require(feature['GT_input'] is False and feature['files']=={feature['tensor']:c.sha(c.path(out,feature['tensor']))},'feature tensor changed')
                expected[t['key']]=c.sha(featurefile)
            require(d['feature_receipts']==expected,'route feature identity changed')
            choices=x.anchors(row,d['decisions']);count+=len(set(choices.values()))*len(x.tiles(row))
            decisions[row['record_id']]={'decision_sha256':c.sha(file),'arms':choices}
        workload.append(count)
    # Prior observed 117 crops / 0.660556 GPU-hours: approximate, not a speed guarantee.
    estimate=sum(workload)*(.6605555555555556/117)
    return {'receipt_sha256':c.sha(out/'receipt.json'),'decisions':decisions,'tiles_by_rank':workload,
            'unique_native_tile_generations':sum(workload),'estimated_render_GPU_hours_from_v4':estimate,
            'world':x.WORLD, 'nodes':x.NODES, 'estimated_wall_hours':estimate/x.WORLD,'estimate_includes_old_overhead_not_new_measurement':True,
            'GT_pixels_read':False,'render_authorized':False}


def plan():
    value=rendering_plan(bound());c.write_new(ROOT/x.OUTPUT/'render_plan.json',value)
    c.write_new(ROOT/f'reports/foundir_official_vlm_router_v4_route_plan{x.SUFFIX}_2026-11-05.json',value)
    print('OFFICIAL_VLM_V4_PLAN_FIXED '+str({k:v for k,v in value.items() if k!='decisions'}),flush=True)


def authorize_render(gpu_hours,note):
    # This records the human's explicit budget approval, not an automatic grant.
    import math
    require(math.isfinite(gpu_hours) and 0<gpu_hours<=400 and len(note.strip())>=12,'explicit user budget and approval note required')
    out=ROOT/x.OUTPUT;expected=rendering_plan(bound())
    require(c.read(out/'render_plan.json')==expected,'fixed route plan required')
    value={'render_plan_sha256':c.sha(out/'render_plan.json'),'approved_render_GPU_hours':gpu_hours,'world':x.WORLD,'nodes':x.NODES,
           'cap_seconds_per_node':int(gpu_hours*3600/x.WORLD),'approval_note':note,'authority':'explicit_user_approval_recorded_by_CPU_agent',
           'no_automatic_extension':True,'route_GPU_hours_counted_separately':True}
    c.write_new(out/'render_budget.json',value)
    print('OFFICIAL_VLM_V4_RENDER_BUDGET_RECORDED '+str(value),flush=True)


def render(node):
    import torch
    import numpy as np
    from PIL import Image
    from lowlight_restoration.five_expert_flux import restore_five_expert
    from smoke.expanded_vlm_h100_runtime_v4 import bounded_vae
    b,impl,rank,device=setup(node);assigned=b['rows'][rank::x.WORLD];out=ROOT/x.OUTPUT
    require(c.read(out/'render_plan.json')==rendering_plan(b),'route plan changed before render')
    budget=c.read(out/'render_budget.json');require(budget['render_plan_sha256']==c.sha(out/'render_plan.json'),'approved plan changed')
    pending=[r for r in assigned if read_result(r,node) is None]
    if pending:
        model,gates,rules,ae,prompts=impl.load_bank(b['original'],device);ae=bounded_vae(ae)
        metrics,metric_id=metric_models(device); require(metric_id==b['metric_identity'],'official metric weights changed')
        for row in pending:
            start=time.monotonic();decfile=out/'decisions'/(row['record_id']+'.json');dec=c.read(decfile)
            choices=x.anchors(row,dec['decisions']);array=input_image(row,'lq');outputs={}
            for anchor in sorted(set(choices.values())):
                images={};weights=torch.tensor([[float(t==anchor) for t in x.ORDER]],device=device)
                tile_list=x.tiles(row)
                for tile_index,tile in enumerate(tile_list):
                    tileitem={**item(row,tile),'key':tile['key']+'__'+anchor,'anchor':anchor}
                    saved=cache(impl,'restored_tiles',tileitem,node)
                    if saved is None:
                        crop,_,_=x.tile_array(array,tile)
                        with torch.no_grad():raw,u8,route=restore_five_expert(model,gates,rules,weights,ae,tensor(crop,device),prompts,seed=tile['seed'])
                        require(route['anchor']==anchor and route['model_calls']==(26 if anchor=='rain' else 50),'wrong expert/readout')
                        save(impl,'restored_tiles',tileitem,{'uint8':u8.cpu()},node,{'route':route,'GT_input':False})
                        del raw,u8
                        saved=cache(impl,'restored_tiles',tileitem,node)
                    require(saved[0]['route']['anchor']==anchor and saved[0]['GT_input'] is False,'wrong cached route')
                    images[tile['key']]=saved[1]['uint8'][0].permute(1,2,0).numpy()
                    if (tile_index+1)%5==0 or tile_index+1==len(tile_list):
                        print(f'OFFICIAL_VLM_V4_TILE rank={rank} id={row["record_id"]} anchor={anchor} tile={tile_index+1}/{len(tile_list)} record_seconds={time.monotonic()-start:.1f}',flush=True)
                image=x.stitch(row,images);path=out/'restored'/(row['record_id']+'__'+anchor+'.png')
                png_once(path,image);outputs[anchor]={'path':path.relative_to(out).as_posix(),'sha256':c.sha(path)}
            # All paired outputs committed BEFORE GT pixels are loaded.
            gt=input_image(row,'gt');target=torch.from_numpy(gt.copy()).permute(2,0,1).unsqueeze(0).float().div(255).to(device)
            scores={}
            with torch.no_grad():
                for anchor,spec in outputs.items():
                    with Image.open(c.path(out,spec['path'])) as im:
                        require(im.size==(int(row['width']),int(row['height'])),'official output resized')
                        pred=torch.from_numpy(np.array(im.convert('RGB'),copy=True)).permute(2,0,1).unsqueeze(0).float().div(255).to(device)
                    scores[anchor]={k:float(m(pred,target).item()) for k,m in metrics.items()}
                    del pred
            result={'receipt_sha256':c.sha(out/'receipt.json'),'runtime_sha256':c.sha(out/f'runtime_node{node}.json'),
                    'row':row,'decision_sha256':c.sha(decfile),'decisions':dec['decisions'],'outputs':outputs,'metrics':scores,
                    'rank':rank,'seconds':time.monotonic()-start,'GT_used_for_deployed_selection':False}
            c.write_new(out/'results'/(row['record_id']+'.json'),result);read_result(row,node)
            print(f'OFFICIAL_VLM_V4_RECORD rank={rank} id={row["record_id"]} scores={scores}',flush=True)
            del target,images,array,gt
    marker={'receipt_sha256':c.sha(out/'receipt.json'),'runtime_sha256':c.sha(out/f'runtime_node{node}.json'),
            'rank':rank,'records':[r['record_id'] for r in assigned]}
    file=out/f'worker{rank}.json'
    if file.exists():require({k:v for k,v in c.read(file).items() if k!='peak_allocated_gib'}==marker,'worker identity changed')
    else:c.write_new(file,{**marker,'peak_allocated_gib':torch.cuda.max_memory_allocated()/1024**3})
    print(f'OFFICIAL_VLM_V4_WORKER_COMPLETE rank={rank} count={len(assigned)}',flush=True)


def costs(out):
    result={}
    for key,gpus in [('prepare',0),('plan',0)]+[(f'{phase}_node{node}',2) for phase in ('route','render') for node in range(x.NODES)]:
        starts=list((out/'attempts').glob(key+'_*.started.json'));ends=list((out/'attempts').glob(key+'_*.ended.json'))
        require(starts and {p.name.removesuffix('.started.json') for p in starts}=={p.name.removesuffix('.ended.json') for p in ends},'incomplete attempt accounting: '+key)
        rows=[c.read(p) for p in ends]
        require(any(r['status']==0 for r in rows) and all(type(r['seconds']) is int and r['seconds']>=0 and r['gpus']==gpus and r['used_gpu_seconds']==gpus*r['seconds'] for r in rows),'invalid compute ledger')
        result[key]={'seconds':sum(r['seconds'] for r in rows),'failed_attempts':sum(r['status']!=0 for r in rows),'used_gpu_seconds':sum(r['used_gpu_seconds'] for r in rows)}
    return {'stages':result,'used_gpu_hours':sum(v['used_gpu_seconds'] for v in result.values())/3600,'finish_CPU_reported_separately':True}


def finish():
    b=bound();out=ROOT/x.OUTPUT;records={};artifacts={}
    require(c.read(out/'render_plan.json')==rendering_plan(b),'render plan changed')
    from smoke.frozen_router_runtime_policy_v1 import execution_signature
    require(all(execution_signature(c.read(out/'runtime_node0.json'))==execution_signature(c.read(out/f'runtime_node{node}.json')) for node in range(x.NODES)), 'cross-node numerical runtime differs')
    for rank in range(x.WORLD):
        assigned=b['rows'][rank::x.WORLD];marker=c.read(out/f'worker{rank}.json')
        expected={'receipt_sha256':c.sha(out/'receipt.json'),'runtime_sha256':c.sha(out/f'runtime_node{rank//2}.json'),
                  'rank':rank,'records':[r['record_id'] for r in assigned]}
        require({k:v for k,v in marker.items() if k!='peak_allocated_gib'}==expected,'worker incomplete')
        for row in assigned:
            rec=read_result(row,rank//2);require(rec is not None and rec['rank']==rank and rec['GT_used_for_deployed_selection'] is False,'wrong worker/GT use')
            records[row['record_id']]=rec;artifacts[row['record_id']]=c.sha(out/'results'/(row['record_id']+'.json'))
    report={'status':'official_vlm_v4_evaluation_complete','run_id':x.RUN,'recipe':x.RECIPE,
            'receipt_sha256':c.sha(out/'receipt.json'),'selected_checkpoint_sha256':x.SELECTED_SHA,
            **x.aggregate(b['rows'],records),'artifacts':artifacts,'compute_costs':costs(out),
            'render_budget':c.read(out/'render_budget.json'),
            'automatic_followup':False,'training':False,'GT_used_for_deployed_selection':False,
            'limitations':['Official set historically used; not a blind test.',
                           'Five singleton categories only, no double/triple claim.',
                           'V4 and frozen image head have different training coverage; not data-matched causal ablation.',
                           'New full-image tiled inference; no claim of bitwise historical output equivalence.']}
    c.write_new(ROOT/x.REPORT,report)
    print('OFFICIAL_VLM_V4_RESULT_VERIFIED '+str(report['checks']),flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('phase',choices=('prepare','hardware','route','plan','authorize_render','render','finish'))
    ap.add_argument('--node',type=int,choices=range(x.NODES),default=0)
    ap.add_argument('--gpu-hours',type=float);ap.add_argument('--approval-note');args=ap.parse_args()
    require(str(ROOT)==ACTIVE,'destination root only; no old-workspace execution')
    if args.phase=='hardware':hardware(args.node,True)
    elif args.phase in ('route','render'):globals()[args.phase](args.node)
    elif args.phase=='authorize_render':
        require(args.gpu_hours is not None and args.approval_note is not None,'budget and user approval required')
        authorize_render(args.gpu_hours,args.approval_note)
    else:globals()[args.phase]()


if __name__=='__main__':main()
