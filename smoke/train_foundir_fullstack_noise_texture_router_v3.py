#!/usr/bin/env python3
"""One-node/two-GPU fixed400-step LQ texture router. CPU preparation only until launch."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
import hashlib
import io
import os
from pathlib import Path
import sys
import tempfile
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from PIL import Image
from smoke import train_foundir_fullstack_noise_physical_router_v1_cpu as old
from smoke.existing_flux_replay_contract_v1 import read,sha,write_new as write
from lowlight_restoration.noise_texture_router_v3 import TextureRouter,views,route,sample_indices
from smoke.train_foundir_fullstack_noise_standalone_router_v2_cpu import admission
OUT=ROOT/'checkpoints/foundir_fullstack_noise_texture_router_v3'
REPORT=ROOT/'reports/foundir_fullstack_noise_texture_router_v3_result_2026-11-06.json'
PREFLIGHT=ROOT/'reports/foundir_fullstack_noise_texture_router_v3_preflight_2026-11-06.json'
SOURCES=['lowlight_restoration/noise_texture_router_v3.py','smoke/train_foundir_fullstack_noise_texture_router_v3.py',
         'smoke/run_foundir_fullstack_noise_texture_router_v3.sh',
         'smoke/train_foundir_fullstack_noise_standalone_router_v2_cpu.py']
RECIPE={'steps':400,'global_batch':32,'world':2,'lr':.001,'weight_decay':.01,'seed':2026110613,
        'input':'fixed_native128_four_corners_center_plus_global_thumbnail','views':6,
        'margin':'mean_six_view_logits','binary_gate':'old_Noise_or_Lowlight_only','threshold':0,
        'optimizer':'fresh_AdamW','checkpoint_selection':'fixed400','new_RGB_network':False,
        'training_GT_pixels':False,'official_inputs':False,'automatic_deployment':False,'wall_cap_seconds':1800}


def need(ok,msg):
    if not ok:raise ValueError(msg)


def prepare():
    start=time.monotonic();parent=old.bound();records=parent['records']
    def extract(r):
        path=OUT/'views'/(r['key']+'.npy');marker=path.with_suffix('.json')
        if marker.exists():
            v=read(marker);need(v['lq_sha256']==r['lq_sha256'] and sha(path)==v['sha256'],'view cache changed');return r['key'],v
        need(not path.exists(),'orphan view file retained; inspect before retry')
        data=(ROOT/r['lq_path']).read_bytes();need(hashlib.sha256(data).hexdigest()==r['lq_sha256'],'LQ changed')
        with Image.open(io.BytesIO(data)) as im:a=views(im)
        need(a.shape==(6,128,128,3) and a.dtype==np.uint8,'bad views')
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as f:np.save(f,a,allow_pickle=False)
        v={'path':str(path.relative_to(OUT)),'sha256':sha(path),'lq_sha256':r['lq_sha256'],'GT_input':False}
        write(marker,v);return r['key'],v
    cache={}
    with ThreadPoolExecutor(max_workers=2) as pool:
        for i,(k,v) in enumerate(pool.map(extract,records)):
            cache[k]=v
            if (i+1)%200==0:print(f'TEXTURE_PREPARE {i+1}/{len(records)}',flush=True)
    b={'recipe':RECIPE,'records':records,'views':cache,'sources':{s:sha(ROOT/s) for s in SOURCES},
       'physical_parent_receipt_sha256':sha(old.OUT/'receipt.json'),
       'legacy_router_exposure_not_erased':True,'new_train_and_holdout_source_disjoint':True}
    write(OUT/'receipt.json',b)
    result={'status':'destination_CPU_preflight_pass','recipe':RECIPE,'receipt_sha256':sha(OUT/'receipt.json'),
            'counts':{role:sum(r['role']==role for r in records) for role in ('fit','calibration','confirmation')},
            'parameters':sum(p.numel() for p in TextureRouter().parameters()),'seconds':time.monotonic()-start,
            'exit_code':0,'GPU_hours':0,'GPU_started':False,'CUDA_qualification':'pending'}
    write(PREFLIGHT,result);print('TEXTURE_PREFLIGHT_PASS '+str(result),flush=True)


def bound():
    b=read(OUT/'receipt.json');need(b['recipe']==RECIPE and b['physical_parent_receipt_sha256']==sha(old.OUT/'receipt.json'),'parent/recipe mismatch')
    parent=old.bound();need(b['records']==parent['records'],'source population changed')
    for name,h in b['sources'].items():need(sha(ROOT/name)==h,'source changed: '+name)
    return b


def load_views(b,r):
    spec=b['views'][r['key']];path=OUT/spec['path'];need(sha(path)==spec['sha256'] and spec['lq_sha256']==r['lq_sha256'],'cached view changed')
    a=np.load(path,allow_pickle=False);need(a.dtype==np.uint8 and a.shape==(6,128,128,3),'view schema')
    return torch.from_numpy(a.copy()).permute(0,3,1,2)


def evaluate(model,b,role,device):
    records=[r for r in b['records'] if r['role']==role];rows=[]
    for offset in range(0,len(records),8):
        batch=records[offset:offset+8];x=torch.stack([load_views(b,r) for r in batch]).to(device).float()/255
        with torch.no_grad():margin=model(x);new=route(torch.tensor([r['base_logits'] for r in batch],device=device),margin)
        for r,n,m in zip(batch,new.tolist(),margin.tolist()):rows.append({'key':r['key'],'task':r['task'],
          'baseline_route':int(torch.tensor(r['base_logits']).argmax()),'candidate_route':n,'margin':m})
    summary={}
    for task,label in [('noise',1),('lowlight',3)]:
        selected=[r for r in rows if r['task']==task]
        summary[task]={'count':len(selected),'baseline_correct':sum(r['baseline_route']==label for r in selected),
                       'candidate_correct':sum(r['candidate_route']==label for r in selected)}
    return {'summary':summary,'rows':rows}


def save_checkpoint(model,opt,step,history):
    target=OUT/'steps'/f'step_{step:06d}.pt';target.parent.mkdir(parents=True,exist_ok=True)
    need(not target.exists(),'preserve existing checkpoint')
    fd,name=tempfile.mkstemp(prefix='.pending-',suffix='.pt',dir=target.parent);os.close(fd)
    torch.save({'model':{k:v.detach().cpu() for k,v in model.state_dict().items()},'optimizer':opt.state_dict(),
                'step':step,'history':history,'receipt_sha256':sha(OUT/'receipt.json'),
                'sampling':'stateless seed + step*101 + rank; no dropout or augmentation RNG'},name)
    with open(name,'rb') as f:os.fsync(f.fileno())
    os.link(name,target);os.unlink(name)
    write(target.with_suffix('.json'),{'step':step,'sha256':sha(target),'file':str(target.relative_to(OUT)),
                                      'receipt_sha256':sha(OUT/'receipt.json')})


def run(node):
    import torch.distributed as dist
    start=time.monotonic();b=bound();rank=int(os.environ['LOCAL_RANK'])
    need(rank in (0,1) and int(os.environ['WORLD_SIZE'])==2,'single node two GPU only')
    torch.cuda.set_device(rank);device=torch.device('cuda',rank)
    need(torch.cuda.get_device_properties(rank).total_memory>=75*1024**3 and any(x in torch.cuda.get_device_name(rank) for x in ('H100','H200')),'Hopper75GiB required')
    dist.init_process_group('nccl',device_id=device,timeout=timedelta(minutes=5))
    try:
        torch.manual_seed(RECIPE['seed']);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
        model=TextureRouter().to(device);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
        rows=[r for r in b['records'] if r['role']=='fit'];labels=torch.tensor([float(r['task']=='noise') for r in rows])
        # Small immutable LQ views only, held in host uint8 memory. No training GT is loaded.
        data=torch.stack([load_views(b,r) for r in rows]);history=[];initial=0
        markers=sorted((OUT/'steps').glob('step_*.json'))
        if markers:
            need([read(p)['step'] for p in markers]==list(range(len(markers))),'checkpoint gap')
            spec=read(markers[-1]);need(sha(OUT/spec['file'])==spec['sha256'],'checkpoint corruption; no rollback')
            state=torch.load(OUT/spec['file'],map_location=device,weights_only=True)
            need(state['receipt_sha256']==sha(OUT/'receipt.json'),'checkpoint receipt mismatch')
            initial=state['step'];model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);history=state['history']
        elif rank==0:save_checkpoint(model,opt,0,[])
        dist.barrier()
        for step in range(initial+1,401):
            need(time.monotonic()-start<1800,'router cap exceeded')
            ids=sample_indices(labels,step,rank);x=data[ids].to(device).float()/255;y=labels[ids].to(device)
            opt.zero_grad();loss=torch.nn.functional.binary_cross_entropy_with_logits(model(x),y);loss.backward()
            for p in model.parameters():dist.all_reduce(p.grad);p.grad.div_(2)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
            loss=loss.detach();dist.all_reduce(loss);history.append(float(loss)/2)
            if rank==0:save_checkpoint(model,opt,step,history)
            dist.barrier()
            if step%20==0:print(f'TEXTURE_TRAIN rank={rank} step={step}/400 loss={history[-1]:.6f}',flush=True)
        model.eval().requires_grad_(False)
        # Rank0 evaluates calibration; rank1 evaluates independent confirmation. Fit is not a gate.
        role=('calibration','confirmation')[rank];classification=evaluate(model,b,role,device)
        write(OUT/(role+'.json'),{'receipt_sha256':sha(OUT/'receipt.json'),
              'checkpoint_sha256':sha(OUT/'steps/step_000400.pt'),'classification':classification})
        write(OUT/f'worker{rank}.json',{'receipt_sha256':sha(OUT/'receipt.json'),'rank':rank,'node':node,
              'seconds':time.monotonic()-start,'peak_allocated_GiB':torch.cuda.max_memory_allocated()/1024**3})
        print(f'TEXTURE_TRAIN_END rank={rank} status=0',flush=True)
    finally:dist.destroy_process_group()


def finish():
    b=bound();digest=sha(OUT/'receipt.json');checkpoint=OUT/'steps/step_000400.pt';spec=read(checkpoint.with_suffix('.json'))
    need(sha(checkpoint)==spec['sha256'] and spec['receipt_sha256']==digest,'final model corrupt')
    for step in range(401):
        p=OUT/'steps'/f'step_{step:06d}.json';r=read(p)
        need(r['step']==step and r['receipt_sha256']==digest and sha(OUT/r['file'])==r['sha256'],'invalid saved step')
    classification={}
    for rank,role in enumerate(('calibration','confirmation')):
        need(read(OUT/f'worker{rank}.json')['receipt_sha256']==digest,'worker incomplete')
        r=read(OUT/(role+'.json'));need(r['receipt_sha256']==digest and r['checkpoint_sha256']==sha(checkpoint),'heldout model changed')
        expected={i['key'] for i in b['records'] if i['role']==role};rows=r['classification']['rows']
        need(len(rows)==len(expected) and {i['key'] for i in rows}==expected,'heldout missing/duplicate rows')
        classification[role]=r['classification']
    result={'status':'fixed400_GPU_router_complete','receipt_sha256':digest,'checkpoint_sha256':sha(checkpoint),
            'classification':classification,'admit_official_evaluation':admission(classification),
            'automatic_deployment':False,'official_evaluated':False,'legacy_router_exposure_not_erased':True,'exit_code':0}
    write(REPORT,result);print('TEXTURE_RESULT_VERIFIED '+str(result['admit_official_evaluation']),flush=True)


if __name__=='__main__':
    torch.set_num_threads(2);p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','run','finish','verify']);p.add_argument('--node',type=int,choices=range(3),default=0);a=p.parse_args()
    if a.phase=='run':run(a.node)
    elif a.phase=='verify':bound();print('TEXTURE_BINDING_VERIFIED')
    else:globals()[a.phase]()
