#!/usr/bin/env python3
"""Rain800/1000 prefix/readout cross on eight existing legal training crops only."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'third_party/flux2/src'))
from smoke import train_foundir_fullstack_rain_continue_v1 as train
from lowlight_restoration import rain_prefix_readout_cross_v1 as cross
c=train.c;parent=train.parent;logic=train.logic
OUT=ROOT/'checkpoints/foundir_fullstack_rain_prefix_cross_v1'
REPORT=ROOT/'reports/foundir_fullstack_rain_prefix_cross_v1_result_2026-11-06.json'
PREFLIGHT=ROOT/'reports/foundir_fullstack_rain_prefix_cross_v1_preflight_2026-11-06.json'
SOURCES=['lowlight_restoration/rain_prefix_readout_cross_v1.py','smoke/diagnose_foundir_fullstack_rain_prefix_cross_v1.py',
         'smoke/run_foundir_fullstack_rain_prefix_cross_v1.sh']
RECIPE={'indices':list(cross.INDICES),'selection':'first_eight_of_existing32_training_probes_not_selected_by_scores',
        'arms':list(cross.ARMS),'prefix_model_calls':25,'readout_model_calls':1,'actual_calls_per_crop':54,
        'original_schedule_steps':50,'reference_time':10,'guidance':1.,'training':False,'GT_input':False,
        'nodes':[0],'local_gpus':2,'world':2,'maximum_wall_seconds':900,
        'automatic_extension':False,'automatic_followup':False,'readout':'original_call25_endpoint',
        'quantization':'original_Rain_raw_float_to_uint8_once','metric':'original_torchmetrics_training_probe'}


def need(ok,msg):
    if not ok:raise ValueError(msg)


def state(step):
    import torch
    record=c.read(train.OUT/'steps'/f'step_{step:06d}.json')
    need(record['step']==step and c.sha(train.OUT/record['file'])==record['sha256'],'checkpoint SHA mismatch')
    saved=torch.load(train.OUT/record['file'],map_location='cpu',weights_only=True)
    train.validate_state(saved,record,c.sha(train.OUT/'receipt.json'))
    return saved['model'],record


def old_probe(index,step,device='cpu'):
    from safetensors.torch import load_file
    p=train.OUT/'probes'/f'step_{step:06d}_image_{index}.json';r=c.read(p)
    need(r['index']==index and r['training_fit_only'] and r['GT_used_for_output_selection'] is False,'invalid parent probe')
    need(r['identity']['receipt_sha256']==c.sha(train.OUT/'receipt.json') and r['identity']['step']==step,'probe binding mismatch')
    need(r['identity']['checkpoint_receipt_sha256']==c.sha(train.OUT/'steps'/f'step_{step:06d}.json'),'probe model mismatch')
    c.validate_record(r,r['identity'],train.OUT)
    values=load_file(str(train.OUT/r['tensor']),device=str(device));need(set(values)=={'uint8'},'probe schema')
    return r,values['uint8']


def prepare():
    begin=time.monotonic();value,receipt,cache=train.bound();sources={**value['sources'],**{s:c.sha(ROOT/s) for s in SOURCES}}
    models={};probes={}
    for step in cross.STEPS:
        weights,r=state(step);models[str(step)]=r;del weights
    for index in cross.INDICES:
        row=receipt['rows'][index]
        need(row['split']=='train' and row['usage']=='train_update' and row['category']=='10Rain','nontraining source')
        need(row['source_group'] not in value['excluded_source_groups'],'excluded source')
        need(all(row[k].startswith('data/foundir/MillionIRData/Train/10Rain/') for k in ('lq_path','gt_path')),'illegal pixel scope')
        data=train.cached(value,receipt,cache,index,'cpu');del data
        for step in cross.STEPS:
            r,u8=old_probe(index,step);need(tuple(u8.shape)==(1,3,512,512),'parent probe dimensions')
            probes[f'{step}_{index}']=c.sha(train.OUT/'probes'/f'step_{step:06d}_image_{index}.json')
    assets={}
    for k in ('flow_weights','autoencoder_weights'):assets[receipt['model'][k]]=receipt['model'][k+'_sha256']
    spec=receipt['rain']['prompt_context_cache'];assets[spec['path']]=spec['sha256']
    spec=receipt['native_checkpoint'];assets[spec['path']]=spec['sha256']
    for path,digest in assets.items():need(c.sha(ROOT/path)==digest,'base asset SHA mismatch')
    b={'recipe':RECIPE,'training_receipt_sha256':c.sha(train.OUT/'receipt.json'),'sources':sources,'checkpoints':models,
       'probe_markers':probes,'assets':assets,'cache_source':value['cache_source'],'metric':receipt['metric'],
       'rows':[receipt['rows'][i] for i in cross.INDICES],'training_fit_only':True}
    c.write_new(OUT/'receipt.json',b)
    report={'status':'destination_CPU_preflight_pass','receipt_sha256':c.sha(OUT/'receipt.json'),
            'checkpoint_hashes_verified':True,'training_cache_tensors_verified':8,'baseline_probe_tensors_verified':16,
            'new_model_calls':432,'new_outputs':32,'official_pixels_read':False,'development_pixels_read':False,
            'new_training':False,'GPU_started':False,'seconds':time.monotonic()-begin,'GPU_hours':0,'exit_code':0}
    c.write_new(PREFLIGHT,report);print('RAIN_PREFIX_CROSS_PREFLIGHT_PASS '+json.dumps(report),flush=True)


def bound():
    b=c.read(OUT/'receipt.json');need(b['recipe']==RECIPE and b['training_receipt_sha256']==c.sha(train.OUT/'receipt.json'),'recipe/parent changed')
    for name,digest in b['sources'].items():need(c.sha(ROOT/name)==digest,'source changed: '+name)
    for label,digest in b['probe_markers'].items():
        step,index=map(int,label.split('_'));need(c.sha(train.OUT/'probes'/f'step_{step:06d}_image_{index}.json')==digest,'baseline changed')
    return b


def validate_record(record,index,rank):
    identity={'receipt_sha256':c.sha(OUT/'receipt.json'),'index':index,'rank':rank}
    c.validate_record(record,identity,OUT)
    need(record['index']==index and record['training_fit_only'] and record['GT_input'] is False and record['actual_model_calls']==54,'output schema mismatch')


def run(node):
    import torch
    from flux2.util import load_ae
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    from smoke.expanded_vlm_h100_runtime_v4 import bounded_vae,POLICY
    b=bound();value,receipt,cache=train.bound();rank=int(os.environ['LOCAL_RANK'])
    need(node==0 and rank in (0,1) and int(os.environ['LOCAL_WORLD_SIZE'])==2 and torch.cuda.device_count()==2,'one node0 two GPUs required')
    torch.cuda.set_device(rank);device=torch.device('cuda',rank)
    need(torch.cuda.get_device_properties(rank).total_memory>=75*1024**3 and any(x in torch.cuda.get_device_name(rank) for x in ('H100','H200')),'75GiB Hopper required')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    runtime={'receipt_sha256':c.sha(OUT/'receipt.json'),'torch':str(torch.__version__),'cuda':torch.version.cuda,
             'device':torch.cuda.get_device_name(rank),'vae_policy':POLICY,
             'metric':parent.metric_impl.metric_runtime({'metric':receipt['metric']})}
    need(runtime['metric']==c.read(train.OUT/'runtime_rank0.json')['metrics'],'training metric runtime mismatch')
    c.write_new(OUT/f'runtime_rank{rank}.json',runtime);begin=time.monotonic()
    for path,digest in b['assets'].items():need(c.sha(ROOT/path)==digest,'asset changed before GPU load')
    os.environ.update(KLEIN_4B_BASE_MODEL_PATH=str(ROOT/receipt['model']['flow_weights']),AE_MODEL_PATH=str(ROOT/receipt['model']['autoencoder_weights']))
    print(f'RAIN_PREFIX_CROSS_LOADING rank={rank}',flush=True)
    model=parent.new_model(receipt,device).eval().requires_grad_(False)
    parameters={n:p for n,p in model.named_parameters() if '.lora_A.' in n or '.lora_B.' in n}
    states={s:state(s)[0] for s in cross.STEPS}
    ae=bounded_vae(parent.hopper.configure_vae(load_ae('flux.2-klein-base-4b',device=device).eval().requires_grad_(False)))
    prompt=parent.load_prompt(receipt,device);metric=LearnedPerceptualImagePatchSimilarity(net_type='alex',normalize=True).to(device).eval().requires_grad_(False)
    need(not any(p.requires_grad for p in model.parameters()),'diagnostic must not train')
    for index in cross.assigned(rank):
        need(time.monotonic()-begin<900,'diagnostic cap reached; preserve partial results')
        marker=OUT/'records'/f'image_{index}.json'
        if marker.exists():validate_record(c.read(marker),index,rank);continue
        data=train.cached(value,receipt,cache,index,device);seed=receipt['config']['seed']+1000003*(index+1)
        calls=[0]
        class Counted:
            def __call__(self,**kwargs):calls[0]+=1;return model(**kwargs)
        counted=Counted();frozen_x={}
        def prefix():return logic.generated_inputs(counted,ae,data['lq'],prompt,seed)
        def endpoint(inputs):
            before=inputs['x'].clone()
            rgb=logic.decode_endpoint(ae,logic.predict_endpoint(counted,inputs),checkpointed=False)
            need(torch.equal(before,inputs['x']),'readout mutated prefix state')
            return parent.quantize(rgb)
        start=time.monotonic()
        with torch.no_grad():
            prefixes,outputs=cross.run_grid(lambda s:cross.copy_adapter(parameters,states[s]),prefix,endpoint)
        need(calls[0]==54,'wrong model call count')
        a,z=prefixes[800],prefixes[1000]
        for k in a:
            if k=='x':continue
            need(torch.equal(a[k],z[k]) if isinstance(a[k],torch.Tensor) else a[k]==z[k],'prefix ancillary input changed: '+k)
        n=a['x'].shape[1]//2;need(torch.equal(a['x'][:,n:],z['x'][:,n:]),'LQ reference tokens changed')
        prefix_rms=float((a['x'][:,:n].float()-z['x'][:,:n].float()).square().mean().sqrt())
        # Publish all four fixed outputs before using training GT for scores. No selection.
        name,digest=parent.replay.save_tensors(OUT,f'cross_{index}',{arm:t.detach().cpu() for arm,t in outputs.items()})
        metrics={arm:parent.scores(metric,u8,data['gt']) for arm,u8 in outputs.items()};parity={}
        for step in cross.STEPS:
            prior,expected=old_probe(index,step,device);arm=f'p{step}_r{step}'
            parity[str(step)]={'max_uint8_difference':int((outputs[arm].int()-expected.int()).abs().max()),
                              'max_metric_difference':max(abs(metrics[arm][k]-prior['metrics'][k]) for k in ('psnr','lpips'))}
        result={'identity':{'receipt_sha256':c.sha(OUT/'receipt.json'),'index':index,'rank':rank},
                'index':index,'seed':seed,'metrics':metrics,'parity':parity,'prefix_generated_tokens_RMS_difference':prefix_rms,
                'tensor':name,'files':{name:digest},'actual_model_calls':calls[0],
                'training_fit_only':True,'GT_input':False,'seconds':time.monotonic()-start}
        c.write_new(marker,result);print('RAIN_PREFIX_CROSS_RECORD '+json.dumps({k:result[k] for k in ('index','metrics','parity','seconds')}),flush=True)
        del data,prefixes,outputs,a,z,expected
    records=[c.read(OUT/'records'/f'image_{i}.json') for i in cross.assigned(rank)]
    for r in records:validate_record(r,r['index'],rank)
    c.write_new(OUT/f'worker{rank}.json',{'rank':rank,'indices':cross.assigned(rank),'receipt_sha256':c.sha(OUT/'receipt.json'),
           'seconds':time.monotonic()-begin,'peak_allocated_GiB':torch.cuda.max_memory_allocated()/1024**3})
    print(f'RAIN_PREFIX_CROSS_WORKER_END rank={rank} status=0',flush=True)


def finish():
    bound();records=[]
    for rank in (0,1):
        worker=c.read(OUT/f'worker{rank}.json');need(worker['rank']==rank and worker['indices']==cross.assigned(rank) and worker['receipt_sha256']==c.sha(OUT/'receipt.json'),'worker incomplete')
        for index in cross.assigned(rank):
            r=c.read(OUT/'records'/f'image_{index}.json');validate_record(r,index,rank);records.append(r)
    need(c.read(OUT/'runtime_rank0.json')==c.read(OUT/'runtime_rank1.json'),'different numerical runtime')
    summary=cross.summarize(records)
    result={'status':'complete' if summary['matched_parent_parity'] else 'completed_but_parent_parity_failed',
            'receipt_sha256':c.sha(OUT/'receipt.json'),'summary':summary,'records':sorted(records,key=lambda r:r['index']),
            'official_or_development_evaluated':False,'new_training':False,'new_model_calls':432,
            'diagnostic_only':True,'automatic_followup':False}
    c.write_new(REPORT,result);print('RAIN_PREFIX_CROSS_RESULT_VERIFIED '+json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','run','finish','verify']);p.add_argument('--node',type=int,default=0);a=p.parse_args()
    if a.phase=='run':run(a.node)
    elif a.phase=='verify':bound();print('RAIN_PREFIX_CROSS_BINDING_VERIFIED')
    else:globals()[a.phase]()
