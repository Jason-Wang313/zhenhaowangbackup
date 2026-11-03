#!/usr/bin/env python3
"""Fixed step400 / call25 GT-free readout on the original full100 development set."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'third_party/flux2/src'))
from smoke import existing_flux_replay_contract_v1 as c
from smoke import existing_flux_replay_cluster_v1 as cluster
from smoke import rain_native50_continue_contract_v1 as parent_contract
from smoke import train_foundir_rain_native50_continue_v1 as parent
from smoke import diagnose_foundir_rain_prediction_gap_v1 as gap
from smoke import rain_midpoint_readout_v1 as readout
from smoke.rain_prediction_gap_math_v1 import RolloutObserver

CONFIG='configs/foundir_rain_midpoint_development_v1.json'
SOURCES=[CONFIG,'smoke/rain_midpoint_readout_v1.py','smoke/evaluate_foundir_rain_midpoint_development_v1.py',
         'smoke/run_foundir_rain_midpoint_development_v1_gpu.sh','smoke/start_foundir_rain_midpoint_development_v1_tmux.sh',
         *gap.SOURCES]


def config():
    cfg=c.read(ROOT/CONFIG)
    expected=dict(checkpoint_step=400,call_index=25,inference_calls=26,schedule_steps=50,indices=[64032,64132],
                  nodes=4,local_gpus=2,world_size=8,training_updates=0,development_pixels_allowed=True,
                  official_or_sealed_pixels_allowed=False,psnr_minimum=34.2,lpips_maximum=.08,
                  wall_cap_seconds=1800,minimum_free_disk_gib=10)
    if any(cfg.get(k)!=v for k,v in expected.items()):raise ValueError('fixed evaluation scope changed')
    if cfg['parent_run']!=gap.config()['parent_run']:raise ValueError('parent run differs')
    return cfg


def output_path(run_id):
    c.run_path(ROOT,run_id)
    return c.path(ROOT,'checkpoints/foundir_rain_midpoint_development_v1/'+run_id)


def make_receipt():
    cfg=config();source=c.path(ROOT,cfg['parent_run']);step=cfg['checkpoint_step']
    for key in ('parent_report','selection_report'):
        if c.sha(ROOT/cfg[key])!=cfg[key+'_sha256']:raise ValueError('frozen evidence changed')
    report=c.read(ROOT/cfg['parent_report'])
    receipt=parent_contract.load_receipt(ROOT,source)
    rows=readout.validate_rows(receipt['rows'])
    marker=source/'steps'/f'step_{step:06d}.json';digest=c.sha(marker);record=c.read(marker)
    if digest!=report['checkpoint_receipt_sha256'][str(step)] or record['step']!=step:
        raise ValueError('wrong frozen checkpoint')
    if record['receipt_hash']!=c.sha(source/'receipt.json') or not c.path(source,record['file']).is_file():
        raise ValueError('checkpoint unavailable or mismatched')
    probes={}
    for index in range(*cfg['indices']):
        marker=source/'probes'/f'step_{step:06d}_image_{index}.json';entry=c.read(marker)
        expected={**parent.identity(source,receipt['rows'][index],index),'step':step,
                  'checkpoint_receipt_sha256':digest,'cache_receipt_sha256':c.sha(parent.cache_marker(source,index))}
        if entry['identity']!=expected or entry['training_fit_only']:
            raise ValueError('wrong frozen development probe')
        if not c.path(source,entry['tensor']).is_file():raise FileNotFoundError(entry['tensor'])
        probes[str(index)]=c.sha(marker)
    return {'config':cfg,'parent_receipt_sha256':c.sha(source/'receipt.json'),'rows':rows,'probes':probes,
            'checkpoints':{str(step):{'receipt_sha256':digest,'record':record}},
            'sources':{**receipt['sources'],**{n:c.sha(ROOT/n) for n in SOURCES}},
            'pixels_read_in_preflight':False,'development_previously_used':True,'new_blind_test':False}


def load_receipt(output):
    bound=c.read(output/'receipt.json')
    if bound['config']!=config():raise ValueError('config changed')
    for name,digest in bound['sources'].items():
        if c.sha(ROOT/name)!=digest:raise ValueError('bound source changed: '+name)
    source=c.path(ROOT,bound['config']['parent_run'])
    if c.sha(source/'receipt.json')!=bound['parent_receipt_sha256']:raise ValueError('parent changed')
    receipt=parent_contract.load_receipt(ROOT,source)
    if readout.validate_rows(receipt['rows'])!=bound['rows']:raise ValueError('development rows changed')
    return bound,receipt,source


def generate_images(model,ae,lq,prompt,seed,device):
    """Only LQ, frozen model, prompt and seed. No GT or quality scorer is available here."""
    import torch
    from flux2.sampling import batched_prc_img
    from smoke.evaluate_foundir_blur_only_flux2_full_development_ddp import restore_batch
    def sampler(current_model):
        return restore_batch(current_model,ae,lq,prompt['cfg_context'],prompt['cfg_ids'],batched_prc_img,
                             seeds=[seed],steps=50,reference_time=10)
    with torch.inference_mode():
        torch.cuda.synchronize(device);start=time.monotonic()
        snap,calls=readout.run_prefix(model,sampler)
        prefix=readout.decode_endpoint(snap,ae,device)
        torch.cuda.synchronize(device);prefix_seconds=time.monotonic()-start
        observer=RolloutObserver(model,selected=(25,));start=time.monotonic()
        full=sampler(observer)
        torch.cuda.synchronize(device);full_seconds=time.monotonic()-start
        if observer.calls!=50 or set(observer.snapshots)!={25}:raise ValueError('wrong full50 trajectory')
        full_readout=readout.decode_endpoint(observer.snapshots[25],ae,device)
        diff=(parent.quantize(prefix).to(torch.int16)-parent.quantize(full_readout).to(torch.int16)).abs()
        controls={'prefix_matches_full_trace_uint8':bool((diff==0).all()),
                  'prefix_full_trace_max_uint8_difference':int(diff.max()),
                  'prefix_full_trace_changed_fraction':float((diff!=0).float().mean())}
    if not torch.isfinite(prefix).all() or not torch.isfinite(full).all():raise ValueError('nonfinite output')
    return prefix,full,controls,{'prefix_calls':calls,'full_calls':observer.calls,
        'total_evaluation_calls':calls+observer.calls,'model_t':float(snap['t'][0]),
        'prefix_seconds':prefix_seconds,'full_seconds':full_seconds,
        'timing_is_ordered_not_randomized_speed_benchmark':True}


def evaluate_one(output,bound,receipt,source,model,ae,prompt,metric,index,device):
    import torch
    from safetensors.torch import load_file
    row=receipt['rows'][index];step=400
    if row['usage']!='development' or index not in range(64032,64132):raise ValueError('wrong sample')
    ident={'receipt_sha256':c.sha(output/'receipt.json'),'step':step,'index':index,'row':row,
           'call_index':25,'parent_checkpoint_sha256':bound['checkpoints']['400']['record']['sha256']}
    marker=output/'images'/f'image_{index}.json'
    if marker.exists():c.validate_record(c.read(marker),ident,output);return
    data,_=parent.cached(source,receipt,index,device)
    # GT is deliberately not passed into generation. Both outputs are fixed before scoring.
    prefix,full,controls,compute=generate_images(model,ae,data['lq'],prompt,
                                                receipt['config']['probe_seed']+index-64000,device)
    prior_marker=source/'probes'/f'step_{step:06d}_image_{index}.json'
    if c.sha(prior_marker)!=bound['probes'][str(index)]:raise ValueError('historical probe changed')
    prior=c.read(prior_marker);c.validate_record(prior,prior['identity'],source)
    old=load_file(str(c.path(source,prior['tensor'])),device=str(device))
    diff=(parent.quantize(full).to(torch.int16)-old['uint8'].to(torch.int16)).abs()
    controls.update(full_matches_historical_uint8=bool((diff==0).all()),
                    full_historical_max_uint8_difference=int(diff.max()),
                    full_historical_changed_fraction=float((diff!=0).float().mean()))
    with torch.inference_mode():
        metrics={name:parent.scores(metric,parent.quantize(rgb),data['gt']) for name,rgb in
                 [('fixed_prefix26',prefix),('matched_full50',full),('original_one_step_teacher',data['teacher_rgb'])]}
    file,digest=parent.replay.save_tensors(output,f'dev_{index}',
        {'fixed_prefix26':parent.quantize(prefix),'matched_full50':parent.quantize(full)})
    record={'identity':ident,'index':index,'files':{file:digest},'tensor':file,'metrics':metrics,
        'controls':controls,'compute':compute,'historical_full50_metrics':prior['metrics']['student_50_step'],
        'GT_used_for_output_selection':False,'training_updates':0,
        'peak_allocated_gib':torch.cuda.max_memory_allocated(device)/1024**3}
    c.write_new(marker,record)
    print('RAIN_MIDPOINT_DEVELOPMENT_IMAGE '+json.dumps({'index':index,'metrics':metrics,'controls':controls}),flush=True)


def summarize(output,bound):
    rows=[]
    for index in range(64032,64132):
        record=c.read(output/'images'/f'image_{index}.json')
        expected={'receipt_sha256':c.sha(output/'receipt.json'),'step':400,'index':index,
                  'row':bound['rows'][index-64032],'call_index':25,
                  'parent_checkpoint_sha256':bound['checkpoints']['400']['record']['sha256']}
        if record['identity']!=expected or record['index']!=index or record['GT_used_for_output_selection']:
            raise ValueError('invalid result identity')
        if record['compute']['prefix_calls']!=26 or record['compute']['full_calls']!=50:
            raise ValueError('wrong inference budget')
        rows.append(record)
    summary=readout.paired_summary(rows)
    replay=sum(r['controls']['full_matches_historical_uint8'] for r in rows)
    prefix=sum(r['controls']['prefix_matches_full_trace_uint8'] for r in rows)
    result={'status':'fixed_midpoint_development_evaluation_complete','receipt_sha256':c.sha(output/'receipt.json'),
        'summary':summary,'full50_historical_exact_count':replay,'prefix_full_trace_exact_count':prefix,
        'matched_comparison_valid':replay==100 and prefix==100,
        'matched_gate_pass':summary['quality_gate_pass'] and replay==100 and prefix==100,
        'checkpoint_step':400,'fixed_call_index':25,'prefix_model_calls_per_image':26,
        'full_model_calls_per_image':50,'total_evaluation_model_calls':sum(r['compute']['total_evaluation_calls'] for r in rows),
        'per_image_metrics':[{'index':r['index'],'record_id':r['identity']['row']['record_id'],
                             'metrics':r['metrics'],'controls':r['controls']} for r in rows],
        'GT_used_for_output_selection':False,'training_updates':0,'official_or_sealed_test_accessed':False,
        'development_previously_used':True,'new_blind_test':False,'automatic_follow_on':False,
        'full_tensor_audit_performed':False,'lpips_reexecuted_in_summary':False}
    c.write_new(output/'result.json',result)
    print('RAIN_MIDPOINT_DEVELOPMENT_RESULT '+json.dumps({k:v for k,v in result.items() if k!='per_image_metrics'},sort_keys=True),flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('phase',choices=['preflight','prepare','hardware','runtime','run','summarize','publish','wait'])
    ap.add_argument('--event',choices=['boot','prepared','ready','done','verified','failed'])
    ap.add_argument('--run-id',default='step400_call25_dev100_v1');ap.add_argument('--launch-id')
    ap.add_argument('--node-rank',type=int,choices=range(4),default=0)
    args=ap.parse_args();cfg=config();output=output_path(args.run_id)
    if str(ROOT)!=cfg['active_root']:raise ValueError('execute only in migrated workspace')
    if args.phase in ('publish','wait'):
        if not args.launch_id or not args.event:raise ValueError('missing launch/event')
        if args.phase=='publish':cluster.publish(output,args.launch_id,args.node_rank,args.event)
        else:
            deadline=time.monotonic()+cfg['wall_cap_seconds']
            while not cluster.check(output,args.launch_id,args.event):
                if time.monotonic()>deadline:raise TimeoutError('four-node wait timed out')
                time.sleep(2)
        return
    if args.phase in ('preflight','prepare'):
        import shutil
        if args.phase=='prepare' and args.node_rank!=0:raise ValueError('node0 only')
        if shutil.disk_usage(ROOT).free<cfg['minimum_free_disk_gib']*1024**3:raise ValueError('need 10GiB free; preserve history')
        bound=make_receipt()
        if args.phase=='prepare':c.write_new(output/'receipt.json',bound)
        print('RAIN_MIDPOINT_DEVELOPMENT_PREFLIGHT_PASS pixels_read=false GPU_used=false',flush=True);return
    if args.phase=='hardware':print(json.dumps(parent.hopper.hardware(True)));return
    bound,receipt,source=load_receipt(output)
    if args.phase=='runtime':gap.runtime(output,receipt,args.node_rank,True);return
    if args.phase=='summarize':
        if args.node_rank!=0 or not cluster.check(output,args.launch_id,'done'):raise ValueError('four-node done required')
        parent.check_runtimes(output);summarize(output,bound);return
    import torch
    rank,local=c.worker_rank(args.node_rank,int(os.environ['LOCAL_RANK']),int(os.environ['WORLD_SIZE']),
                             int(os.environ['LOCAL_WORLD_SIZE']),int(os.environ['RANK']))
    gap.runtime(output,receipt,args.node_rank,False);parent.check_runtimes(output)
    torch.cuda.set_device(local);device=torch.device('cuda',local)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    os.environ.update(KLEIN_4B_BASE_MODEL_PATH=str(c.path(ROOT,receipt['model']['flow_weights'])),
        AE_MODEL_PATH=str(c.path(ROOT,receipt['model']['autoencoder_weights'])),TORCH_HOME=str(ROOT/'models/torch'),
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    from flux2.util import load_ae
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    model=parent.new_model(receipt,device).eval().requires_grad_(False)
    ae=parent.hopper.configure_vae(load_ae('flux.2-klein-base-4b',device=device).eval().requires_grad_(False))
    prompt=parent.load_prompt(receipt,device)
    metric=LearnedPerceptualImagePatchSimilarity(net_type='alex',normalize=True).to(device).eval().requires_grad_(False)
    gap.load_weights(model,source,bound,400)
    for index in readout.assigned_indices(rank):evaluate_one(output,bound,receipt,source,model,ae,prompt,metric,index,device)
    print(f'RAIN_MIDPOINT_DEVELOPMENT_RANK_DONE rank={rank}',flush=True)


if __name__=='__main__':main()
