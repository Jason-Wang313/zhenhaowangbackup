#!/usr/bin/env python3
"""Frozen Rain: GT-path endpoint vs own-path endpoint vs unchanged real 50-step output."""
import argparse
import gc
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'third_party/flux2/src'))
from smoke import existing_flux_replay_contract_v1 as c
from smoke import existing_flux_replay_cluster_v1 as cluster
from smoke import rain_native50_continue_contract_v1 as parent_contract
from smoke import train_foundir_rain_native50_continue_v1 as parent
from smoke import rain_native50_continue_state_v1 as state
from smoke import rain_prediction_gap_math_v1 as logic

CONFIG = 'configs/foundir_rain_prediction_gap_v1.json'
SOURCES = [CONFIG, 'smoke/rain_prediction_gap_math_v1.py',
           'smoke/diagnose_foundir_rain_prediction_gap_v1.py',
           'smoke/run_foundir_rain_prediction_gap_v1_gpu.sh',
           'smoke/start_foundir_rain_prediction_gap_v1_tmux.sh',
           'smoke/rain_hopper_memory_v1.py']


def output_path(run_id):
    c.run_path(ROOT, run_id)
    return c.path(ROOT, 'checkpoints/foundir_rain_prediction_gap_v1/'+run_id)


def config():
    cfg = c.read(ROOT/CONFIG)
    if (cfg['checkpoints'] != [400, 1500] or cfg['indices'] != [64000, 64032]
        or cfg['trace_indices'] != [0, 10, 25, 40, 49] or cfg['training_updates'] != 0
        or cfg['world_size'] != 8 or cfg['nodes'] != 4 or cfg['local_gpus'] != 2
        or not cfg['training_fit_only'] or cfg['development_pixels_allowed']
        or cfg['official_test_pixels_allowed']):
        raise ValueError('diagnostic scope changed')
    return cfg


def make_receipt():
    cfg = config(); source = c.path(ROOT, cfg['parent_run'])
    if c.sha(ROOT/cfg['parent_report']) != cfg['parent_report_sha256']:
        raise ValueError('milestone report changed')
    report = c.read(ROOT/cfg['parent_report'])
    receipt = parent_contract.load_receipt(ROOT, source)
    rows = logic.validate_indices(receipt['rows'])
    checkpoints = {}
    probes = {}
    for step in cfg['checkpoints']:
        name = f'steps/step_{step:06d}.json'; marker = source/name
        digest = c.sha(marker)
        if digest != report['checkpoint_receipt_sha256'][str(step)]:
            raise ValueError('checkpoint receipt not the published milestone')
        record = c.read(marker)
        if record['step'] != step or record['receipt_hash'] != c.sha(source/'receipt.json'):
            raise ValueError('parent checkpoint identity mismatch')
        if not c.path(source, record['file']).is_file():raise FileNotFoundError(record['file'])
        checkpoints[str(step)] = {'receipt_sha256': digest, 'record': record}
        for index in range(*cfg['indices']):
            probe = source/'probes'/f'step_{step:06d}_image_{index:03d}.json'
            entry = c.read(probe)
            expected = {**parent.identity(source, receipt['rows'][index], index), 'step': step,
                        'checkpoint_receipt_sha256': digest,
                        'cache_receipt_sha256': c.sha(parent.cache_marker(source,index))}
            if entry['identity'] != expected or not entry['training_fit_only']:
                raise ValueError('probe identity mismatch')
            if not c.path(source,entry['tensor']).is_file():raise FileNotFoundError(entry['tensor'])
            probes[f'{step}_{index}'] = c.sha(probe)
    exposure = {}
    for step in cfg['checkpoints']:
        updates=receipt['rows'][:step*32]
        exposure[str(step)]=[{key:sum(all(u[k]==row[k] for k in fields) for u in updates)
                              for key,fields in [('source_presentations',['source_group']),
                                                 ('record_presentations',['record_id']),
                                                 ('exact_crop_presentations',['record_id','crop_xy','augmentation_code'])]}
                             for row in rows]
    return {'config': cfg, 'parent_receipt_sha256': c.sha(source/'receipt.json'),
            'sources': {**receipt['sources'], **{n:c.sha(ROOT/n) for n in SOURCES}},
            'checkpoints': checkpoints, 'probes': probes, 'rows': rows, 'training_exposure':exposure,
            'pixels_read': False, 'GPU_used_in_preflight': False}


def load_receipt(output):
    result = c.read(output/'receipt.json')
    if result['config'] != config():raise ValueError('diagnostic config changed')
    for name,digest in result['sources'].items():
        if c.sha(ROOT/name) != digest:raise ValueError('bound source changed: '+name)
    source=c.path(ROOT,result['config']['parent_run'])
    if c.sha(source/'receipt.json') != result['parent_receipt_sha256']:
        raise ValueError('parent receipt changed')
    receipt=parent_contract.load_receipt(ROOT,source)
    if logic.validate_indices(receipt['rows'])!=result['rows']:raise ValueError('diagnostic rows changed')
    return result,receipt,source


def runtime(output, receipt, node, idle):
    value=parent.hopper.hardware(idle)
    value['metrics']=parent.metric_impl.metric_runtime({'metric':receipt['metric']})
    value['memory_policy']=receipt['config']['memory_policy']
    previous=c.read(c.path(ROOT,config()['parent_run'])/f'runtime_node{node}.json')
    if parent.hopper.execution_signature(value)!=parent.hopper.execution_signature(previous):
        raise ValueError('software/metric runtime differs from compared run')
    marker=output/f'runtime_node{node}.json'
    if idle:c.write_new(marker,value)
    elif c.read(marker)!=value:raise ValueError('runtime changed within diagnostic')


def load_weights(model, source, bound, step):
    import torch
    entry=bound['checkpoints'][str(step)]; record=entry['record']
    if c.sha(source/'steps'/f'step_{step:06d}.json')!=entry['receipt_sha256']:
        raise ValueError('step receipt changed')
    file=c.path(source,record['file'])
    if c.sha(file)!=record['sha256']:raise ValueError('step tensor hash mismatch')
    saved=torch.load(file,map_location='cpu',weights_only=True)
    if (saved['step']!=step or saved['receipt_hash']!=bound['parent_receipt_sha256']
        or state.parameter_hash(saved['model'])!=record['parameter_hash']):
        raise ValueError('wrong checkpoint payload')
    values=dict(model.named_parameters())
    names=[n for n,p in values.items() if '.lora_A.' in n or '.lora_B.' in n]
    if set(names)!=set(saved['model']):raise ValueError('LoRA parameter mismatch')
    with torch.no_grad():
        for name in names:
            value=saved['model'][name];dest=values[name]
            if value.shape!=dest.shape or value.dtype!=dest.dtype or not torch.isfinite(value).all():
                raise ValueError('invalid frozen LoRA')
            dest.copy_(value.to(dest.device))
    model.eval().requires_grad_(False)
    # No optimizer instantiation or restore; frozen diagnostic only.


def measure(metric, rgb, gt):
    value=parent.scores(metric,parent.quantize(rgb),gt)
    value['continuous_mse_0_1']=float(((rgb.float()-gt.float())*.5).square().mean())
    value['continuous_lpips']=float(metric((rgb.float()+1)*.5,(gt.float()+1)*.5));metric.reset()
    return value


def diagnose_one(output,bound,receipt,source,model,ae,prompt,metric,index,step,device):
    import torch
    from safetensors.torch import load_file
    from flux2.sampling import batched_prc_img,get_schedule
    from lowlight_restoration.rain_flow_adaptation import endpoint_estimate
    from smoke.evaluate_foundir_blur_only_flux2_full_development_ddp import restore_batch
    cfg=bound['config'];row=receipt['rows'][index]
    if row['usage']!='train_probe' or row['split']!='train':raise ValueError('protected image')
    ident={'receipt_sha256':c.sha(output/'receipt.json'),'step':step,'index':index,'row':row,
           'parent_checkpoint_sha256':bound['checkpoints'][str(step)]['record']['sha256']}
    marker=output/'images'/f'step_{step:06d}_image_{index}.json'
    if marker.exists():
        c.validate_record(c.read(marker),ident,output);return
    data,_=parent.cached(source,receipt,index,device)
    start=time.monotonic()
    with torch.inference_mode():
        observer=logic.RolloutObserver(model,cfg['trace_indices'])
        final=restore_batch(observer,ae,data['lq'],prompt['cfg_context'],prompt['cfg_ids'],batched_prc_img,
                            seeds=[receipt['config']['probe_seed']+index-64000],steps=50,reference_time=10)
        if observer.calls!=50 or set(observer.snapshots)!=set(cfg['trace_indices']):
            raise ValueError('unexpected sampling trajectory')
        prior_marker=source/'probes'/f'step_{step:06d}_image_{index}.json'
        if c.sha(prior_marker)!=bound['probes'][f'{step}_{index}']:raise ValueError('prior probe changed')
        prior=c.read(prior_marker);c.validate_record(prior,prior['identity'],source)
        old=load_file(str(c.path(source,prior['tensor'])),device=str(device))
        diff=(parent.quantize(final).to(torch.int16)-old['uint8'].to(torch.int16)).abs()
        matching={'uint8_exact':bool((diff==0).all()),'max_uint8_difference':int(diff.max()),
                  'changed_fraction':float((diff!=0).float().mean()),'historical_metrics':prior['metrics']['student_50_step']}
        del old,diff
        # Evaluation caches deliberately have no encoded GT. Encode ONLY these training GTs here.
        with torch.autocast(device.type,enabled=False):clean=ae.encode(data['gt'].float()).float()
        shape=clean.shape;n=shape[2]*shape[3]
        first=observer.snapshots[0]['x'][:,:n].to(device).float()
        epsilon=first.transpose(1,2).reshape(shape)
        def decode(z):
            with torch.autocast(device.type,dtype=torch.bfloat16):
                return ae.decode(z).float().clamp(-1,1)
        reconstruction=decode(clean)
        scores={'final_50_step':measure(metric,final,data['gt']),
                'original_one_step_teacher':measure(metric,data['teacher_rgb'],data['gt']),
                'GT_VAE_roundtrip_not_an_upper_bound':measure(metric,reconstruction,data['gt'])}
        tensors={'final_50_step':parent.quantize(final),'GT_VAE_roundtrip':parent.quantize(reconstruction)}
        traces=[];schedule=get_schedule(50,n)
        for call in cfg['trace_indices']:
            snap={k:v.to(device) for k,v in observer.snapshots[call].items()}
            t=snap['t'].float()
            # Same actual model timestep and epsilon. GT-path carries clean-answer information.
            gt_est,terms=logic.gt_path_prediction(model,clean,data['reference_latent'],epsilon,t,
                                                 prompt['context'],prompt['context_ids'],batched_prc_img)
            with torch.autocast(device.type,dtype=torch.bfloat16):
                own_pred=model(x=snap['x'],x_ids=snap['x_ids'],timesteps=t,
                               ctx=prompt['context'],ctx_ids=prompt['context_ids'],guidance=None)[:,:n]
            own=endpoint_estimate(snap['x'][:,:n],own_pred,t,shape)
            guided=endpoint_estimate(snap['x'][:,:n],snap['guided'],t,shape)
            arms={}
            for name,z in [('GT_path_single_prediction',gt_est),('own_path_single_prediction',own),
                           ('own_path_deployment_CFG_prediction',guided)]:
                rgb=decode(z)
                if not torch.isfinite(rgb).all():raise ValueError('nonfinite prediction')
                arms[name]=measure(metric,rgb,data['gt'])
                tensors[f'call{call}_{name}']=parent.quantize(rgb)
                if name=='GT_path_single_prediction':
                    terms.update(gt_rgb_mse=arms[name]['continuous_mse_0_1']*4,
                                 gt_lpips=arms[name]['continuous_lpips'],
                                 teacher_rgb_l1=float((rgb-data['teacher_rgb']).abs().mean()))
                    terms['weighted_training_objective']=sum(receipt['config']['loss_weights'][k]*terms[k]
                                                              for k in receipt['config']['loss_weights'])
            gt_tokens=batched_prc_img((1-t.view(-1,1,1,1))*clean+t.view(-1,1,1,1)*epsilon)[0]
            traces.append({'call':call,'schedule_t':schedule[call],'model_t':float(t[0]),'metrics':arms,
                'GT_path_terms':terms,'state_rms_from_GT_path':float((snap['x'][:,:n].float()-gt_tokens).square().mean().sqrt()),
                'reference_rms_from_cached_training_reference':float((snap['x'][:,n:].float()-batched_prc_img(data['reference_latent'])[0]).square().mean().sqrt()),
                'single_vs_CFG_velocity_rms':float((own_pred.float()-snap['guided'].float()).square().mean().sqrt()),
                'conditional_vs_CFG_arithmetic_rms':float((snap['conditional'].float()-snap['guided'].float()).square().mean().sqrt())})
        name,digest=parent.replay.save_tensors(output,f'gap_{step}_{index}',tensors)
    result={'identity':ident,'tensor':name,'files':{name:digest},'training_fit_only':True,
            'GT_assisted_not_deployable':True,'metrics':scores,'trace':traces,'rollout_reproduction':matching,
            'training_exposure':bound['training_exposure'][str(step)][index-64000],
            'elapsed_seconds':time.monotonic()-start,'peak_allocated_gib':torch.cuda.max_memory_allocated(device)/1024**3}
    c.write_new(marker,result)
    print('RAIN_PREDICTION_GAP_IMAGE '+json.dumps({'step':step,'index':index,'metrics':scores,'reproduction':matching}),flush=True)


def summarize(output,bound):
    summary={};all_match=True
    for step in bound['config']['checkpoints']:
        rows=[c.read(output/'images'/f'step_{step:06d}_image_{i}.json') for i in range(64000,64032)]
        for i,r in enumerate(rows,64000):
            expected={'receipt_sha256':c.sha(output/'receipt.json'),'step':step,'index':i,'row':bound['rows'][i-64000],
                      'parent_checkpoint_sha256':bound['checkpoints'][str(step)]['record']['sha256']}
            if r['identity']!=expected or not r['training_fit_only']:raise ValueError('incomplete/wrong result')
            if [t['call'] for t in r['trace']]!=bound['config']['trace_indices']:raise ValueError('trace incomplete')
        def means(values):
            keys=set(values[0])
            if any(set(v)!=keys or any(not math.isfinite(x) for x in v.values()) for v in values):
                raise ValueError('nonfinite/inconsistent metrics')
            return {k:sum(v[k] for v in values)/len(values) for k in sorted(keys)}
        traces=[]
        for j,call in enumerate(bound['config']['trace_indices']):
            traces.append({'call':call,'model_t':rows[0]['trace'][j]['model_t'],
                'metrics':{arm:means([r['trace'][j]['metrics'][arm] for r in rows]) for arm in rows[0]['trace'][j]['metrics']},
                'GT_path_terms':means([r['trace'][j]['GT_path_terms'] for r in rows]),
                'state_rms_mean':sum(r['trace'][j]['state_rms_from_GT_path'] for r in rows)/32})
        exact=sum(r['rollout_reproduction']['uint8_exact'] for r in rows);all_match &= exact==32
        summary[str(step)]={'count':32,'exact_rollout_reproductions':exact,'trace':traces,
            'metrics':{arm:means([r['metrics'][arm] for r in rows]) for arm in rows[0]['metrics']}}
    result={'status':'diagnostic_complete','receipt_sha256':c.sha(output/'receipt.json'),'summary':summary,
        'training_fit_only':True,'training_updates':0,'GT_path_is_deployable':False,'all_rollouts_uint8_exact':all_match,
        'development_or_test_pixels_accessed':False,'automatic_follow_on':False,'full_tensor_audit_performed':False,
        'training_exposure':bound['training_exposure'],
        'interpretation':'GT-assisted endpoint quality is not deployment success. Differences diagnose, not prove a unique causal root. Inspect numerical/protocol controls before attributing a gap to rollout states.'}
    c.write_new(output/'result.json',result)
    print('RAIN_PREDICTION_GAP_RESULT '+json.dumps(result,sort_keys=True),flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('phase',choices=['preflight','prepare','hardware','runtime','run','summarize','publish','wait'])
    ap.add_argument('--event',choices=['boot','prepared','ready','done','verified','failed'])
    ap.add_argument('--run-id',default='trainprobe32_steps400_1500_v1')
    ap.add_argument('--launch-id');ap.add_argument('--node-rank',type=int,choices=range(4),default=0)
    args=ap.parse_args();cfg=config();output=output_path(args.run_id)
    if str(ROOT)!=cfg['active_root']:raise ValueError('execute only in migrated GPU workspace')
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
        if shutil.disk_usage(ROOT).free<cfg['minimum_free_disk_gib']*1024**3:raise ValueError('need 10GiB free; do not delete history')
        bound=make_receipt()
        if args.phase=='prepare':c.write_new(output/'receipt.json',bound)
        print('RAIN_PREDICTION_GAP_PREFLIGHT_PASS pixels_read=false GPU_used=false',flush=True);return
    if args.phase=='hardware':print(json.dumps(parent.hopper.hardware(True)));return
    bound,receipt,source=load_receipt(output)
    if args.phase=='runtime':runtime(output,receipt,args.node_rank,True);return
    if args.phase=='summarize':
        if args.node_rank!=0 or not cluster.check(output,args.launch_id,'done'):raise ValueError('four-node done required')
        parent.check_runtimes(output);summarize(output,bound);return
    import torch
    rank,local=c.worker_rank(args.node_rank,int(os.environ['LOCAL_RANK']),int(os.environ['WORLD_SIZE']),
                             int(os.environ['LOCAL_WORLD_SIZE']),int(os.environ['RANK']))
    runtime(output,receipt,args.node_rank,False);parent.check_runtimes(output)
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
    for step in cfg['checkpoints']:
        load_weights(model,source,bound,step)
        for index in logic.assigned_indices(rank):
            diagnose_one(output,bound,receipt,source,model,ae,prompt,metric,index,step,device)
        gc.collect();torch.cuda.empty_cache()
    print(f'RAIN_PREDICTION_GAP_RANK_DONE rank={rank}',flush=True)


if __name__=='__main__':main()
