#!/usr/bin/env python3
"""One cached training sample, full three-update backward; no optimizer update.

A single-GPU capacity diagnostic, not a qualification or batch-size change.
Reads the completed native candidate cache. Never resumes the failed model.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'third_party/flux2/src'))
from smoke import flux_native_energy_contract_v1 as contract


def validate_cache_receipt(previous, current):
    unsigned=dict(previous);claimed=unsigned.pop('protocol_sha256')
    if hashlib.sha256(json.dumps(unsigned,sort_keys=True).encode()).hexdigest()!=claimed:
        raise RuntimeError('cache producer receipt is corrupt')
    for key in ('spec','selected','assets','base_spec','plan','prompt_root','config_sha256'):
        if previous[key]!=current[key]:raise RuntimeError('cache producer differs in '+key)


def run(run_id, cache_run):
    import torch
    from flux2.sampling import get_schedule
    from lowlight_restoration.flux_native_energy import (
        FluxNativeEnergy,unroll,direct_quality_loss,pareto_ranking_loss,energy_attention_mode,
    )
    from smoke import flux_native_energy_runtime_v1 as runtime
    from smoke.train_foundir_flux_native_energy_v1 import sample_index,action_pair
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise RuntimeError('expose exactly one idle GPU for this capacity probe')
    device=torch.device('cuda',0);torch.cuda.set_device(device)
    props=torch.cuda.get_device_properties(device);free,total=torch.cuda.mem_get_info(device)
    if not any(k in props.name for k in ('H100','H200')) or total<75*1024**3 or total-free>2*1024**3:
        raise RuntimeError('requires one idle H100/H200 with at least 75 GiB')
    # Validate names before joining paths; create a fresh diagnostic directory.
    contract.run_path(run_id);cache=contract.run_path(cache_run)
    output=ROOT/'checkpoints/foundir_flux_native_energy_memory_v1'/run_id
    output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();rows=[]
    result=dict(status='running',diagnostic_only=True,optimizer_updates=0,
        training_steps_completed=0,qualification_pass=False,run_id=run_id,
        gpu_name=props.name,gpu_uuid=str(props.uuid),total_gib=total/1024**3,
        torch=torch.__version__,cuda=torch.version.cuda,python=sys.version,
        cache_run=cache_run,stages=rows,recipe_changed=False,
        numerical_implementation='full_softmax_query_tiles_128_checkpointed_vjp')
    def observe(stage):
        torch.cuda.synchronize(device)
        row=dict(stage=stage,seconds=time.monotonic()-started,
            allocated_gib=torch.cuda.memory_allocated(device)/1024**3,
            reserved_gib=torch.cuda.memory_reserved(device)/1024**3,
            interval_peak_allocated_gib=torch.cuda.max_memory_allocated(device)/1024**3)
        rows.append(row)
        with (output/'memory.jsonl').open('a') as handle:
            handle.write(json.dumps(row)+'\n');handle.flush();os.fsync(handle.fileno())
        print('NATIVE_ENERGY_MEMORY '+json.dumps(row),flush=True)
        torch.cuda.reset_peak_memory_stats(device)
    try:
        current=contract.preflight(6)
        previous=json.loads((cache/'preflight.json').read_text())
        validate_cache_receipt(previous,current)
        result['cache_producer_protocol_sha256']=previous['protocol_sha256']
        result['consumer_protocol_sha256']=current['protocol_sha256']
        result['sources']=dict(current['sources'])
        for name in ('smoke/profile_foundir_flux_native_energy_memory_v1.py',
                     'smoke/run_foundir_flux_native_energy_memory_v1_gpu.sh'):
            result['sources'][name]=contract.sha(ROOT/name)
        spec=current['spec'];index=sample_index(0,spec['seed']);result['training_index']=index
        runtime.cache_metadata(cache,previous,index)
        observe('assets_verified')
        flow,ae,perceptual=runtime.load_models(current,device)
        observe('frozen_models_loaded')
        record=runtime.load_record(cache,previous,index,device)
        contexts=runtime.prompt(current,'noise',device)
        torch.manual_seed(spec['seed']);random.seed(spec['seed'])
        model=FluxNativeEnergy(flow,spec['energy_rank'],spec['energy_alpha'],expected_adapters=50)
        result['architecture']=model.architecture_receipt()
        result['tokens']=dict(candidate=record['initial'].shape[1],reference=record['reference'].shape[1],
                              proposal=record['actions'].shape[1],text=contexts['cfg_context'].shape[1]+1)
        observe('energy_and_record_loaded')
        t=get_schedule(50,record['initial'].shape[1])[48]
        args=runtime.energy_arguments(record,contexts,t)
        # Same real 4B/BF16 state: compare first derivatives without constructing
        # the old large higher-order graph. This is numerical validation only.
        parity=[]
        for tiled in (False,True):
            z=record['initial'].float().detach().requires_grad_(True)
            with energy_attention_mode(tiled):
                score=model(z,**args)
                gradient=torch.autograd.grad(score.sum(),z,create_graph=False)[0]
            parity.append((score.detach(),gradient.detach()))
            observe('tiled_first_gradient' if tiled else 'reference_first_gradient')
        ref_score,ref_grad=parity[0];new_score,new_grad=parity[1]
        relative_error=float((new_grad-ref_grad).norm()/ref_grad.norm().clamp_min(1e-12))
        result['attention_parity']=dict(score_abs_error=float((new_score-ref_score).abs().max()),
            gradient_relative_l2_error=relative_error,gradient_relative_l2_limit=.01,
            score_atol=1e-5,score_rtol=.01)
        torch.testing.assert_close(new_score,ref_score,atol=1e-5,rtol=.01)
        if not bool(ref_grad.isfinite().all() and new_grad.isfinite().all()) or relative_error>.01:
            raise RuntimeError('real BF16 attention gradient parity exceeded fixed 1% tolerance')
        print('NATIVE_ENERGY_ATTENTION_PARITY '+json.dumps(result['attention_parity']),flush=True)
        del parity,z,score,gradient,ref_score,ref_grad,new_score,new_grad
        states,scores=unroll(lambda z:model(z,**args),record['initial'].float(),training=True,
            updates=spec['updates'],step_size=spec['step_size'],radius=spec['latent_rms_radius'],
            observe=observe)
        observe('unroll_complete')
        images=[]
        for i,z in enumerate(states[1:],1):
            images.append(runtime.continuous_output(record,runtime.render_tail(model,ae,record,contexts,z)))
            observe(f'render_{i}_done')
        direct,diagnostics=direct_quality_loss(images,record['target'],record['proposal'],perceptual)
        observe('direct_quality_loss')
        with torch.no_grad():
            costs=torch.stack([runtime.quality(record['proposal'],record['target'],perceptual)]+
                              [runtime.quality(image,record['target'],perceptual) for image in images])
        trajectory=pareto_ranking_loss(torch.cat(scores),costs,margin=spec['ranking_margin'])
        pair=action_pair(0);endpoint=runtime.energy_arguments(record,contexts,0.)
        action_scores=torch.cat([model(record['actions'][k:k+1],**endpoint) for k in pair])
        action=pareto_ranking_loss(action_scores,record['costs'][list(pair)],margin=spec['ranking_margin'])
        loss=direct+spec['ranking_weight']*(trajectory+action)
        if not bool(loss.isfinite()):raise RuntimeError('nonfinite loss')
        observe('all_losses_before_backward')
        # One sample's contribution to the declared global mean of eight. No
        # Adam step: seven samples are absent, so this is NOT a training step.
        (loss/spec['global_batch']).backward()
        observe('full_backward_done')
        grads={n:p.grad for n,p in model.named_parameters() if p.requires_grad}
        if any(g is None or not bool(g.isfinite().all()) for g in grads.values()):
            raise RuntimeError('missing or nonfinite trainable gradient')
        lora_norm=sum(float(g.square().sum()) for n,g in grads.items() if '.energy_b.' in n)**.5
        if lora_norm<=0:raise RuntimeError('zero energy LoRA gradient')
        model.architecture_receipt()
        result.update(status='single_sample_full_backward_pass',loss=float(loss.detach()),
            energy_lora_gradient_norm=lora_norm,training_metrics=diagnostics,
            max_stage_peak_allocated_gib=max(r['interval_peak_allocated_gib'] for r in rows),
            gradient_finite=True,full_three_updates=True)
        print('NATIVE_ENERGY_MEMORY_BACKWARD_PASS; not a qualification or optimizer step',flush=True)
    except BaseException as error:
        result.update(status='failed',error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
        if isinstance(error,torch.OutOfMemoryError):
            result['oom_allocated_gib']=torch.cuda.memory_allocated(device)/1024**3
            result['oom_peak_allocated_gib']=torch.cuda.max_memory_allocated(device)/1024**3
            (output/'cuda_memory_summary.txt').write_text(torch.cuda.memory_summary(device=device))
        raise
    finally:
        result['wall_seconds']=time.monotonic()-started
        runtime.atomic_json(output/'result.json',result)
        print('NATIVE_ENERGY_MEMORY_RESULT '+str(output/'result.json'),flush=True)


def interrupted(signum, frame):
    raise InterruptedError('memory diagnostic received signal '+str(signum))


if __name__=='__main__':
    signal.signal(signal.SIGTERM,interrupted)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--cache-run',default='qualification_3x2_v1')
    options=parser.parse_args();run(options.run_id,options.cache_run)
