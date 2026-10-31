"""FLUX-only candidate construction and real frozen-generator rollout.

Only the native VAE and the fixed LPIPS metric contain convolution. No old
energy, Restormer, refiner, distilled output, or CNN-generated candidate is read.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from smoke import flux_native_energy_contract_v1 as contract
from lowlight_restoration.lora import inject_lora, load_lora_checkpoint
from lowlight_restoration.scalar_residual import quantized_scalar_residual
from lowlight_restoration.flux2_trajectory_endpoint import _cfg_velocity


def atomic_write(path, writer, *, replace=False):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix='.'+path.name+'.',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as handle:
            writer(handle); handle.flush(); os.fsync(handle.fileno())
        if replace: os.replace(tmp,path)
        else:
            os.link(tmp,path); os.unlink(tmp)
        fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
        try: os.fsync(fd)
        finally: os.close(fd)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def atomic_json(path, data, *, replace=False):
    encoded=(json.dumps(data,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
    atomic_write(path,lambda h:h.write(encoded),replace=replace)


def load_models(receipt, device):
    from flux2.util import load_flow_model,load_ae
    os.environ['KLEIN_4B_BASE_MODEL_PATH']=str(contract.path(receipt['base_spec']['flow']))
    os.environ['AE_MODEL_PATH']=str(contract.path(receipt['base_spec']['ae']))
    os.environ['TORCH_HOME']=str(contract.ROOT/'models/torch')
    flow=load_flow_model('flux.2-klein-base-4b',device=device)
    names=inject_lora(flow,['double_blocks.*.img_attn.qkv','double_blocks.*.img_attn.proj',
        'single_blocks.*.linear1','single_blocks.*.linear2'],32,32,0.)
    if len(names)!=50 or flow.hidden_size!=3072 or len(flow.double_blocks)!=5 or len(flow.single_blocks)!=20:
        raise RuntimeError('not the bound Klein 4B architecture')
    load_lora_checkpoint(flow,contract.path(receipt['plan']['baseline_checkpoint']))
    flow.eval().requires_grad_(False)
    ae=load_ae('flux.2-klein-base-4b',device=device).eval().requires_grad_(False)
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    perceptual=LearnedPerceptualImagePatchSimilarity(net_type='alex').to(device).eval().requires_grad_(False).net
    return flow,ae,perceptual


def prompt(receipt, task, device):
    from safetensors.torch import load_file
    return load_file(str(contract.path(receipt['prompt_root'])/f'prompt_{task}.safetensors'),device=str(device))


def deployed(observation, raw, alpha=.3):
    _,raw_quantized=quantized_scalar_residual(observation,raw,1.)
    return quantized_scalar_residual(observation,raw_quantized,alpha)


def quality(image,target,perceptual):
    mse=((image-target)/2).square().mean()
    lpips=perceptual(image,target).mean()
    return torch.stack((mse,lpips))


def metrics(image,target,perceptual):
    with torch.no_grad():
        costs=quality(image,target,perceptual)
        return dict(psnr=float(-10*costs[0].clamp_min(1e-12).log10()),lpips=float(costs[1]))


def read_crop(identity):
    import numpy as np
    from PIL import Image
    row=identity['row']
    if identity['role']!='train' or row['split']!='train' or row['category']!='05Noise':
        raise RuntimeError('protected/nontraining pixel access refused')
    x,y=identity['crop_xy']; result={}
    for key,field in (('observation','lq'),('target','gt')):
        filename=contract.bound(row[field+'_path'],row[field+'_sha256'])
        with Image.open(filename) as image:
            if image.size!=(int(row['width']),int(row['height'])):
                raise ValueError('image geometry changed')
            result[key]=torch.from_numpy(np.array(image.convert('RGB').crop((x,y,x+512,y+512)),copy=True)).permute(2,0,1)
    return result


def encode(ae, image):
    from flux2.sampling import batched_prc_img
    with torch.no_grad():
        return batched_prc_img(ae.encode(image).to(torch.bfloat16))


def decode(ae, state, ids):
    from flux2.sampling import scatter_ids
    def call(value):
        latent=torch.cat(scatter_ids(value,ids)).squeeze(2)
        return ae.decode(latent).float().clamp(-1,1)
    return checkpoint(call,state,use_reentrant=False) if torch.is_grad_enabled() and state.requires_grad else call(state)


def velocity(model, state, ids, reference, reference_ids, context, context_ids, time):
    value=state.to(torch.bfloat16)
    return _cfg_velocity(model,torch.cat((value,value)),torch.cat((ids,ids)),context,context_ids,
        torch.cat((reference,reference)),torch.cat((reference_ids,reference_ids)),time,1.).chunk(2)[0]


@torch.no_grad()
def restore_capture(flow,ae,image,contexts,seed,*,check_native=False):
    from flux2.sampling import batched_prc_img,denoise_cfg,get_schedule
    latent=ae.encode(image).to(torch.bfloat16)
    reference,reference_ids=batched_prc_img(latent)
    reference_ids=reference_ids.clone();reference_ids[...,0]=10
    noise=torch.randn(latent.shape,generator=torch.Generator(device=image.device).manual_seed(seed),
                      device=image.device,dtype=torch.bfloat16)
    initial,ids=batched_prc_img(noise);state=initial.clone()
    schedule=get_schedule(50,state.shape[1]);saved=None
    for index,(current,following) in enumerate(zip(schedule[:-1],schedule[1:])):
        if index==48:saved=state.clone()
        state=state+(following-current)*velocity(flow,state,ids,reference,reference_ids,
            contexts['cfg_context'],contexts['cfg_ids'],current)
    if check_native:
        native=denoise_cfg(flow,initial,ids,contexts['cfg_context'],contexts['cfg_ids'],schedule,1.,
            img_cond_seq=reference,img_cond_seq_ids=reference_ids)
        if not torch.equal(native,state):
            raise RuntimeError('same-environment native entry/captured rollout mismatch')
    return decode(ae,state,ids),dict(initial=saved,ids=ids,reference=reference,reference_ids=reference_ids)


def cache_metadata(output,receipt,index):
    info=json.loads((output/'cache'/f'{index:03d}.json').read_text())
    if info['identity']!=receipt['selected'][index] or info['protocol_sha256']!=receipt['protocol_sha256']:
        raise RuntimeError('candidate cache identity changed')
    expected=f'{index:03d}.safetensors'
    if info['file']!=expected or contract.sha(output/'cache'/expected)!=info['sha256']:
        raise RuntimeError('candidate cache file/hash mismatch')
    if info['action_names']!=receipt['spec']['action_names']:
        raise RuntimeError('action identity mismatch')
    return info


def build_cache(output,receipt,rank,world,flow,ae,perceptual,device):
    from safetensors.torch import save,load_file
    directory=output/'cache';directory.mkdir(exist_ok=True)
    contexts=prompt(receipt,'noise',device)
    assigned=list(range(24))[rank::world]
    pending=[]
    for index in assigned:
        if (directory/f'{index:03d}.json').exists():cache_metadata(output,receipt,index)
        else:pending.append(index)
    # Offline, no live autograd graph: generate true wrong-FLUX-expert outputs
    # first, then restore the exact original Noise weights before any training.
    if pending:
        load_lora_checkpoint(flow,contract.path(receipt['base_spec']['wrong_expert']))
        wrong_context=prompt(receipt,'haze',device)
        for index in pending:
            name=directory/f'{index:03d}_wrong.safetensors'
            meta_path=directory/f'{index:03d}_wrong.json'
            if meta_path.exists():
                info=json.loads(meta_path.read_text())
                if info['protocol_sha256']!=receipt['protocol_sha256'] or contract.sha(name)!=info['sha256']:
                    raise RuntimeError('wrong-expert stage changed')
                continue
            identity=receipt['selected'][index]
            x=read_crop(identity)['observation'][None].to(device).float()/127.5-1
            raw,_=restore_capture(flow,ae,x,wrong_context,identity['seed'])
            wrong,_=deployed(x,raw,alpha=1.)
            payload=save({'wrong':wrong[0].cpu().contiguous()})
            atomic_write(name,lambda h:h.write(payload),replace=name.exists())
            atomic_json(meta_path,dict(protocol_sha256=receipt['protocol_sha256'],sha256=contract.sha(name)))
            print(f'NATIVE_ENERGY_WRONG_FLUX_READY index={index}',flush=True)
        load_lora_checkpoint(flow,contract.path(receipt['plan']['baseline_checkpoint']))
    for index in pending:
        identity=receipt['selected'][index];images=read_crop(identity)
        x=images['observation'][None].to(device).float()/127.5-1
        target=images['target'][None].to(device).float()/127.5-1
        raw,state=restore_capture(flow,ae,x,contexts,identity['seed'],check_native=index==pending[0])
        uint8,p=deployed(x,raw)
        repeated_raw,_=restore_capture(flow,ae,p,contexts,identity['seed'])
        _,repeated=deployed(p,repeated_raw)
        wrong=load_file(str(directory/f'{index:03d}_wrong.safetensors'),device=str(device))['wrong'][None].float()/127.5-1
        candidates=[p,x,target,(x+.5*(p-x)).clamp(-1,1),(x+1.5*(p-x)).clamp(-1,1),
                    F.avg_pool2d(F.pad(p,(2,2,2,2),mode='reflect'),5,stride=1),repeated,wrong]
        latents=[]; costs=[]
        with torch.no_grad():
            for image in candidates:
                _,image=quantized_scalar_residual(image,image,1.)
                tokens,_=encode(ae,image)
                latents.append(tokens[0]);costs.append(quality(image,target,perceptual))
        tensors=dict(observation=images['observation'],target=images['target'],proposal=uint8[0].cpu(),
            initial=state['initial'][0],ids=state['ids'][0],reference=state['reference'][0],
            reference_ids=state['reference_ids'][0],actions=torch.stack(latents),costs=torch.stack(costs))
        tensors={k:v.detach().cpu().contiguous() for k,v in tensors.items()}
        name=directory/f'{index:03d}.safetensors';payload=save(tensors)
        atomic_write(name,lambda h:h.write(payload),replace=name.exists())
        atomic_json(directory/f'{index:03d}.json',dict(identity=identity,file=name.name,
            sha256=contract.sha(name),protocol_sha256=receipt['protocol_sha256'],
            action_names=receipt['spec']['action_names'],zero_reference_checked=index==pending[0],
            cnn_outputs_used=False,restormer_outputs_used=False))
        print(f'NATIVE_ENERGY_CACHE_READY index={index} actions=8',flush=True)


def load_record(output,receipt,index,device):
    from safetensors.torch import load_file
    cache_metadata(output,receipt,index)
    data=load_file(str(output/'cache'/f'{index:03d}.safetensors'),device=str(device))
    for key in ('observation','target','proposal'):
        data[key]=data[key][None].float()/127.5-1
    for key in ('initial','ids','reference','reference_ids'):
        data[key]=data[key][None]
    data['descriptor']=torch.tensor([contract.descriptor(receipt['selected'][index]['row'])],device=device)
    return data


def energy_arguments(record,contexts,time):
    prop_ids=record['ids'].clone();prop_ids[...,0]=20
    # Only observables are passed. target/costs/actions are intentionally absent.
    return dict(ids=record['ids'],timestep=torch.tensor([time],device=record['initial'].device),
        reference=record['reference'],reference_ids=record['reference_ids'],
        proposal_reference=record['actions'][0:1],proposal_ids=prop_ids,
        context=contexts['cfg_context'][1:2],context_ids=contexts['cfg_ids'][1:2],descriptor=record['descriptor'])


def render_tail(model,ae,record,contexts,state):
    from flux2.sampling import get_schedule
    schedule=get_schedule(50,state.shape[1])
    for current,following in zip(schedule[48:-1],schedule[49:]):
        state=state+(following-current)*velocity(model.generation,state,record['ids'],record['reference'],
            record['reference_ids'],contexts['cfg_context'],contexts['cfg_ids'],current)
    return decode(ae,state,record['ids'])


def continuous_output(record,raw):
    return (record['observation']+.3*(raw.clamp(-1,1)-record['observation'])).clamp(-1,1)
