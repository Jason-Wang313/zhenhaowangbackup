"""Exact historical checkpoint and derivative/delta checks; never perform optimizer updates."""
from pathlib import Path
import torch
from smoke.existing_flux_replay_contract_v1 import read,sha
from smoke.blur_delta_state_v1 import parameter_hash
from lowlight_restoration.blur_delta_velocity_v1 import sample_indices


def load_exact_step(output,step,receipt_sha):
    output=Path(output);r=read(output/'steps'/f'step_{step:06d}.json');p=output/r['file']
    if p.resolve().parent!=(output/'steps').resolve() or sha(p)!=r['sha256']:raise ValueError('checkpoint path/SHA mismatch')
    v=torch.load(p,map_location='cpu',weights_only=True)
    for k in ('step','world','next_global_slot','receipt_sha256','parameter_hash'):
        if v[k]!=r[k]:raise ValueError('checkpoint receipt differs')
    if (v['step']!=step or v['world']!=2 or v['next_global_slot']!=step*8 or v['receipt_sha256']!=receipt_sha
      or len(v['history'])!=step or len(v['rank_rng'])!=2 or parameter_hash(v['model'])!=v['parameter_hash']):raise ValueError('checkpoint scope mismatch')
    if list(v['model'])!=v['parameter_order'] or any(t.dtype!=torch.float32 or not torch.isfinite(t).all() for t in v['model'].values()):raise ValueError('invalid saved parameters')
    states=v['optimizer']['state']
    if (step>0 and len(states)!=len(v['model'])) or (step==0 and states):raise ValueError('invalid Adam initialization/state count')
    for s in states.values():
        if float(s['step'])!=step or any(s[k].dtype!=torch.float32 or not torch.isfinite(s[k]).all() for k in ('exp_avg','exp_avg_sq')):raise ValueError('invalid Adam state')
    return r,v


def original_batch(rows,step):
    if len(rows)!=1024 or any(r['role']!='train' for r in rows):raise ValueError('original1024 training sources required')
    indices={str(rank):sample_indices(step,rank,count=1024) for rank in (0,1)}
    if len({i for values in indices.values() for i in values})!=8:raise ValueError('global8 duplicated')
    return {'indices_by_rank':indices,'items_by_rank':{r:[rows[i] for i in indices[r]] for r in indices}}


def tensor_dot(a,b):
    if len(a)!=len(b) or not a:raise ValueError('tensor layout differs')
    sums=[]
    for x,y in zip(a,b):
        if x.shape!=y.shape or not torch.isfinite(x).all() or not torch.isfinite(y).all():raise ValueError('invalid tensor pair')
        sums.append((x.detach().double()*y.detach().double()).sum())
    return float(torch.stack(sums).sum())


def replay_check(values,norm,expected):
    errors={k:abs(values[k]-expected[k]) for k in ('loss','mse','lpips')}
    norm_error=abs(norm-expected['gradient_norm'])/max(1.,abs(expected['gradient_norm']))
    return {'passed':all(v<=1e-7 for v in errors.values()) and norm_error<=3e-4,
      'loss_absolute_errors':errors,'normalized_gradient_norm_error':norm_error,
      'limits':{'loss_absolute':1e-7,'gradient_norm_normalized':3e-4},'actual':values,
      'actual_gradient_norm':norm,'original_logged':expected}
