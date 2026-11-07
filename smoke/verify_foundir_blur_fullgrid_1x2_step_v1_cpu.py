#!/usr/bin/env python3
"""One actual saved continuation checkpoint; no images or CUDA."""
import argparse
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from smoke import train_foundir_blur_fullgrid_1x2_v1 as t
from smoke.existing_flux_replay_contract_v1 import sha,write_new


def run(step):
    started=time.monotonic();t.bound()
    value,marker=t.state.load_step(t.OUT,step,sha(t.OUT/'receipt.json'))
    migration_checked = False
    if step == 3688:
        import torch
        parent,_ = t.previous.state.load_step(t.previous.OUT,t.START,sha(t.previous.OUT/'receipt.json'))
        initial,_ = t.state.load_step(t.OUT,t.START,sha(t.OUT/'receipt.json'))
        def equal(a,b):
            if isinstance(a,torch.Tensor): return isinstance(b,torch.Tensor) and torch.equal(a,b)
            if isinstance(a,dict): return isinstance(b,dict) and a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
            if isinstance(a,(tuple,list)): return type(a)==type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
            return a==b
        for key in ('model','optimizer','parameter_order','history','next_global_slot','parameter_hash'):
            if not equal(initial[key],parent[key]): raise ValueError('migration changed '+key)
        for rank,old_rank in enumerate(t.state.PARENT_RNG_RANKS):
            if not equal(initial['rank_rng'][rank],parent['rank_rng'][old_rank]): raise ValueError('RNG migration differs')
        migration_checked = True
    result={'parent_model_Adam_history_sampler_RNG_mapping_exact':migration_checked,'step':step,'checkpoint':marker,'actual_SHA_Adam_RNG_sampler_verified':True,
            'Adam_states':len(value['optimizer']['state']),'rank_RNG_count':len(value['rank_rng']),
            'next_global_slot':value['next_global_slot'],'same_world_state_preserved':False,'model_Adam_sampler_preserved':True,'bitwise_four_rank_continuation':False, 'new_data_views':True,
            'official_Blur_pass':False,'seconds':time.monotonic()-started,'exit_code':0,'GPU_hours':0}
    write_new(ROOT/f'reports/foundir_blur_fullgrid_1x2_v1_step{step}_state_2026-11-07.json',result)
    print('BLUR_FULLGRID_1X2_CHECKPOINT_VERIFIED '+json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('step',type=int,choices=[3688,3712,5632,7552,9472]);run(p.parse_args().step)
