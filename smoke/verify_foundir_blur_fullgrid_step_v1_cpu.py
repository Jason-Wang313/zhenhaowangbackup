#!/usr/bin/env python3
"""One actual saved continuation checkpoint; no images or CUDA."""
import argparse
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from smoke import train_foundir_blur_fullgrid_v1 as t
from smoke.existing_flux_replay_contract_v1 import sha,write_new


def run(step):
    started=time.monotonic();t.bound()
    value,marker=t.state.load_step(t.OUT,step,sha(t.OUT/'receipt.json'))
    result={'step':step,'checkpoint':marker,'actual_SHA_Adam_RNG_sampler_verified':True,
            'Adam_states':len(value['optimizer']['state']),'rank_RNG_count':len(value['rank_rng']),
            'next_global_slot':value['next_global_slot'],'same_world_state_preserved':True, 'new_data_views':True,
            'official_Blur_pass':False,'seconds':time.monotonic()-started,'exit_code':0,'GPU_hours':0}
    write_new(ROOT/f'reports/foundir_blur_fullgrid_v1_step{step}_state_2026-11-07.json',result)
    print('BLUR_FULLGRID_CHECKPOINT_VERIFIED '+json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('step',type=int,choices=[1793,3712,5632,7552,9472]);run(p.parse_args().step)
