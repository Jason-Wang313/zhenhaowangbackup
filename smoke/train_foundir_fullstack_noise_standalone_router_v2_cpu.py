#!/usr/bin/env python3
"""Fixed CPU binary pilot: replace the inherited binary margin, never alter experts."""
import argparse
from collections import Counter
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from smoke import train_foundir_fullstack_noise_physical_router_v1_cpu as old
from lowlight_restoration.noise_lowlight_standalone_router_v2 import StandaloneRouter
from smoke.existing_flux_replay_contract_v1 import read, sha, write_new as write
OUT = ROOT/'checkpoints/foundir_fullstack_noise_standalone_router_v2'
REPORT = ROOT/'reports/foundir_fullstack_noise_standalone_router_v2_2026-11-06.json'
SOURCES = ['smoke/train_foundir_fullstack_noise_standalone_router_v2_cpu.py',
           'lowlight_restoration/noise_lowlight_standalone_router_v2.py',
           'lowlight_restoration/noise_lowlight_physical_router_v1.py']
RECIPE = dict(steps=400, batch=64, lr=.001, weight_decay=.01, seed=2026110611,
              hidden=32, no_base_logit_in_margin=True, no_official_training=True,
              decision_threshold=0, maximum_CPU_wall_seconds=1800, automatic_deployment=False)


def admission(classification):
    return all(v['candidate_correct'] >= v['baseline_correct']
               for role in ('calibration', 'confirmation')
               for v in classification[role]['summary'].values())


def prepare():
    b = old.bound()
    features = {}
    for row in b['records']:
        path = old.OUT/'features'/(row['key']+'.json')
        v = read(path)
        old.require(v['lq_sha256'] == row['lq_sha256'] and v['GT_input'] is False, 'cached physical feature changed')
        features[row['key']] = sha(path)
    write(OUT/'receipt.json', dict(recipe=RECIPE, sources={s:sha(ROOT/s) for s in SOURCES},
          parent_receipt_sha256=sha(old.OUT/'receipt.json'), features=features,
          source_counts=dict(Counter(r['role']+'_'+r['task'] for r in b['records']))))
    print('STANDALONE_ROUTER_PREPARED', flush=True)


def bound():
    b=read(OUT/'receipt.json'); parent=old.bound()
    old.require(b['recipe']==RECIPE and b['parent_receipt_sha256']==sha(old.OUT/'receipt.json'),'router recipe/parent changed')
    for s,h in b['sources'].items(): old.require(sha(ROOT/s)==h,'router source changed: '+s)
    values={}
    for row in parent['records']:
        p=old.OUT/'features'/(row['key']+'.json');old.require(sha(p)==b['features'][row['key']],'feature changed')
        values[row['key']]=read(p)['values']
    return parent['records'],values


def run():
    start=time.monotonic();old.require(not REPORT.exists(),'preserve completed pilot')
    records,values=bound();fit=[r for r in records if r['role']=='fit']
    x=torch.tensor([values[r['key']] for r in fit]);z=torch.tensor([r['base_logits'] for r in fit])
    y=torch.tensor([float(r['task']=='noise') for r in fit]);torch.manual_seed(RECIPE['seed'])
    model=StandaloneRouter(x.mean(0),x.std(0,unbiased=False));opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    groups=[torch.where(y==label)[0] for label in (0,1)];history=[]
    for step in range(RECIPE['steps']):
        old.require(time.monotonic()-start<1800,'fixed CPU cap')
        ids=torch.cat([g[torch.randint(len(g),(32,))] for g in groups]);margin,_,_=model(z[ids],x[ids])
        loss=torch.nn.functional.binary_cross_entropy_with_logits(margin,y[ids]);opt.zero_grad();loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();history.append(float(loss.detach()))
    model.eval().requires_grad_(False);results={}
    for role in ('fit','calibration','confirmation'):
        rows=[r for r in records if r['role']==role]
        results[role]=old.evaluate(model,rows,torch.tensor([values[r['key']] for r in rows]))
    checkpoint=OUT/'fixed_step400.pt';old.require(not checkpoint.exists(),'preserve checkpoint')
    torch.save(dict(model=model.state_dict(),recipe=RECIPE,receipt_sha256=sha(OUT/'receipt.json')),checkpoint)
    result=dict(status='CPU_pilot_complete_not_deployed',recipe=RECIPE,classification=results,history=history,
                validation_admission=admission(results),parameters=sum(p.numel() for p in model.parameters()),
                checkpoint_sha256=sha(checkpoint),receipt_sha256=sha(OUT/'receipt.json'),
                seconds=time.monotonic()-start,exit_code=0,GPU_hours=0,official_evaluated=False,
                frozen_router_legacy_source_exposure_not_erased=True,automatic_deployment=False)
    write(REPORT,result)
    print('STANDALONE_ROUTER_END '+str({k:result[k] for k in ('validation_admission','seconds','parameters')}),flush=True)
    print({r:v['summary'] for r,v in results.items()},flush=True)


if __name__=='__main__':
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','run']);globals()[p.parse_args().phase]()
