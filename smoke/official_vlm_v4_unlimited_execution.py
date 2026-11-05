#!/usr/bin/env python3
"""User-approved no-deadline launch overlay; immutable scientific producer unchanged."""
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from smoke import evaluate_foundir_official_vlm_router_v4 as producer
c,x,require=producer.c,producer.x,producer.require
PLAN_SHA='5f3823ed90320639bf804e7e29d0a891a5e0edb92e8b2761b51ac55fd90ea53d'
RECEIPT_SHA='98eb74a6534eb234e75a164a2460f43b0a2f6beaeb08a686e1812f85a7615157'
AUTH='reports/advisor_governance/foundir_official_vlm_v4_unlimited_authorization_2026-11-05.json'
SOURCES=[AUTH,'smoke/official_vlm_v4_unlimited_execution.py',
         'smoke/run_foundir_official_vlm_v4_unlimited.sh',
         'smoke/start_foundir_official_vlm_v4_unlimited_tmux.sh']


def scope():
    require(x.NODES==4 and x.WORLD==8 and x.RUN=='official_vlm_v4_frozen4800_v1_4x2','only completed four-node run authorized')
    producer.bound()  # Original bound code/config/manifest checks remain intact.
    out=ROOT/x.OUTPUT
    require(c.sha(out/'receipt.json')==RECEIPT_SHA and c.sha(out/'render_plan.json')==PLAN_SHA,'approved plan/producer changed')
    p=c.read(out/'render_plan.json')
    require(p['world']==8 and p['nodes']==4 and len(p['decisions'])==600 and
            p['unique_native_tile_generations']==21828 and sum(p['tiles_by_rank'])==21828,'wrong official workload')
    auth=c.read(ROOT/AUTH)
    require(auth['status']=='authorized' and auth['user_request']=='no need for max hour limit, just execute and start the exp' and
            auth['render_plan_sha256']==PLAN_SHA and auth['maximum_wall_hours'] is None and
            auth['maximum_GPU_hours'] is None and auth['run_id']==x.RUN,'explicit unlimited authorization missing')
    return out


def budget():
    return {'render_plan_sha256':PLAN_SHA,'approved_render_GPU_hours':None,'world':8,'nodes':4,
            'cap_seconds_per_node':None,'maximum_wall_hours':None,'time_limit_disabled':True,
            'approval_note':'User: no need for max hour limit, just execute and start the exp',
            'authority':'explicit_user_approval_recorded_by_CPU_agent','authorization_sha256':c.sha(ROOT/AUTH),
            'scope':'existing_frozen600_render_and_finish_only','route_GPU_hours_counted_separately':True}


def authorize():
    out=scope()
    # Refuse to overwrite any conflicting previous grant; all historical records survive.
    c.write_new(out/'render_budget.json',budget())
    value={'receipt_sha256':RECEIPT_SHA,'render_plan_sha256':PLAN_SHA,
           'render_budget_sha256':c.sha(out/'render_budget.json'),
           'execution_sources':{p:c.sha(ROOT/p) for p in SOURCES},
           'producer_sources_unchanged':True,'automatic_retry':False,'no_elapsed_time_kill':True}
    c.write_new(out/'unlimited_execution.json',value)
    verify()
    print('OFFICIAL_VLM_V4_UNLIMITED_AUTHORIZED existing_caches_reused=true no_new_route=true',flush=True)


def verify():
    out=scope();a=c.read(out/'unlimited_execution.json')
    require(c.read(out/'render_budget.json')==budget(),'unlimited grant changed')
    require(a=={'receipt_sha256':RECEIPT_SHA,'render_plan_sha256':PLAN_SHA,
                'render_budget_sha256':c.sha(out/'render_budget.json'),
                'execution_sources':{p:c.sha(ROOT/p) for p in SOURCES},
                'producer_sources_unchanged':True,'automatic_retry':False,'no_elapsed_time_kill':True},'execution overlay changed')
    return out


def account_check(key):
    out=verify()
    require(key=='finish' or key in {f'render_node{n}' for n in range(4)},'render/finish only')
    starts=list((out/'attempts').glob(key+'_*.started.json'));ends=list((out/'attempts').glob(key+'_*.ended.json'))
    require({p.name.removesuffix('.started.json') for p in starts}=={p.name.removesuffix('.ended.json') for p in ends},
            'interrupted accounting: preserve progress and reconcile elapsed time before resume')
    gpus=0 if key=='finish' else 2
    values=[c.read(p) for p in ends]
    require(all(type(r['seconds']) is int and r['seconds']>=0 and type(r['status']) is int and
                r['gpus']==gpus and r['used_gpu_seconds']==r['seconds']*gpus for r in values),'invalid historical attempt')
    print(f'OFFICIAL_VLM_V4_UNLIMITED_ACCOUNT key={key} prior_seconds={sum(r["seconds"] for r in values)} no_cap=true',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=('authorize','verify','account'))
    p.add_argument('--key');args=p.parse_args()
    require(str(ROOT)==producer.ACTIVE,'destination workspace only')
    if args.phase=='account':account_check(args.key)
    else:globals()[args.phase]()


if __name__=='__main__':main()
