#!/usr/bin/env python3
"""Official600 unprocessed-input baseline: metric evaluation only, no FLUX rollout."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# Only identifies the immutable four-node parent; current execution is six workers.
os.environ['FOUNDIR_OFFICIAL_NNODES'] = '4'
from smoke import official_vlm_v4_metric_identity_fix as fix
c, x, require = fix.c, fix.x, fix.require
OUTPUT = ROOT/'checkpoints/foundir_fullstack_official_input_baseline_3x2_v2'
REPORT = ROOT/'reports/foundir_fullstack_official_input_baseline_3x2_v2_2026-11-06.json'
SOURCES = ['smoke/audit_foundir_fullstack_official_baseline_3x2_v2.py',
           'smoke/run_foundir_fullstack_official_baseline_3x2_v2.sh']

NODES = 3
WORLD = NODES * 2


def assigned(rows, rank):
    require(type(rank) is int and 0 <= rank < WORLD, 'invalid six-worker rank')
    return rows[rank::WORLD]


def prepare():
    parent = fix.verify()
    terminal = ROOT/x.REPORT
    value = {'official_receipt_sha256':c.sha(parent/'receipt.json'),
             'parent_terminal_sha256':c.sha(terminal),
             'metricfix_binding_sha256':c.sha(parent/'metric_identity_fix.json'),
             'sources':{name:c.sha(ROOT/name) for name in SOURCES},
             'world':WORLD,'nodes':NODES,'records':600,'candidate':'unprocessed_LQ',
             'training':False,'FLUX_inference':False}
    c.write_new(OUTPUT/'receipt.json',value)
    print('FULLSTACK_OFFICIAL_BASELINE_PREPARED',flush=True)


def bound():
    parent = fix.verify(); b = c.read(OUTPUT/'receipt.json')
    require(b['official_receipt_sha256']==c.sha(parent/'receipt.json') and
            b['metricfix_binding_sha256']==c.sha(parent/'metric_identity_fix.json') and
            b['parent_terminal_sha256']==c.sha(ROOT/x.REPORT),'baseline parent changed')
    require(b['sources']=={name:c.sha(ROOT/name) for name in SOURCES},'baseline source changed')
    require(b['world']==WORLD and b['nodes']==NODES, 'baseline topology changed')
    original = c.read(parent/'receipt.json')
    require(len(original['rows'])==600 and len({r['record_id'] for r in original['rows']})==600, 'official rows incomplete or duplicated')
    return original


def pixels(row, side, device):
    import numpy as np
    import torch
    from PIL import Image
    raw = (ROOT/row[side+'_path']).read_bytes()
    require(hashlib.sha256(raw).hexdigest()==row[side+'_sha256'],'official pixels changed')
    with Image.open(io.BytesIO(raw)) as im:
        require(im.size==(int(row['width']),int(row['height'])),'official size changed')
        array = np.array(im.convert('RGB'),copy=True)
    return torch.from_numpy(array).permute(2,0,1).unsqueeze(0).to(device).float()/255


def run(node):
    require(type(node) is int and node in range(NODES), 'invalid new node')
    import torch
    b = bound(); local = int(os.environ['LOCAL_RANK']); rank = 2*node+local
    require(local in (0,1) and int(os.environ['LOCAL_WORLD_SIZE'])==2,'two local workers required')
    receipt_sha = c.sha(OUTPUT/'receipt.json')
    rows = assigned(b['rows'],rank)
    marker = OUTPUT/f'worker{rank}.json'
    if marker.exists():
        completed = c.read(marker)
        require(completed['receipt_sha256']==receipt_sha and completed['rank']==rank and
                completed['records']==[r['record_id'] for r in rows],'completed worker identity changed')
        for row in rows:
            prior = c.read(OUTPUT/'records'/(row['record_id']+'.json'))
            require(prior['receipt_sha256']==receipt_sha and prior['row']==row and prior['rank']==rank,
                    'completed worker record identity changed')
        print(f'FULLSTACK_OFFICIAL_BASELINE_WORKER_REUSED rank={rank}',flush=True)
        return
    torch.cuda.set_device(local); device = torch.device('cuda',local)
    models,identity = fix.checked_metric_models(device,b['metric_identity'])
    start = time.monotonic()
    for number,row in enumerate(rows):
        target = OUTPUT/'records'/(row['record_id']+'.json')
        if target.exists():
            prior = c.read(target)
            require(prior['receipt_sha256']==receipt_sha and prior['row']==row and prior['rank']==rank,'baseline resume identity changed')
            continue
        lq,gt = pixels(row,'lq',device),pixels(row,'gt',device)
        with torch.no_grad(): scores = {name:float(model(lq,gt).item()) for name,model in models.items()}
        require(all(__import__('math').isfinite(v) for v in scores.values()),'nonfinite baseline score')
        c.write_new(target,{'receipt_sha256':receipt_sha,'row':row,'rank':rank,'metrics':scores})
        del lq,gt
        if number%10==0: print(f'FULLSTACK_OFFICIAL_BASELINE rank={rank} completed={number+1}/{len(rows)}',flush=True)
    c.write_new(OUTPUT/f'worker{rank}.json',{'receipt_sha256':receipt_sha,'rank':rank,'records':[r['record_id'] for r in rows],
                 'seconds':time.monotonic()-start,'peak_allocated_gib':torch.cuda.max_memory_allocated()/1024**3})
    print(f'FULLSTACK_OFFICIAL_BASELINE_WORKER_COMPLETE rank={rank}',flush=True)


def finish():
    b = bound();receipt_sha=c.sha(OUTPUT/'receipt.json');records=[]
    for rank in range(WORLD):
        rows=assigned(b['rows'],rank);marker=c.read(OUTPUT/f'worker{rank}.json')
        require(marker['receipt_sha256']==receipt_sha and marker['rank']==rank and marker['records']==[r['record_id'] for r in rows],'baseline worker incomplete')
        for row in rows:
            value=c.read(OUTPUT/'records'/(row['record_id']+'.json'))
            require(value['row']==row and value['receipt_sha256']==receipt_sha and value['rank']==rank,'baseline record changed')
            records.append(value)
    summary={}
    for category in sorted({r['row']['category'] for r in records}):
        rows=[r for r in records if r['row']['category']==category]
        summary[category]={'count':len(rows),**{metric:sum(r['metrics'][metric] for r in rows)/len(rows) for metric in ('psnr','lpips','ssim')}}
    c.write_new(REPORT,{'status':'official_input_baseline_complete','receipt_sha256':receipt_sha,
                       'categories':summary,'records':600,'FLUX_inference':False,'training':False})
    print('FULLSTACK_OFFICIAL_BASELINE_VERIFIED '+json.dumps(summary),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=('prepare','run','finish'))
    parser.add_argument('--node',type=int,choices=range(NODES),default=0);args=parser.parse_args()
    require(str(ROOT)==fix.producer.ACTIVE,'destination workspace only')
    if args.phase=='run':run(args.node)
    else:globals()[args.phase]()
