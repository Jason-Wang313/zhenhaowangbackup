#!/usr/bin/env python3
"""Supplement failed deterministic controls; never rerun successful parent fits."""
from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from smoke import audit_foundir_blur_only_zimage_repro_batch as batch
from lowlight_restoration.zimage_padding_compat import PIPELINE_SHA256, build_padding_function, padding_operator_probe
from lowlight_restoration.zimage_repro_batch import PROTECTED


def load_repair(path,verify_artifacts=True):
    repair = json.loads(path.read_text())
    if any(repair.get(k) is not False for k in PROTECTED):
        raise ValueError('forbidden repair scope')
    if repair['worker_ranks'] != [2,3,6,7] or repair['reuse_ranks'] != [0,1,4,5]:
        raise ValueError('only failed deterministic workers may be rerun')
    for key in ('parent_config','parent_result'):
        if batch.sha256(ROOT/repair[key]) != repair[key+'_sha256']:
            raise ValueError(f'parent identity mismatch: {key}')
    if repair['upstream_pipeline_sha256'] != PIPELINE_SHA256:
        raise ValueError('unexpected pipeline identity')
    from diffsynth.pipelines import z_image
    _,_,patch = build_padding_function(z_image)
    parent_s,old,e,c = batch.load_config(ROOT/repair['parent_config'])
    parent = json.loads((ROOT/repair['parent_result']).read_text())
    if parent['config_sha256'] != repair['parent_config_sha256'] or parent['completed_fits'] != 12:
        raise ValueError('unexpected parent result')
    by_rank = {w['rank']:w for w in parent['workers']}
    if len(by_rank) != 8 or len(parent['workers']) != 8:
        raise ValueError('parent worker coverage changed')
    reused = {}
    for rank in repair['reuse_ranks']:
        w = by_rank[rank]
        if w['status'] != 'complete' or w['deterministic'] is not False:
            raise ValueError('cannot reuse incomplete/non-ordinary controls')
        for run in w['runs']:
            if run['status'] != 'complete':
                raise ValueError('parent phase incomplete')
            if verify_artifacts:
                batch.verify_run_artifacts(run,parent_s)
        reused[rank] = (w,parent_s,repair['parent_config_sha256'])
    for rank in repair['worker_ranks']:
        w = by_rank[rank]
        if w['status'] != 'partial_failure' or any(r['status'] != 'failed' for r in w['runs']):
            raise ValueError('refuse to rerun a successful parent deterministic phase')
    output = (ROOT/repair['output_dir']).resolve()
    if output != ROOT/'checkpoints/foundir_blur_only_zimage_repro_repair_v1':
        raise ValueError('unexpected repair output root')
    s = dict(parent_s,output_dir=repair['output_dir'],padding_repair=True)
    return repair,s,old,e,c,reused,patch


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--worker',type=int)
    parser.add_argument('--cpu-smoke',action='store_true')
    args=parser.parse_args()
    repair,s,old,e,c,reused,patch=load_repair(args.config,verify_artifacts=args.worker is None)
    if args.preflight:
        batch.verify_model_artifacts(c,large_hashes=True)
        batch.load_frozen_rows(c,verify_images=True)
        for name,mem in [('NVIDIA H100 80GB HBM3',81920),('NVIDIA H200',143000)]:
            batch.validate_gpu_inventory([name]*8,[mem]*8,c)
        padding_operator_probe('cpu')
        print('REPRO_REPAIR_PREFLIGHT_PASS',batch.sha256(args.config),json.dumps(patch),flush=True)
        return 0
    if args.worker is not None:
        if args.worker not in repair['worker_ranks']:
            raise ValueError('ordinary worker rerun prohibited')
        if args.cpu_smoke:
            s['output_dir']=str(Path(tempfile.mkdtemp(prefix='zimage_padding_cpu_',dir=ROOT/'logs')).relative_to(ROOT))
        return batch.worker(args,s,old,e,c)
    if args.cpu_smoke:
        raise ValueError('CPU smoke requires --worker')
    batch.inventory(c)
    batch.verify_model_artifacts(c,large_hashes=True)
    output=ROOT/s['output_dir']
    output.mkdir(parents=True,exist_ok=False)
    sources = ['smoke/repair_foundir_blur_only_zimage_repro_batch.py',
               'smoke/audit_foundir_blur_only_zimage_repro_batch.py',
               'lowlight_restoration/zimage_padding_compat.py',
               'lowlight_restoration/zimage_repro_batch.py',
               'lowlight_restoration/zimage_optimizer_trace.py']
    batch.write_json(output/'execution_identity.json',dict(
        config_sha256=batch.sha256(args.config),parent_result=repair['parent_result'],
        parent_result_sha256=repair['parent_result_sha256'],patch=patch,
        reused_worker_ranks=repair['reuse_ranks'],new_worker_ranks=repair['worker_ranks'],
        sources={p:batch.sha256(ROOT/p) for p in sources},
        git_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        command=sys.argv,executable=sys.executable))
    codes=[0]*8
    with contextlib.ExitStack() as stack:
        processes={}
        for rank in repair['worker_ranks']:
            log=stack.enter_context((output/f'worker_{rank}.log').open('x'))
            processes[rank]=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--config',str(args.config),'--worker',str(rank)],
                                             cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        while any(p.poll() is None for p in processes.values()):
            print('REPRO_REPAIR_WORKERS',json.dumps({r:p.poll() for r,p in processes.items()}),flush=True)
            time.sleep(15)
        for rank,p in processes.items():
            codes[rank]=p.wait()
    return batch.aggregate(output,s,batch.sha256(args.config),codes,reused=reused)


if __name__=='__main__':
    raise SystemExit(main())
