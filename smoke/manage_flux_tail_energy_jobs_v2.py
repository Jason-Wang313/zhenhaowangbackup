#!/usr/bin/env python3
"""Stop only the explicitly superseded four-GPU qualification on this GPU node."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from smoke.flux_tail_checkpoint_v2 import atomic_json,sha

LEGACY_ID='qualification_4gpu_v1'
LEGACY_SCRIPT='qualify_foundir_flux_tail_energy_v1.py'


def matches(argv):
    for i,arg in enumerate(argv):
        if Path(arg).name==LEGACY_SCRIPT and argv[i+1:i+2]==['train']:
            try:
                return argv[argv.index('--run-id')+1]==LEGACY_ID and argv[argv.index('--gpus')+1]=='4'
            except (ValueError,IndexError):return False
    return False


def process_identity(pid):
    root=Path('/proc')/str(pid)
    try:
        argv=[s.decode() for s in (root/'cmdline').read_bytes().split(b'\0') if s]
        stat=(root/'stat').read_text().rsplit(')',1)[1].split()
        cwd=(root/'cwd').resolve()
    except (FileNotFoundError,ProcessLookupError,PermissionError):return None
    if not matches(argv) or cwd!=ROOT:return None
    return dict(pid=pid,start_ticks=stat[19],torchrun='torch.distributed.run' in argv)


def find_jobs():
    result=[]
    for p in Path('/proc').iterdir():
        if p.name.isdigit():
            item=process_identity(int(p.name))
            if item is not None:result.append(item)
    return result


def switch(output,gpus):
    legacy=ROOT/'checkpoints/foundir_flux_tail_energy_qualification_v1'/LEGACY_ID
    if not (output/'preflight.json').exists():raise RuntimeError('prepare and verify reusable cache before stopping')
    if (legacy/'energy_step_000025.pt').exists():raise RuntimeError('legacy checkpoint exists: do not discard it')
    devices=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader,nounits'],text=True).splitlines()
    if len(devices)<gpus:raise RuntimeError('not enough allocated GPUs')
    chosen=[line.rsplit(',',1) for line in devices[:gpus]]
    if len({v[0].strip() for v in chosen})!=1 or any(not any(k in name for k in ('H100','H200')) or float(memory)<75*1024 for name,memory in chosen):
        raise RuntimeError('requires homogeneous H100/H200 >=75GiB')
    jobs=find_jobs()
    active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).splitlines()
    active_pids={int(v.strip()) for v in active if v.strip().isdigit()}
    if active_pids-{j['pid'] for j in jobs}:
        raise RuntimeError('other GPU processes exist; refusing automatic stop or overlap')
    now=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    record=dict(utc=now,legacy_run=LEGACY_ID,matched_processes=jobs,
                legacy_in_memory_optimizer_checkpoint_available=False,
                cache_preserved=True,new_training_replays_unsaved_steps=True)
    atomic_json(output/f'legacy_switch_request_{now}.json',record)
    parents=[j for j in jobs if j['torchrun']]
    targets=parents if parents else jobs
    for j in targets:
        if process_identity(j['pid'])!=j:raise RuntimeError('PID changed before signal')
        os.kill(j['pid'],signal.SIGTERM)
    deadline=time.monotonic()+45
    while find_jobs() and time.monotonic()<deadline:time.sleep(.5)
    if find_jobs():raise RuntimeError('legacy did not stop cleanly; no broad kill or second launch')
    old_log=ROOT/f'logs/foundir_flux_tail_energy_qualification_v1_{LEGACY_ID}.log'
    if old_log.exists():
        snapshot=output/f'legacy_log_{now}.log'
        with old_log.open('rb') as source,snapshot.open('xb') as dest:
            shutil.copyfileobj(source,dest);dest.flush();os.fsync(dest.fileno())
        rows=[]
        for line in snapshot.read_text().splitlines():
            if line.startswith('TAIL_ENERGY_TRAIN '):rows.append(json.loads(line.split(' ',1)[1]))
        record.update(legacy_last_logged_step=max((r['step'] for r in rows),default=0),
                      log_snapshot=snapshot.name,log_sha256=sha(snapshot))
    atomic_json(output/f'legacy_switch_complete_{now}.json',record)
    print('TAIL_ENERGY_V2_LEGACY_STOPPED '+json.dumps(record),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True);p.add_argument('--gpus',type=int,choices=[4,8],default=8)
    a=p.parse_args();output=Path(a.output).resolve()
    output.relative_to(ROOT/'checkpoints/foundir_flux_tail_energy_resumable_v2')
    switch(output,a.gpus)


if __name__=='__main__':main()
