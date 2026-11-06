#!/usr/bin/env python3
"""Wait for bounded training, evaluate only Rain100, publish small reports and stop."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

ROOT=Path(__file__).resolve().parents[1]
PY=ROOT/'.conda/gpu-train-stable/bin/python'
QUEUE=ROOT/'checkpoints/foundir_fullstack_executor_3x2_h100_v2'
TRAIN=ROOT/'checkpoints/foundir_fullstack_rain_continue_3x2_v1'
TRAIN_JOB=QUEUE/'jobs/round4_rain_continue_3x2_v1'
OUT=ROOT/'checkpoints/foundir_fullstack_rain_step1000_v1'
MONITOR=OUT/'monitor'
JOB_ID='round5_rain_step1000_3x2_v1'
JOB=QUEUE/'jobs'/JOB_ID
SCRIPT='smoke/evaluate_foundir_fullstack_rain_step1000_v1.py'
ENTRY='smoke/run_foundir_fullstack_rain_step1000_v1.sh'
SELF='smoke/watch_foundir_fullstack_rain_step1000_v1_cpu.py'
RESULT='reports/foundir_fullstack_rain_step1000_v1_result_2026-11-06.json'
PREFLIGHT='reports/foundir_fullstack_rain_step1000_v1_preflight_2026-11-06.json'


def read(p):return json.loads(p.read_text())


def write_new(p,v):
    if p.exists():return read(p)
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+f'.tmp.{os.getpid()}.{time.time_ns()}')
    with tmp.open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,p);return v


def env():
    return {**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1',
            'PYTHONPATH':str(ROOT/'.conda/gpu-train-stable/lib/python3.12/site-packages')+':'+str(ROOT)}


def run_once(name,args):
    marker=MONITOR/(name+'.json')
    if marker.exists():value=read(marker)
    else:
        start=time.monotonic()
        p=subprocess.run(args,cwd=ROOT,env=env(),capture_output=True,text=True)
        value=write_new(marker,{'status':p.returncode,'seconds':time.monotonic()-start,'stdout':p.stdout,'stderr':p.stderr})
    if value['status']!=0:raise RuntimeError(f'{name} failed; preserved evidence at {marker}: '+value['stderr'][-2000:])
    return value


def snapshot(event,detail=None):
    records=list((OUT/'records').glob('*.json'))
    rain=sum(not read(p)['reused_parent'] for p in records)
    nodes=[]
    for n in range(3):
        nodes.append({'node':n,'started':read(JOB/f'node{n}.started.json') if (JOB/f'node{n}.started.json').exists() else None,
                      'ended':read(JOB/f'node{n}.ended.json') if (JOB/f'node{n}.ended.json').exists() else None})
    value={'event':event,'utc_unix':time.time(),'detail':detail,'job':JOB_ID,'fixed_step':1000,
           'completed_Rain_records':rain,'complete_summary_records':len(records),'tiles':len(list((OUT/'tiles').glob('*.json'))),
           'nodes':nodes,'new_training':False,'all600_end_to_end_rerun':False,'independent_blind_test':False,
           'legacy_GT_source_overlap':'3 sources shared with 30 official Rain records; earlier weights already exposed',
           'preserve_accepted_Haze_and_other500':True,'automatic_retry':False}
    ends=[n['ended'] for n in nodes if n['ended']]
    value['completed_node_GPU_hours']=sum(e['allocated_GPU_hours'] for e in ends)
    if len(ends)==3:
        value['wall_seconds']=max(n['started']['utc_unix']+n['ended']['seconds'] for n in nodes)-min(n['started']['utc_unix'] for n in nodes)
    for name in ('prepare','finish'):
        p=MONITOR/(name+'.json')
        if p.exists():value[name+'_CPU']=read(p)
    if (ROOT/RESULT).exists():value['result']=read(ROOT/RESULT)
    return value


def publish(event,detail=None):
    marker=MONITOR/(event+'.published.json')
    if marker.exists():return
    stem=f'reports/foundir_fullstack_rain_step1000_v1_{event}_2026-11-06'
    value=write_new(ROOT/(stem+'.json'),snapshot(event,detail))
    md=ROOT/(stem+'.md')
    if not md.exists():
        text=f"# 固定step1000 Rain评价：{event}\n\nRain新输出 {value['completed_Rain_records']}/100；图块 {value['tiles']}/4680。其余500图保留已接受结果，含Haze修复。\n\n"
        text+='仅本轮Rain评价，不新增训练、不按图GT选择、不启动统一600端到端复跑。旧权重有3来源/30张官方Rain的GT来源曝光，本轮不是独立盲测。\n\n'
        if 'result' in value:
            r=value['result'];m=r['per_category']['10Rain']
            text+=f"Rain PSNR {m['psnr']:.6f} / LPIPS {m['lpips']:.6f}；双门槛结果 {r['checks']['rain']}。完整十项和退步均见JSON。\n\n"
        text+=f"已完成节点费用 {value['completed_node_GPU_hours']:.6f} GPU-hours；退出码、CPU耗时、未完成节点见JSON。\n"
        if detail:text+='\n'+detail+'\n'
        md.write_text(text.rstrip()+'\n')
    files=[stem+'.json',stem+'.md']
    if event=='terminal':files.extend([RESULT,PREFLIGHT])
    for attempt in range(3):
        try:
            subprocess.run(['git','fetch','origin','main'],cwd=ROOT,check=True,capture_output=True,text=True)
            wt=Path(tempfile.mkdtemp(prefix='rain1000-publish-'))
            subprocess.run(['git','worktree','add','--quiet','--detach',str(wt),'origin/main'],cwd=ROOT,check=True,capture_output=True,text=True)
            for name in files:(wt/name).write_bytes((ROOT/name).read_bytes())
            subprocess.run(['git','add','-f','--',*files],cwd=wt,check=True,capture_output=True,text=True)
            subprocess.run(['git','diff','--cached','--check'],cwd=wt,check=True,capture_output=True,text=True)
            if subprocess.run(['git','diff','--cached','--quiet'],cwd=wt).returncode:
                subprocess.run(['git','commit','-m',f'Record fixed Rain step1000 evaluation {event}'],cwd=wt,check=True,capture_output=True,text=True)
                subprocess.run(['git','push','origin','HEAD:main'],cwd=wt,check=True,capture_output=True,text=True)
            commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=wt,text=True).strip()
            subprocess.run(['git','fetch','origin','main'],cwd=ROOT,check=True,capture_output=True,text=True)
            subprocess.run(['git','merge-base','--is-ancestor',commit,'origin/main'],cwd=ROOT,check=True)
            write_new(marker,{'commit':commit,'files':files,'utc_unix':time.time()})
            print('RAIN1000_REPORT_PUSHED',event,commit,flush=True);return
        except subprocess.CalledProcessError as e:
            print('RAIN1000_REPORT_RETRY',attempt,str(e),e.stdout,e.stderr,flush=True);time.sleep(10)
    raise RuntimeError('report publication failed; no forced push')


def wait_training():
    while True:
        ends=[read(TRAIN_JOB/f'node{n}.ended.json') if (TRAIN_JOB/f'node{n}.ended.json').exists() else None for n in range(3)]
        if any(e and e['status']!=0 for e in ends):raise RuntimeError('parent training failed; no evaluation submitted')
        terminal=TRAIN/'monitor/terminal.published.json'
        if all(e and e['status']==0 for e in ends) and terminal.exists():return
        print('RAIN1000_WAIT_FOR_TRAINING_AND_CPU_FINISH',flush=True);time.sleep(30)


def main(commit):
    MONITOR.mkdir(parents=True,exist_ok=True)
    lock=(MONITOR/'observer.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        wait_training()
        run_once('prepare',[str(PY),'-S',SCRIPT,'prepare'])
        run_once('submit',[str(PY),'-S','smoke/foundir_fullstack_executor_3x2_v2.py','submit',
                          '--job-id',JOB_ID,'--entry',ENTRY,'--commit',commit,
                          '--purpose','Fixed step1000 Rain100 only; preserve accepted Haze and other500; no more training',
                          '--nodes','0','1','2','--sources',SCRIPT,ENTRY,SELF])
        publish('queued')
        while True:
            ends=[read(JOB/f'node{n}.ended.json') if (JOB/f'node{n}.ended.json').exists() else None for n in range(3)]
            if any(e and e['status']!=0 for e in ends):raise RuntimeError('Rain evaluation node failed; no automatic retry')
            if not (MONITOR/'first_output.published.json').exists():
                if any(not read(p)['reused_parent'] for p in (OUT/'records').glob('*.json')):publish('first_output')
            if all(e and e['status']==0 for e in ends):
                run_once('finish',[str(PY),'-S',SCRIPT,'finish']);publish('terminal');return
            print('RAIN1000_WAIT tiles='+str(len(list((OUT/'tiles').glob('*.json')))),flush=True);time.sleep(30)
    except Exception as error:
        publish('failed',str(error));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--commit',required=True);main(p.parse_args().commit)
