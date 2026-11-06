#!/usr/bin/env python3
"""Explicit user-requested recovery after assistant cancellation; preserve both attempts."""
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from smoke.existing_flux_replay_contract_v1 import read,sha,write_new as write
PY=ROOT/'.conda/gpu-train-stable/bin/python'
OUT=ROOT/'checkpoints/foundir_fullstack_rain_step1000_v1'
QUEUE=ROOT/'checkpoints/foundir_fullstack_executor_3x2_h100_v2/jobs'
OLD='round5_rain_step1000_3x2_v1';JOB='round5_rain_step1000_3x2_resume_v1'
MONITOR=OUT/'resume_monitor'
REPORT='reports/foundir_fullstack_rain_step1000_resume_v1_2026-11-06.json'
RESULT='reports/foundir_fullstack_rain_step1000_v1_result_2026-11-06.json'
SELF='smoke/watch_foundir_fullstack_rain_step1000_resume_v1_cpu.py'


def command(args,cwd=ROOT):
    return subprocess.run(args,cwd=cwd,check=True,capture_output=True,text=True,timeout=60)


def publish(value):
    write(ROOT/REPORT,value)
    md=ROOT/REPORT.replace('.json','.md')
    text='# Rain误取消后的授权续跑\n\n用户是在询问统一复现，并未要求取消。代理错误发出取消，原三节点均-15；用户随后明确要求继续至完成。原记录和费用完整保留。\n\n'
    text+=f"恢复状态：{value['status']}。复用同一receipt、seed、权重和输出目录，只补未完成图块；没有训练重跑。全部尝试实际占用费用 {value['total_attempt_GPU_hours']:.6f} GPU-hours。\n\n"
    text+='旧统一入口仍绑定Rain step800，本轮独立评价绑定step1000。统一复现必须显式绑定最终采用权重、路由、26调用读出、量化及拼接；单类通过不能代替统一端到端验收。\n\n'
    text+='原result的used_GPU_hours按最终worker记录求和，不能涵盖被中断的所有计算；总成本以本报告两个attempt的ended.json汇总为准。\n'
    if value.get('result'):
        r=value['result'];text+=f"\nRain最终指标：{r['per_category']['10Rain']}；原门槛判断：{r['checks']['rain']}。\n"
    if not md.exists():md.write_text(text)
    files=[REPORT,str(md.relative_to(ROOT)),SELF]
    if value['status']=='complete':files.append(RESULT)
    for attempt in range(3):
        try:
            command(['git','fetch','origin','main']);wt=Path(tempfile.mkdtemp(prefix='rain-recovery-publish-'))
            command(['git','worktree','add','--quiet','--detach',str(wt),'origin/main'])
            for name in files:(wt/name).parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,wt/name)
            command(['git','add','-f','--',*files],wt);command(['git','diff','--cached','--check'],wt)
            if subprocess.run(['git','diff','--cached','--quiet'],cwd=wt).returncode:
                command(['git','commit','-m','Record user-requested Rain recovery and all attempt costs'],wt)
                command(['git','push','origin','HEAD:main'],wt)
            commit=command(['git','rev-parse','HEAD'],wt).stdout.strip();command(['git','fetch','origin','main']);command(['git','merge-base','--is-ancestor',commit,'origin/main'])
            write(MONITOR/'published.json',{'commit':commit,'report':REPORT,'report_sha256':sha(ROOT/REPORT),'status':value['status']})
            print('RAIN_RECOVERY_REPORT_PUSHED '+commit,flush=True);return
        except (subprocess.SubprocessError,OSError) as exc:print('PUBLISH_RETRY',str(exc),flush=True);time.sleep(10)
    raise RuntimeError('recovery report push failed')


def main():
    MONITOR.mkdir(parents=True,exist_ok=True);lock=(MONITOR/'observer.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    while True:
        ends=[read(QUEUE/JOB/f'node{n}.ended.json') if (QUEUE/JOB/f'node{n}.ended.json').exists() else None for n in range(3)]
        if all(ends):break
        print('RAIN_RECOVERY_WAIT tiles='+str(len(list((OUT/'tiles').glob('*.json')))),flush=True);time.sleep(10)
    ok=all(e['status']==0 for e in ends);cpu=None
    if ok:
        marker=MONITOR/'finish.json'
        if not marker.exists():
            start=time.monotonic();env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'2',
              'PYTHONPATH':str(ROOT/'.conda/gpu-train-stable/lib/python3.12/site-packages')+':'+str(ROOT)}
            p=subprocess.run([str(PY),'-S','smoke/evaluate_foundir_fullstack_rain_step1000_v1.py','finish'],cwd=ROOT,env=env,capture_output=True,text=True,timeout=1800)
            write(marker,{'status':p.returncode,'seconds':time.monotonic()-start,'stdout':p.stdout,'stderr':p.stderr})
        cpu=read(marker);ok=cpu['status']==0
    attempts={job:[{'node':n,'start':read(QUEUE/job/f'node{n}.started.json'),'end':read(QUEUE/job/f'node{n}.ended.json')} for n in range(3)] for job in (OLD,JOB)}
    value={'status':'complete' if ok else 'failed','explicit_user_request':'do not stop; let existing evaluation finish',
           'interruption_cause':'assistant incorrectly interpreted explanation request as instruction to cancel',
           'receipt_sha256':sha(OUT/'receipt.json'),'attempts':attempts,
           'total_attempt_GPU_hours':sum(a['end']['allocated_GPU_hours'] for rows in attempts.values() for a in rows),
           'old_assets_preserved':True,'original_cancel_record_preserved':True,'training_restarted':False,
           'new_experiments_started':False,'CPU_finish':cpu,
           'result':read(ROOT/RESULT) if ok else None,'result_sha256':sha(ROOT/RESULT) if ok else None}
    publish(value)
    if not ok:raise RuntimeError('recovery failed; no automatic retries')

if __name__=='__main__':main()
