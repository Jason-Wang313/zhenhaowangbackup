#!/usr/bin/env python3
"""Explicitly armed finite queue: wait Rain -> Noise router/cache/score -> Blur cache/score."""
import argparse
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
from smoke.existing_flux_replay_contract_v1 import read,write_new as write,sha
PY=ROOT/'.conda/gpu-train-stable/bin/python'
QUEUE=ROOT/'checkpoints/foundir_fullstack_executor_3x2_h100_v2'
OUT=ROOT/'checkpoints/foundir_fullstack_noise_blur_pipeline_v1'
ENTRY='smoke/run_foundir_fullstack_frequency_repair_v1_gpu.sh'
SCRIPT='smoke/run_foundir_fullstack_frequency_repair_v1.py'
TEXTURE='smoke/train_foundir_fullstack_noise_texture_router_v3.py'
SELF='smoke/watch_foundir_fullstack_noise_blur_v1_cpu.py'
RAIN='round5_rain_step1000_3x2_v1'


def env():return {**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'1',
    'PYTHONPATH':str(ROOT/'.conda/gpu-train-stable/lib/python3.12/site-packages')+':'+str(ROOT)}


def command(args,timeout=600,cwd=ROOT):
    return subprocess.run(args,cwd=cwd,env=env(),check=True,capture_output=True,text=True,timeout=timeout)


def cpu(name,args):
    marker=OUT/'cpu'/(name+'.json')
    if not marker.exists():
        begin=time.monotonic();p=subprocess.run([str(PY),'-S',*args],cwd=ROOT,env=env(),capture_output=True,text=True,timeout=1800)
        write(marker,{'status':p.returncode,'seconds':time.monotonic()-begin,'stdout':p.stdout,'stderr':p.stderr})
    value=read(marker)
    if value['status']!=0:raise RuntimeError('CPU stage failed: '+str(marker))


def status(job,nodes):
    root=QUEUE/'jobs'/job;ends=[read(root/f'node{n}.ended.json') if (root/f'node{n}.ended.json').exists() else None for n in nodes]
    if any(e and e['status']!=0 for e in ends):raise RuntimeError('GPU stage failed; no retry: '+job)
    return ends


def wait(job,nodes):
    while True:
        if all(status(job,nodes)):return
        print('NOISE_BLUR_WAIT '+job,flush=True);time.sleep(30)


def publish(event,files,detail):
    marker=OUT/'published'/(event+'.json')
    if marker.exists():return
    snapshot=OUT/'snapshots'/(event+'.json')
    if not snapshot.exists():
        jobs=[]
        for p in sorted((QUEUE/'jobs').glob('round6_*_v1/job.json')):
            j=read(p);ends=[read(p.parent/f'node{n}.ended.json') if (p.parent/f'node{n}.ended.json').exists() else None for n in j['nodes']]
            starts=[read(p.parent/f'node{n}.started.json') if (p.parent/f'node{n}.started.json').exists() else None for n in j['nodes']]
            jobs.append({'id':j['id'],'nodes':j['nodes'],'started':starts,'ended':ends,
                         'completed_GPU_hours':sum(e['allocated_GPU_hours'] for e in ends if e)})
        write(snapshot,{'event':event,'detail':detail,'utc_unix':time.time(),'jobs':jobs,'automatic_official_evaluation':False,
              'automatic_extension':False,'CPU_stages':{p.stem:read(p) for p in sorted((OUT/'cpu').glob('*.json'))}})
    stem=f'reports/foundir_fullstack_noise_blur_pipeline_v1_{event}_2026-11-06'
    small=ROOT/(stem+'.json');write(small,read(snapshot));md=ROOT/(stem+'.md')
    if not md.exists():md.write_text(f'# Noise/Blur有限修复：{event}\n\n{detail}\n\nGPU退出码、耗时及CPU阶段见JSON；不自动官方评价、部署、扩大训练或重试。\n')
    files=[*files,stem+'.json',stem+'.md']
    for attempt in range(3):
        try:
            command(['git','fetch','origin','main'],timeout=60)
            wt=Path(tempfile.mkdtemp(prefix='noise-blur-publish-'))
            command(['git','worktree','add','--quiet','--detach',str(wt),'origin/main'])
            for name in files:
                (wt/name).parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,wt/name)
            command(['git','add','-f','--',*files],cwd=wt);command(['git','diff','--cached','--check'],cwd=wt)
            if subprocess.run(['git','diff','--cached','--quiet'],cwd=wt).returncode:
                command(['git','commit','-m','Record bounded Noise Blur repair '+event],cwd=wt)
                command(['git','push','origin','HEAD:main'],timeout=60,cwd=wt)
            commit=command(['git','rev-parse','HEAD'],cwd=wt).stdout.strip();command(['git','fetch','origin','main'],timeout=60)
            command(['git','merge-base','--is-ancestor',commit,'origin/main'])
            write(marker,{'commit':commit,'files':files,'utc_unix':time.time()});print('NOISE_BLUR_REPORT_PUSHED '+event+' '+commit,flush=True);return
        except (subprocess.SubprocessError,OSError) as exc:
            print('NOISE_BLUR_REPORT_RETRY '+str(exc),flush=True);time.sleep(10)
    raise RuntimeError('Git publication failed; no forced push')


def submit(job,entry,args,nodes,commit,sources):
    path=QUEUE/'jobs'/job/'job.json'
    if path.exists():
        value=read(path)
        if value['entry']!=entry or value['args']!=args or value['nodes']!=nodes or value['source_commit']!=commit:
            raise ValueError('existing job identity differs')
        return
    for node in nodes:
        h=read(QUEUE/f'workers/node{node}/heartbeat.json')
        if time.time()-h['utc_unix']>60 or h['state']!='idle':raise RuntimeError('worker not verified idle: '+str(node))
    command([str(PY),'-S','smoke/foundir_fullstack_executor_3x2_v2.py','submit','--job-id',job,
             '--entry',entry,'--commit',commit,'--purpose','Finite Noise then Blur repair; no official run or expansion',
             '--nodes',*map(str,nodes),'--sources',*sources,'--arguments',*args])


def ready_idle(nodes):
    # Allow the worker a few polling cycles after its preceding process exits.
    end=time.monotonic()+60
    while time.monotonic()<end:
        hs=[read(QUEUE/f'workers/node{n}/heartbeat.json') for n in nodes]
        if all(time.time()-h['utc_unix']<60 and h['state']=='idle' for h in hs):return
        time.sleep(5)
    raise RuntimeError('nodes not idle; preserve other jobs')


def main(commit):
    OUT.mkdir(parents=True,exist_ok=True);lock=(OUT/'observer.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        # This process is not launched by prepare. Arming it explicitly authorizes these finite jobs.
        wait(RAIN,[0,1,2])
        while not (ROOT/'checkpoints/foundir_fullstack_rain_step1000_v1/monitor/terminal.published.json').exists():
            print('NOISE_BLUR_WAIT_RAIN_CPU_TERMINAL',flush=True);time.sleep(30)
        cpu('verify_texture',[TEXTURE,'verify'])
        for task in ('noise','blur'):cpu('verify_'+task,[SCRIPT,'verify','--task',task])
        ready_idle([0]);job='round6_noise_texture_v3_v1'
        submit(job,'smoke/run_foundir_fullstack_noise_texture_router_v3.sh',[],[0],commit,
               [TEXTURE,SELF,'lowlight_restoration/noise_texture_router_v3.py','smoke/train_foundir_fullstack_noise_standalone_router_v2_cpu.py'])
        publish('queued',[],'已提交单节点Noise纹理路由，固定400步；之后依次Noise和Blur缓存/拟合/开发评分。')
        wait(job,[0]);cpu('texture_finish',[TEXTURE,'finish'])
        publish('router_terminal',['reports/foundir_fullstack_noise_texture_router_v3_result_2026-11-06.json'],
                '路由工程终态已核验；是否通过独立验证见JSON，不自动部署。')
        for task in ('noise','blur'):
            b=read(ROOT/f'checkpoints/foundir_fullstack_frequency_repair_v1/{task}/receipt.json');nodes=b['nodes']
            ready_idle(nodes);cache_job=f'round6_{task}_frequency_cache_v1'
            submit(cache_job,ENTRY,[task,'cache'],nodes,commit,[SCRIPT,SELF,'lowlight_restoration/flux_frequency_repair_v1.py'])
            wait(cache_job,nodes);cpu(task+'_fit',[SCRIPT,'fit','--task',task])
            ready_idle([0]);score_job=f'round6_{task}_frequency_score_v1'
            submit(score_job,ENTRY,[task,'score'],[0],commit,[SCRIPT,SELF,'lowlight_restoration/flux_frequency_repair_v1.py'])
            wait(score_job,[0]);cpu(task+'_finish',[SCRIPT,'finish','--task',task])
            publish(task+'_terminal',[f'reports/foundir_fullstack_{task}_frequency_repair_v1_{phase}_2026-11-06.json' for phase in ('fit','result')],
                    task+'固定系数与开发评价已结束；通过/退步均保留，不自动官方测试或部署。')
        publish('terminal',[],'本轮Noise路由、Noise/Blur有限频段修复已全部收尾；停止，不扩大预算。')
    except Exception as exc:
        publish('failed',[],str(exc));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--commit',required=True);main(p.parse_args().commit)
