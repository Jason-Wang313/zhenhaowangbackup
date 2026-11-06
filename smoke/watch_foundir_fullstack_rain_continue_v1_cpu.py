#!/usr/bin/env python3
"""CPU-only durable observer: publish first-five, step900 and terminal reports."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'checkpoints/foundir_fullstack_rain_continue_3x2_v1'
QUEUE=ROOT/'checkpoints/foundir_fullstack_executor_3x2_h100_v2'
JOB=QUEUE/'jobs/round4_rain_continue_3x2_v1'
MONITOR=OUT/'monitor'
PY=ROOT/'.conda/gpu-train-stable/bin/python'


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():return read(path)
    temp=path.with_name(path.name+f'.tmp.{os.getpid()}')
    with temp.open('x') as f:json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(temp,path);return value


def cpu_env():
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=''
    env['PYTHONPATH']=str(ROOT/'.conda/gpu-train-stable/lib/python3.12/site-packages')+':'+str(ROOT)
    env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    return env


def verify_checkpoint(step):
    marker=MONITOR/f'checkpoint_{step}_CPU.json'
    if marker.exists():
        result=read(marker)
        if result['status']!=0:raise RuntimeError('prior CPU checkpoint verification failed; preserve evidence')
        return result
    source="""import json,torch
from smoke import train_foundir_fullstack_rain_continue_v1 as task
step=int(__import__('sys').argv[1]);record=task.c.read(task.OUT/'steps'/f'step_{step:06d}.json')
file=task.OUT/record['file'];assert task.c.sha(file)==record['sha256']
saved=torch.load(file,map_location='cpu',weights_only=True)
task.validate_state(saved,record,task.c.sha(task.OUT/'receipt.json'))
print('CPU_CHECKPOINT_VERIFIED',step,'world',saved['world'],'next_slot',saved['next_global_slot'])
"""
    start=time.monotonic();p=subprocess.run([str(PY),'-S','-c',source,str(step)],cwd=ROOT,env=cpu_env(),capture_output=True,text=True)
    result=write_new(marker,{'status':p.returncode,'seconds':time.monotonic()-start,'stdout':p.stdout,'stderr':p.stderr})
    if p.returncode:raise RuntimeError('CPU checkpoint verification failed: '+p.stderr)
    return result


def snapshot(event):
    markers=sorted((OUT/'steps').glob('step_*.json'))
    last=read(markers[-1]) if markers else None
    result={'event':event,'observed_utc_unix':time.time(),'job':'round4_rain_continue_3x2_v1',
            'source_commit':read(JOB/'job.json')['source_commit'],'latest_checkpoint':last,
            'new_updates':max(0,last['step']-800) if last else 0,'nodes':[],'milestones':{},
            'official_Rain_evaluated_this_round':False,'training_fit_only':True,
            'legacy_official_GT_source_overlap':'3 sources / 30 official Rain records; 6 historical on-policy presentations confirmed',
            'new_training_excludes_all_detected_official_source_groups':True,
            'world_change_bitwise_equivalent':False,'automatic_more_training':False}
    losses=[]
    for node in range(3):
        start=JOB/f'node{node}.started.json';end=JOB/f'node{node}.ended.json';log=JOB/f'node{node}.log'
        n={'node':node,'start':read(start) if start.exists() else None,
           'end':read(end) if end.exists() else None,'heartbeat':read(QUEUE/f'workers/node{node}/heartbeat.json')}
        if log.exists():
            lines=log.read_text().splitlines();n['log_sha256']=sha(log)
            n['NCCL_verified_count']=sum('RAIN_FULLSTACK_NCCL_VERIFIED' in s for s in lines)
            n['restored_count']=sum('RAIN_FULLSTACK_RESTORED' in s for s in lines)
            if event=='failed':n['tail']=lines[-25:]
            if node==0:
                for s in lines:
                    if 'RAIN_FULLSTACK_TRAIN ' in s:
                        try:losses.append(json.loads(s.split('RAIN_FULLSTACK_TRAIN ',1)[1]))
                        except json.JSONDecodeError:pass
        result['nodes'].append(n)
    result['latest_training_logs']=losses[-5:]
    for step in (800,900,1000):
        p=OUT/'milestones'/f'step_{step:06d}.json'
        if p.exists():result['milestones'][str(step)]=read(p)
    if event=='first5':result['independent_checkpoint_verification']=verify_checkpoint(805)
    if event in ('terminal','failed'):
        result['completed_node_GPU_hours']=sum(n['end']['allocated_GPU_hours'] for n in result['nodes'] if n['end'])
        p=JOB/'cpu_finish.json'
        if p.exists():result['CPU_finish']=read(p)
    return result


def publish(event):
    done=MONITOR/f'{event}.published.json'
    if done.exists():return
    stem=f'reports/foundir_fullstack_rain_continue_v1_{event}_2026-11-06'
    rp=ROOT/(stem+'.json');result=read(rp) if rp.exists() else write_new(rp,snapshot(event))
    text=f"# Rain续训观察：{event}\n\n当前最后完整step：{result['latest_checkpoint']['step'] if result['latest_checkpoint'] else '尚无'}；新增{result['new_updates']}/200步。完整节点状态、日志SHA、loss与训练探针见JSON。\n\n"
    text+='固定step800→1000、global32、三节点六卡；非bitwise八卡恢复。新训练排除已知官方重叠来源，但旧权重的3来源/30张官方Rain历史GT来源曝光仍保留，不能称干净独立泛化。当前探针均training_fit_only，不表示官方Rain已修复。\n\n'
    for step,m in result['milestones'].items():text+=f"step{step}训练32图：PSNR {m['means']['psnr']:.6f} / LPIPS {m['means']['lpips']:.6f}。\n\n"
    if event=='terminal':text+='200步任务完成后停在1000，不自动追加训练；随后单独准备本轮Rain100官方评价。Haze单独复跑与全部600端到端复跑仍未启动。\n'
    if event=='failed':text+='任务失败，保留最后完整状态、日志及所有历史资产；没有自动重试或扩预算。\n'
    md=ROOT/(stem+'.md')
    if not md.exists():md.write_text(text.rstrip()+'\n')
    files=[stem+'.json',stem+'.md']
    if event=='terminal':files.append('reports/foundir_fullstack_rain_continue_v1_result_2026-11-06.json')
    for attempt in range(3):
        try:
            subprocess.run(['git','fetch','origin','main'],cwd=ROOT,check=True,capture_output=True,text=True)
            wt=Path(tempfile.mkdtemp(prefix='rain-monitor-publish-'))
            subprocess.run(['git','worktree','add','--quiet','--detach',str(wt),'origin/main'],cwd=ROOT,check=True,capture_output=True,text=True)
            for name in files:
                content=(ROOT/name).read_bytes()
                # Historical local report is preserved; normalize only the published Markdown.
                if name.endswith('.md'):content=content.rstrip()+b'\n'
                (wt/name).write_bytes(content)
            subprocess.run(['git','add','-f','--',*files],cwd=wt,check=True,capture_output=True,text=True)
            subprocess.run(['git','diff','--cached','--check'],cwd=wt,check=True,capture_output=True,text=True)
            if subprocess.run(['git','diff','--cached','--quiet'],cwd=wt).returncode==0:
                commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=wt,text=True).strip()
            else:
                subprocess.run(['git','commit','-m',f'Record Rain continuation {event} observation and exposure limits'],cwd=wt,check=True,capture_output=True,text=True)
                commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=wt,text=True).strip()
                subprocess.run(['git','push','origin','HEAD:main'],cwd=wt,check=True,capture_output=True,text=True)
            remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/main'],cwd=ROOT,text=True).split()[0]
            if remote!=commit:
                subprocess.run(['git','fetch','origin','main'],cwd=ROOT,check=True,capture_output=True,text=True)
                subprocess.run(['git','merge-base','--is-ancestor',commit,'origin/main'],cwd=ROOT,check=True)
            write_new(done,{'commit':commit,'remote_at_verification':remote,'files':files,'utc_unix':time.time()})
            print('RAIN_MONITOR_PUBLISHED',event,commit,flush=True);return
        except subprocess.CalledProcessError as e:
            print('RAIN_MONITOR_GIT_RETRY',event,attempt,str(e),getattr(e,'stdout',''),getattr(e,'stderr',''),flush=True);time.sleep(10)
    raise RuntimeError('report push failed after three safe attempts')


def finish():
    target=JOB/'cpu_finish.json'
    if target.exists():
        if read(target)['status']!=0:raise RuntimeError('prior finish failed; preserve report')
        return
    start=time.monotonic();p=subprocess.run([str(PY),'-S','smoke/train_foundir_fullstack_rain_continue_v1.py','finish'],
        cwd=ROOT,env=cpu_env(),capture_output=True,text=True)
    write_new(target,{'status':p.returncode,'seconds':time.monotonic()-start,'stdout':p.stdout,'stderr':p.stderr})
    if p.returncode:raise RuntimeError('CPU finish failed: '+p.stderr)


def main():
    MONITOR.mkdir(parents=True,exist_ok=True)
    lock=(MONITOR/'observer.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    while True:
        ends=[read(JOB/f'node{i}.ended.json') if (JOB/f'node{i}.ended.json').exists() else None for i in range(3)]
        if any(e and e['status']!=0 for e in ends):publish('failed');return
        if (OUT/'steps/step_000805.json').exists():publish('first5')
        if (OUT/'milestones/step_000900.json').exists():publish('step900')
        if all(e and e['status']==0 for e in ends):finish();publish('terminal');return
        markers=sorted((OUT/'steps').glob('step_*.json'))
        print('RAIN_MONITOR_WAIT last_step='+str(read(markers[-1])['step'] if markers else None),flush=True)
        time.sleep(30)


if __name__=='__main__':main()
