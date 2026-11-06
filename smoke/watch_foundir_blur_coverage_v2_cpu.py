#!/usr/bin/env python3
"""Fixed data-coverage control: cache -> fresh200 -> official150; independent report uploader."""
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
from smoke.existing_flux_replay_contract_v1 import read,sha,write_new as write
OUT=ROOT/'checkpoints/foundir_blur_delta_coverage_v2'
OFF=ROOT/'checkpoints/foundir_blur_coverage_official_v2'
QUEUE=ROOT/'checkpoints/foundir_fullstack_executor_3x2_h100_v2'
PY=ROOT/'.conda/gpu-train-stable/bin/python'
TRAIN='smoke/run_foundir_fullstack_blur_coverage_v2.py'
EVAL='smoke/evaluate_foundir_blur_coverage_official_v2.py'
ENTRY='smoke/run_foundir_fullstack_blur_coverage_v2_gpu.sh'
SOURCES=[TRAIN,EVAL,'lowlight_restoration/blur_delta_velocity_v1.py','smoke/blur_delta_state_v1.py']
JOBS={p:'round10_blur_coverage_'+p+'_v2' for p in ('cache','train','official')}
REQUESTS=OUT/'publication_requests'


def command(args,cwd=ROOT,**kwargs):
    if args[0]=='git':kwargs.setdefault('timeout',45)
    return subprocess.run(args,cwd=cwd,text=True,capture_output=True,check=True,**kwargs)


def env():
    return dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',
      TORCH_HOME=str(ROOT/'models/torch'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
      PYTHONPATH=str(ROOT)+':'+str(ROOT/'third_party/flux2/src'))


def cpu(phase):
    marker=OUT/'cpu'/(phase+'.json')
    if not marker.exists():
        start=time.monotonic()
        proc=subprocess.run([str(PY),'-u',EVAL,phase],cwd=ROOT,env=env(),text=True,capture_output=True)
        write(marker,{'status':proc.returncode,'seconds':time.monotonic()-start,'stdout':proc.stdout,'stderr':proc.stderr})
    if read(marker)['status']!=0:raise RuntimeError('CPU stage failed: '+phase)


def enqueue(event,detail):
    request=REQUESTS/(event+'.json')
    if request.exists():return
    jobs={}
    for phase,name in JOBS.items():
        p=QUEUE/'jobs'/name
        if (p/'job.json').exists():jobs[phase]={f.stem:read(f) for f in p.glob('*.json')}
    logs=[];log=QUEUE/'jobs'/JOBS['train']/'node3.log'
    if log.exists():
        for line in log.read_text(errors='replace').splitlines():
            if line.startswith('BLUR_DELTA_TRAIN {'):logs.append(json.loads(line.split(' ',1)[1]))
    steps=sorted((OUT/'pilot/steps').glob('step_*.json'))
    v={'event':event,'detail':detail,'utc_unix':time.time(),'jobs':jobs,'training_logs':logs,
       'latest_step':read(steps[-1]) if steps else None,'receipt_sha256':sha(OUT/'receipt.json'),
       'CPU_stages':{p.stem:read(p) for p in (OUT/'cpu').glob('*.json')},
       'completed_GPU_hours':sum(j['node3.ended']['allocated_GPU_hours'] for j in jobs.values() if 'node3.ended' in j),
       'prior_delta_pilot_and_path_GPU_hours':1.2959202243505916,
       'cost_complete':all('node3.ended' in j for j in jobs.values()),
       'official_only_quality_gate':True,'other_categories_submitted':False,'new_development_test':False,
       'official_Blur_pass':False,'fixed_steps':200,'training_sources':1024,'training_exposures':1600,
       'adaptive_official_benchmark_not_independent_blind_test':True}
    result=ROOT/'reports/foundir_blur_coverage_official_v2_result_2026-11-06.json'
    if result.exists():
        r=read(result);v['official_result_sha256']=sha(result)
        for k in ('means','gate_checks','official_Blur_pass','PSNR_delta_vs_old','LPIPS_relative_reduction_vs_old_percent'):
            v[k]=r[k]
    stem=f'reports/foundir_blur_coverage_v2_{event}_2026-11-06'
    write(ROOT/(stem+'.json'),v)
    metrics=('\n\n官方150终态：'+json.dumps(v['means'],ensure_ascii=False)+'；联合通过='+str(v['official_Blur_pass'])) if 'means' in v else '\n\n本轮尚无完整官方指标，不能声称修复。'
    (ROOT/(stem+'.md')).write_text('# Blur1080p训练来源覆盖对照：'+event+'\n\n'+detail+
      '\n\n固定1024不同训练来源、200步/global8，与旧128对照同1600张训练曝光；来源分辨率和多样性共同变化，不能分离二者因果。fresh原LoRA/Adam，未从旧step200续训。'+
      '\n\n仅官方完整150作25.9/.18门槛，无其它开发集gate；官方GT不拟合/逐图挑选，已有反复反馈曝光。node3双H100，工程成功不等于官方成功。'+
      f'\n\n本轮已结束阶段费用{v["completed_GPU_hours"]:.9f} GPU-hours；记录中阶段费用是否齐全：{v["cost_complete"]}。逐rank状态和CPU退出码/耗时见JSON。历史另计，全部资产保留。'+metrics+'\n')
    files=[stem+'.json',stem+'.md']
    for name in ['reports/foundir_blur_delta_coverage_v2_preflight_2026-11-06.json',
                 'reports/foundir_blur_coverage_official_v2_preflight_2026-11-06.json',
                 'reports/foundir_blur_coverage_official_v2_result_2026-11-06.json']:
        if (ROOT/name).exists():files.append(name)
    write(request,{'event':event,'detail':detail,'utc_unix':time.time(),'files':{n:sha(ROOT/n) for n in files}})
    print('BLUR_COVERAGE_REPORT_DURABLE '+event,flush=True)


def submit(phase,commit):
    p=QUEUE/'jobs'/JOBS[phase]/'job.json'
    if p.exists():
        j=read(p)
        if (j['nodes'],j['entry'],j['args'],j['source_commit'])!=([3],ENTRY,[phase],commit):
            raise RuntimeError('existing job differs; preserve')
        return
    hb=read(QUEUE/'workers/node3/heartbeat.json')
    if time.time()-hb['utc_unix']>60 or hb['state']!='idle':raise RuntimeError('node3 not freshly idle')
    for f in (QUEUE/'jobs').glob('*/job.json'):
        j=read(f)
        if 3 in j['nodes'] and not (f.parent/'node3.ended.json').exists() and not (f.parent/'cancel.json').exists():
            raise RuntimeError('other unfinished node3 job; avoid duplicate launch')
    p=command([str(PY),'-S','smoke/foundir_fullstack_node3_extension_v1.py','submit','--job-id',JOBS[phase],
       '--entry',ENTRY,'--commit',commit,'--purpose','Blur-only fixed200 data-coverage control: '+phase+'; official150 sole quality gate',
       '--sources',*SOURCES,'--arguments',phase])
    print(p.stdout,flush=True)


def wait(phase):
    folder=QUEUE/'jobs'/JOBS[phase]
    while not (folder/'node3.ended.json').exists():
        hb=read(QUEUE/'workers/node3/heartbeat.json')
        print('BLUR_COVERAGE_WAIT '+phase+' age='+str(round(time.time()-hb['utc_unix'],1))+' state='+hb['state'],flush=True)
        if time.time()-hb['utc_unix']>300:raise RuntimeError('stale worker; preserve job, no retry')
        if phase=='train':
            for step in (1,50,100,200):
                if (OUT/'pilot/steps'/f'step_{step:06d}.json').exists():enqueue('step'+str(step),f'固定200步训练已保存step{step}，尚非质量通过。')
        time.sleep(20)
    e=read(folder/'node3.ended.json')
    if e['status']!=0:raise RuntimeError(phase+' exited '+str(e['status']))
    for _ in range(12):
        hb=read(QUEUE/'workers/node3/heartbeat.json')
        if hb['state']=='idle':return
        time.sleep(5)
    raise RuntimeError('completed job worker not idle; no duplicate launch')


def monitor(commit):
    lock=(OUT/'monitor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        cpu('plan')
        enqueue('queued','目的地CPU预检完成，固定缓存→fresh200→官方150已授权链开始。')
        for phase in ('cache','train','official'):
            if phase=='official':cpu('prepare')
            submit(phase,commit);wait(phase)
            if phase=='train':
                for step in (1,50,100,200):enqueue('step'+str(step),f'固定200步已保存step{step}。')
            enqueue(phase+'_ended',phase+'所需GPU launcher退出0；详细工程/指标须专用CPU核验。')
        cpu('finish');enqueue('terminal','完整官方150生成和CPU完整性核验结束；按同一最终200输出判断双门槛。')
        print('BLUR_COVERAGE_MONITOR_END status=0',flush=True)
    except Exception as exc:
        enqueue('failed',str(exc)+'；无自动重试，保留全部运行和资产。')
        raise


def publish(request):
    marker=OUT/'published'/(request['event']+'.json')
    if marker.exists():return
    command(['git','fetch','origin','main'])
    w=Path(tempfile.mkdtemp(prefix='blur-coverage-report-'))
    command(['git','worktree','add','--quiet','--detach',str(w),'origin/main'])
    files=list(request['files'])
    for name,digest in request['files'].items():
        if sha(ROOT/name)!=digest:raise RuntimeError('report changed before publication')
        (w/name).parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,w/name)
    if request['event'] in ('terminal','failed'):
        head='## Blur1024训练来源对照终态 — 2026-11-06\n\n'+request['detail']+'\n'
        result=ROOT/'reports/foundir_blur_coverage_official_v2_result_2026-11-06.json'
        if result.exists():
            v=read(result);head+=f'完整官方150指标：{v["means"]["candidate"]}；25.9/.18联合通过={v["official_Blur_pass"]}。\n'
        head+='仅官方FoundIR作质量门槛，Blur未过前不推进其它类别。不自动追加步数/切换任务。查看reports/foundir_blur_coverage_v2_'+request['event']+'_2026-11-06.{json,md}。保留全部资产和旧失败。\n\n'
        for name in ('AGENTS.md','agents.md','agent.md'):
            p=w/name;old=p.read_text() if p.exists() else '';p.write_text(head+old);files.append(name)
            local=ROOT/name;s=local.read_text()
            if not s.startswith(head):local.write_text(head+s)
    command(['git','add','-f','--',*files],cwd=w);command(['git','diff','--cached','--check'],cwd=w)
    if subprocess.run(['git','diff','--cached','--quiet'],cwd=w).returncode:
        command(['git','commit','-m','Report Blur coverage control '+request['event']],cwd=w)
        command(['git','push','origin','HEAD:main'],cwd=w)
    commit=command(['git','rev-parse','HEAD'],cwd=w).stdout.strip()
    command(['git','fetch','origin','main']);command(['git','merge-base','--is-ancestor',commit,'origin/main'])
    write(marker,{'commit':commit,'event':request['event'],'files':request['files']})
    print('BLUR_COVERAGE_REPORT_PUSHED '+request['event']+' '+commit,flush=True)


def upload():
    lock=(OUT/'upload.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    while True:
        for p in sorted(REQUESTS.glob('*.json'),key=lambda p:read(p)['utc_unix']):
            if (OUT/'published'/p.name).exists():continue
            try:publish(read(p))
            except (OSError,subprocess.SubprocessError,RuntimeError) as exc:
                print('BLUR_COVERAGE_UPLOAD_PENDING '+str(exc),flush=True);break
        if any((REQUESTS/(e+'.json')).exists() for e in ('terminal','failed')) and all((OUT/'published'/p.name).exists() for p in REQUESTS.glob('*.json')):
            print('BLUR_COVERAGE_ALL_REPORTS_PUSHED',flush=True);return
        time.sleep(30)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['monitor','upload']);p.add_argument('--commit');a=p.parse_args()
    if a.phase=='upload':upload()
    else:
        if not a.commit:raise ValueError('published commit required')
        monitor(a.commit)
