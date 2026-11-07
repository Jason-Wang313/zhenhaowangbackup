#!/usr/bin/env python3
"""Durable fixed1280 -> official150 chain and independent small-report uploader."""
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
OUT=ROOT/'checkpoints/foundir_blur_exposure_2x2_v2'
OFF=ROOT/'checkpoints/foundir_blur_exposure_official_2x2_v2'
QUEUE=ROOT/'checkpoints/foundir_fullstack_executor_3x2_h100_v2'
PY=ROOT/'.conda/gpu-train-stable/bin/python'
TRAIN='smoke/train_foundir_blur_exposure_2x2_v2.py'
EVAL='smoke/evaluate_foundir_blur_exposure_official_2x2_v2.py'
SOURCES=[TRAIN,EVAL,'smoke/blur_exposure_state_2x2_v2.py','lowlight_restoration/blur_delta_velocity_v1.py',
         'smoke/run_foundir_fullstack_blur_coverage_v2.py','lowlight_restoration/rain_global_batch_v1.py']
JOBS={p:'round14b_blur_exposure_'+p+'_v2' for p in ('train','official')}
ENTRIES={'train':'smoke/run_foundir_fullstack_blur_exposure_2x2_v2.sh','official':'smoke/run_foundir_fullstack_blur_exposure_official_2x2_v2.sh'}
REQUESTS=OUT/'publication_requests'


def command(args,cwd=ROOT,**kwargs):
    if args[0]=='git':kwargs.setdefault('timeout',45)
    return subprocess.run(args,cwd=cwd,text=True,capture_output=True,check=True,**kwargs)


def env():
    return dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',
                TORCH_HOME=str(ROOT/'models/torch'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                PYTHONPATH=str(ROOT)+':'+str(ROOT/'third_party/flux2/src'))


def cpu(name,script,*args):
    marker=OUT/'cpu'/(name+'.json')
    if not marker.exists():
        start=time.monotonic();p=subprocess.run([str(PY),'-u',script,*args],cwd=ROOT,env=env(),text=True,capture_output=True)
        write(marker,{'status':p.returncode,'seconds':time.monotonic()-start,'stdout':p.stdout,'stderr':p.stderr})
    if read(marker)['status']!=0:raise RuntimeError('CPU failed: '+name)


def jobdir(phase,node):
    return QUEUE/'jobs'/(JOBS[phase]+f'_node{node}')


def enqueue(event,detail):
    request=REQUESTS/(event+'.json')
    if request.exists():return
    jobs={}
    for phase in JOBS:
        for node in (1,3):
            folder=jobdir(phase,node)
            if (folder/'job.json').exists():jobs[phase+f'_node{node}']={p.stem:read(p) for p in folder.glob('*.json')}
    latest=sorted((OUT/'steps').glob('step_*.json'));logs=[];p=jobdir('train',1)/'node1.log'
    if p.exists():
        for line in p.read_text(errors='replace').splitlines():
            if line.startswith('BLUR_EXPOSURE_TRAIN {'):logs.append(json.loads(line.split(' ',1)[1]))
    value={'event':event,'detail':detail,'utc_unix':time.time(),'jobs':jobs,'latest_step':read(latest[-1]) if latest else None,
           'training_logs':logs,'CPU_stages':{p.stem:read(p) for p in (OUT/'cpu').glob('*.json')},
           'completed_GPU_hours':sum(v['allocated_GPU_hours'] for j in jobs.values() for k,v in j.items() if k.endswith('.ended')),
           'cost_complete_for_submitted_jobs':all(any(k.endswith('.ended') for k in j) for j in jobs.values()),
           'fixed_final_step':1280,'training_sources':1024,'total_exposures_per_source':10,'global_batch':8,
           'physical_nodes':[1,3],'world':4,'bitwise_2_to4_resume':False,'official_Blur_pass':False,'only_official_quality_gate':True}
    official=ROOT/'reports/foundir_blur_exposure_official_2x2_v2_result_2026-11-07.json'
    if official.exists():
        v=read(official);value['official_result_sha256']=sha(official)
        for k in ('means','gate_checks','official_Blur_pass','PSNR_delta_vs_old','LPIPS_relative_reduction_vs_old_percent'):value[k]=v[k]
    stem=f'reports/foundir_blur_exposure_2x2_v2_{event}_2026-11-07'
    write(ROOT/(stem+'.json'),value)
    metrics=('完整官方150终态：'+json.dumps(value['means']['candidate'],ensure_ascii=False)+'；25.9/.18联合通过='+str(value['official_Blur_pass'])) if 'means' in value else '本轮完整官方结果尚未产生，不能声称修复成功。'
    (ROOT/(stem+'.md')).write_text('# Blur1024来源十遍曝光：'+event+'\n\n'+detail+
        '\n\n原step200模型/Adam迁移到当前健康的物理node1/node3，各两H100。global8/采样位置保持；四rank RNG重建，不称逐位等价两卡续训。固定最终1280→完整官方150，不用8图拟合权重、不选中途官方最好检查点。'+
        '\n\n已结束子job费用合计'+str(value['completed_GPU_hours'])+' GPU-hours；本轮已提交子job费用是否齐全='+str(value['cost_complete_for_submitted_jobs'])+'。历史另计保留。'+
        '\n\n'+metrics+'\n')
    files=[stem+'.json',stem+'.md']
    for p in [ROOT/'reports/foundir_blur_exposure_2x2_v2_preflight_2026-11-07.json',ROOT/'reports/foundir_blur_exposure_2x2_v2_train_result_2026-11-07.json',ROOT/'reports/foundir_blur_exposure_official_2x2_v2_preflight_2026-11-07.json',official,*sorted((ROOT/'reports').glob('foundir_blur_exposure_2x2_v2_step*_state_2026-11-07.json'))]:
        if p.exists():files.append(str(p.relative_to(ROOT)))
    write(request,{'event':event,'detail':detail,'utc_unix':time.time(),'files':{n:sha(ROOT/n) for n in files}})
    print('BLUR_EXPOSURE_REPORT_DURABLE '+event,flush=True)


def submit(phase,commit):
    needed=[]
    for node in (1,3):
        folder=jobdir(phase,node)
        if (folder/'job.json').exists():
            j=read(folder/'job.json')
            if (j['nodes'],j['entry'],j['source_commit'],j['args'])!=([node],ENTRIES[phase],commit,[]):raise RuntimeError('existing paired job differs')
        else:needed.append(node)
    for node in needed:
        h=read(QUEUE/'workers'/f'node{node}/heartbeat.json')
        if time.time()-h['utc_unix']>60 or h['state']!='idle':raise RuntimeError('worker not fresh idle: '+str(node))
        for p in (QUEUE/'jobs').glob('*/job.json'):
            j=read(p)
            if node in j['nodes'] and not (p.parent/f'node{node}.ended.json').exists() and not (p.parent/'cancel.json').exists():raise RuntimeError('other unfinished node job')
    for node in needed:
        script='smoke/foundir_fullstack_node3_extension_v1.py' if node==3 else 'smoke/foundir_fullstack_executor_3x2_v2.py'
        args=[str(PY),'-S',script,'submit','--job-id',JOBS[phase]+f'_node{node}','--entry',ENTRIES[phase],
              '--commit',commit,'--purpose','Blur fixed1024 step200->1280 then official150; paired nodes1/3: '+phase]
        if node==1:args+=['--nodes','1']
        args+=['--sources',*SOURCES]
        print(command(args).stdout,flush=True)


def cancel(phase,reason,commit=None):
    for node in (1,3):
        folder=jobdir(phase,node);spec=folder/'job.json'
        if not spec.exists() or (folder/f'node{node}.ended.json').exists() or (folder/'cancel.json').exists():continue
        job=read(spec)
        if job['nodes']==[node] and job['entry']==ENTRIES[phase] and (commit is None or job['source_commit']==commit):
            write(folder/'cancel.json',{'reason':reason,'automatic_retry':False})


def wait(phase):
    while True:
        ends=[read(jobdir(phase,n)/f'node{n}.ended.json') for n in (1,3) if (jobdir(phase,n)/f'node{n}.ended.json').exists()]
        if any(e['status']!=0 for e in ends):
            cancel(phase,'paired_peer_failed');raise RuntimeError('paired GPU job failed: '+phase)
        if phase=='train':
            for step in (201,512,1024,1280):
                if (OUT/'steps'/f'step_{step:06d}.json').exists() and not (REQUESTS/f'step{step}.json').exists():
                    cpu('verify'+str(step),'smoke/verify_foundir_blur_exposure_step_2x2_v2_cpu.py',str(step))
                    enqueue('step'+str(step),f'完整step{step}实际checkpoint/Adam/四rank RNG/采样核验；不是质量通过。')
        if len(ends)==2:break
        time.sleep(15)
    for _ in range(12):
        if all(read(QUEUE/'workers'/f'node{n}/heartbeat.json')['state']=='idle' for n in (1,3)):return
        time.sleep(5)
    raise RuntimeError('paired workers not idle after completion')


def monitor(commit):
    lock=(OUT/'monitor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        cpu('official_plan',EVAL,'plan')
        submit('train',commit);enqueue('queued','两节点四卡固定曝光链已提交，真实运行以step/心跳为准。');wait('train')
        cpu('train_finish',TRAIN,'finish');enqueue('train_ended','固定1280步训练结束并CPU核验，转最终权重官方150。')
        cpu('official_prepare',EVAL,'prepare');submit('official',commit);wait('official')
        cpu('official_finish',EVAL,'finish');enqueue('terminal','完整官方150结束并CPU核验，按同一最终1280输出判断双门槛。')
        print('BLUR_EXPOSURE_MONITOR_END status=0',flush=True)
    except Exception as exc:
        for phase in JOBS:cancel(phase,'controller_failed_preserve_assets',commit)
        enqueue('failed',str(exc)+'；保留全部资产，无自动重试。');raise
    finally:lock.close()


def publish(request):
    marker=OUT/'published'/(request['event']+'.json')
    if marker.exists():return
    command(['git','fetch','origin','main']);w=Path(tempfile.mkdtemp(prefix='blur-exposure-report-'))
    command(['git','worktree','add','--quiet','--detach',str(w),'origin/main'])
    files=list(request['files'])
    for name,digest in request['files'].items():
        if sha(ROOT/name)!=digest:raise RuntimeError('report changed before publication')
        (w/name).parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,w/name)
    if request['event'] in ('terminal','failed'):
        header='## Blur1024来源十遍曝光终态 — 2026-11-07\n\n'+request['detail']+'\n'
        result=ROOT/'reports/foundir_blur_exposure_official_2x2_v2_result_2026-11-07.json'
        if result.exists():
            v=read(result);header+=f'完整官方150：{v["means"]["candidate"]}；25.9/.18联合通过={v["official_Blur_pass"]}。\n'
        header+='只用官方作质量门槛，Blur未过不推进其它类别。完整自主修复授权继续；本轮不自动追加旧步数。见reports/foundir_blur_exposure_2x2_v2_'+request['event']+'_2026-11-07.{json,md}。全部资产保留。\n\n'
        for name in ('AGENTS.md','agents.md','agent.md'):
            p=w/name;s=p.read_text() if p.exists() else '';p.write_text(header+s);files.append(name)
            local=ROOT/name;s=local.read_text()
            if not s.startswith(header):local.write_text(header+s)
    command(['git','add','-f','--',*files],cwd=w);command(['git','diff','--cached','--check'],cwd=w)
    if subprocess.run(['git','diff','--cached','--quiet'],cwd=w).returncode:
        command(['git','commit','-m','Report fixed1024 Blur exposure '+request['event']],cwd=w)
        command(['git','push','origin','HEAD:main'],cwd=w)
    commit=command(['git','rev-parse','HEAD'],cwd=w).stdout.strip()
    command(['git','fetch','origin','main']);command(['git','merge-base','--is-ancestor',commit,'origin/main'])
    write(marker,{'commit':commit,'event':request['event'],'files':request['files']})
    print('BLUR_EXPOSURE_REPORT_PUSHED '+request['event']+' '+commit,flush=True)


def upload():
    lock=(OUT/'upload.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    while True:
        for p in sorted(REQUESTS.glob('*.json'),key=lambda p:read(p)['utc_unix']):
            if (OUT/'published'/p.name).exists():continue
            try:publish(read(p))
            except (OSError,subprocess.SubprocessError,RuntimeError) as exc:
                print('BLUR_EXPOSURE_UPLOAD_PENDING '+str(exc),flush=True);break
        if any((REQUESTS/(e+'.json')).exists() for e in ('terminal','failed')) and all((OUT/'published'/p.name).exists() for p in REQUESTS.glob('*.json')):
            print('BLUR_EXPOSURE_ALL_REPORTS_PUSHED',flush=True);return
        time.sleep(30)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['monitor','upload']);p.add_argument('--commit');a=p.parse_args()
    if a.phase=='upload':upload()
    else:
        if not a.commit:raise ValueError('published commit required')
        monitor(a.commit)
