#!/usr/bin/env python3
"""Observe only the authorized eight-crop cross; finalize, publish, stop."""
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from smoke.existing_flux_replay_contract_v1 import read,write_new as write,sha
OUT=ROOT/'checkpoints/foundir_fullstack_rain_prefix_cross_v1';MONITOR=OUT/'monitor'
JOB=ROOT/'checkpoints/foundir_fullstack_executor_3x2_h100_v2/jobs/round7_rain_prefix_cross_v1'
RESULT='reports/foundir_fullstack_rain_prefix_cross_v1_result_2026-11-06.json'
REPORT='reports/foundir_fullstack_rain_prefix_cross_v1_terminal_2026-11-06.json'
SELF='smoke/watch_foundir_fullstack_rain_prefix_cross_v1_cpu.py'


def command(args,cwd=ROOT):return subprocess.run(args,cwd=cwd,check=True,capture_output=True,text=True,timeout=60)


def publish(value):
    write(ROOT/REPORT,value);md=ROOT/REPORT.replace('.json','.md')
    text=f"# Rain前缀/读出四组诊断终态\n\n状态：{value['status']}；node0退出码：{value['ended']['status']}，{value['ended']['seconds']:.3f}秒，{value['ended']['allocated_GPU_hours']:.6f} GPU-hours。\n\n"
    text+='仅固定8张历史训练裁图、四组各8输出；没有训练、开发/官方测试读取、逐图选输出或自动后续任务。不是泛化或官方通过证据。\n\n'
    if value.get('result'):
        s=value['result']['summary'];text+='| 前缀 / 末次权重 | PSNR | LPIPS |\n|---|---:|---:|\n'
        for arm,m in s['means'].items():text+=f"| {arm} | {m['psnr']:.8f} | {m['lpips']:.8f} |\n"
        text+=f"\n原版本逐像素/指标匹配：{s['matched_parent_parity']}。本8图是否呈现预定前缀抵消特征：{s['prefix_cancellation_signature_in_this_eight_crop_probe']}。该布尔量不是显著性检验、通用根因结论或训练授权；完整条件差及逐图数据见JSON。\n"
    if not md.exists():md.write_text(text.rstrip()+'\n')
    files=[REPORT,str(md.relative_to(ROOT)),SELF]
    if value.get('result'):files.append(RESULT)
    for attempt in range(3):
        try:
            command(['git','fetch','origin','main']);wt=Path(tempfile.mkdtemp(prefix='rain-cross-report-'))
            command(['git','worktree','add','--quiet','--detach',str(wt),'origin/main'])
            for name in files:(wt/name).parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,wt/name)
            command(['git','add','-f','--',*files],wt);command(['git','diff','--cached','--check'],wt)
            if subprocess.run(['git','diff','--cached','--quiet'],cwd=wt).returncode:
                command(['git','commit','-m','Report fixed Rain prefix readout intervention and actual cost'],wt);command(['git','push','origin','HEAD:main'],wt)
            commit=command(['git','rev-parse','HEAD'],wt).stdout.strip();command(['git','fetch','origin','main']);command(['git','merge-base','--is-ancestor',commit,'origin/main'])
            write(MONITOR/'published.json',{'commit':commit,'report_sha256':sha(ROOT/REPORT),'status':value['status']});print('RAIN_CROSS_REPORT_PUSHED '+commit,flush=True);return
        except (OSError,subprocess.SubprocessError) as exc:print('PUBLISH_RETRY',str(exc),flush=True);time.sleep(10)
    raise RuntimeError('report push failed; no forced update')


def main():
    MONITOR.mkdir(parents=True,exist_ok=True);lock=(MONITOR/'observer.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    while not (JOB/'node0.ended.json').exists():
        print('RAIN_CROSS_WAIT records='+str(len(list((OUT/'records').glob('*.json')))),flush=True);time.sleep(10)
    end=read(JOB/'node0.ended.json');finish=None;result=None
    if end['status']==0:
        marker=MONITOR/'finish.json'
        if not marker.exists():
            start=time.monotonic();env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'2',
                'PYTHONPATH':str(ROOT/'.conda/gpu-train-stable/lib/python3.12/site-packages')+':'+str(ROOT)}
            p=subprocess.run([str(ROOT/'.conda/gpu-train-stable/bin/python'),'-S','smoke/diagnose_foundir_fullstack_rain_prefix_cross_v1.py','finish'],
                             cwd=ROOT,env=env,capture_output=True,text=True,timeout=300)
            write(marker,{'status':p.returncode,'seconds':time.monotonic()-start,'stdout':p.stdout,'stderr':p.stderr})
        finish=read(marker)
        if finish['status']==0:result=read(ROOT/RESULT)
    value={'status':'complete' if result else 'failed','job':'round7_rain_prefix_cross_v1','ended':end,
           'started':read(JOB/'node0.started.json'),'CPU_finish':finish,'result':result,
           'result_sha256':sha(ROOT/RESULT) if result else None,'new_training':False,'official_evaluated':False,'automatic_followup':False}
    publish(value)
    if not result:raise RuntimeError('diagnostic failed; no automatic retry')

if __name__=='__main__':main()
