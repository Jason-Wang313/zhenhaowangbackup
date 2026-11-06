#!/usr/bin/env python3
"""Fixed Blur-only pipeline, automatic engineering gates and honest small reports."""
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
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from smoke.existing_flux_replay_contract_v1 import read, sha, write_new as write
OUT = ROOT/'checkpoints/foundir_blur_delta_velocity_v1'
QUEUE = ROOT/'checkpoints/foundir_fullstack_executor_3x2_h100_v2'
PY = ROOT/'.conda/gpu-train-stable/bin/python'
ENTRY = 'smoke/run_foundir_fullstack_blur_delta_v1_gpu.sh'
SCRIPT = 'smoke/run_foundir_fullstack_blur_delta_v1.py'
SOURCES = [SCRIPT, 'smoke/blur_delta_state_v1.py', 'lowlight_restoration/blur_delta_velocity_v1.py']
JOBS = {phase: 'round9_blur_delta_'+phase+'_v1' for phase in ('qualify', 'cache', 'train', 'evaluate')}


def command(args, cwd=ROOT, **kwargs):
    return subprocess.run(args, cwd=cwd, check=True, text=True, capture_output=True, **kwargs)


def cpu(phase):
    path = OUT/'cpu'/(phase+'.json')
    if not path.exists():
        begin = time.monotonic()
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1',
                   TORCH_HOME=str(ROOT/'models/torch'), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                   PYTHONPATH=str(ROOT)+':'+str(ROOT/'third_party/flux2/src'))
        result = subprocess.run([str(PY), '-u', SCRIPT, phase], cwd=ROOT, env=env,
                                text=True, capture_output=True)
        write(path, {'status': result.returncode, 'seconds': time.monotonic()-begin,
                     'stdout': result.stdout, 'stderr': result.stderr})
    if read(path)['status']:
        raise RuntimeError('CPU verification failed: '+str(path))


def submit(phase, commit):
    file = QUEUE/'jobs'/JOBS[phase]/'job.json'
    if file.exists():
        job = read(file)
        if (job['entry'], job['nodes'], job['args'], job['source_commit']) != (ENTRY, [3], [phase], commit):
            raise ValueError('existing job differs; preserve it')
        return
    heartbeat = read(QUEUE/'workers/node3/heartbeat.json')
    if time.time()-heartbeat['utc_unix'] > 60 or heartbeat['state'] != 'idle':
        raise RuntimeError('node3 not freshly idle')
    for path in (QUEUE/'jobs').glob('*/job.json'):
        job = read(path)
        if 3 in job['nodes'] and not (path.parent/'node3.ended.json').exists() and not (path.parent/'cancel.json').exists():
            raise RuntimeError('another node3 job pending; do not double-submit')
    result = command([str(PY), '-S', 'smoke/foundir_fullstack_node3_extension_v1.py', 'submit',
                      '--job-id', JOBS[phase], '--entry', ENTRY, '--commit', commit,
                      '--purpose', 'Blur only: '+phase+' fixed delta-velocity pilot; other types blocked on official Blur pass',
                      '--sources', *SOURCES, '--arguments', phase])
    print(result.stdout, flush=True)


def publish(event, detail):
    marker = OUT/'published'/(event+'.json')
    if marker.exists():
        return
    stem = f'reports/foundir_blur_delta_velocity_v1_{event}_2026-11-06'
    path = ROOT/(stem+'.json')
    if not path.exists():
        jobs = {}
        for phase, name in JOBS.items():
            folder = QUEUE/'jobs'/name
            if (folder/'job.json').exists():
                jobs[phase] = {p.stem: read(p) for p in folder.glob('*.json') if p.name != 'job.json'}
        stages = {p.stem: read(p) for p in (OUT/'cpu').glob('*.json')}
        # Full optimizer states remain on shared storage; reports include receipts/history only.
        steps = sorted((OUT/'pilot/steps').glob('step_*.json'))
        value = {'event': event, 'detail': detail, 'utc_unix': time.time(),
                 'receipt_sha256': sha(OUT/'receipt.json'), 'jobs': jobs, 'CPU_stages': stages,
                 'latest_pilot_step_receipt': read(steps[-1]) if steps else None,
                 'completed_GPU_hours': sum(x['node3.ended']['allocated_GPU_hours'] for x in jobs.values() if 'node3.ended' in x),
                 'cost_complete': all('node3.ended' in x for x in jobs.values()),
                 'official_evaluated': False, 'other_categories_submitted': False,
                 'cumulative_GPU_hour_cap': None, 'official_Blur_pass_required_before_other_types': True}
        write(path, value)
    value = read(path)
    md = ROOT/(stem+'.md')
    if not md.exists():
        md.write_text(f'# Blur delta-velocity：{event}\n\n{detail}\n\n'
                      '本轮仅Blur，单节点node3两张H100；其它类别等待完整官方150张25.9/.18通过。'
                      '工程通过、训练拟合和开发结果均不等于官方通过。\n\n'
                      f'已结束阶段费用：{value["completed_GPU_hours"]:.8f} GPU-hours；'
                      f'费用是否完整：{value["cost_complete"]}。详细退出码、耗时及进度见同名JSON。\n')
    files = [stem+'.json', stem+'.md']
    for phase in ('preflight', 'qualification', 'result'):
        name = f'reports/foundir_blur_delta_velocity_v1_{phase}_2026-11-06.json'
        if (ROOT/name).exists():
            files.append(name)
    for attempt in range(3):
        try:
            command(['git', 'fetch', 'origin', 'main'])
            worktree = Path(tempfile.mkdtemp(prefix='blur-delta-report-'))
            command(['git', 'worktree', 'add', '--quiet', '--detach', str(worktree), 'origin/main'])
            for name in files:
                shutil.copyfile(ROOT/name, worktree/name)
            command(['git', 'add', '-f', '--', *files], cwd=worktree)
            command(['git', 'diff', '--cached', '--check'], cwd=worktree)
            if subprocess.run(['git', 'diff', '--cached', '--quiet'], cwd=worktree).returncode:
                command(['git', 'commit', '-m', 'Report Blur delta pilot '+event], cwd=worktree)
                command(['git', 'push', 'origin', 'HEAD:main'], cwd=worktree)
            commit = command(['git', 'rev-parse', 'HEAD'], cwd=worktree).stdout.strip()
            command(['git', 'fetch', 'origin', 'main'])
            command(['git', 'merge-base', '--is-ancestor', commit, 'origin/main'])
            write(marker, {'commit': commit, 'event': event, 'report_sha256': sha(path)})
            print('BLUR_DELTA_REPORT_PUSHED '+event+' '+commit, flush=True)
            return
        except (OSError, subprocess.SubprocessError) as exc:
            print('BLUR_DELTA_REPORT_RETRY '+str(exc), flush=True)
            time.sleep(10)
    raise RuntimeError('small report publication failed; no forced push')


def wait(phase):
    folder = QUEUE/'jobs'/JOBS[phase]
    while not (folder/'node3.ended.json').exists():
        hb = read(QUEUE/'workers/node3/heartbeat.json')
        print(f'BLUR_DELTA_WAIT phase={phase} heartbeat_age={time.time()-hb["utc_unix"]:.1f} state={hb["state"]}', flush=True)
        if phase == 'train':
            for step in (1, 50, 100, 200):
                if (OUT/'pilot/steps'/f'step_{step:06d}.json').exists():
                    publish(f'step{step}', f'固定200步Blur小试验已完整落盘step{step}；尚无本轮官方结果。')
        if time.time()-hb['utc_unix'] > 300:
            raise RuntimeError('node3 heartbeat stale; preserve running job, no duplicate launch')
        time.sleep(20)
    ended = read(folder/'node3.ended.json')
    if ended['status'] != 0:
        raise RuntimeError(f'{phase} exited status={ended["status"]}; no automatic retry')
    # Worker writes ended before its next idle heartbeat; wait briefly, not resubmit.
    for _ in range(12):
        if read(QUEUE/'workers/node3/heartbeat.json')['state'] == 'idle':
            return
        time.sleep(5)
    raise RuntimeError('job ended but worker not idle')


def main(commit):
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT/'observer.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        submit('qualify', commit)
        publish('queued', '已提交两步Blur工程验收。通过CPU收尾后依次缓存、fresh200步、完整160图评价；失败停链，不提交官方或其它类别。')
        wait('qualify')
        cpu('verify-qualification')
        publish('qualified', '两步CUDA工程验收及CPU独立核验通过；此结论只覆盖初始化、梯度、两卡汇总及保存恢复。')
        submit('cache', commit)
        wait('cache')
        submit('train', commit)
        wait('train')
        submit('evaluate', commit)
        wait('evaluate')
        cpu('finish')
        publish('terminal', '固定Blur200步训练与128训练/32开发评价结束；完整指标见result JSON，不自动开始官方评价。')
    except Exception as exc:
        publish('failed', str(exc))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--commit', required=True)
    main(parser.parse_args().commit)
