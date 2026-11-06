#!/usr/bin/env python3
"""Blur-only fixed cache/fit/development chain; no other category is submitted."""
import argparse
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from smoke import watch_foundir_fullstack_noise_blur_v1_cpu as queue
from smoke.existing_flux_replay_contract_v1 import read, write_new as write, sha

OUT = ROOT / 'checkpoints/foundir_fullstack_blur_first_v1'
SELF = 'smoke/watch_foundir_fullstack_blur_first_v1_cpu.py'
SCRIPT = 'smoke/run_foundir_fullstack_frequency_repair_v1.py'
ENTRY = 'smoke/run_foundir_fullstack_frequency_repair_v1_gpu.sh'
CACHE_JOB = 'round8_blur_frequency_cache_v1'
SCORE_JOB = 'round8_blur_frequency_score_v1'
RESULT = 'reports/foundir_fullstack_blur_frequency_repair_v1_result_2026-11-06.json'
SOURCES = [SCRIPT, SELF, 'lowlight_restoration/flux_frequency_repair_v1.py',
           'smoke/watch_foundir_fullstack_noise_blur_v1_cpu.py']


def cpu(phase):
    if phase not in ('verify', 'fit', 'finish'):
        raise ValueError('only Blur verify/fit/finish phases permitted')
    path = OUT / 'cpu' / (phase + '.json')
    if not path.exists():
        start = time.monotonic()
        result = subprocess.run([str(queue.PY), '-S', SCRIPT, phase, '--task', 'blur'],
                                cwd=ROOT, env=queue.env(), text=True, capture_output=True, timeout=1800)
        write(path, {'status': result.returncode, 'seconds': time.monotonic() - start,
                     'stdout': result.stdout, 'stderr': result.stderr})
    if read(path)['status'] != 0:
        raise RuntimeError('Blur CPU stage failed: ' + str(path))


def submit(phase, nodes, commit):
    if (phase, nodes) not in (('cache', [0, 1, 2]), ('score', [0])):
        raise ValueError('fixed Blur cache/score topology required')
    job = CACHE_JOB if phase == 'cache' else SCORE_JOB
    path = queue.QUEUE / 'jobs' / job / 'job.json'
    if path.exists():
        value = read(path)
        if (value['entry'] != ENTRY or value['args'] != ['blur', phase]
                or value['nodes'] != nodes or value['source_commit'] != commit):
            raise ValueError('existing Blur job identity differs; preserve it')
        return job
    queue.ready_idle(nodes)
    queue.command([str(queue.PY), '-S', 'smoke/foundir_fullstack_executor_3x2_v2.py', 'submit',
                   '--job-id', job, '--entry', ENTRY, '--commit', commit,
                   '--purpose', 'Blur only: fixed training/development experiment; other types gated on official Blur pass',
                   '--nodes', *map(str, nodes), '--sources', *SOURCES, '--arguments', 'blur', phase])
    return job


def publish(event, detail):
    marker = OUT / 'published' / (event + '.json')
    if marker.exists():
        return
    stem = f'reports/foundir_fullstack_blur_first_v1_{event}_2026-11-06'
    path = ROOT / (stem + '.json')
    if not path.exists():
        jobs = []
        for name in (CACHE_JOB, SCORE_JOB):
            directory = queue.QUEUE / 'jobs' / name
            if not (directory / 'job.json').exists():
                continue
            job = read(directory / 'job.json')
            ends = [read(directory / f'node{n}.ended.json') if (directory / f'node{n}.ended.json').exists()
                    else None for n in job['nodes']]
            starts = [read(directory / f'node{n}.started.json') if (directory / f'node{n}.started.json').exists()
                      else None for n in job['nodes']]
            jobs.append({'job_id': name, 'nodes': job['nodes'], 'started': starts, 'ended': ends,
                         'completed_node_GPU_hours': sum(e['allocated_GPU_hours'] for e in ends if e),
                         'cost_complete': all(e is not None for e in ends)})
        value = {'event': event, 'detail': detail, 'utc_unix': time.time(), 'jobs': jobs,
                 'CPU_stages': {p.stem: read(p) for p in sorted((OUT / 'cpu').glob('*.json'))},
                 'task': 'blur', 'other_categories_submitted': False,
                 'global_GPU_hour_limit': None, 'official_pass_required_before_other_categories': True,
                 'official_evaluated_by_this_chain': False}
        if event == 'development_terminal':
            value['result'] = read(ROOT / RESULT)
            value['result_sha256'] = sha(ROOT / RESULT)
        write(path, value)
    value = read(path)
    md = ROOT / (stem + '.md')
    if not md.exists():
        text = f'# Blur优先：{event}\n\n{detail}\n\n'
        text += '只执行Blur。其它类别等待Blur官方PSNR≥25.9且LPIPS≤.18；开发通过不能代替官方通过。\n\n'
        text += '逐节点退出码、实际费用、CPU耗时和结果见JSON。无累计GPU-hours上限，保留原固定实验范围和故障停止条件。\n'
        if value.get('result'):
            text += '\n| 集合 | 原PSNR | 新PSNR | 原LPIPS | 新LPIPS |\n|---|---:|---:|---:|---:|\n'
            for role, summary in value['result']['summary'].items():
                old, new = summary['baseline'], summary['candidate']
                text += f'| {role} | {old["psnr"]:.8f} | {new["psnr"]:.8f} | {old["lpips"]:.8f} | {new["lpips"]:.8f} |\n'
        md.write_text(text.rstrip() + '\n')
    files = [stem + '.json', stem + '.md']
    if value.get('result'):
        files += [RESULT, 'reports/foundir_fullstack_blur_frequency_repair_v1_fit_2026-11-06.json']
    for attempt in range(3):
        try:
            queue.command(['git', 'fetch', 'origin', 'main'])
            worktree = Path(tempfile.mkdtemp(prefix='blur-first-report-'))
            queue.command(['git', 'worktree', 'add', '--quiet', '--detach', str(worktree), 'origin/main'])
            for name in files:
                shutil.copyfile(ROOT / name, worktree / name)
            queue.command(['git', 'add', '-f', '--', *files], cwd=worktree)
            queue.command(['git', 'diff', '--cached', '--check'], cwd=worktree)
            if subprocess.run(['git', 'diff', '--cached', '--quiet'], cwd=worktree).returncode:
                queue.command(['git', 'commit', '-m', 'Report Blur-first ' + event], cwd=worktree)
                queue.command(['git', 'push', 'origin', 'HEAD:main'], cwd=worktree)
            commit = queue.command(['git', 'rev-parse', 'HEAD'], cwd=worktree).stdout.strip()
            queue.command(['git', 'fetch', 'origin', 'main'])
            queue.command(['git', 'merge-base', '--is-ancestor', commit, 'origin/main'])
            write(marker, {'commit': commit, 'report_sha256': sha(path), 'event': event})
            print('BLUR_FIRST_REPORT_PUSHED ' + event + ' ' + commit, flush=True)
            return
        except (OSError, subprocess.SubprocessError) as exc:
            print('BLUR_FIRST_PUBLICATION_RETRY ' + str(exc), flush=True)
            time.sleep(10)
    raise RuntimeError('Blur report publication failed; no forced push')


def main(commit):
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT / 'observer.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        cpu('verify')
        job = submit('cache', [0, 1, 2], commit)
        publish('queued', '已提交Blur160图固定专家缓存，随后仅训练侧六系数拟合和开发评分。')
        queue.wait(job, [0, 1, 2])
        cpu('fit')
        queue.wait(submit('score', [0], commit), [0])
        cpu('finish')
        publish('development_terminal', 'Blur固定128训练/32开发对照完成；根据完整结果决定是否进入官方150图评价。')
    except Exception as exc:
        publish('failed', str(exc))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--commit', required=True)
    main(parser.parse_args().commit)
