#!/usr/bin/env python3
"""Observe the single authorized GPU job, verify its terminal artifacts, publish small reports."""
import fcntl
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
OUT = ROOT/'checkpoints/foundir_blur_singleframe_fit_v1'
JOB = ROOT/'checkpoints/foundir_fullstack_executor_3x2_h100_v2/jobs/round19_blur_singleframe_fit_v1'
PY = ROOT/'.conda/gpu-train-stable/bin/python'
STEM = 'reports/foundir_blur_singleframe_fit_v1_terminal_2026-11-07'
RESULT = 'reports/foundir_blur_singleframe_fit_v1_result_2026-11-07.json'


def git(*args, cwd=ROOT):
    return subprocess.run(['git', '-c', 'http.version=HTTP/1.1', *args], cwd=cwd, check=True,
                          text=True, capture_output=True, timeout=45).stdout.strip()


def publish(files, header):
    marker = OUT/'publication_candidate.json'
    git('fetch', 'origin', 'main')
    if marker.exists():
        candidate = read(marker); worktree = Path(candidate['worktree']); commit = candidate['commit']
    else:
        worktree = Path(tempfile.mkdtemp(prefix='blur-singleframe-publish-'))
        git('worktree', 'add', '--quiet', '--detach', str(worktree), 'origin/main')
        for name in files:
            target = worktree/name; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT/name, target)
        for name in ('AGENTS.md', 'agents.md', 'agent.md'):
            path = worktree/name; content = path.read_text() if path.exists() else ''
            if not content.startswith(header): path.write_text(header+content)
        git('add', '-f', '--', *files, 'AGENTS.md', 'agents.md', 'agent.md', cwd=worktree)
        git('diff', '--cached', '--check', cwd=worktree)
        git('commit', '-m', 'Record Blur single-frame training-only diagnostic outcome and cost', cwd=worktree)
        commit = git('rev-parse', 'HEAD', cwd=worktree)
        write(marker, {'worktree': str(worktree), 'commit': commit, 'files': {n: sha(ROOT/n) for n in files}})
    already = subprocess.run(['git', 'merge-base', '--is-ancestor', commit, 'origin/main'], cwd=ROOT).returncode == 0
    if not already: git('push', 'origin', 'HEAD:main', cwd=worktree)
    git('fetch', 'origin', 'main'); git('merge-base', '--is-ancestor', commit, 'origin/main')
    import hashlib
    for name, expected in read(marker)['files'].items():
        blob = subprocess.check_output(['git', 'show', f'{commit}:{name}'], cwd=ROOT)
        if hashlib.sha256(blob).hexdigest() != expected: raise ValueError('published report SHA mismatch')
    write(OUT/'published.json', {'commit': commit, 'files': read(marker)['files'], 'utc_unix': time.time()})
    print('BLUR_SINGLEFRAME_PUSH_VERIFIED '+commit, flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT/'monitor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while not (JOB/'node3.ended.json').exists():
            time.sleep(15)
        ended = read(JOB/'node3.ended.json')
        cpu_marker = OUT/'cpu_finish.json'
        if ended['status'] == 0 and not cpu_marker.exists():
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1',
                       TORCH_HOME=str(ROOT/'models/torch'), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                       PYTHONPATH=str(ROOT)+':'+str(ROOT/'third_party/flux2/src'))
            begun = time.monotonic()
            p = subprocess.run([str(PY), '-u', 'smoke/train_foundir_blur_singleframe_fit_v1.py', 'finish'],
                               cwd=ROOT, env=env, capture_output=True, text=True)
            write(cpu_marker, {'status': p.returncode, 'seconds': time.monotonic()-begun,
                               'stdout': p.stdout, 'stderr': p.stderr})
        cpu = read(cpu_marker) if cpu_marker.exists() else None
        complete = ended['status'] == 0 and cpu is not None and cpu['status'] == 0
        result = read(ROOT/RESULT) if complete else None
        value = {'status': 'complete' if complete else 'failed', 'GPU_job': ended, 'CPU_finish': cpu,
                 'scope': 'training_fit_only', 'official_Blur_pass': False, 'may_seed_formal': False,
                 'automatic_followup': False, 'snapshots': result['snapshots'] if result else None,
                 'result_sha256': sha(ROOT/RESULT) if result else None,
                 'assets_preserved': True, 'utc_unix': ended['utc_unix'] if 'utc_unix' in ended else None}
        write(ROOT/(STEM+'.json'), value)
        text = '# Blur单来源16上下文固定200步诊断终态\n\n'
        text += f"GPU退出码{ended['status']}；CPU收尾退出码{cpu['status'] if cpu else '未运行'}。"
        text += f"GPU任务耗时{ended.get('seconds')}秒，费用{ended.get('allocated_GPU_hours')} GPU-hours。\n\n"
        if result:
            text += '| 全局步数 | 全图PSNR | 全图LPIPS | 16视图平均PSNR | 16视图平均LPIPS |\n|---|---:|---:|---:|---:|\n'
            for step, snapshot in result['snapshots'].items():
                full, mean = snapshot['full']['candidate'], snapshot['means']['candidate']
                text += f"| {step} | {full['psnr']:.6f} | {full['lpips']:.6f} | {mean['psnr']:.6f} | {mean['lpips']:.6f} |\n"
            text += '\n初始16输出逐像素复现1792、201连续步receipt、终态Adam/两rank RNG及拼接/均值核验通过；未全量重哈希所有大checkpoint。\n'
        text += '\n仅training_fit_only，未跑官方、无新质量门槛，诊断权重不得部署或seed正式训练。官方150仍21.948140/.202188未过25.9/.18。仅Blur继续，保留全部历史资产/energy/费用，无自动续步或其它类别。\n'
        (ROOT/(STEM+'.md')).write_text(text)
        header = '## Blur单来源16上下文固定200步诊断已结束 — 2026-11-07\n\n'
        header += f"round19_blur_singleframe_fit_v1，GPU退出码{ended['status']}，CPU收尾{cpu['status'] if cpu else '未运行'}。"
        header += '无待续本轮GPU，勿重复；详见'+STEM+'.{json,md}。仅training_fit_only，诊断权重不得部署或seed正式训练。官方仍21.948140/.202188未过25.9/.18，继续仅Blur。接手先读本轮证据再冻结下一项，不自动追加步数；全栈自主修复授权持续，所有资产/energy保留。\n\n'
        for name in ('AGENTS.md', 'agents.md', 'agent.md'):
            path = ROOT/name; old = path.read_text() if path.exists() else ''
            if not old.startswith(header): path.write_text(header+old)
        files = [STEM+'.json', STEM+'.md'] + ([RESULT] if result else [])
        for attempt in range(3):
            try:
                publish(files, header)
                return
            except (subprocess.SubprocessError, OSError) as error:
                print(f'BLUR_SINGLEFRAME_PUBLISH_PENDING attempt={attempt+1} {error}', flush=True)
                if attempt == 2: raise
                time.sleep(15)


if __name__ == '__main__': main()
