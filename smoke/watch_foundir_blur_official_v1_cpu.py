#!/usr/bin/env python3
"""User-directed continuation: completed200 -> official150 only -> verify/publish."""
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
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from smoke import watch_foundir_blur_delta_network_v2_cpu as prior
from smoke.existing_flux_replay_contract_v1 import read, sha, write_new as write
OUT = ROOT/'checkpoints/foundir_blur_delta_official_v1'
QUEUE = prior.prior.QUEUE
PY = prior.prior.PY
SCRIPT = 'smoke/evaluate_foundir_blur_delta_official_v1.py'
ENTRY = 'smoke/run_foundir_fullstack_blur_delta_official_v1.sh'
JOB = 'round9_blur_delta_official150_v1'


def command(args, cwd=ROOT):
    return subprocess.run(args, cwd=cwd, check=True, text=True, capture_output=True, timeout=45)


def cpu(phase):
    file = OUT/'cpu'/(phase+'.json')
    if not file.exists():
        begin = time.monotonic()
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1',
                   TORCH_HOME=str(ROOT/'models/torch'), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                   PYTHONPATH=str(ROOT)+':'+str(ROOT/'third_party/flux2/src'))
        r = subprocess.run([str(PY), '-u', SCRIPT, phase], cwd=ROOT, env=env, text=True, capture_output=True)
        write(file, {'status': r.returncode, 'seconds': time.monotonic()-begin,
                     'stdout': r.stdout, 'stderr': r.stderr})
    if read(file)['status'] != 0: raise RuntimeError('official CPU phase failed: '+phase)


def submit(commit):
    file = QUEUE/'jobs'/JOB/'job.json'
    if file.exists():
        j = read(file)
        if (j['entry'], j['nodes'], j['args'], j['source_commit']) != (ENTRY, [3], [], commit):
            raise ValueError('existing official job differs')
        return
    heartbeat = read(QUEUE/'workers/node3/heartbeat.json')
    if time.time()-heartbeat['utc_unix'] > 60 or heartbeat['state'] != 'idle':
        raise RuntimeError('node3 not freshly idle')
    result = command([str(PY), '-S', 'smoke/foundir_fullstack_node3_extension_v1.py', 'submit',
              '--job-id', JOB, '--entry', ENTRY, '--commit', commit,
              '--purpose', 'User requests official Blur150 only after fixed200; no development set or other categories',
              '--sources', SCRIPT, 'lowlight_restoration/blur_delta_velocity_v1.py',
              'smoke/run_foundir_fullstack_blur_delta_v1.py', 'smoke/blur_delta_state_v1.py'])
    print(result.stdout, flush=True)


def publish(event, detail):
    marker = OUT/'published'/(event+'.json')
    if marker.exists(): return
    stem = f'reports/foundir_blur_delta_official_v1_{event}_2026-11-06'
    path = ROOT/(stem+'.json'); result_path = ROOT/'reports/foundir_blur_delta_official_v1_result_2026-11-06.json'
    if not path.exists():
        jobs = {}
        for name in [*prior.prior.JOBS.values(), JOB]:
            p = QUEUE/'jobs'/name
            if not (p/'job.json').exists(): continue
            jobs[name] = {f.stem: read(f) for f in p.glob('node3.*.json')}
        value = {'event': event, 'detail': detail, 'utc_unix': time.time(), 'jobs': jobs,
                 'completed_GPU_hours': sum(v['node3.ended']['allocated_GPU_hours'] for v in jobs.values() if 'node3.ended' in v),
                 'cost_complete': all('node3.ended' in v for v in jobs.values()),
                 'CPU_stages': {f.stem: read(f) for f in (OUT/'cpu').glob('*.json')},
                 'official_only_per_latest_user': True, 'other_evaluation_populations_run': False,
                 'other_categories_run': False, 'independent_blind_test': False}
        if result_path.exists():
            value['result_sha256'] = sha(result_path)
            result = read(result_path)
            value['means'] = result['means']; value['official_Blur_pass'] = result['official_Blur_pass']
        write(path, value)
    value = read(path); md = ROOT/(stem+'.md')
    if not md.exists():
        text = '# Blur最终200步：官方150 '+event+'\n\n'+detail+'\n\n'
        text += '依用户最新指令，仅官方Blur150作为验收，不运行其它验证集。官方GT仅在输出固定后评分；反复开发使用，非独立盲测。\n\n'
        text += f'已结束阶段费用 {value["completed_GPU_hours"]:.8f} GPU-hours；费用是否完整：{value["cost_complete"]}。逐阶段退出码及耗时见JSON。\n'
        if 'means' in value:
            text += '\n| 方法 | PSNR | LPIPS | SSIM |\n|---|---:|---:|---:|\n'
            for method, m in value['means'].items(): text += f'| {method} | {m["psnr"]:.8f} | {m["lpips"]:.8f} | {m["ssim"]:.8f} |\n'
            text += f'\n官方25.9/.18联合通过：{value["official_Blur_pass"]}。不得只报较好的单项。\n'
        md.write_text(text)
    files = [stem+'.json', stem+'.md']
    for name in ['reports/foundir_blur_delta_official_v1_preflight_2026-11-06.json',
                 'reports/foundir_blur_delta_official_v1_result_2026-11-06.json']:
        if (ROOT/name).exists(): files.append(name)
    last_error = None
    for _ in range(3):
        try:
            command(['git', 'fetch', 'origin', 'main'])
            w = Path(tempfile.mkdtemp(prefix='blur-official-report-'))
            command(['git', 'worktree', 'add', '--quiet', '--detach', str(w), 'origin/main'])
            for name in files: shutil.copyfile(ROOT/name, w/name)
            command(['git', 'add', '-f', '--', *files], w); command(['git', 'diff', '--cached', '--check'], w)
            if subprocess.run(['git', 'diff', '--cached', '--quiet'], cwd=w).returncode:
                command(['git', 'commit', '-m', 'Report Blur official150 '+event], w)
                command(['git', 'push', 'origin', 'HEAD:main'], w)
            commit = command(['git', 'rev-parse', 'HEAD'], w).stdout.strip()
            command(['git', 'fetch', 'origin', 'main']); command(['git', 'merge-base', '--is-ancestor', commit, 'origin/main'])
            write(marker, {'commit': commit, 'report_sha256': sha(path)})
            print('BLUR_OFFICIAL_REPORT_PUSHED '+event+' '+commit, flush=True); return
        except (OSError, subprocess.SubprocessError) as exc:
            last_error = exc; print('BLUR_OFFICIAL_REPORT_RETRY '+str(exc), flush=True); time.sleep(10)
    raise RuntimeError('report pending: '+str(last_error))


def monitor(commit):
    lock = (OUT/'observer.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    # Reuse only the live training wait, never the old train/development main().
    prior.wait('train')
    prior.enqueue('step200', '固定200步训练已完成；按最新指令跳过32开发评价，接官方150。')
    prior.enqueue('terminal', '200步训练完成；旧开发评价未提交，用户改为仅官方150验收。此报告仅为训练终态。')
    cpu('prepare')
    submit(commit)
    # Uploads can be retried independently by the upload mode; never block rendering.
    write(OUT/'publication_requests/queued.json', {'event': 'queued', 'detail': '已提交最终step200的完整官方150张恢复评价，单节点两H100，无其它验证集。'})
    folder = QUEUE/'jobs'/JOB
    while not (folder/'node3.ended.json').exists():
        hb = read(QUEUE/'workers/node3/heartbeat.json')
        print(f'BLUR_OFFICIAL_WAIT records={len(list((OUT/"records").glob("*.json")))}/150 tiles={len(list((OUT/"tiles").glob("*.json")))} heartbeat_age={time.time()-hb["utc_unix"]:.1f}', flush=True)
        if time.time()-hb['utc_unix'] > 300: raise RuntimeError('heartbeat stale; preserve running job')
        time.sleep(20)
    status = read(folder/'node3.ended.json')['status']
    if status: raise RuntimeError(f'official GPU exited {status}; no automatic retry')
    cpu('finish')
    write(OUT/'publication_requests/terminal.json', {'event': 'terminal', 'detail': '官方Blur150张完整恢复、两rank退出0及CPU完整性核验结束。最终联合门槛以完整result JSON为准。'})
    print('BLUR_OFFICIAL_CPU_COMPLETE', flush=True)


def upload():
    lock = (OUT/'uploader.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    while True:
        for p in sorted((OUT/'publication_requests').glob('*.json'), key=lambda a:a.stat().st_mtime):
            r = read(p)
            if (OUT/'published'/p.name).exists(): continue
            try: publish(r['event'], r['detail'])
            except RuntimeError as exc: print(str(exc), flush=True); break
        if any((OUT/'published'/f'{k}.json').exists() for k in ('terminal','failed')): return
        time.sleep(30)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('phase', choices=['monitor','upload']); p.add_argument('--commit')
    a = p.parse_args(); OUT.mkdir(parents=True, exist_ok=True)
    if a.phase == 'upload': upload()
    else:
        if not a.commit: raise ValueError('published full commit required')
        try: monitor(a.commit)
        except Exception as exc:
            write(OUT/'publication_requests/failed.json', {'event': 'failed', 'detail': str(exc)})
            raise
