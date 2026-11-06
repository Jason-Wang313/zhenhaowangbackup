#!/usr/bin/env python3
"""Same published Blur experiment; durable reports upload independently of GPU scheduling."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from smoke import watch_foundir_blur_delta_v1_cpu as prior
from smoke.existing_flux_replay_contract_v1 import read, sha, write_new as write
REQUESTS = prior.OUT/'publication_requests'
original_publish = prior.publish
original_command = prior.command


def bounded_command(args, **kwargs):
    if args[0] == 'git':
        kwargs.setdefault('timeout', 45)
    return original_command(args, **kwargs)


def enqueue(event, detail):
    path = REQUESTS/(event+'.json')
    if path.exists():
        return
    report = ROOT/f'reports/foundir_blur_delta_velocity_v1_{event}_2026-11-06.json'
    if not report.exists():
        jobs = {}
        for phase, name in prior.JOBS.items():
            folder = prior.QUEUE/'jobs'/name
            if (folder/'job.json').exists():
                jobs[phase] = {p.stem: read(p) for p in folder.glob('*.json') if p.name != 'job.json'}
        steps = sorted((prior.OUT/'pilot/steps').glob('step_*.json'))
        training_logs = []
        logfile = prior.QUEUE/'jobs'/prior.JOBS['train']/'node3.log'
        if logfile.exists():
            for line in logfile.read_text(errors='replace').splitlines():
                if line.startswith('BLUR_DELTA_TRAIN {'):
                    try: training_logs.append(json.loads(line.split(' ', 1)[1]))
                    except json.JSONDecodeError: pass
        write(report, {'event': event, 'detail': detail, 'utc_unix': time.time(),
              'receipt_sha256': sha(prior.OUT/'receipt.json'), 'jobs': jobs,
              'CPU_stages': {p.stem: read(p) for p in (prior.OUT/'cpu').glob('*.json')},
              'latest_pilot_step_receipt': read(steps[-1]) if steps else None,
              'training_logs_available_at_report_time': training_logs,
              'completed_GPU_hours': sum(v['node3.ended']['allocated_GPU_hours'] for v in jobs.values() if 'node3.ended' in v),
              'cost_complete': all('node3.ended' in v for v in jobs.values()),
              'official_evaluated': False, 'other_categories_submitted': False,
              'cumulative_GPU_hour_cap': None, 'official_Blur_pass_required_before_other_types': True,
              'upload_may_be_deferred_by_network': True})
    write(path, {'event': event, 'detail': detail, 'requested_utc_unix': time.time(), 'report_sha256': sha(report)})
    print('BLUR_DELTA_REPORT_DURABLE '+event, flush=True)


def wait(phase):
    folder = prior.QUEUE/'jobs'/prior.JOBS[phase]
    while not (folder/'node3.ended.json').exists():
        hb = read(prior.QUEUE/'workers/node3/heartbeat.json')
        print(f'BLUR_DELTA_WAIT phase={phase} heartbeat_age={time.time()-hb["utc_unix"]:.1f} state={hb["state"]}', flush=True)
        if phase == 'train':
            for step in (1, 50, 100, 200):
                if (prior.OUT/'pilot/steps'/f'step_{step:06d}.json').exists():
                    enqueue(f'step{step}', f'固定200步Blur小试验step{step}已落盘，尚无官方结果。')
        if time.time()-hb['utc_unix'] > 300:
            raise RuntimeError('node3 heartbeat stale; preserve job and do not relaunch')
        time.sleep(20)
    ended = read(folder/'node3.ended.json')
    if ended['status'] != 0:
        raise RuntimeError(f'{phase} status={ended["status"]}; no automatic retry')
    # When resuming the observer, a subsequent phase may already be running.
    # Its live heartbeat is not evidence that this completed phase failed.
    for _ in range(12):
        hb = read(prior.QUEUE/'workers/node3/heartbeat.json')
        if hb['state'] == 'idle' or hb.get('job') != prior.JOBS[phase]:
            return
        time.sleep(5)
    raise RuntimeError('completed job still has its own running heartbeat')


def upload():
    lock = (prior.OUT/'uploader.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    while True:
        for path in sorted(REQUESTS.glob('*.json'), key=lambda p: read(p)['requested_utc_unix']):
            request = read(path)
            if (prior.OUT/'published'/(request['event']+'.json')).exists():
                continue
            try:
                original_publish(request['event'], request['detail'])
            except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
                print('BLUR_DELTA_UPLOAD_PENDING '+request['event']+' '+str(exc), flush=True)
                break
        terminal = [p for p in REQUESTS.glob('*.json') if p.stem in ('terminal', 'failed')]
        if terminal and all((prior.OUT/'published'/p.name).exists() for p in REQUESTS.glob('*.json')):
            print('BLUR_DELTA_ALL_REPORTS_PUSHED', flush=True)
            return
        time.sleep(30)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['monitor', 'upload'])
    parser.add_argument('--commit')
    args = parser.parse_args()
    prior.command = bounded_command
    if args.phase == 'upload':
        upload()
    else:
        if not args.commit:
            raise ValueError('published full experiment commit required')
        prior.publish = enqueue
        prior.wait = wait
        prior.main(args.commit)
