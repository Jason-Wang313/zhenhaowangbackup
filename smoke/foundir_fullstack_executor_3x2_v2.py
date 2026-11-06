#!/usr/bin/env python3
"""Shared-filesystem GPU executor for the newly mapped three H100 nodes (isolated from v1)."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
ACTIVE = '/inspire/hdd/project/multimodal-discrete-diffusion/liuxiaohong-25080/zhenhaowang'
QUEUE = ROOT / 'checkpoints/foundir_fullstack_executor_3x2_h100_v2'
NODES = 3
NODE_MAP = {
    0: {'notebook': '37497631-24c8-4a19-907e-494c701599ab', 'ip': '10.244.211.17'},
    1: {'notebook': '2551e049-18b3-4c01-8490-735ced1e58cd', 'ip': '10.244.47.55'},
    2: {'notebook': '55ef95b9-862f-4ffa-a2ed-6eaa7156ccad', 'ip': '10.245.39.185'},
}
STOP = False


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value, replace=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + '\n'
    if path.exists() and not replace:
        if path.read_text() != data:
            raise ValueError(f'immutable record differs: {path}')
        return
    tmp = path.with_name(path.name + f'.tmp.{os.getpid()}.{time.time_ns()}')
    with tmp.open('x') as f:
        f.write(data); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def lock(path, nonblocking=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    f = path.open('a')
    fcntl.flock(f, fcntl.LOCK_EX | (fcntl.LOCK_NB if nonblocking else 0))
    return f


def local_file(name):
    if not isinstance(name, str) or Path(name).is_absolute() or '..' in Path(name).parts:
        raise ValueError('project-relative file required')
    path = ROOT / name
    if not path.resolve().is_relative_to(ROOT.resolve()) or not path.is_file():
        raise ValueError('source file missing or escapes project')
    return path


def validate(job):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,95}', job['id']):
        raise ValueError('invalid job id')
    if not job['entry'].startswith('smoke/run_foundir_fullstack_') or not job['entry'].endswith('.sh'):
        raise ValueError('only dedicated fullstack launchers allowed')
    if job['entry'] not in job['sources'] or not job['nodes'] or len(set(job['nodes'])) != len(job['nodes']):
        raise ValueError('entry hash and unique nodes required')
    if any(type(n) is not int or n not in range(NODES) for n in job['nodes']):
        raise ValueError('invalid node')
    if not isinstance(job['args'], list) or any(not isinstance(a, str) or '\0' in a for a in job['args']):
        raise ValueError('literal argument list required')
    for name, expected in job['sources'].items():
        if digest(local_file(name)) != expected:
            raise ValueError(f'published source changed: {name}')
    return job


def submit(args):
    commit = subprocess.check_output(['git','rev-parse',args.commit], cwd=ROOT, text=True).strip()
    subprocess.run(['git','merge-base','--is-ancestor',commit,'origin/main'], cwd=ROOT, check=True)
    sources = {name: digest(local_file(name)) for name in sorted(set([args.entry, 'smoke/foundir_fullstack_executor_3x2_v2.py'] + args.sources))}
    for name, expected in sources.items():
        blob = subprocess.check_output(['git','show',f'{commit}:{name}'], cwd=ROOT)
        if hashlib.sha256(blob).hexdigest() != expected:
            raise ValueError(f'local source not identical to published commit: {name}')
    job = {'id':args.job_id, 'entry':args.entry, 'args':args.arguments, 'nodes':args.nodes,
           'sources':sources, 'source_commit':commit, 'purpose':args.purpose,
           'scope':'authorized FoundIR full-stack iteration; official600 evaluation only; train data for training',
           'automatic_retry':False}
    validate(job)
    with lock(QUEUE/'submission.lock'):
        write(QUEUE/'jobs'/job['id']/'job.json', job)
    print('FULLSTACK_JOB_SUBMITTED '+json.dumps(job, ensure_ascii=False), flush=True)


def inventory(node):
    raw = subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.total,uuid',
                                   '--format=csv,noheader,nounits'], text=True)
    cards = []
    for line in raw.splitlines():
        index, name, memory, uuid = [v.strip() for v in line.split(',')]
        if index in ('0','1'):
            cards.append({'index':int(index),'name':name,'memory_mib':int(memory),'uuid':uuid})
    cards.sort(key=lambda d:d['index'])
    if len(cards) != 2 or cards[0]['name'] != cards[1]['name'] or any(c['memory_mib'] < 75*1024 for c in cards):
        raise ValueError('two same-model Hopper GPUs >=75GiB required')
    expected = 'H100'
    if any(expected not in c['name'] for c in cards):
        raise ValueError(f'node{node} must use its confirmed {expected} pair')
    addresses = subprocess.check_output(['hostname', '-I'], text=True).split()
    if NODE_MAP[node]['ip'] not in addresses:
        raise ValueError(f'wrong node mapping: node{node} requires IP {NODE_MAP[node]["ip"]}, observed {addresses}')
    return {'node':node, 'hostname':socket.gethostname(), 'cards':cards,
            'confirmed_mapping':NODE_MAP[node], 'observed_addresses':sorted(addresses)}


def register(node):
    value = inventory(node)
    with lock(QUEUE/'registration.lock'):
        for other in (QUEUE/'workers').glob('node*/registration.json'):
            prior = read(other)
            if prior['node'] != node and {c['uuid'] for c in prior['cards']} & {c['uuid'] for c in value['cards']}:
                raise ValueError('same physical GPUs registered as two different nodes')
        write(QUEUE/'workers'/f'node{node}'/'registration.json', value)
    return value


def heartbeat(node, state, job=None):
    write(QUEUE/'workers'/f'node{node}'/'heartbeat.json',
          {'utc_unix':time.time(),'pid':os.getpid(),'hostname':socket.gethostname(),
           'state':state,'job':job,'worker_source_sha256':digest(Path(__file__))}, replace=True)


def run_job(folder, node):
    job = read(folder/'job.json')
    started, ended = folder/f'node{node}.started.json', folder/f'node{node}.ended.json'
    if node not in job['nodes'] or ended.exists():
        return
    validate(job)
    if started.exists():
        raise ValueError(f'unfinished attempt needs reconciliation; no automatic retry: {folder}')
    start = time.time()
    write(started, {'utc_unix':start,'job_sha256':digest(folder/'job.json'),'node':node,
                    'hostname':socket.gethostname(),'gpus':2})
    status = 1; error = None; process = None; cancel_at = None
    try:
        if (folder/'cancel.json').exists():
            status = 143
        else:
            with (folder/f'node{node}.log').open('xb') as log:
                env = {**os.environ, 'CUDA_VISIBLE_DEVICES':'0,1','FOUNDIR_FULLSTACK_NODE':str(node)}
                process = subprocess.Popen(['bash',str(local_file(job['entry'])),str(node),*job['args']],
                    cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                while process.poll() is None:
                    heartbeat(node,'running',job['id'])
                    if STOP or (folder/'cancel.json').exists():
                        if cancel_at is None:
                            os.killpg(process.pid, signal.SIGTERM); cancel_at = time.monotonic()
                        elif time.monotonic()-cancel_at > 30:
                            os.killpg(process.pid, signal.SIGKILL)
                    time.sleep(2)
                status = process.returncode
    except Exception as exc:
        error = repr(exc)
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try: process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL); process.wait()
    finally:
        seconds = time.time()-start
        write(ended, {'status':status,'error':error,'seconds':seconds,'gpus':2,
                      'allocated_GPU_hours':2*seconds/3600,'automatic_retry':False})
    print(f'FULLSTACK_JOB_END job={job["id"]} node={node} status={status}', flush=True)


def worker(node):
    global STOP
    def stop(*_):
        global STOP
        STOP = True
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    with lock(QUEUE/'workers'/f'node{node}.lock', nonblocking=True):
        register(node)
        print(f'FULLSTACK_WORKER_READY node={node}', flush=True)
        try:
            while not STOP:
                heartbeat(node,'idle')
                for spec in sorted((QUEUE/'jobs').glob('*/job.json')):
                    if STOP: break
                    run_job(spec.parent,node)
                time.sleep(5)
        except Exception as exc:
            heartbeat(node,'blocked: '+str(exc))
            raise
        finally:
            if STOP: heartbeat(node,'stopped')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='phase', required=True)
    w = sub.add_parser('worker'); w.add_argument('node', type=int, choices=range(NODES))
    i = sub.add_parser('inventory'); i.add_argument('node', type=int, choices=range(NODES))
    s = sub.add_parser('submit'); s.add_argument('--job-id',required=True); s.add_argument('--entry',required=True)
    s.add_argument('--commit',required=True); s.add_argument('--purpose',required=True)
    s.add_argument('--nodes',type=int,nargs='+',default=list(range(NODES))); s.add_argument('--sources',nargs='+',required=True)
    s.add_argument('--arguments',nargs='*',default=[])
    q = sub.add_parser('cancel'); q.add_argument('job_id'); q.add_argument('--reason',required=True)
    args = parser.parse_args()
    if str(ROOT) != ACTIVE: raise ValueError('destination workspace only')
    if args.phase == 'worker': worker(args.node)
    elif args.phase == 'inventory': print('FULLSTACK_NODE_VERIFIED '+json.dumps(inventory(args.node)))
    elif args.phase == 'submit': submit(args)
    else:
        folder = QUEUE/'jobs'/args.job_id
        job = read(folder/'job.json'); validate(job)
        if folder.resolve().parent != (QUEUE/'jobs').resolve(): raise ValueError('invalid job path')
        write(folder/'cancel.json', {'reason':args.reason,'requested_utc_unix':time.time()})


if __name__ == '__main__': main()
