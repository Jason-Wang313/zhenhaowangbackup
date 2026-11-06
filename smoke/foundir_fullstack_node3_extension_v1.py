#!/usr/bin/env python3
"""Add independent node3 jobs without editing or restarting the three live v2 workers.

Same shared job format/queue. v2 workers skip jobs whose nodes list is [3]
before validation. Four-node DDP jobs must NOT be submitted to the legacy workers.
"""
import argparse
import os
from pathlib import Path
import socket
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from smoke import foundir_fullstack_executor_3x2_v2 as base
SELF='smoke/foundir_fullstack_node3_extension_v1.py'
NODE={'notebook':'e087f2fc-829f-4a0d-8aa5-5df18b6cd56a','ip':'10.244.90.116'}


def validate_extension_scope(job):
    if 3 in job['nodes']:
        if job['nodes']!=[3]:raise ValueError('node3 extension permits independent node3-only jobs; legacy workers cannot accept four-node jobs')
        if SELF not in job['sources']:raise ValueError('node3 extension source must be hash-bound')


def configure():
    if str(ROOT)!=base.ACTIVE:raise ValueError('destination shared directory only')
    base.NODES=4
    base.NODE_MAP={**base.NODE_MAP,3:NODE}
    original=base.validate
    def validate(job):
        validate_extension_scope(job)
        return original(job)
    def heartbeat(node,state,job=None):
        base.write(base.QUEUE/'workers'/f'node{node}'/'heartbeat.json',
          {'utc_unix':time.time(),'pid':os.getpid(),'hostname':socket.gethostname(),'state':state,'job':job,
           'worker_source_sha256':base.digest(Path(base.__file__)),'extension_source_sha256':base.digest(ROOT/SELF),
           'capability':'independent_node3_two_H100_jobs_only'},replace=True)
    base.validate=validate;base.heartbeat=heartbeat


def main():
    p=argparse.ArgumentParser(description=__doc__);subs=p.add_subparsers(dest='phase',required=True)
    for phase in ('worker','inventory'):
        sp=subs.add_parser(phase);sp.add_argument('node',type=int,choices=[3])
    sp=subs.add_parser('submit');sp.add_argument('--job-id',required=True);sp.add_argument('--entry',required=True)
    sp.add_argument('--commit',required=True);sp.add_argument('--purpose',required=True)
    sp.add_argument('--sources',nargs='+',required=True);sp.add_argument('--arguments',nargs='*',default=[])
    a=p.parse_args();configure()
    if a.phase=='submit':
        a.nodes=[3];a.sources=[*a.sources,SELF];base.submit(a)
    elif a.phase=='inventory':print('FULLSTACK_NODE_VERIFIED '+__import__('json').dumps(base.inventory(3)),flush=True)
    else:base.worker(3)

if __name__=='__main__':main()
