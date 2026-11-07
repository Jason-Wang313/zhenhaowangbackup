#!/usr/bin/env python3
"""After the fixed official result is published, compare all fixed rounds and publish only small reports."""
import fcntl
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from smoke import compare_foundir_blur_fullgrid_official_v1_cpu as comparison
from smoke import watch_foundir_blur_fullgrid_v1_cpu as watcher
from smoke.existing_flux_replay_contract_v1 import read, sha


def main():
    out = watcher.OUT
    with (out/'comparison.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while not (out/'published/terminal.json').exists():
            if (out/'publication_requests/failed.json').exists():
                print('BLUR_FULLGRID_COMPARISON_NOT_RUN upstream_failed; preserve_all_assets', flush=True)
                return
            time.sleep(15)
        for node in (1,3):
            ended = read(watcher.jobdir('official',node)/f'node{node}.ended.json')
            if ended['status'] != 0: raise RuntimeError('official workers did not finish normally')
        stem = 'reports/foundir_blur_fullgrid_v1_official_comparison_2026-11-07'
        result = ROOT/(stem+'.json')
        if not result.exists():
            subprocess.run([str(watcher.PY),'-S','smoke/compare_foundir_blur_fullgrid_official_v1_cpu.py'],
                           cwd=ROOT,check=True)
        saved = read(result)
        expected = comparison.summarize({name:read(ROOT/path) for name,path in comparison.INPUTS.items()})
        if any(saved[k] != value for k,value in expected.items()): raise RuntimeError('comparison evidence differs')
        for name,path in comparison.INPUTS.items():
            if saved['inputs'][name] != {'path':path,'sha256':sha(ROOT/path)}:
                raise RuntimeError('comparison input changed')
        request = {'event':'official_comparison','files':{n:sha(ROOT/n) for n in (stem+'.json',stem+'.md')}}
        while not (out/'published/official_comparison.json').exists():
            try:
                watcher.publish(request)
            except (OSError,subprocess.SubprocessError,RuntimeError) as exc:
                print('BLUR_FULLGRID_COMPARISON_UPLOAD_PENDING '+str(exc),flush=True)
                time.sleep(30)
        print('BLUR_FULLGRID_COMPARISON_PUBLISHED',flush=True)


if __name__=='__main__':main()
