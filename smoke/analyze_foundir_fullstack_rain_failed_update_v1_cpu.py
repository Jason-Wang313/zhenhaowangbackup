#!/usr/bin/env python3
"""Read existing metrics only; report failed update, not a new candidate or GT selection."""
import hashlib
import json
import math
from pathlib import Path
import statistics
import time
from collections import defaultdict
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/foundir_fullstack_rain_failed_update_v1_2026-11-06.json'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    begin=time.monotonic()
    final_path=ROOT/'reports/foundir_fullstack_rain_step1000_v1_result_2026-11-06.json'
    train_path=ROOT/'reports/foundir_fullstack_rain_continue_v1_result_2026-11-06.json'
    recovery_path=ROOT/'reports/foundir_fullstack_rain_step1000_resume_v1_2026-11-06.json'
    final,train,recovery=map(read,(final_path,train_path,recovery_path))
    if final['status']!='complete' or recovery['status']!='complete' or recovery['result_sha256']!=sha(final_path):raise ValueError('terminal identity mismatch')
    records=[]
    for p in sorted((ROOT/'checkpoints/foundir_fullstack_rain_step1000_v1/records').glob('*.json')):
        r=read(p)
        if r['reused_parent']:continue
        if r['receipt_sha256']!=final['receipt_sha256'] or r['item']['row']['category']!='10Rain':raise ValueError('wrong official population')
        records.append(r)
    if len(records)!=100 or len({r['item']['row']['record_id'] for r in records})!=100:raise ValueError('incomplete Rain100')
    groups=defaultdict(list)
    for r in records:groups[r['item']['row']['gt_sha256']].append(r)
    group_report=[]
    for digest,rows in groups.items():
        entry={'GT_file_sha256':digest,'record_ids':[r['item']['row']['record_id'] for r in rows],'count':len(rows)}
        for name,get in [('baseline',lambda r:r['item']['accepted']['metrics']),('candidate',lambda r:r['metrics'])]:
            entry[name]={m:statistics.mean(get(r)[m] for r in rows) for m in ('psnr','lpips')}
        entry['delta']={m:entry['candidate'][m]-entry['baseline'][m] for m in ('psnr','lpips')};group_report.append(entry)
    new={m:math.fsum(r['metrics'][m] for r in records)/100 for m in ('psnr','lpips')}
    old={m:math.fsum(r['item']['accepted']['metrics'][m] for r in records)/100 for m in new}
    for m in new:
        if abs(new[m]-final['per_category']['10Rain'][m])>1e-10:raise ValueError('independent mean mismatch')
    history=train['history_new_updates'];windows={label:{k:statistics.mean(r['losses'][k] for r in rows) for k in rows[0]['losses']}
        for label,rows in [('first20',history[:20]),('last20',history[-20:])]}
    value={'status':'read_only_failure_diagnosis_complete','source_sha256':sha(Path(__file__)),
      'inputs':{str(p.relative_to(ROOT)):sha(p) for p in (final_path,train_path,recovery_path)},
      'official_records':100,'unique_GT_files':len(groups),'baseline':old,'candidate':new,
      'delta':{m:new[m]-old[m] for m in new},'PSNR_gap_to_34_2':34.2-new['psnr'],
      'LPIPS_relative_improvement_percent':100*(old['lpips']-new['lpips'])/old['lpips'],
      'both_metrics_better_images':sum(r['metrics']['psnr']>r['item']['accepted']['metrics']['psnr'] and r['metrics']['lpips']<r['item']['accepted']['metrics']['lpips'] for r in records),
      'source_groups':sorted(group_report,key=lambda r:r['record_ids'][0]),
      'PSNR_improved_GT_groups':sum(g['delta']['psnr']>0 for g in group_report),
      'training_probe':{s:train['milestones'][s]['means'] for s in ('800','900','1000')},
      'training_loss_windows':windows,'loss_windows_use_different_presentations_not_a_controlled_comparison':True,
      'training_fixed_probe_PSNR_delta':train['milestones']['1000']['delta_vs_matched800']['psnr'],
      'code_observation':'Same LoRA controls prefix25 and call26; prefix regenerated every presentation but detached from backward. Omitted prefix derivative is a hypothesis, not established cause.',
      'next_bounded_diagnostic':'fixed legal training crops; compare prefix800/readout800, prefix800/readout1000, prefix1000/readout800, prefix1000/readout1000. No new training, no official rerun, single node only.',
      'decision':'Do not adopt step1000 or append more identical training. Keep step800 reference. Isolate prefix/readout coupling before choosing any repair.',
      'evaluation_attempt_GPU_hours':recovery['total_attempt_GPU_hours'],
      'all_expense_including_training':False,'official_is_adaptive_not_blind':True,
      'official_GT_pixels_read':False,'pixels_decoded':0,'new_GPU_started':False,'new_training':False,
      'seconds':time.monotonic()-begin,'GPU_hours':0,'exit_code':0}
    if OUT.exists():raise ValueError('preserve prior report')
    OUT.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:value[k] for k in ('status','candidate','delta','PSNR_gap_to_34_2','PSNR_improved_GT_groups','seconds')}))
if __name__=='__main__':main()
