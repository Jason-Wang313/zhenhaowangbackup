#!/usr/bin/env python3
"""Read existing score/identity records in Noise then Blur order; no pixels or fitting."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

ROOT=Path(__file__).resolve().parents[1]
PARENT=ROOT/'checkpoints/foundir_official_vlm_router_v4/official_vlm_v4_frozen4800_v1_4x2'
INPUT=ROOT/'checkpoints/foundir_fullstack_official_input_baseline_3x2_v2'
AUDIT=ROOT/'checkpoints/foundir_fullstack_round1_cpu_audit'
TASKS={'noise':('05Noise',100,38.9,.10,'checkpoints/foundir_noise_flux2_scalar_residual_full_development_v1'),
       'blur':('01Blur',150,25.9,.18,'checkpoints/foundir_blur_only_flux2_scalar_residual_full_development_v1')}


def require(ok,message):
    if not ok:raise ValueError(message)


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path,hashes):
    data=path.read_bytes();hashes[str(path.relative_to(ROOT))]=hashlib.sha256(data).hexdigest();return json.loads(data)


def mean(values):return math.fsum(values)/len(values)


def metrics(rows,key):return {m:mean([v[key][m] for v in rows]) for m in ('psnr','lpips')}


def distribution(values):
    q=statistics.quantiles(values,n=4,method='inclusive')
    return {'min':min(values),'q25':q[0],'median':q[1],'q75':q[2],'max':max(values),'mean':mean(values)}


def analyze(task):
    started=time.monotonic();hashes={};category,count,psnr_goal,lpips_goal,development=TASKS[task]
    parent=read(PARENT/'receipt.json',hashes);parent_hash=hashes[str((PARENT/'receipt.json').relative_to(ROOT))]
    require(parent_hash=='98eb74a6534eb234e75a164a2460f43b0a2f6beaeb08a686e1812f85a7615157','wrong official parent')
    read(INPUT/'receipt.json',hashes);input_hash=hashes[str((INPUT/'receipt.json').relative_to(ROOT))]
    rows=[v for v in parent['rows'] if v['category']==category]
    require(len(rows)==count and len({v['record_id'] for v in rows})==count,'official cardinality changed')
    records=[];order=['blur','noise','haze','lowlight','rain']
    for row in rows:
        rid=row['record_id'];v=read(PARENT/'results'/(rid+'.json'),hashes)
        inp=read(INPUT/'records'/(rid+'.json'),hashes);audit=read(AUDIT/'records'/(rid+'.json'),hashes)
        require(v['row']==inp['row']==row and v['receipt_sha256']==parent_hash and inp['receipt_sha256']==input_hash,'official record identity changed')
        require(audit['identity']['receipt_sha256']==parent_hash and audit['saved_known_class_psnr']==v['metrics'][task]['psnr'],'old diagnostic differs')
        decision=v['decisions']['conditional_v4'];route=decision['anchor']
        selected=v['metrics'][route];known=v['metrics'][task];input_score=inp['metrics']
        require(all(math.isfinite(s[m]) for s in (selected,known,input_score) for m in ('psnr','lpips')),'nonfinite metric')
        require(abs(audit['input_psnr_diagnostic']-input_score['psnr'])<1e-4,'input scoring audit differs')
        mse=10**(-audit['known_class_psnr_diagnostic']/10)-1e-8
        bias=mean([e*e for e in audit['mean_rgb_error']])
        require(0<=bias<=mse+1e-10,'invalid bias/error decomposition')
        margin=decision['logits'][order.index(task)]-max(v for i,v in enumerate(decision['logits']) if i!=order.index(task))
        records.append({'record_id':rid,'GT_file_sha256':row['gt_sha256'],'route':route,'image_only_route':v['decisions']['image_only']['anchor'],
                        'correct_class_logit_margin':margin,'width':int(row['width']),'height':int(row['height']),
                        'input':{m:input_score[m] for m in ('psnr','lpips')},'deployed':{m:selected[m] for m in ('psnr','lpips')},
                        'known_class_reference':{m:known[m] for m in ('psnr','lpips')},
                        'expert_PSNR_gain_over_input':known['psnr']-input_score['psnr'],
                        'expert_LPIPS_change_over_input':known['lpips']-input_score['lpips'],
                        'whole_image_RGB_bias_fraction_of_MSE':bias/mse,
                        'edge_vs_interior_output_MSE_ratio':audit['regions']['tile_edge_bands']['output_mse']/audit['regions']['interior']['output_mse']})
    groups=defaultdict(list)
    for row in records:groups[row['GT_file_sha256']].append(row)
    wrong=[r for r in records if r['route']!=task]
    means={k:metrics(records,k) for k in ('input','deployed','known_class_reference')}
    stats={'count':count,'identical_GT_file_groups':len(groups),'means':means,
           'route_counts':dict(Counter(r['route'] for r in records)),
           'image_only_route_counts':dict(Counter(r['image_only_route'] for r in records)),
           'misrouted_records':len(wrong),'misrouted_GT_file_groups':len({r['GT_file_sha256'] for r in wrong}),
           'known_class_reference_is_deployable_router':False,
           'known_class_vs_deployed_delta':{m:means['known_class_reference'][m]-means['deployed'][m] for m in ('psnr','lpips')},
           'known_class_remaining_PSNR_gap':max(0,psnr_goal-means['known_class_reference']['psnr']),
           'known_class_remaining_LPIPS_absolute_gap':max(0,means['known_class_reference']['lpips']-lpips_goal),
           'known_class_required_relative_LPIPS_reduction_percent':100*max(0,1-lpips_goal/means['known_class_reference']['lpips']),
           'known_class_PSNR_worse_than_input':sum(r['expert_PSNR_gain_over_input']<0 for r in records),
           'known_class_LPIPS_worse_than_input':sum(r['expert_LPIPS_change_over_input']>0 for r in records),
           'mean_whole_image_RGB_bias_fraction_of_MSE':mean([r['whole_image_RGB_bias_fraction_of_MSE'] for r in records]),
           'edge_vs_interior_output_MSE_ratio':distribution([r['edge_vs_interior_output_MSE_ratio'] for r in records]),
           'input_PSNR_distribution':distribution([r['input']['psnr'] for r in records]),
           'expert_PSNR_gain_distribution':distribution([r['expert_PSNR_gain_over_input'] for r in records]),
           'groups':[{'GT_file_sha256':key,'records':[r['record_id'] for r in group],
                      'route_counts':dict(Counter(r['route'] for r in group)),
                      'input':metrics(group,'input'),'deployed':metrics(group,'deployed'),
                      'known_class_reference':metrics(group,'known_class_reference')} for key,group in sorted(groups.items())]}
    dev=ROOT/development;summary=read(dev/'result.json',hashes);config=read(ROOT/summary['config'],hashes)
    require(hashes[summary['config']]==summary['config_sha256'],'development config mismatch')
    spec=parent['original']['parent']['experts'][task]['checkpoint']
    require(summary['checkpoint_sha256']==spec['sha256']==digest(ROOT/spec['path']),'development/deployment checkpoint differs')
    sidecars=[read(p,hashes) for p in sorted((dev/'per_record').glob('*.json'))]
    require(len(sidecars)==summary['records'] and len({r['record_id'] for r in sidecars})==len(sidecars),'development missing/duplicate records')
    require(all(s['checkpoint_sha256']==spec['sha256'] and s['config_sha256']==summary['config_sha256'] for s in sidecars),'development binding changed')
    devmeans={k:metrics(sidecars,k) for k in ('input','base','candidate')}
    require(all(abs(devmeans['candidate'][m]-summary['candidate_metrics'][m])<1e-10 for m in ('psnr','lpips')),'development aggregate mismatch')
    devstats={'count':len(sidecars),'source_groups':len({r['source_group'] for r in sidecars}),'means':devmeans,
              'same_checkpoint_as_official':True,'frozen_alpha':summary['frozen_alpha'],
              'PSNR_gain_over_input':devmeans['candidate']['psnr']-devmeans['input']['psnr'],
              'candidate_PSNR_worse_than_input':sum(r['candidate']['psnr']<r['input']['psnr'] for r in sidecars),
              'input_PSNR_distribution':distribution([r['input']['psnr'] for r in sidecars]),
              'old_source_isolation_does_not_prove_official_disjointness':True}
    result={'status':'CPU_existing_records_diagnosis_complete','task':task,'official':stats,'development':devstats,
            'official_vs_development':{
                'input_PSNR_delta':means['input']['psnr']-devmeans['input']['psnr'],
                'known_class_output_PSNR_delta':means['known_class_reference']['psnr']-devmeans['candidate']['psnr'],
                'difference_in_expert_PSNR_gain':means['known_class_reference']['psnr']-means['input']['psnr']-devstats['PSNR_gain_over_input'],
                'causal_attribution_from_unmatched_sets':False},
            'protocol_differences':{'development':'one deterministic512 crop/source record, historical seed, torchmetrics LPIPS',
                                    'official':'full-resolution512/128 tiled output, official tile seed, pyiqa LPIPS',
                                    'matched_protocol_comparison_completed':False},
            'records':records,'evidence_hashes':hashes,'script_sha256':digest(Path(__file__)),
            'seconds':time.monotonic()-started,'exit_code':0,'GPU_hours':0,'pixels_decoded':0,
            'new_training':False,'new_restorations':False,'GT_fitting_or_output_selection':False,
            'official_is_adaptive_benchmark':True,'independent_blind_test':False,
            'training_official_overlap_audit_completed_for_this_expert':False}
    name=ROOT/f'reports/foundir_fullstack_{task}_preparation_v1_2026-11-06.json'
    require(not name.exists(),'preserve old report; do not overwrite')
    tmp=name.with_suffix('.json.tmp');tmp.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');tmp.replace(name)
    print('PREPARATION_DIAGNOSIS',task,json.dumps({'official':{k:v for k,v in stats.items() if k not in ['groups']},'development':devstats,'seconds':result['seconds']},ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('task',choices=['noise','blur']);analyze(p.parse_args().task)
