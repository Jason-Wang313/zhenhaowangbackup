#!/usr/bin/env python3
"""Read-only router error/coverage audit. No training, threshold search or renders."""
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from smoke import evaluate_foundir_supervised_router_restoration_v1 as quality
s = quality.supervised
c, require, ORDER = s.c, s.require, s.ORDER
QUALITY_SHA = 'e6006f7bad915ecddfb4beacdee9e44da93afcfa4031f22105ce73d943cda390'
SCORE_REPORT = 'reports/foundir_router_score_audit_2026-11-04.json'
COVERAGE_REPORT = 'reports/foundir_router_feature_coverage_2026-11-04.json'
CATEGORIES = dict(zip(ORDER,('01Blur','05Noise','08Haze','14Lowlight','10Rain')))


def argmax(values):
    require(len(values)==5 and all(quality.analysis.number(v) for v in values),'invalid five logits')
    return max(range(5),key=lambda j:values[j])


def score_audit(cpu,restoration):
    rows = cpu['arms']['image_plus_vlm']['development']['rows']
    image_only = {r['key']:r for r in cpu['arms']['image_only']['development']['rows']}
    restored = {r['key']:r for r in restoration['arms']['image_plus_vlm']['per_image']}
    require(len(rows)==len(image_only)==len(restored)==40 and {r['key'] for r in rows}==set(image_only)==set(restored),
            'fixed matched population required')
    records=[]
    for row in rows:
        key=row['key']; true=row['task_for_reporting_only']; target=ORDER.index(true)
        image=row['image_logits']; delta=row['vlm_delta']; logits=row['logits']; k=argmax(logits)
        require(ORDER[k]==row['predicted_anchor'] and restored[key]['predicted_anchor']==ORDER[k]
                and restored[key]['task']==true and image_only[key]['task_for_reporting_only']==true,'prediction/label mismatch')
        require(all(math.isclose(a+b,z,rel_tol=2e-6,abs_tol=2e-6) for a,b,z in zip(image,delta,logits)),
                'combined score is not image + VLM')
        im=ORDER[argmax(image)]; vlm=ORDER[argmax(delta)]
        exps=[math.exp(v-max(logits)) for v in logits]; probs=[v/sum(exps) for v in exps]
        require(all(abs(a-b)<2e-6 for a,b in zip(probs,row['soft_weights_diagnostic_only'])),'probability mismatch')
        records.append({'key':key,'task_for_diagnosis_only':true,'combined':ORDER[k],
            'same_model_image_branch':im,'vlm_delta_argmax_not_standalone_model':vlm,
            'separately_trained_image_only':image_only[key]['predicted_anchor'],
            'combined_confidence':probs[k],'correct':ORDER[k]==true,
            'flip_helped':im!=true and ORDER[k]==true,'flip_hurt':im==true and ORDER[k]!=true,
            'image_logits':image,'vlm_delta':delta,'combined_logits':logits,
            'selected_minus_true_margin':{'image':image[k]-image[target],'vlm':delta[k]-delta[target],
                                          'combined':logits[k]-logits[target]},
            'restoration':restored[key]})
    return {'status':'published_router_score_audit_complete','count':40,
            'combined_correct':sum(r['correct'] for r in records),
            'same_model_image_branch_correct':sum(r['same_model_image_branch']==r['task_for_diagnosis_only'] for r in records),
            'flip_helped':sum(r['flip_helped'] for r in records),'flip_hurt':sum(r['flip_hurt'] for r in records),
            'training':{a:cpu['arms'][a]['train'] for a in quality.ARMS},
            'development_ce':{a:cpu['arms'][a]['development']['ce'] for a in quality.ARMS},
            'per_image':records,'errors':[r for r in records if not r['correct']],
            'limitations':['Branch scores are jointly trained; removing a branch is not a matched retraining ablation.',
                'VLM score dominance is not proof of a feature or label bug.',
                'No coefficient sweep, thresholds, early checkpoint selection or development-label training.',
                'No new restoration metrics, GPU work, expert changes or formal training.']}


def reports():
    require(c.sha(ROOT/s.REPORT)==quality.CPU_REPORT_SHA and c.sha(ROOT/quality.REPORT)==QUALITY_SHA,
            'published terminal report changed')
    cpu=c.read(ROOT/s.REPORT); restored=c.read(ROOT/quality.REPORT)
    old=c.read(ROOT/quality.analysis.OUTPUT); hard=c.read(ROOT/quality.prior_hard.REPORT)
    require(c.sha(ROOT/quality.analysis.OUTPUT)==quality.prior_hard.BREAKDOWN_SHA
            and c.sha(ROOT/quality.prior_hard.REPORT)==quality.HARD_REPORT_SHA,'parent reports changed')
    plan=quality.make_plan(cpu,old,hard)
    arms,diffs=quality.aggregate(plan,old,{k:v['metrics'] for k,v in restored['artifacts'].items()})
    require(arms==restored['arms'] and diffs==restored['vlm_minus_image_only'],'restoration summary mismatch')
    return cpu,restored


def metadata_audit(items):
    s.validate_split(items)
    for item in items:
        r=item['row']; x,y=r['crop_xy']
        require(r['category']==CATEGORIES[item['task']] and item['key']==item['split']+'_'+r['record_id'],
                'category/key label mismatch')
        require(type(x) is int and type(y) is int and 0<=x<=int(r['width'])-512 and 0<=y<=int(r['height'])-512,
                'crop outside input')
    train=[i for i in items if i['split']=='train']; dev=[i for i in items if i['split']=='validation']
    overlaps={field:len({i['row'][field] for i in train}&{i['row'][field] for i in dev})
              for field in ('source_group','gt_sha256','lq_sha256')}
    require(not any(overlaps.values()),'train/development identity overlap')
    exposure=Counter(j for step in range(400) for j in s.training_indices(train,step))
    require(len(exposure)==640 and set(exposure.values())=={20},'unequal/missing training exposure')
    return {'records':len(items),'train':len(train),'development':len(dev),'overlaps':overlaps,
            'exposures_per_training_record':20,
            'unique_sources':{split:len({i['row']['source_group'] for i in rows}) for split,rows in [('train',train),('validation',dev)]},
            'train_unique_sources_per_task':{t:len({i['row']['source_group'] for i in train if i['task']==t}) for t in ORDER},
            'category_labels_match_manifest':True,'crop_bounds_valid':True,
            'visual_label_correctness_verified':False}


def feature_coverage(train_items,train,dev_items,dev):
    """Fixed centered-cosine distances; all neighbours come only from training."""
    import torch
    require(len(train_items)==len(train) and len(dev_items)==len(dev)
            and all(i['split']=='train' for i in train_items) and all(i['split']=='validation' for i in dev_items),
            'wrong feature population/split')
    require(not ({i['row']['source_group'] for i in train_items}&{i['row']['source_group'] for i in dev_items}),
            'train/development source overlap')
    def normalize(x):
        require(x.ndim==2 and torch.isfinite(x).all(),'invalid feature matrix')
        x=x.double(); x=x-x.mean(-1,keepdim=True); norm=x.norm(dim=-1,keepdim=True)
        require(bool((norm>1e-12).all()),'constant feature: no meaningful centered cosine')
        return x/norm
    x=normalize(train); y=normalize(dev)
    tt=(1-x@x.T).clamp(0,2); dt=(1-y@x.T).clamp(0,2)
    # Exclude ALL records sharing a source, not merely the same row.
    for j,i in enumerate(train_items):
        for k,o in enumerate(train_items):
            if i['row']['source_group']==o['row']['source_group']: tt[j,k]=float('inf')
    refs={}; summary={}
    for task in ORDER:
        ids=[j for j,i in enumerate(train_items) if i['task']==task]
        require(len(ids)>=2,'at least two sources per task required')
        distances=tt[ids][:,ids].min(dim=1).values
        require(torch.isfinite(distances).all(),'no different-source reference neighbour')
        refs[task]=distances
        summary[task]={'count':len(ids),'leave_source_out_same_class_distance_quantiles':
                       dict(zip(('q10','q50','q90','q95'),torch.quantile(distances,torch.tensor([.1,.5,.9,.95],dtype=torch.float64)).tolist()))}
    rows=[]
    for j,item in enumerate(dev_items):
        def neighbor(k):
            i=train_items[k]
            return {'key':i['key'],'task':i['task'],'source_group':i['row']['source_group'],'distance':float(dt[j,k])}
        top=sorted(range(len(train_items)),key=lambda k:(float(dt[j,k]),train_items[k]['key']))[:5]
        nearest={}
        for task in ORDER:
            ids=[k for k,i in enumerate(train_items) if i['task']==task]
            nearest[task]=neighbor(min(ids,key=lambda k:(float(dt[j,k]),train_items[k]['key'])))
        task=item['task']; distance=nearest[task]['distance']
        rows.append({'key':item['key'],'task_for_diagnosis_only':task,'top5_training_neighbors':list(map(neighbor,top)),
                     'nearest_training_by_class':nearest,
                     'same_class_distance_percentile_in_training_reference':float((refs[task]<=distance).double().mean())})
    return {'distance':'one_minus_centered_cosine_FP64','training_reference':summary,'development_rows':rows,
            'used_to_change_routing':False,'interpretation':'Distance is descriptive, not a calibrated OOD detector or proof that more data fixes an error.'}


def cache_identity_audit(bound):
    import torch
    from safetensors import safe_open
    parent=ROOT/'checkpoints/foundir_frozen_expert_router_v1'/quality.analysis.RUN
    for item in bound['items']:
        key=item['key']; spec=bound['features'][key]
        marker=parent/'features'/(key+'.json'); row=c.read(marker)
        require(c.sha(marker)==spec['marker_sha256'] and row['identity']=={
            'receipt_sha256':bound['parent_receipt_sha256'],'item':item,'arm':'features'}
            and row['GT_input'] is False,'feature input/identity mismatch')
        vm=parent/'vlm'/(key+'.json'); v=c.read(vm)
        require(c.sha(vm)==row['vlm_receipt_sha256'] and v['identity']=={
            'receipt_sha256':bound['parent_receipt_sha256'],'item':item,'arm':'vlm'}
            and v['GT_input'] is False,'VLM crop/input identity mismatch')
        file=c.path(parent,v['tensor'])
        require(v['files']=={v['tensor']:c.sha(file)},'VLM artifact changed')
        with safe_open(str(file),framework='pt',device='cpu') as f: original=f.get_tensor('vlm')
        with safe_open(str(c.path(ROOT,spec['path'])),framework='pt',device='cpu') as f: copied=f.get_tensor('vlm')
        require(torch.equal(original,copied),'VLM feature copied from a different input')
    return {'verified_records':len(bound['items']),'vlm_copy_exact':True,'shared_crop_and_input_metadata':True,
            'GT_pixels_read':False,'latent_reference_loaded':False,'pixels_regenerated':False}


def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('phase',choices=['scores','coverage']); args=ap.parse_args()
    cpu,restored=reports(); scores=score_audit(cpu,restored)
    if args.phase=='scores':
        c.write_new(ROOT/SCORE_REPORT,{'inputs':{'cpu':quality.CPU_REPORT_SHA,'restoration':QUALITY_SHA},**scores})
        print('ROUTER_SCORE_AUDIT_PASS '+json.dumps({k:scores[k] for k in ('combined_correct','same_model_image_branch_correct','flip_helped','flip_hurt')})); return
    require(str(ROOT)==quality.analysis.ACTIVE_ROOT,'coverage runs only in destination workspace')
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU-only audit: hide all CUDA devices')
    import torch
    start=time.monotonic(); torch.set_num_threads(8)
    s.verify(); bound=s.load_bound(); metadata=metadata_audit(bound['items'])
    train,ti,tv,_=s.load_features(bound,'train'); dev,di,dv,_=s.load_features(bound,'validation')
    identity=cache_identity_audit(bound)
    coverage={name:feature_coverage(train,x,dev,y) for name,x,y in [('image',ti,di),('vlm',tv,dv)]}
    require(not torch.cuda.is_initialized(),'unexpected CUDA use')
    value={'status':'router_feature_coverage_audit_complete','training':False,'GPU_used':False,'automatic_followup':False,
           'inputs':{'cpu':quality.CPU_REPORT_SHA,'restoration':QUALITY_SHA,'cpu_receipt':c.sha(ROOT/s.OUTPUT/'receipt.json')},
           'code_sha256':c.sha(Path(__file__)),'metadata':metadata,'cache_identity':identity,'coverage':coverage,
           'seconds':time.monotonic()-start,'cpu_threads':8,'errors':[r['key'] for r in scores['errors']],
           'limitations':['All 40 development records examined, not only two errors.',
              'No pixel inspection: manifest labels are not a visual ground-truth audit.',
              'No retraining, coefficient sweep, threshold fitting, checkpoint selection, or new restoration outputs.']}
    # An existing report is not silently overwritten on a repeat with a different elapsed time.
    target=ROOT/COVERAGE_REPORT
    if target.exists():
        previous=c.read(target)
        require({k:v for k,v in previous.items() if k!='seconds'}=={k:v for k,v in value.items() if k!='seconds'},'audit results changed')
    else: c.write_new(target,value)
    print('ROUTER_FEATURE_COVERAGE_PASS records=680 development=40 GPU_used=false',flush=True)


if __name__=='__main__': main()
