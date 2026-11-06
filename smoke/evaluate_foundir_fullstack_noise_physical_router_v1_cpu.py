#!/usr/bin/env python3
"""Frozen CPU pilot inference on official LQ; reuse GPU outputs only where available."""
import hashlib
import io
import json
from pathlib import Path
import sys
import time
from collections import Counter

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from PIL import Image
from lowlight_restoration.noise_lowlight_physical_router_v1 import features,BinaryCorrection
from smoke.train_foundir_fullstack_noise_physical_router_v1_cpu import OUT,REPORT as TRAIN_REPORT,bound,read,sha,write,require
PARENT=ROOT/'checkpoints/foundir_official_vlm_router_v4/official_vlm_v4_frozen4800_v1_4x2'
AUDIT=ROOT/'checkpoints/foundir_fullstack_noise_router_audit_v1'
REPORT=ROOT/'reports/foundir_fullstack_noise_physical_router_v1_official_2026-11-06.json'
ORDER=('blur','noise','haze','lowlight','rain')


def main():
    start=time.monotonic();require(not REPORT.exists(),'preserve fixed official result');b=bound();training=read(TRAIN_REPORT)
    require(training['receipt_sha256']==sha(OUT/'receipt.json') and training['checkpoint_sha256']==sha(OUT/'fixed_step400.pt'),'fixed CPU model changed')
    saved=torch.load(OUT/'fixed_step400.pt',map_location='cpu',weights_only=True)
    require(saved['recipe']==training['recipe'] and saved['receipt_sha256']==training['receipt_sha256'],'CPU checkpoint binding changed')
    state=saved['model'];model=BinaryCorrection(state['mean'],state['scale']);model.load_state_dict(state,strict=True);model.eval().requires_grad_(False)
    records=[];parent=read(PARENT/'receipt.json')
    for row in parent['rows']:
        decision=read(PARENT/'decisions'/(row['record_id']+'.json'))['decisions']['conditional_v4'];old=decision['anchor']
        if old not in ('noise','lowlight'):continue
        require(row['category'] in ('05Noise','14Lowlight'),'unexpected gated population; preserve other categories')
        audit=read(AUDIT/'records'/(row['record_id']+'.json'))
        require(audit['mean_logit_route']==old and max(abs(a-b) for a,b in zip(audit['mean_logits'],decision['logits']))<1e-4,'old decision replay changed')
        path=OUT/'official_features'/(row['record_id']+'.json')
        if path.exists():feature=read(path);require(feature['lq_sha256']==row['lq_sha256'],'LQ feature identity changed')
        else:
            data=(ROOT/row['lq_path']).read_bytes();require(hashlib.sha256(data).hexdigest()==row['lq_sha256'],'official LQ hash changed')
            with Image.open(io.BytesIO(data)) as image:f=features(image).tolist()
            feature={'lq_sha256':row['lq_sha256'],'values':f,'GT_input':False};write(path,feature)
        with torch.no_grad():margin,new,delta=model(torch.tensor([decision['logits']]),torch.tensor([feature['values']]))
        anchor=ORDER[int(new[0])]
        # Publish the LQ-only decision before consulting category/score/output metadata.
        prediction={'record_id':row['record_id'],'lq_sha256':row['lq_sha256'],'checkpoint_sha256':training['checkpoint_sha256'],
                    'base_logits':decision['logits'],'baseline_route':old,'candidate_route':anchor,'delta':float(delta[0]),
                    'GT_input':False,'class_label_input':False}
        write(OUT/'official_predictions'/(row['record_id']+'.json'),prediction)
        prior=read(PARENT/'results'/(row['record_id']+'.json'));available=anchor in prior['outputs'] and anchor in prior['metrics']
        record={**prediction,'category_for_scoring_only':row['category'],'baseline_metrics':prior['metrics'][old],
                'candidate_output_available':available,'candidate_metrics':None}
        if available:
            output=prior['outputs'][anchor];require(sha(PARENT/output['path'])==output['sha256'],'cached GPU output changed')
            record.update(candidate_metrics=prior['metrics'][anchor],candidate_output={'path':str((PARENT/output['path']).relative_to(ROOT)),'sha256':output['sha256']})
        records.append(record)
        if len(records)%25==0:print(f'PHYSICAL_ROUTER_OFFICIAL {len(records)}/150',flush=True)
    require(len(records)==150,'expected gated150');summary={}
    for category in ('05Noise','14Lowlight'):
        group=[r for r in records if r['category_for_scoring_only']==category];target='noise' if category=='05Noise' else 'lowlight'
        entry={'count':len(group),'baseline_routes':dict(Counter(r['baseline_route'] for r in group)),
               'candidate_routes':dict(Counter(r['candidate_route'] for r in group)),
               'correct_routes':sum(r['candidate_route']==target for r in group),
               'changed':sum(r['candidate_route']!=r['baseline_route'] for r in group),
               'missing_GPU_outputs':[r['record_id'] for r in group if not r['candidate_output_available']]}
        if not entry['missing_GPU_outputs']:
            entry['candidate_metrics']={m:sum(r['candidate_metrics'][m] for r in group)/len(group) for m in ('psnr','lpips','ssim')}
        summary[category]=entry
    result={'status':'fixed_CPU_route_diagnosis_not_deployed','training_report_sha256':sha(TRAIN_REPORT),'checkpoint_sha256':training['checkpoint_sha256'],
            'source_sha256':sha(Path(__file__)),'parent_official_receipt_sha256':sha(PARENT/'receipt.json'),
            'summary':summary,'records':records,'seconds':time.monotonic()-start,'exit_code':0,'GPU_hours':0,
            'new_GPU_generations':0,'GT_pixels_decoded':0,'GT_or_labels_used_for_prediction':False,
            'old_GPU_outputs_reused_only_after_LQ_decision':True,'other450_not_regenerated_or_replaced':True,
            'new_Rain_result_not_overwritten':True,'automatic_deployment':False,'official_is_adaptive_benchmark':True,
            'legacy_router_source_overlap_not_erased':True}
    write(REPORT,result);print('PHYSICAL_ROUTER_OFFICIAL_END '+json.dumps({'summary':summary,'seconds':result['seconds']}),flush=True)


if __name__=='__main__':
    torch.set_num_threads(2);torch.set_num_interop_threads(1);main()
