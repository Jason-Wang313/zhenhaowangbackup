#!/usr/bin/env python3
"""Conservative source screening and deterministic future data plans; never train."""
import csv
import hashlib
import io
import json
from collections import Counter,defaultdict
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from PIL import Image
from smoke.check_low_light_source_pool import perceptual_hash
OUT=ROOT/'checkpoints/foundir_fullstack_noise_blur_sources_v1'
REPORT=ROOT/'reports/foundir_fullstack_noise_blur_sources_v1_2026-11-06.json'
MANIFEST=ROOT/'manifests/foundir_five_type_pairs_v1.csv'
MANIFEST_SHA='e8c018879029c0bcc2da8d074cc4999541158ca97d07f48aadbcdddcdccc3ca7'
OFFICIAL=ROOT/'checkpoints/foundir_official_vlm_router_v4/official_vlm_v4_frozen4800_v1_4x2/receipt.json'
SEED='foundir_noise_blur_sources_20261106_v1'


def read(path):return json.loads(path.read_text())
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()
def require(ok,text):
    if not ok:raise ValueError(text)
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():require(read(path)==value,'existing immutable plan differs');return
    temp=path.with_name(path.name+'.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temp.replace(path)
def stable(*values):return hashlib.sha256(json.dumps([SEED,*values],sort_keys=True).encode()).hexdigest()


def main():
    start=time.monotonic();require(not REPORT.exists(),'preserve completed screen')
    require(sha(MANIFEST)==MANIFEST_SHA,'manifest changed')
    protected=read(OFFICIAL)['rows'];require(len(protected)==600,'official600 required')
    by_gt=defaultdict(list)
    for row in protected:by_gt[row['gt_sha256']].append(row)
    images=[]
    for i,(digest,rows) in enumerate(sorted(by_gt.items())):
        path=ROOT/rows[0]['gt_path'];marker=OUT/'protected_index'/(digest+'.json')
        if marker.exists():entry=read(marker);require(entry['file_sha256']==digest,'protected index changed')
        else:
            data=path.read_bytes();require(hashlib.sha256(data).hexdigest()==digest,'official GT file changed')
            with Image.open(io.BytesIO(data)) as image:
                rgb=image.convert('RGB');p=perceptual_hash(rgb)
                entry={'file_sha256':digest,'phash64':f'{p:016x}','size':list(rgb.size),
                       'canonical_RGB_sha256':hashlib.sha256(rgb.tobytes()).hexdigest(),
                       'record_ids':[r['record_id'] for r in rows],
                       'purpose':'source_exclusion_only_no_loss_no_parameter_fit'}
            write(marker,entry)
        images.append(entry)
        if (i+1)%40==0:print(f'SOURCE_SCREEN protected_unique_GT={i+1}/{len(by_gt)}',flush=True)
    protected_phashes=[(int(v['phash64'],16),v['record_ids']) for v in images]
    exact={r[k] for r in protected for k in ('lq_sha256','gt_sha256')};blocked={};phash_cache={}
    all_rows=[]
    with MANIFEST.open(newline='') as f:
        for row in csv.DictReader(f):
            if row['category'] not in ('01Blur','05Noise') or row['composition_size']!='1':continue
            all_rows.append(row);p=row['gt_phash']
            if p not in phash_cache:
                n=int(p,16);matches=[{'official_record_ids':ids,'phash_distance':(n-h).bit_count()} for h,ids in protected_phashes if (n-h).bit_count()<=4]
                phash_cache[p]=matches
            exact_match=[k for k in ('lq_sha256','gt_sha256') if row[k] in exact]
            if phash_cache[p] or exact_match:
                blocked.setdefault(row['source_group'],{'example_record_id':row['record_id'],'exact_fields':exact_match,'phash_matches':phash_cache[p]})
    print('SOURCE_SCREEN metadata_rows='+str(len(all_rows))+' blocked_groups='+str(len(blocked)),flush=True)
    pools={};plans={}
    for task,category in [('noise','05Noise'),('blur','01Blur')]:
        pools[task]={};plans[task]={}
        for split,target in [('train',128),('validation',32)]:
            groups=defaultdict(list)
            for row in all_rows:
                if row['category']==category and row['split']==split and row['source_group'] not in blocked and min(int(row['width']),int(row['height']))>=512:groups[row['source_group']].append(row)
            pools[task][split]=len(groups);require(len(groups)>=target,'insufficient screened sources')
            chosen=[]
            for group in sorted(groups,key=lambda g:stable(task,split,g))[:target]:
                row=min(groups[group],key=lambda row:stable('record',row['record_id']));bits=bytes.fromhex(stable('crop',row['record_id']))
                chosen.append({'row':row,'crop_xy':[int.from_bytes(bits[:8],'big')%(int(row['width'])-511),int.from_bytes(bits[8:16],'big')%(int(row['height'])-511)],'inference_seed':int.from_bytes(bits[16:20],'big')})
            plans[task][split]=chosen
        require(not ({v['row']['source_group'] for v in plans[task]['train']} & {v['row']['source_group'] for v in plans[task]['validation']}),'plan splits overlap')
    router=read(ROOT/'checkpoints/foundir_expanded_vlm_router_v4/expanded_conditional_v4/receipt.json')
    router_hits=[]
    for item in router['items']:
        if item['role']!='fit' or item['task'] not in ('noise','lowlight'):continue
        row=item['row'];p=int(row['gt_phash'],16)
        matches=[{'official_record_ids':ids,'phash_distance':(p-h).bit_count()} for h,ids in protected_phashes if (p-h).bit_count()<=4]
        if matches or any(row[k] in exact for k in ('lq_sha256','gt_sha256')):
            router_hits.append({'key':item['key'],'task':item['task'],'source_group':row['source_group'],'phash_matches':matches,'exact_fields':[k for k in ('lq_sha256','gt_sha256') if row[k] in exact]})
    value={'status':'screened_metadata_plan_not_GPU_preflight','seed':SEED,'manifest_sha256':MANIFEST_SHA,
           'official_receipt_sha256':sha(OFFICIAL),'phash_implementation_sha256':sha(ROOT/'smoke/check_low_light_source_pool.py'),
           'script_sha256':sha(Path(__file__)),'phash_max_distance':4,
           'phash_threshold_origin':'same fixed threshold as original training source clustering; not tuned on scores',
           'official_unique_GT_files':len(images),'official_GT_used_only_for_source_exclusion':True,
           'official_metrics_used_in_source_selection':False,'semantic_disjointness_guaranteed':False,
           'false_positive_and_false_negative_phash_matches_possible':True,
           'blocked_source_groups':blocked,'eligible_sources':pools,'plans':plans,
           'legacy_router_fit_possible_overlap':router_hits,'legacy_router_overlap_is_not_proven_training_GT_exposure':True,
           'plan_is_not_authorization_to_train_or_expand_budget':True,
           'GPU_jobs_submitted':False,'new_training':False,'GT_used_in_loss':False,
           'validation_previously_used_in_development':True,'seconds':time.monotonic()-start,'GPU_hours':0,'exit_code':0}
    write(OUT/'plan.json',value);write(REPORT,value)
    print('SOURCE_SCREEN_END '+json.dumps({'seconds':value['seconds'],'eligible':pools,'blocked_groups':len(blocked),'possible_legacy_router_fit_overlap_crops':len(router_hits)}),flush=True)


if __name__=='__main__':main()
