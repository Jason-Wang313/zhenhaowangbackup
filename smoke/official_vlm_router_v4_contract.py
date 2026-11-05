"""Official-only SOW600 evaluation; no training, tuning, or other populations."""
from collections import Counter
import math
from pathlib import PurePosixPath
from smoke import existing_flux_replay_contract_v1 as c

RUN = 'official_vlm_v4_frozen4800_v1'
OUTPUT = 'checkpoints/foundir_official_vlm_router_v4/' + RUN
REPORT = 'reports/foundir_official_vlm_router_v4_terminal_2026-11-05.json'
PREFLIGHT = 'reports/foundir_official_vlm_router_v4_preflight_2026-11-05.json'
MANIFEST = 'manifests/foundir_official_sow600_qwen_v1.csv'
MANIFEST_SHA = 'dfe5a05ba13f941ac57734e2a96eb21048ad463f4248f306d2fb2d9c6d48866e'
TRAIN_REPORT_SHA = '09fa4b7998ddbfe46f52d4aa3952a6f40624b6fc72b2eb4c182d4e2f1db3e695'
# Fixed independently of a mutable latest pointer.
SELECTED_SHA = '383d40b63e31d79283ec5a33193413fee914a5082a8888c5c918737ccdd6d971'
ORDER = ('blur', 'noise', 'haze', 'lowlight', 'rain')
CATEGORIES = dict(zip(ORDER, ('01Blur', '05Noise', '08Haze', '14Lowlight', '10Rain')))
COUNTS = {'01Blur':150, '05Noise':100, '08Haze':200, '14Lowlight':50, '10Rain':100}
TARGETS = dict(zip(ORDER, ((25.9,.18), (38.9,.10), (22.1,.19), (18.9,.28), (34.2,.08))))
ARMS = ('image_only', 'conditional_v4', 'known_class_reference')
WORLD = 4
TILE, OVERLAP = 512, 128
RECIPE = {'run':RUN, 'nodes':2, 'local_gpus':2, 'world':WORLD, 'selected_step':4800,
          'arms':list(ARMS), 'tile':TILE, 'overlap':OVERLAP,
          'route':'mean_crop_logits_then_one_global_top1_no_labels',
          'padding':'numpy_reflect_right_bottom_singleton_edge',
          'stitch':'native_uint8_tiles_sqrt_hann_floor_1e-3_then_round_uint8',
          'seed':'sha256_2026110502_record_xy_shared_across_arms',
          'metrics':'pyiqa_0.1.16_defaults_psnr_lpips_ssim_full_RGB_no_crop_resize_shave',
          'frozen_experts':True, 'training':False, 'checkpoint_selection':False,
          'route_GPU_cap_seconds_per_node':10800, 'route_used_GPU_hours_cap':12,
          'render_budget':'separate_explicit_user_approval_after_exact_route_plan',
          'CPU_prepare_cap_seconds':1800, 'CPU_finish_cap_seconds':900}
SOURCES = ['smoke/official_vlm_router_v4_contract.py', 'smoke/evaluate_foundir_official_vlm_router_v4.py',
           'smoke/run_foundir_official_vlm_router_v4.sh', 'smoke/start_foundir_official_vlm_router_v4_tmux.sh',
           'smoke/expanded_vlm_h100_runtime_v4.py', 'lowlight_restoration/five_expert_flux.py',
           'lowlight_restoration/conditional_vlm_router_v4.py',
           'reports/advisor_governance/foundir_official_vlm_router_v4_authorization_2026-11-05.json']


def require(ok, message):
    if not ok: raise ValueError(message)


def validate_rows(rows):
    require(len(rows)==600 and Counter(r['category'] for r in rows)==Counter(COUNTS), 'official SOW600 scope only')
    require(len({r['record_id'] for r in rows})==600, 'duplicate official ID')
    for r in rows:
        require(r['record_id']=='official_'+r['category']+'_'+r['pair_id'], 'wrong official record identity')
        require(r['pair_id'].isdigit() and int(r['width'])>0 and int(r['height'])>0, 'invalid dimensions/ID')
        for side in ('lq','gt'):
            expected=f"downloads/foundir_official/extracted_{side}/{r['category']}/{r['pair_id']}.png"
            require(r[side+'_path']==expected and '..' not in PurePosixPath(expected).parts, 'nonofficial path forbidden')
            h=r[side+'_sha256']; require(len(h)==64 and all(a in '0123456789abcdef' for a in h), 'invalid image hash')


def origins(length):
    require(type(length) is int and length>0, 'positive integer dimension required')
    if length<=TILE: return [0]
    result=list(range(0,length-TILE+1,TILE-OVERLAP))
    if result[-1]!=length-TILE: result.append(length-TILE)
    return result


def tiles(row):
    return [{'key':f"{row['record_id']}__x{x}_y{y}", 'x':x, 'y':y,
             'seed':int(c.digest([2026110502,row['record_id'],x,y])[:8],16)}
            for y in origins(int(row['height'])) for x in origins(int(row['width']))]


def tile_array(array, tile):
    import numpy as np
    require(array.ndim==3 and array.shape[2]==3 and array.dtype==np.uint8, 'RGB uint8 input required')
    x,y=tile['x'],tile['y']; cut=array[y:y+TILE,x:x+TILE]
    h,w=cut.shape[:2]; require(h>0 and w>0, 'empty tile')
    # Axis-wise padding supports tiny images; torch reflect-pad cannot pad >= input size.
    cut=np.pad(cut,((0,TILE-h),(0,0),(0,0)),mode='reflect' if h>1 else 'edge')
    cut=np.pad(cut,((0,0),(0,TILE-w),(0,0)),mode='reflect' if w>1 else 'edge')
    return cut.copy(),h,w


def stitch(row, predictions):
    import numpy as np
    expected=tiles(row); require(set(predictions)=={t['key'] for t in expected}, 'missing/extra output tile')
    h,w=int(row['height']),int(row['width']); total=np.zeros((h,w,3),np.float32); norm=np.zeros((h,w),np.float32)
    for t in expected:
        value=predictions[t['key']]
        require(value.shape==(TILE,TILE,3) and value.dtype==np.uint8, 'native uint8 RGB tiles required')
        x,y=t['x'],t['y']; th,tw=min(TILE,h-y),min(TILE,w-x)
        vy=np.hanning(th).astype(np.float32) if th>2 else np.ones(th,np.float32)
        vx=np.hanning(tw).astype(np.float32) if tw>2 else np.ones(tw,np.float32)
        weight=np.maximum(np.sqrt(np.outer(vy,vx)),np.float32(1e-3))
        total[y:y+th,x:x+tw]+=value[:th,:tw].astype(np.float32)*weight[...,None]
        norm[y:y+th,x:x+tw]+=weight
    require(np.isfinite(total).all() and (norm>0).all(), 'uncovered/nonfinite stitch')
    return np.round(total/norm[...,None]).clip(0,255).astype(np.uint8)


def decisions(image_logits, vlm_logits):
    # No task/category argument. Mean crop logits preserve the learned crop-level head.
    def choose(rows):
        require(rows and all(len(z)==5 and all(math.isfinite(v) for v in z) for z in rows), 'invalid logits')
        z=[sum(r[j] for r in rows)/len(rows) for j in range(5)]
        return {'logits':z,'anchor':ORDER[max(range(5),key=lambda j:z[j])]}
    require(len(image_logits)==len(vlm_logits), 'unequal crop population')
    return {'image_only':choose(image_logits),'conditional_v4':choose(vlm_logits)}


def anchors(row, decision):
    # Known category is added only AFTER the two label-free decisions are fixed.
    require(set(decision)==set(ARMS[:2]), 'two deployed arms required')
    for value in decision.values():
        z=value['logits']
        require(len(z)==5 and all(isinstance(v,(int,float)) and math.isfinite(v) for v in z), 'invalid saved decision scores')
        require(value['anchor']==ORDER[max(range(5),key=lambda j:z[j])], 'saved anchor differs from argmax')
    result={arm:decision[arm]['anchor'] for arm in ARMS[:2]}
    require(all(v in ORDER for v in result.values()), 'invalid expert')
    result['known_class_reference']={v:k for k,v in CATEGORIES.items()}[row['category']]
    return result


def aggregate(rows, records):
    validate_rows(rows); require(set(records)=={r['record_id'] for r in rows}, 'incomplete official results')
    arms={}
    for arm in ARMS:
        values=[]
        for row in rows:
            rec=records[row['record_id']]; a=anchors(row,rec['decisions'])[arm]; m=rec['metrics'][a]
            require(set(m)=={'psnr','lpips','ssim'} and all(isinstance(v,(int,float)) and math.isfinite(v) for v in m.values()), 'invalid official metrics')
            values.append({'record_id':row['record_id'],'category':row['category'],'anchor':a,**m})
        def mean(sub): return {m:sum(r[m] for r in sub)/len(sub) for m in ('psnr','lpips','ssim')}
        arms[arm]={'count':600,'overall':mean(values),'per_category':{cat:mean([r for r in values if r['category']==cat]) for cat in COUNTS},
                   'correct_routes':sum(CATEGORIES[r['anchor']]==r['category'] for r in values),'rows':values}
    checks={t:{'psnr':arms['conditional_v4']['per_category'][CATEGORIES[t]]['psnr']>=TARGETS[t][0],
               'lpips':arms['conditional_v4']['per_category'][CATEGORIES[t]]['lpips']<=TARGETS[t][1]} for t in ORDER}
    return {'arms':arms,'checks':checks,'all_ten_SoW_checks_pass':all(all(v.values()) for v in checks.values()),
            'official_singleton_pairs':600,'full_20_category_test':False,'independent_blind_test':False}
