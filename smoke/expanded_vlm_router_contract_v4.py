"""Metadata-only fixed expansion and paired rendering contract; no test pixels."""
from collections import Counter
import math
from pathlib import Path
from smoke import repair_foundir_vlm_preserve_router_v3 as old

c, require, ORDER = old.c, old.require, old.ORDER
RUN = 'expanded_conditional_v4'
OUTPUT = 'checkpoints/foundir_expanded_vlm_router_v4/' + RUN
TRAIN_REPORT = 'reports/foundir_expanded_vlm_router_v4_training_2026-11-05.json'
REPORT = 'reports/foundir_expanded_vlm_router_v4_terminal_2026-11-05.json'
NODES, LOCAL_GPUS, WORLD_SIZE = 2, 2, 4
SEED = 2026110501
STEPS, CHECKS, VIEWS, MAX_SOURCES = 6400, (1600, 3200, 4800, 6400), 3, 1024
ARMS = ('image_only', 'previous_v3', 'conditional_v4', 'known_class_reference')
TARGETS = dict(zip(ORDER, ((25.9,.18), (38.9,.10), (22.1,.19), (18.9,.28), (34.2,.08))))
SOURCES = ['lowlight_restoration/conditional_vlm_router_v4.py', 'smoke/expanded_vlm_router_contract_v4.py',
           'smoke/expanded_vlm_router_state_v4.py', 'smoke/expanded_vlm_h100_runtime_v4.py', 'smoke/run_foundir_expanded_vlm_router_v4.py',
           'smoke/run_foundir_expanded_vlm_router_v4.sh', 'smoke/start_foundir_expanded_vlm_router_v4_tmux.sh']
RECIPE = {'steps': STEPS, 'checks': list(CHECKS), 'seed': SEED, 'batch': 32, 'views_per_record_cap': VIEWS,
          'source_cap_per_class': MAX_SOURCES, 'lr': .001, 'weight_decay': .01, 'label_smoothing': .05,
          'preserve_weight': 1., 'margin_cap': .2, 'clip': 1., 'frozen_image_and_experts': True,
          'selection': 'minimum_calibration_CE_then_earliest_no_per_class_classification_gate',
          'classification_gate_blocks_restoration': False, 'GPU_features_cap_per_node': 10800,
          'GPU_render_cap_per_node': 5400, 'CPU_train_cap': 3600, 'CPU_finish_cap': 600,
          'nodes': NODES, 'local_gpus': LOCAL_GPUS, 'world_size': WORLD_SIZE, 'H100_qualified': False,
          'GPU_execution_policy': 'vae_native_conv_fp32_query_chunk256_v1',
          'H100_status': 'code_and_CPU_checks_only_pending_CUDA',
          'legacy_cache_runtime_difference_recorded': True}


def expand(rows, previous, confirmation, cap=MAX_SOURCES, views=VIEWS):
    rows = list(rows)
    fields = ('source_group', 'gt_sha256', 'lq_sha256')
    blocked = {f: {r[f] for r in rows if r['split'] != 'train'} |
               {i['row'][f] for i in previous if i['role'] != 'fit'} |
               {i['row'][f] for i in confirmation} for f in fields}
    categories = {v:k for k,v in old.parent.audit.CATEGORIES.items()}
    chosen = [dict(i) for i in previous if i['role'] == 'fit']
    require(all(not any(i['row'][f] in blocked[f] for f in fields) for i in chosen), 'old fit overlaps protected population')
    pools = {t: {} for t in ORDER}
    used = {t: {i['row']['source_group'] for i in chosen if i['task'] == t} for t in ORDER}
    for r in rows:
        if (r['split'] != 'train' or r['category'] not in categories or int(r['composition_size']) != 1 or
                min(int(r['width']), int(r['height'])) < 512 or any(r[f] in blocked[f] for f in fields)):
            continue
        t = categories[r['category']]
        if r['source_group'] in used[t]:
            continue
        prior = pools[t].get(r['source_group'])
        if prior is None or c.stable(SEED, 'record', r['record_id']) < c.stable(SEED, 'record', prior['record_id']):
            pools[t][r['source_group']] = r
    for t in ORDER:
        require(len(used[t]) <= cap, 'cap below inherited source count')
        for source, row in sorted(pools[t].items(), key=lambda kv: c.stable(SEED, 'source', t, kv[0])):
            if len(used[t]) == cap:
                break
            raw = bytes.fromhex(c.stable(SEED, 'basecrop', row['record_id']))
            r = dict(row)
            r['crop_xy'] = [int.from_bytes(raw[:8], 'big') % (int(r['width'])-511),
                            int.from_bytes(raw[8:16], 'big') % (int(r['height'])-511)]
            chosen.append({'key': 'expanded_' + r['record_id'], 'task': t, 'role': 'fit', 'split': 'train',
                           'row': r, 'seed': int.from_bytes(raw[16:20], 'big')})
            used[t].add(source)
    result = []
    previous_keys = {i['key'] for i in previous}
    for i in chosen:
        base = {**i, 'parent_record': i['row']['record_id'], 'view_index': 0,
                'reuse_key': i['key'] if i['key'] in previous_keys else None}
        result.append(base)
        width, height = int(i['row']['width'])-511, int(i['row']['height'])-511
        total = width*height
        used_crops = {i['row']['crop_xy'][1]*width+i['row']['crop_xy'][0]}
        for v in range(1, min(views, total)):
            raw = bytes.fromhex(c.stable(SEED, 'view', i['row']['record_id'], v))
            position = int.from_bytes(raw[:8], 'big') % total
            while position in used_crops:
                position = (position+1) % total
            used_crops.add(position)
            result.append({**base, 'key': i['key'] + f'_view{v}', 'view_index': v, 'reuse_key': None,
                           'seed': int.from_bytes(raw[8:12], 'big'),
                           'row': {**i['row'], 'crop_xy': [position % width, position // width]}})
    for i in previous:
        if i['role'] == 'calibration':
            result.append({**i, 'reuse_key': i['key']})
    for i in confirmation:
        result.append({**i, 'role': 'confirmation', 'reuse_key': None})
    result.sort(key=lambda i: (('fit', 'calibration', 'confirmation').index(i['role']), ORDER.index(i['task']), i['key']))
    validate(result, previous, confirmation)
    return result


def validate(items, previous, confirmation):
    require(len({i['key'] for i in items}) == len(items), 'duplicate key')
    for field in ('source_group', 'gt_sha256', 'lq_sha256'):
        groups = {role: {i['row'][field] for i in items if i['role'] == role} for role in ('fit','calibration','confirmation')}
        require(all(not groups[a] & groups[b] for a in groups for b in groups if a != b), 'role leakage')
        hold = {i['row'][field] for i in previous if i['role'] == 'holdout'}
        require(all(not group & hold for group in groups.values()), 'old holdout leaked')
    old_map = {i['key']:i for i in previous if i['role'] in ('fit','calibration')}
    reused = {i['reuse_key']:i for i in items if i['reuse_key']}
    require(set(reused) == set(old_map), 'old fit/calibration must be retained exactly')
    for k,i in reused.items():
        require(i['row'] == old_map[k]['row'] and i['seed'] == old_map[k]['seed'] and i['task'] == old_map[k]['task'], 'reused input changed')
    expected = [{**i, 'role':'confirmation', 'reuse_key':None} for i in confirmation]
    require(sorted((i for i in items if i['role']=='confirmation'),key=lambda i:i['key']) ==
            sorted(expected,key=lambda i:i['key']), 'reserved confirmation changed')
    for i in items:
        r = i['row']; x,y = r['crop_xy']
        require(0 <= x <= int(r['width'])-512 and 0 <= y <= int(r['height'])-512, 'crop outside image')
        require(r['split'] == 'train' and int(r['composition_size']) == 1, 'only training-collection singleton pixels')
        for side in ('lq_path','gt_path'):
            require(r[side].startswith('data/foundir/MillionIRData/Train/'+r['category']+'/') and '..' not in Path(r[side]).parts,
                    'invalid collection path')


def schedule(items):
    """Source-balanced within class, then alternate available crops; no missing-data fallback."""
    fit = [i for i in items if i['role']=='fit']
    groups = {t:{} for t in ORDER}
    for j,i in enumerate(fit):
        groups[i['task']].setdefault(i['row']['source_group'], []).append(j)
    require(all(groups.values()), 'every class must have training sources')
    groups = {t:[sorted(v,key=lambda j:fit[j]['view_index']) for _,v in
                 sorted(g.items(),key=lambda kv:c.stable(SEED,'sample',t,kv[0]))] for t,g in groups.items()}
    return fit, groups


def batch_indices(groups, step):
    require(type(step) is int and 0 <= step < STEPS, 'outside fixed budget')
    result = []
    for slot in range(step*32, (step+1)*32):
        sources = groups[ORDER[slot%5]]; n = slot//5
        crops = sources[n%len(sources)]
        result.append(crops[(n//len(sources))%len(crops)])
    return result


def render_plan(items, predictions):
    confirms = {i['key']:i for i in items if i['role']=='confirmation'}
    require(Counter(i['task'] for i in confirms.values()) == Counter({t:20 for t in ORDER}), '100 fixed confirmation images')
    routes, arms = {}, {}
    require(set(predictions)==set(ARMS[:-1]), 'three saved model predictions required')
    for arm in ARMS:
        if arm == 'known_class_reference':
            choices = {k:i['task'] for k,i in confirms.items()}
        else:
            rows = predictions[arm]['rows']
            require(len(rows)==100 and {r['key'] for r in rows}==set(confirms), 'prediction population mismatch')
            choices = {}
            for r in rows:
                i=confirms[r['key']]; z=r['logits']
                require(len(z)==5 and all(math.isfinite(x) for x in z), 'invalid prediction')
                anchor=ORDER[max(range(5),key=lambda k:z[k])]
                require((r['seed'],r['crop_xy'],r['predicted_anchor'])==(i['seed'],i['row']['crop_xy'],anchor), 'prediction input mismatch')
                choices[r['key']]=anchor
        arms[arm]=[]
        for key in sorted(confirms):
            anchor=choices[key]; rid=key+'__'+anchor
            routes[rid]={'render_id':rid,'key':key,'anchor':anchor,'seed':confirms[key]['seed'],
                         'weights':[float(t==anchor) for t in ORDER]}
            arms[arm].append(rid)
    return {'arms':arms,'routes':routes,'unique_renders':len(routes),
            'reference_uses_known_class_not_deployed':True, 'selection_uses_restoration_metrics':False}


def aggregate(plan, items, metrics):
    require(set(metrics)==set(plan['routes']), 'missing/extra render metrics')
    require(all(set(v)=={'psnr','lpips'} and all(isinstance(z,(int,float)) and math.isfinite(z) for z in v.values())
                for v in metrics.values()), 'nonfinite or incomplete metrics')
    confirms={i['key']:i for i in items if i['role']=='confirmation'}
    results={}
    for arm,rids in plan['arms'].items():
        rows=[{'key':plan['routes'][rid]['key'],'task':confirms[plan['routes'][rid]['key']]['task'],
               'anchor':plan['routes'][rid]['anchor'],**metrics[rid]} for rid in rids]
        def mean(rs): return {m:sum(r[m] for r in rs)/len(rs) for m in ('psnr','lpips')}
        results[arm]={'count':len(rows),'overall':mean(rows),
                     'per_task':{t:mean([r for r in rows if r['task']==t]) for t in ORDER},'rows':rows}
    def delta(a,b): return {'psnr_gain_db':a['psnr']-b['psnr'],'lpips_reduction':b['lpips']-a['lpips']}
    diffs={b:{'overall':delta(results['conditional_v4']['overall'],results[b]['overall']),
              'per_task':{t:delta(results['conditional_v4']['per_task'][t],results[b]['per_task'][t]) for t in ORDER}}
           for b in ARMS if b!='conditional_v4'}
    targets={t:{'psnr_min':TARGETS[t][0],'lpips_max':TARGETS[t][1],
                'pass_on_this_population':results['conditional_v4']['per_task'][t]['psnr']>=TARGETS[t][0] and
                results['conditional_v4']['per_task'][t]['lpips']<=TARGETS[t][1]} for t in ORDER}
    joint=all(diffs[b]['overall']['psnr_gain_db']>=0 and diffs[b]['overall']['lpips_reduction']>=0 and
              (diffs[b]['overall']['psnr_gain_db']>0 or diffs[b]['overall']['lpips_reduction']>0)
              for b in ('image_only','previous_v3'))
    return {'arms':results,'vlm_v4_minus_comparators':diffs,'joint_mean_improvement_over_image_and_previous_vlm':joint,
            'historical_numeric_targets_on_new100':targets,'all_numeric_targets_on_new100_pass':all(v['pass_on_this_population'] for v in targets.values()),
            'official_SoW_pass':None,'new100_is_official_test':False}
