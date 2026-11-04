#!/usr/bin/env python3
"""Complete matched 40-image top-1 evaluation using four missing renders only."""
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from smoke import analyze_foundir_frozen_router_errors_v1 as analysis
from smoke import diagnose_foundir_router_hard_counterfactual_v1 as prior_hard
from smoke import train_foundir_router_supervised_repair_v1 as supervised
c = supervised.c
require = analysis.require
ORDER = analysis.ORDER
ARMS = ('image_only', 'image_plus_vlm')
RUN = 'matched_top1_four_missing_v1'
OUTPUT = 'checkpoints/foundir_supervised_router_restoration_v1/' + RUN
REPORT = 'reports/foundir_supervised_router_restoration_terminal_2026-11-04.json'
CPU_REPORT_SHA = 'fb074f9dc53d6294cd3c432fb1529adbfc684418aab76edf7cc4d1b8e00320e1'
HARD_REPORT_SHA = '96ec2bf6b3d243584c524396b5bfce43f87fa56fe18cfdd3480c8f2c0607f1aa'
EXPECTED_NEW = {f'validation_foundir_train_{suffix}__haze' for suffix in ('0007534','0009805','0377282','0391222')}
SOURCES = ['smoke/evaluate_foundir_supervised_router_restoration_v1.py',
           'smoke/run_foundir_supervised_router_restoration_v1_gpu.sh',
           'smoke/start_foundir_supervised_router_restoration_v1_tmux.sh',
           'smoke/frozen_router_runtime_policy_v1.py']


def route_id(key, anchor):
    require(anchor in ORDER and key.startswith('validation_foundir_train_')
            and key.removeprefix('validation_foundir_train_').isdigit(), 'unsafe route identifier')
    return key + '__' + anchor


def make_plan(cpu, breakdown, hard):
    require(cpu['status'] == 'matched_cpu_router_training_complete' and cpu['run_id'] == supervised.RUN
            and cpu['steps_per_arm'] == 400 and cpu['deployment_policy'] == 'predicted_top1_singleton_only'
            and cpu['restoration_images_generated'] == 0, 'wrong trained-router report')
    old = {r['key']: r for r in breakdown['per_image']}
    require(len(old) == 40 and len(breakdown['per_image']) == 40
            and Counter(r['task'] for r in old.values()) == Counter({t:8 for t in ORDER}), 'fixed 40-image population required')
    available = {route_id(r['key'],r['task']) for r in old.values()}
    available.update(route_id(r['key'],r['predicted_anchor']) for r in hard['per_image'])
    predictions = {}; unique = {}
    require(set(cpu['arms']) == set(ARMS), 'two matched arms required')
    for arm in ARMS:
        rows = cpu['arms'][arm]['development']['rows']
        require(len(rows) == 40 and {r['key'] for r in rows} == set(old), 'arm population mismatch')
        predictions[arm] = []
        for row in rows:
            reference = old[row['key']]; logits = row['logits']
            require(len(logits) == 5 and all(analysis.number(v) for v in logits), 'invalid saved logits')
            k = max(range(5), key=lambda j: logits[j]); anchor = ORDER[k]
            weights = [float(j == k) for j in range(5)]
            require(row['predicted_anchor'] == anchor and row['hard_weights'] == weights
                    and row['seed'] == reference['seed'] and row['crop_xy'] == reference['crop_xy']
                    and row['task_for_reporting_only'] == reference['task'], 'changed prediction/input; never substitute GT anchor')
            rid = route_id(row['key'], anchor)
            plan = {'render_id': rid, 'key': row['key'], 'predicted_anchor': anchor,
                    'hard_weights_from_model_argmax': weights, 'seed': row['seed'], 'crop_xy': row['crop_xy']}
            if rid in unique: require(unique[rid] == plan, 'same route has inconsistent identity')
            unique[rid] = plan
            predictions[arm].append({'key': row['key'], 'render_id': rid})
    new = sorted(set(unique)-available)
    require(set(new) == EXPECTED_NEW and len(unique) == 47, 'scope changed: exactly four new and 47 unique prediction routes')
    return {'arms': predictions, 'routes': unique, 'new_plans': [unique[k] for k in new],
            'prediction_rows': 80, 'unique_prediction_routes': 47, 'reused_unique_prediction_routes': 43,
            'reused_prediction_rows': sum(x['render_id'] in available for a in predictions.values() for x in a),
            'new_render_count': 4}


def validate_artifact(root, folder, row):
    tensor = c.path(folder,row['tensor'])
    require(row['files'] == {row['tensor']: c.sha(tensor)}, 'cached image artifact changed')
    return {'tensor': tensor.relative_to(root).as_posix(), 'tensor_sha256': row['files'][row['tensor']]}


def cache_catalog(root, bound, breakdown, hard):
    """Match exact route/input identity, never compare metrics to choose a cache."""
    parent = root/'checkpoints/foundir_frozen_expert_router_v1'/analysis.RUN
    items = {i['key']:i for i in bound['items'] if i['split'] == 'validation'}
    old = {r['key']:r for r in breakdown['per_image']}
    prompts = {t:bound['parent']['experts'][t]['prompt'] for t in ORDER}
    result = {}
    for key,item in items.items():
        file = parent/'evaluation/known_class_control'/(key+'.json'); row = c.read(file)
        relative = file.relative_to(root).as_posix()
        require(c.sha(file) == breakdown['input_json_sha256'][relative], 'published baseline metadata changed')
        require(row['identity'] == {'receipt_sha256': breakdown['receipt_sha256'], 'item': item,
                                   'arm': 'evaluation/known_class_control'}
                and row['known_class_control_not_deployed'] is True, 'baseline input identity changed')
        analysis.check_route(row['route'],item['task'],control=True,prompts=prompts)
        require(row['metrics'] == old[key]['baseline'], 'baseline metrics changed')
        result[route_id(key,row['route']['anchor'])] = {
            'record': relative, 'record_sha256': c.sha(file), 'metrics': row['metrics'], 'route': row['route'],
            'source': 'existing_onehot_cache', **validate_artifact(root,parent,row)}
    hard_folder = root/prior_hard.OUTPUT
    require(c.sha(hard_folder/'receipt.json') == hard['receipt_sha256']
            and c.read(hard_folder/'runtime.json') == hard['runtime'], 'previous hard run identity changed')
    for saved in hard['per_image']:
        key = saved['key']; file = hard_folder/'hard'/(key+'.json'); row = c.read(file)
        prior_plan = next(p for p in breakdown['missing_counterfactual']['items'] if p['key'] == key)
        require(row['identity'] == {'plan':prior_plan,'receipt_sha256':hard['receipt_sha256'],
                                   'runtime_sha256':c.sha(hard_folder/'runtime.json')}
                and c.sha(file) == hard['artifacts'][key]['record_sha256']
                and row['files'] == hard['artifacts'][key]['files'] and row['metrics'] == saved['hard'],
                'previous predicted-onehot cache changed')
        analysis.check_route(row['route'],saved['predicted_anchor'],control=True,prompts=prompts)
        result[route_id(key,saved['predicted_anchor'])] = {
            'record':file.relative_to(root).as_posix(),'record_sha256':c.sha(file),'metrics':row['metrics'],
            'route':row['route'],'source':'previous_predicted_onehot',**validate_artifact(root,hard_folder,row)}
    require(len(result) == 44, 'expected 40 baseline plus four hard cache entries')
    return result


def prepare():
    require(c.sha(ROOT/supervised.REPORT) == CPU_REPORT_SHA and c.sha(ROOT/prior_hard.REPORT) == HARD_REPORT_SHA,
            'published parent report changed')
    supervised.verify()  # CPU JSON + two final checkpoint hashes; never resumes training.
    cpu = c.read(ROOT/supervised.REPORT); breakdown = c.read(ROOT/analysis.OUTPUT); hard = c.read(ROOT/prior_hard.REPORT)
    require(c.sha(ROOT/analysis.OUTPUT) == prior_hard.BREAKDOWN_SHA and analysis.diagnose(ROOT) == breakdown,
            'original metrics/identities changed')
    parent = ROOT/'checkpoints/foundir_frozen_expert_router_v1'/analysis.RUN
    bound = c.read(parent/'receipt.json'); plan = make_plan(cpu,breakdown,hard)
    for name,stat in bound['asset_stats'].items():
        actual = c.path(ROOT,name).stat()
        require([actual.st_size,actual.st_mtime_ns] == stat, 'asset metadata changed: '+name)
    catalog = cache_catalog(ROOT,bound,breakdown,hard)
    require(set(plan['routes'])-set(catalog) == EXPECTED_NEW, 'cache coverage changed')
    sources = {**bound['sources'], **supervised.load_bound()['sources'],
               **{name:c.sha(ROOT/name) for name in SOURCES}}
    for name,digest in sources.items(): require(c.sha(ROOT/name) == digest, 'source changed: '+name)
    runtime_files = [parent/f'runtime_node{n}.json' for n in range(4)] + [ROOT/prior_hard.OUTPUT/'runtime.json']
    value = {'run_id':RUN,'plan':plan,'catalog':catalog,'sources':sources,
             'cpu_report_sha256':CPU_REPORT_SHA,'hard_report_sha256':HARD_REPORT_SHA,
             'parent_receipt_sha256':breakdown['receipt_sha256'],
             'runtime_inputs':{f.relative_to(ROOT).as_posix():c.sha(f) for f in runtime_files},
             'training':False,'GT_selects_expert':False,'nodes':1,'gpus':2,'wall_cap_seconds':900,
             'new_render_count':4,'automatic_followup':False}
    c.write_new(ROOT/OUTPUT/'receipt.json',value)
    print('MATCHED_RESTORATION_PREFLIGHT_PASS new=4 reused_prediction_rows=76 total_prediction_rows=80 pixels_decoded=false',flush=True)


def bound_inputs():
    b = c.read(ROOT/OUTPUT/'receipt.json')
    require(b['run_id'] == RUN and b['cpu_report_sha256'] == c.sha(ROOT/supervised.REPORT) == CPU_REPORT_SHA
            and b['hard_report_sha256'] == c.sha(ROOT/prior_hard.REPORT) == HARD_REPORT_SHA, 'receipt/report mismatch')
    for name,digest in {**b['sources'],**b['runtime_inputs']}.items():
        require(c.sha(ROOT/name) == digest, 'bound input changed: '+name)
    parent = ROOT/'checkpoints/foundir_frozen_expert_router_v1'/analysis.RUN
    require(c.sha(parent/'receipt.json') == b['parent_receipt_sha256'], 'parent receipt changed')
    require(c.sha(ROOT/analysis.OUTPUT) == prior_hard.BREAKDOWN_SHA, 'published breakdown changed')
    expected = make_plan(c.read(ROOT/supervised.REPORT),c.read(ROOT/analysis.OUTPUT),c.read(ROOT/prior_hard.REPORT))
    require(b['plan'] == expected, 'fixed output plan changed')
    return b,parent


def runtime_check(idle):
    import torch
    from smoke import train_foundir_frozen_expert_router_mixed_driver_v1 as impl
    from smoke.frozen_router_runtime_policy_v1 import execution_signature
    _,parent = bound_inputs(); bound = impl.contract.load_receipt(parent)
    actual = impl.native.hopper.hardware(idle)
    require('H200' in actual['gpu_name'] and all(torch.cuda.get_device_properties(i).total_memory >= 139*1024**3 for i in range(2)),
            'observed inference peak ~138GiB: two H200 >=139GiB required; H10080GB is not qualified')
    actual['metrics'] = impl.parent.four_impl.metric_runtime({'metric':bound['parent']['metric']})
    actual['memory_policy'] = {'inherited':bound['parent']['memory_policy'],'router_explicit_gate_checkpoint':True}
    histories = [c.read(parent/f'runtime_node{n}.json') for n in range(4)]
    histories.append(c.read(ROOT/prior_hard.OUTPUT/'runtime.json')['inherited_comparison_signature'])
    require(all(execution_signature(actual) == execution_signature(h) for h in histories), 'cached/new numerical runtime mismatch')
    value = {'inherited_comparison_signature':actual,'actual_nodes':1,'actual_gpus':2,'training':False,
             'driver_difference_allowed':True,'H100_qualified':False}
    file = ROOT/OUTPUT/'runtime.json'
    if idle: c.write_new(file,value)
    else: require(c.read(file) == value, 'runtime changed during/resuming evaluation')
    return impl,bound


def saved_new(plan):
    output = ROOT/OUTPUT; file = output/'new'/(plan['render_id']+'.json')
    if not file.exists(): return None
    row = c.read(file)
    require(row['identity'] == {'plan':plan,'receipt_sha256':c.sha(output/'receipt.json'),
                               'runtime_sha256':c.sha(output/'runtime.json')}, 'new result identity changed')
    require(row['route']['anchor'] == plan['predicted_anchor'] and row['route']['weights'] == [plan['hard_weights_from_model_argmax']]
            and row['GT_used_for_generation_or_selection'] is False, 'new route changed')
    validate_artifact(ROOT,output,row)
    return row


def run():
    import torch
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    from lowlight_restoration.five_expert_flux import restore_five_expert
    rank = int(os.environ['LOCAL_RANK'])
    require(rank in (0,1) and int(os.environ['WORLD_SIZE']) == 2 and int(os.environ['LOCAL_WORLD_SIZE']) == 2
            and int(os.environ['RANK']) == rank, 'one node, two inference workers only')
    receipt,_ = bound_inputs(); impl,bound = runtime_check(False)
    torch.cuda.set_device(rank); device = torch.device('cuda',rank)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    os.environ.update(KLEIN_4B_BASE_MODEL_PATH=str(c.path(ROOT,bound['parent']['model']['flow_weights'])),
                      AE_MODEL_PATH=str(c.path(ROOT,bound['parent']['model']['autoencoder_weights'])))
    pending = [p for p in receipt['plan']['new_plans'][rank::2] if saved_new(p) is None]
    if not pending:
        print(f'MATCHED_RESTORATION_WORKER_COMPLETE rank={rank} reused_new=2',flush=True); return
    model,gates,rules,ae,prompts = impl.load_bank(bound,device)
    metric = LearnedPerceptualImagePatchSimilarity(net_type='alex',normalize=True).to(device).eval().requires_grad_(False)
    items = {i['key']:i for i in bound['items'] if i['split'] == 'validation'}; output = ROOT/OUTPUT
    for plan in pending:
        start = time.monotonic(); item = items[plan['key']]
        with torch.no_grad():
            lq = impl.pixels(item,'lq',device)
            w = torch.tensor([plan['hard_weights_from_model_argmax']],device=device)
            raw,u8,route = restore_five_expert(model,gates,rules,w,ae,lq,prompts,seed=plan['seed'])
            require(route['anchor'] == plan['predicted_anchor'], 'wrong selected expert')
            gt = impl.pixels(item,'gt',device)  # Only after the predetermined output exists.
            image,target = u8.float()/255.,(gt+1)*.5
            mse = float((image-target).square().mean())
            metrics = {'psnr':100. if mse <= 1e-12 else -10*math.log10(mse),'lpips':float(metric(image,target))}
            metric.reset()
        require(all(analysis.number(v) for v in metrics.values()), 'nonfinite output metric')
        tensor,digest = impl.native.replay.save_tensors(output,plan['render_id'],{'uint8':u8})
        value = {'identity':{'plan':plan,'receipt_sha256':c.sha(output/'receipt.json'),'runtime_sha256':c.sha(output/'runtime.json')},
                 'tensor':tensor,'files':{tensor:digest},'route':route,'metrics':metrics,
                 'GT_used_for_generation_or_selection':False,'seconds':time.monotonic()-start,'rank':rank,
                 'peak_allocated_gib':torch.cuda.max_memory_allocated(device)/1024**3}
        c.write_new(output/'new'/(plan['render_id']+'.json'),value)
        print('MATCHED_RESTORATION_IMAGE_COMPLETE '+json.dumps({'render_id':plan['render_id'],'metrics':metrics}),flush=True)
        del lq,gt,image,target,raw,u8
    print(f'MATCHED_RESTORATION_WORKER_COMPLETE rank={rank}',flush=True)


def aggregate(plan, breakdown, metrics):
    """Join by exact input+predicted route, with no metric-dependent selection."""
    old = {r['key']:r for r in breakdown['per_image']}; arms = {}
    require(len(old) == len(breakdown['per_image']) == 40
            and Counter(r['task'] for r in old.values()) == Counter({t:8 for t in ORDER}), 'incomplete reference population')
    require(set(plan['arms']) == set(ARMS), 'two matched arms required')
    require(set(plan['routes']) <= set(metrics), 'missing prediction metric')
    for arm in ARMS:
        require(len(plan['arms'][arm]) == 40 and {r['key'] for r in plan['arms'][arm]} == set(old),
                'incomplete/duplicate arm population')
        rows = []
        for entry in plan['arms'][arm]:
            item = old[entry['key']]; p = plan['routes'][entry['render_id']]; values = metrics[entry['render_id']]
            require(p['key'] == entry['key'] and route_id(p['key'],p['predicted_anchor']) == entry['render_id'],
                    'metric joined to wrong input/route')
            require(set(values) == {'psnr','lpips'} and all(analysis.number(v) for v in values.values()), 'invalid metric')
            rows.append({'key':entry['key'],'task':item['task'],'predicted_anchor':p['predicted_anchor'],
                         'render_id':entry['render_id'],'metrics':values,'correct':p['predicted_anchor'] == item['task'],
                         'known_class_control':item['baseline'],'old_step408_soft':item['step408']['metrics']})
        def group(selected):
            return {'count':len(selected),'correct':sum(r['correct'] for r in selected),
                    'means':{name:{m:analysis.mean(r[name][m] for r in selected) for m in ('psnr','lpips')}
                             for name in ('metrics','known_class_control','old_step408_soft')},
                    'regressions_vs_known_class':{m:sum((r['metrics'][m] < r['known_class_control'][m]) if m == 'psnr'
                                                       else (r['metrics'][m] > r['known_class_control'][m]) for r in selected)
                                                  for m in ('psnr','lpips')}}
        arms[arm] = {'overall':group(rows),'per_task':{t:group([r for r in rows if r['task'] == t]) for t in ORDER},'per_image':rows}
    def gain(a,b): return {'psnr_gain_db':b['psnr']-a['psnr'],'lpips_reduction':a['lpips']-b['lpips']}
    comparisons = {'overall':gain(arms['image_only']['overall']['means']['metrics'],arms['image_plus_vlm']['overall']['means']['metrics']),
                   'per_task':{t:gain(arms['image_only']['per_task'][t]['means']['metrics'],
                                      arms['image_plus_vlm']['per_task'][t]['means']['metrics']) for t in ORDER}}
    return arms,comparisons


def summarize():
    receipt,parent = bound_inputs(); bound = c.read(parent/'receipt.json')
    breakdown = c.read(ROOT/analysis.OUTPUT); hard = c.read(ROOT/prior_hard.REPORT)
    catalog = cache_catalog(ROOT,bound,breakdown,hard)
    require(catalog == receipt['catalog'], 'reused cache changed after preflight')
    require({p.stem for p in (ROOT/OUTPUT/'new').glob('*.json')} == EXPECTED_NEW, 'exactly four new outputs required')
    prompts = {t:bound['parent']['experts'][t]['prompt'] for t in ORDER}
    artifacts = dict(catalog); new_diagnostics = []
    for plan in receipt['plan']['new_plans']:
        row = saved_new(plan); require(row is not None,'new output missing')
        analysis.check_route(row['route'],plan['predicted_anchor'],control=True,prompts=prompts)
        file = ROOT/OUTPUT/'new'/(plan['render_id']+'.json')
        artifacts[plan['render_id']] = {'record':file.relative_to(ROOT).as_posix(),'record_sha256':c.sha(file),
            'metrics':row['metrics'],'route':row['route'],'source':'new_render',**validate_artifact(ROOT,ROOT/OUTPUT,row)}
        new_diagnostics.append({'render_id':plan['render_id'],'seconds':row['seconds'],'peak_allocated_gib':row['peak_allocated_gib']})
    arms,comparison = aggregate(receipt['plan'],breakdown,{k:v['metrics'] for k,v in artifacts.items()})
    cpu = c.read(ROOT/supervised.REPORT)
    require(all(arms[a]['overall']['correct'] == cpu['arms'][a]['development']['correct'] for a in ARMS), 'router count mismatch')
    value = {'status':'matched_top1_restoration_evaluation_complete','run_id':RUN,
             'receipt_sha256':c.sha(ROOT/OUTPUT/'receipt.json'),'cpu_report_sha256':CPU_REPORT_SHA,
             'runtime':c.read(ROOT/OUTPUT/'runtime.json'),'arms':arms,'vlm_minus_image_only':comparison,
             'plan_counts':{k:receipt['plan'][k] for k in ('prediction_rows','unique_prediction_routes',
                            'reused_unique_prediction_routes','reused_prediction_rows','new_render_count')},
             'artifacts':artifacts,'new_render_diagnostics':new_diagnostics,'training':False,'GT_selects_expert':False,
             'automatic_followup':False,'official_test':False,'singleton_development_only':True,
             'interpretation':['Same 40 inputs and same predicted top-1 policy for both matched trained router arms.',
                'Cache lookup uses input+route identity, never quality or a GT choice of expert.',
                'Known-class controls are references, not deployable GT routing.',
                'Comparison to old step408 soft also changes training and policy; do not attribute that whole difference to VLM.',
                'Small single-seed development set; no official SoW, Rain100 or mixed-degradation gate claim.',
                'Driver differences recorded; no cross-driver bitwise equivalence claimed.']}
    c.write_new(ROOT/REPORT,value)
    print('MATCHED_RESTORATION_RESULT_VERIFIED '+json.dumps(comparison),flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('phase',choices=['prepare','hardware','run','summarize'])
    args = ap.parse_args(); require(str(ROOT) == analysis.ACTIVE_ROOT,'destination workspace only')
    if args.phase == 'prepare': prepare()
    elif args.phase == 'hardware': runtime_check(True)
    elif args.phase == 'run': run()
    else: summarize()


if __name__ == '__main__': main()
