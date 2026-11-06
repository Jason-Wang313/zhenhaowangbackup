#!/usr/bin/env python3
"""One fixed GT-free Haze tile synchronization; CPU generation, one-node scoring."""
import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ['FOUNDIR_OFFICIAL_NNODES'] = '4'  # Frozen parent identity, not execution topology.
from smoke import audit_foundir_fullstack_blur_strength_v1 as base

c, fix, require = base.c, base.fix, base.require
PARENT = base.PARENT
OUT = ROOT/'checkpoints/foundir_fullstack_haze_consensus_v1'
REPORT = ROOT/'reports/foundir_fullstack_haze_consensus_v1_result_2026-11-06.json'
SOURCES = ['smoke/evaluate_foundir_fullstack_haze_consensus_v1.py',
           'smoke/run_foundir_fullstack_haze_consensus_v1.sh',
           'lowlight_restoration/haze_overlap_consensus_v1.py',
           'smoke/audit_foundir_fullstack_blur_strength_v1.py']
POLICY = {'method': 'adjacent_overlap_constant_RGB_weighted_least_squares',
          'edge_weight': 'intersection_pixel_count', 'gauge': 'zero_mean_blended_correction_field',
          'coefficient': 1, 'parameter_sweep': False, 'output': 'original_unrounded_stitch_minus_field_then_round_clip',
          'mean_preservation': 'preclip_only; report actual postquantization shift',
          'scope': 'existing predicted haze anchor only; other routes exactly reused',
          'GT_input': False, 'training': False, 'new_FLUX_generation': False,
          'world': 2, 'nodes': [0]}


def manifest():
    old = base.manifest()
    value = {k: old[k] for k in ('receipt_sha256', 'parent_report_sha256', 'metricfix_sha256', 'rows', 'metric_identity')}
    value.update(policy=POLICY, sources={s: c.sha(ROOT/s) for s in SOURCES})
    require(sum(i['deployed']['anchor']=='haze' for i in value['rows']) == 200, '200 predicted haze records required')
    return value


def bound():
    value = c.read(OUT/'receipt.json')
    require(value == manifest(), 'source/parent/policy changed')
    return value


def prepare():
    c.write_new(OUT/'receipt.json', manifest())
    print('HAZE_CONSENSUS_PREPARED records=600 affected=200 world=2', flush=True)


def candidate(item, receipt):
    row = item['row']; value = c.read(OUT/'candidates'/(row['record_id']+'.json'))
    require(value['receipt_sha256']==receipt and value['item']==item, 'candidate identity changed')
    require(c.sha(OUT/value['output']['path'])==value['output']['sha256'], 'candidate PNG changed')
    require(value['GT_input'] is False and value['parent_pixel_exact'] is True, 'candidate contract changed')
    return value


def generate():
    import numpy as np
    from PIL import Image
    from safetensors.numpy import load
    from lowlight_restoration.haze_overlap_consensus_v1 import synchronize
    from smoke import official_vlm_router_v4_contract as contract
    value = bound(); receipt = c.sha(OUT/'receipt.json'); start = time.monotonic()
    items = [i for i in value['rows'] if i['deployed']['anchor']=='haze']
    if (OUT/'generation.json').exists():
        old = c.read(OUT/'generation.json')
        require(old['receipt_sha256']==receipt and old['records']==200 and old['status']==0, 'generation identity changed')
        for item in items:
            candidate(item, receipt)
        print('HAZE_CONSENSUS_GENERATION_REUSED', flush=True)
        return
    for index, item in enumerate(items):
        row = item['row']; dest = OUT/'candidates'/(row['record_id']+'.json')
        if dest.exists():
            candidate(item, receipt); continue
        tiles = contract.tiles(row); predictions = {}; identities = {}
        parent_record = c.read(PARENT/'results'/(row['record_id']+'.json'))
        for tile in tiles:
            path = PARENT/'restored_tiles'/(tile['key']+'__haze.json'); meta = c.read(path)
            expected = {'key':tile['key']+'__haze', 'record_id':row['record_id'], 'lq_sha256':row['lq_sha256'],
                        'tile':tile, 'anchor':'haze'}
            require(meta['identity']['item']==expected and meta['identity']['receipt_sha256']==value['receipt_sha256'],
                    'parent tile identity changed')
            require(meta['identity']['runtime_sha256']==parent_record['runtime_sha256'] and meta['GT_input'] is False,
                    'parent runtime changed')
            require(set(meta['files'])=={meta['tensor']}, 'unexpected tensor files')
            data = (PARENT/meta['tensor']).read_bytes()
            require(hashlib.sha256(data).hexdigest()==meta['files'][meta['tensor']], 'tile hash changed')
            tensors = load(data); require(set(tensors)=={'uint8'}, 'unexpected tile tensor')
            arr = tensors['uint8']; require(arr.shape==(1,3,512,512) and arr.dtype==np.uint8, 'invalid native tile')
            predictions[tile['key']] = arr[0].transpose(1,2,0); identities[tile['key']] = c.sha(path)
        image, parent_image, stats = synchronize(row, tiles, predictions)
        original = base.load_png(PARENT/item['output']['path'], item['output']['sha256'])
        require(np.array_equal(parent_image, original), 'parent stitch not pixel exact')
        # This phase never reads GT pixels. Only a single frozen candidate is generated.
        image_path = OUT/'images'/(row['record_id']+'.png')
        buffer = io.BytesIO(); Image.fromarray(image).save(buffer, format='PNG'); data = buffer.getvalue()
        digest = hashlib.sha256(data).hexdigest(); image_path.parent.mkdir(parents=True, exist_ok=True)
        if image_path.exists():
            require(c.sha(image_path)==digest, 'conflicting existing image; preserve and stop')
        else:
            tmp = image_path.with_name(image_path.name+f'.tmp.{os.getpid()}.{time.time_ns()}')
            with tmp.open('xb') as f:
                f.write(data); f.flush(); os.fsync(f.fileno())
            os.replace(tmp, image_path)
        spec = {'path':'images/'+row['record_id']+'.png', 'sha256':digest}
        c.write_new(dest, {'item':item, 'receipt_sha256':receipt, 'output':spec, 'stats':stats,
                          'parent_pixel_exact':True, 'GT_input':False, 'tile_metadata_hashes':identities})
        if (index+1)%10==0:
            print(f'HAZE_CONSENSUS_GENERATED records={index+1}/200 seconds={time.monotonic()-start:.1f}', flush=True)
    c.write_new(OUT/'generation.json', {'receipt_sha256':receipt, 'records':200,
                                      'seconds':time.monotonic()-start, 'gpus':0, 'status':0})
    print('HAZE_CONSENSUS_GENERATION_END status=0', flush=True)


def check_record(record, item, receipt, rank):
    require(record['item']==item and record['receipt_sha256']==receipt and record['rank']==rank, 'record identity changed')
    require(set(record['metrics'])=={'psnr','lpips','ssim'} and all(math.isfinite(v) for v in record['metrics'].values()),
            'invalid metrics')
    if item['deployed']['anchor']=='haze':
        require(record['output']==candidate(item,receipt)['output'] and record['reused_parent'] is False, 'candidate changed')
    else:
        require(record['reused_parent'] is True and record['output']==item['output'], 'unchanged arm changed')
        require(record['metrics']=={k:item['deployed'][k] for k in ('psnr','lpips','ssim')}, 'reused metrics changed')


def run(node):
    require(node==0, 'single-node task')
    import torch
    value = bound(); receipt = c.sha(OUT/'receipt.json')
    generation = c.read(OUT/'generation.json')
    require(generation['receipt_sha256']==receipt and generation['status']==0 and generation['records']==200, 'generation incomplete')
    rank = int(os.environ['LOCAL_RANK']); require(rank in (0,1) and int(os.environ['LOCAL_WORLD_SIZE'])==2, 'two local workers')
    torch.cuda.set_device(rank); device = torch.device('cuda', rank)
    models, _ = fix.checked_metric_models(device, value['metric_identity'])
    start = time.monotonic(); assigned = value['rows'][rank::2]
    def tensor(array):
        return torch.from_numpy(array).permute(2,0,1).unsqueeze(0).to(device).float()/255
    for index, item in enumerate(assigned):
        row = item['row']; path = OUT/'records'/(row['record_id']+'.json')
        if path.exists():
            check_record(c.read(path), item, receipt, rank); continue
        reuse = item['deployed']['anchor']!='haze'
        if reuse:
            metrics = {k:item['deployed'][k] for k in ('psnr','lpips','ssim')}; spec = item['output']
        else:
            cand = candidate(item, receipt); spec = cand['output']
            pred = tensor(base.load_png(OUT/spec['path'], spec['sha256']))
            gt = tensor(base.load_png(ROOT/row['gt_path'], row['gt_sha256']))
            with torch.no_grad():
                metrics = {k:float(model(pred,gt).item()) for k,model in models.items()}
            del pred, gt
        record = {'item':item, 'receipt_sha256':receipt, 'rank':rank, 'metrics':metrics,
                  'output':spec, 'reused_parent':reuse}
        check_record(record,item,receipt,rank); c.write_new(path,record)
        if (index+1)%10==0:
            print(f'HAZE_CONSENSUS_PROGRESS rank={rank} records={index+1}/300 seconds={time.monotonic()-start:.1f}', flush=True)
    c.write_new(OUT/f'worker{rank}.json', {'rank':rank, 'receipt_sha256':receipt,
                'records':[i['row']['record_id'] for i in assigned], 'seconds':time.monotonic()-start,
                'peak_allocated_gib':torch.cuda.max_memory_allocated()/1024**3})
    print(f'HAZE_CONSENSUS_WORKER_COMPLETE rank={rank}', flush=True)


def finish():
    value = bound(); receipt = c.sha(OUT/'receipt.json'); records = []
    for rank in range(2):
        items = value['rows'][rank::2]; marker = c.read(OUT/f'worker{rank}.json')
        require(marker['rank']==rank and marker['receipt_sha256']==receipt and
                marker['records']==[i['row']['record_id'] for i in items], 'worker incomplete')
        for item in items:
            record = c.read(OUT/'records'/(item['row']['record_id']+'.json'))
            check_record(record,item,receipt,rank); records.append(record)
    require(len(records)==len({r['item']['row']['record_id'] for r in records})==600, '600 unique results required')
    categories = {}; checks = {}
    for cat, (psnr, lpips) in base.THRESHOLDS.items():
        group = [r for r in records if r['item']['row']['category']==cat]
        require(len(group)==fix.x.COUNTS[cat], 'category cardinality changed')
        avg = {k:math.fsum(r['metrics'][k] for r in group)/len(group) for k in ('psnr','lpips','ssim')}
        categories[cat] = avg; checks[cat] = {'psnr':avg['psnr']>=psnr, 'lpips':avg['lpips']<=lpips}
    haze = [r for r in records if r['item']['deployed']['anchor']=='haze']
    report = {'status':'complete', 'records':600, 'affected_records':200, 'receipt_sha256':receipt,
              'policy':POLICY, 'per_category':categories, 'checks':checks,
              'all_ten_pass':all(all(x.values()) for x in checks.values()),
              'parent':c.read(base.PARENT_REPORT)['arms']['conditional_v4']['per_category'],
              'haze_PSNR_wins':sum(r['metrics']['psnr']>r['item']['deployed']['psnr'] for r in haze),
              'haze_LPIPS_wins':sum(r['metrics']['lpips']<r['item']['deployed']['lpips'] for r in haze),
              'haze_worst_PSNR_delta':min(r['metrics']['psnr']-r['item']['deployed']['psnr'] for r in haze),
              'independent_blind_test':False, 'GT_used_for_selection':False,
              'selected_for_deployment':None}
    c.write_new(REPORT, report)
    print('HAZE_CONSENSUS_RESULT_VERIFIED '+json.dumps(report), flush=True)


if __name__=='__main__':
    p = argparse.ArgumentParser(); p.add_argument('phase',choices=['prepare','generate','run','finish'])
    p.add_argument('--node',type=int,default=0); args=p.parse_args()
    if args.phase=='run': run(args.node)
    else: globals()[args.phase]()
