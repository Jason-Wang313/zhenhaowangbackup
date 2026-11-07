"""Descriptive first8 training full frames at fixed epoch boundaries, never a quality gate."""
import math
import numpy as np
from PIL import Image
from smoke import blur_fullgrid_1x2_data_v1 as data
from smoke import blur_fullgrid_1x2_state_v1 as state
from smoke.existing_flux_replay_contract_v1 import read, sha, write_new as write
from smoke.evaluate_foundir_official_vlm_router_v4 import png_once

ROOT, OUT, t, need = data.ROOT, data.OUT, data.base, data.need


def png(path, digest):
    need(sha(path) == digest, 'snapshot PNG changed')
    with Image.open(path) as image: return np.array(image.convert('RGB'),copy=True)


def summarize(records, b, step, digest):
    need(step in state.SNAPSHOTS and len(records) == 8, 'fixed8 snapshot required')
    by_index = {r['source_index']:r for r in records}
    need(set(by_index) == set(range(8)), 'missing/duplicate training frame')
    for index in range(8):
        r = by_index[index]
        need(r['item'] == b['rows'][index] and r['rank'] == index%2 and r['step'] == step
             and r['parameter_hash'] == digest and r['receipt_sha256'] == data.receipt_sha()
             and r['scope'] == 'training_fit_only', 'snapshot identity changed')
    means = {arm:{k:math.fsum(r['metrics'][arm][k] for r in records)/8
                  for k in ('psnr','lpips','ssim')} for arm in ('lq','candidate')}
    need(all(math.isfinite(v) for arm in means.values() for v in arm.values()), 'nonfinite snapshot metrics')
    return {'step':step,'parameter_hash':digest,'scope':'training_fit_only','quality_gate':False,
            'means':means,'records':sorted(records,key=lambda r:r['source_index']),
            'both_better_than_LQ':sum(r['metrics']['candidate']['psnr']>r['metrics']['lq']['psnr']
                                     and r['metrics']['candidate']['lpips']<r['metrics']['lq']['lpips'] for r in records),
            'receipt_sha256':data.receipt_sha()}


def evaluate(model,ae,prompt,metrics,b,step,rank,device):
    import json
    import torch
    import torch.distributed as dist
    rng = t.state.rng_state(device); before = t.state.parameter_hash(t.state.trainable_state(model))
    need(before == read(OUT/'steps'/f'step_{step:06d}.json')['parameter_hash'], 'snapshot weights differ')
    records = []; model.eval()
    try:
        for source in range(rank,8,2):
            item = b['rows'][source]; row = item['row']; tiles = {}; outputs = []
            for view in b['views'][source*15:source*15+15]:
                tensors = data.cached(view,device); need(tensors is not None, 'snapshot reference missing')
                with torch.no_grad():
                    pred = t.prediction(model,ae,prompt,tensors)
                    need(torch.isfinite(pred).all(),'nonfinite snapshot')
                    array = t.core.uint8_output(pred)[0].permute(1,2,0).cpu().numpy()
                if step == 1792:
                    spec = b['initial_tiles'][view['tile_key']]
                    need(np.array_equal(array,png(data.CONTEXT/spec['path'],spec['sha256'])),
                         'initial1792 cached tile pixel mismatch: '+view['tile_key'])
                file = OUT/'snapshots'/f'step_{step:06d}'/(view['tile_key']+'.png'); png_once(file,array)
                outputs.append({'tile_key':view['tile_key'],'path':str(file.relative_to(OUT)),'sha256':sha(file)})
                tiles[view['tile_key']] = array
                del tensors,pred
            full = data.geometry.stitch(row,tiles)
            if step == 1792:
                spec = b['initial_full'][row['record_id']]
                need(np.array_equal(full,png(data.CONTEXT/spec['path'],spec['sha256'])), 'initial full1792 differs')
            file = OUT/'snapshots'/f'step_{step:06d}'/(row['record_id']+'_full.png');png_once(file,full)
            # Fixed full prediction precedes GT decoding; no alternative-output selection.
            lq = png(ROOT/row['lq_path'],row['lq_sha256']); gt = png(ROOT/row['gt_path'],row['gt_sha256'])
            def tensor(array):return torch.from_numpy(array.copy()).permute(2,0,1).unsqueeze(0).to(device).float()/255
            with torch.no_grad():
                target = tensor(gt)
                values = {name:{k:float(m(tensor(array),target).mean()) for k,m in metrics.items()}
                          for name,array in [('lq',lq),('candidate',full)]}
            r = {'source_index':source,'item':item,'rank':rank,'step':step,'parameter_hash':before,
                 'receipt_sha256':data.receipt_sha(),'scope':'training_fit_only','metrics':values,
                 'tiles':outputs,'output':{'path':str(file.relative_to(OUT)),'sha256':sha(file)}}
            write(file.with_suffix('.json'),r);records.append(r)
            del target,tiles
        gathered = [None]*2;dist.all_gather_object(gathered,records)
        summary = summarize([r for group in gathered for r in group],b,step,before)
        need(t.state.parameter_hash(t.state.trainable_state(model)) == before,'snapshot changed weights')
        if rank == 0:
            write(OUT/'snapshots'/f'step_{step:06d}_summary.json',summary)
            print('BLUR_FULLGRID_1X2_TRAINING_FIT '+json.dumps({k:v for k,v in summary.items() if k!='records'}),flush=True)
    finally:
        t.state.restore_rng(rng,device);model.train()
    dist.barrier()


def verify(b):
    result = {}
    for step in state.SNAPSHOTS:
        summary = read(OUT/'snapshots'/f'step_{step:06d}_summary.json')
        digest = read(OUT/'steps'/f'step_{step:06d}.json')['parameter_hash']
        records = [read(OUT/'snapshots'/f'step_{step:06d}'/(i['row']['record_id']+'_full.json')) for i in b['rows'][:8]]
        need(summarize(records,b,step,digest) == summary, 'snapshot means/identity differ')
        for r in records:
            full = png(OUT/r['output']['path'],r['output']['sha256'])
            tiles = {s['tile_key']:png(OUT/s['path'],s['sha256']) for s in r['tiles']}
            need(np.array_equal(data.geometry.stitch(r['item']['row'],tiles),full),'snapshot stitch differs')
        result[str(step)] = summary
    return result
