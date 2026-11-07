"""All15 deployment contexts per original training source; reuse frozen reference caches."""
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import json
import os
from pathlib import Path
import shutil
import time
from smoke import train_foundir_blur_multiview_v1 as previous
from smoke import official_vlm_router_v4_contract as geometry
from smoke.existing_flux_replay_contract_v1 import read, sha, write_new as write

ROOT, base, need = previous.ROOT, previous.parent, previous.need
OUT = ROOT/'checkpoints/foundir_blur_fullgrid_v1'
CONTEXT = ROOT/'checkpoints/foundir_blur_multiview_context_v1'
EVIDENCE = ROOT/'reports/foundir_blur_singleframe_fit_v1_interpretation_2026-11-07.json'
RECIPE = {**previous.RECIPE, 'version': 5, 'start_step': 1792, 'train_steps': 9472,
          'new_updates': 7680, 'views_per_source': 15, 'training_views': 15360,
          'fullgrid_epochs': 4, 'new_exposures_per_view': 4, 'final_exposures_per_source': 74,
          'sample_order': 'all15360_view_shuffle_without_replacement_seed62017; four_complete_epochs',
          'crop_selection': 'all15_deployment_grid_crops; no_GT_or_scores',
          'initialization': 'complete1792_model_Adam_four_rank_RNG; never_singleframe1992',
          'migration': 'same_world4_state_preserved; new_fullgrid_data_not_bitwise_old_training',
          'evaluation': 'only_final9472_complete_official150; no_checkpoint_search',
          'training_fit_snapshots': [1792,3712,5632,7552,9472],
          'training_fit_population': 'fixed_original_first8_sources_all15_tiles',
          'training_fit_is_quality_gate': False}
SOURCES = ['smoke/blur_fullgrid_data_v1.py', 'smoke/blur_fullgrid_state_v1.py',
           'smoke/train_foundir_blur_fullgrid_v1.py', 'smoke/blur_fullgrid_snapshots_v1.py',
           'smoke/run_foundir_fullstack_blur_fullgrid_v1.sh',
           'smoke/run_foundir_fullstack_blur_fullgrid_cache_v1.sh']


def key(item):
    x,y = item['crop_xy']
    return f"train_{item['row']['record_id']}__grid{item['view']:02d}_x{x}_y{y}"


def views(rows):
    result = []
    for source, old in enumerate(rows):
        row = old['row']
        need(old['role'] == 'train' and row['split'] == 'train' and row['category'] == '01Blur'
             and (int(row['width']),int(row['height'])) == (1920,1080), 'same training1080p only')
        grid = geometry.tiles(row); need(len(grid) == 15, 'deployment grid changed')
        for view, tile in enumerate(grid):
            result.append({'role': 'train', 'row': row, 'crop_xy': [tile['x'],tile['y']],
                           'view': view, 'source_index': source, 'tile_key': tile['key']})
    need(len({key(i) for i in result}) == len(result), 'cache keys collide')
    return result


def reuse_map(old_views, new_views):
    old = {(i['row']['record_id'],tuple(i['crop_xy'])): i for i in old_views}
    need(len(old) == len(old_views), 'duplicate parent view')
    matches = {}
    for i in new_views:
        prior = old.get((i['row']['record_id'],tuple(i['crop_xy'])))
        if prior is not None:
            need(prior['row'] == i['row'] and prior['role'] == i['role'] == 'train', 'reuse source identity changed')
            matches[key(i)] = prior
    need(len(matches) == len(old_views), 'parent view missing from full grid')
    return matches


@lru_cache(maxsize=1)
def receipt_sha():
    # Immutable receipt validated by bound() at process entry; avoid hashing ~20MB per microbatch.
    return sha(OUT/'receipt.json')


def prepare():
    from PIL import Image
    begun = time.monotonic(); old = previous.bound()
    saved, marker = previous.state.load_step(previous.OUT,1792,sha(previous.OUT/'receipt.json'))
    need(saved['step'] == 1792 and len(saved['optimizer']['state']) == 100, 'complete formal1792 required')
    for rank in range(4):
        w = read(previous.OUT/f'worker{rank}.json')
        need(w['step'] == 1792 and w['exit_code'] == 0 and w['parameter_hash'] == saved['parameter_hash'], 'parent incomplete')
    evidence = read(EVIDENCE)
    need(evidence['status'] == 'fixed_snapshots_training_context_analysis_complete'
         and evidence['scope'] == 'training_fit_only', 'diagnostic evidence required, not its weights')
    items = views(old['rows']); reuse = reuse_map(old['views'],items)
    need(len(items) == 15360 and len(reuse) == 4096, 'fixed1024 full grid required')
    def check_image(item):
        row = item['row']
        for side in ('lq','gt'):
            p = ROOT/row[side+'_path']; need(sha(p) == row[side+'_sha256'], 'training image changed')
            with Image.open(p) as im: need(im.size == (1920,1080), 'training dimensions changed')
    def check_cache(pair):
        new_key, item = pair
        path = previous.OUT/'cache'/(previous.data.key(item)+'.json'); r = read(path)
        file = previous.OUT/r['tensor']
        need(r['item'] == item and r['receipt_sha256'] == sha_parent and r['GT_input'] is False,
             'parent reference identity changed')
        need(file.resolve().parent == (previous.OUT/'tensors').resolve() and sha(file) == r['sha256'],
             'parent reference SHA/path changed')
        return new_key, {'marker': str(path.relative_to(ROOT)), 'marker_sha256': sha(path),
                         'tensor': str(file.relative_to(ROOT)), 'sha256': r['sha256']}
    sha_parent = sha(previous.OUT/'receipt.json')
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(check_image,old['rows']))
        imports = dict(pool.map(check_cache,reuse.items()))
    ctx = read(CONTEXT/'receipt.json')
    need(ctx['parameter_hash'] == saved['parameter_hash'] and ctx['items'] == old['rows'][:8], 'initial context parent differs')
    initial_tiles, initial_full = {}, {}
    for item in old['rows'][:8]:
        r = read(CONTEXT/'records'/(base.parent.key(item)+'.json'))
        need(r['item'] == item and r['parameter_hash'] == saved['parameter_hash'], 'initial full record differs')
        initial_full[item['row']['record_id']] = r['output']
        for tile in geometry.tiles(item['row']):
            tr = read(CONTEXT/'tiles'/(tile['key']+'.json'))
            need(tr['identity']['item'] == item and tr['GT_input'] is False, 'initial tile differs')
            initial_tiles[tile['key']] = tr['png']
    for spec in [*initial_tiles.values(), *initial_full.values()]:
        need(sha(CONTEXT/spec['path']) == spec['sha256'], 'initial PNG changed')
    for name,digest in old['assets'].items(): need(sha(ROOT/name) == digest, 'asset changed')
    need(shutil.disk_usage(ROOT).free > 4*1024**4, 'need4TiB to preserve all per-step states and old assets')
    b = {k: old[k] for k in ('rows','assets','model','lora','prompt','metric_identity')}
    b.update(recipe=RECIPE, views=items, imports=imports, parent_receipt_sha256=sha_parent,
             parent_checkpoint=marker, evidence_sha256=sha(EVIDENCE),
             context_receipt_sha256=sha(CONTEXT/'receipt.json'), initial_tiles=initial_tiles, initial_full=initial_full,
             sources={n: sha(ROOT/n) for n in sorted(set(old['sources'])|set(SOURCES))})
    write(OUT/'receipt.json', b)
    # Immutable hardlinks share bytes only; never modify either parent or imported tensor.
    directory = OUT/'tensors'; directory.mkdir(parents=True,exist_ok=True)
    for item in items:
        origin = imports.get(key(item))
        if origin is None: continue
        file = directory/('imported_'+key(item)+'.safetensors')
        if file.exists(): need(sha(file) == origin['sha256'], 'existing imported tensor changed')
        else: os.link(ROOT/origin['tensor'],file)
        write(OUT/'cache'/(key(item)+'.json'), {'item':item, 'rank':None,'GT_input':False,
              'receipt_sha256':receipt_sha(),'tensor':str(file.relative_to(OUT)), 'sha256':origin['sha256'],
              'origin':origin})
    result = {'status':'destination_CPU_preflight_pass','training_sources':1024,'training_views':15360,
              'imported_reference_tensors_full_SHA_verified':4096,'remaining_new_caches':11264,
              'parent_model_Adam_four_RNG_sampler_SHA_verified':True,'singleframe_weights_used':False,
              'all2048_training_files_SHA_verified':True,'initial120_tiles8_full_PNG_SHA_verified':True,
              'old_cache_bytes_modified':False,'official_pixels_decoded':False,
              'recipe':RECIPE,'receipt_sha256':receipt_sha(),'seconds':time.monotonic()-begun,'GPU_hours':0,'exit_code':0}
    write(ROOT/'reports/foundir_blur_fullgrid_v1_preflight_2026-11-07.json',result)
    print('BLUR_FULLGRID_PREFLIGHT_PASS '+json.dumps(result),flush=True)


def bound():
    b = read(OUT/'receipt.json'); old = previous.bound()
    need(b['recipe'] == RECIPE and b['rows'] == old['rows'] and b['views'] == views(old['rows'])
         and b['assets'] == old['assets'], 'fullgrid scope changed')
    need(b['parent_receipt_sha256'] == sha(previous.OUT/'receipt.json')
         and b['parent_checkpoint'] == read(previous.OUT/'steps/step_001792.json')
         and b['evidence_sha256'] == sha(EVIDENCE)
         and b['context_receipt_sha256'] == sha(CONTEXT/'receipt.json'), 'parent/evidence changed')
    need(receipt_sha() == sha(OUT/'receipt.json'), 'receipt changed in process')
    for n,h in b['sources'].items(): need(sha(ROOT/n) == h, 'bound source changed: '+n)
    return b


rgb = previous.data.rgb


def cached(item,device):
    import torch
    from safetensors.torch import load_file
    marker = OUT/'cache'/(key(item)+'.json')
    if not marker.exists(): return None
    r = read(marker); file = OUT/r['tensor']
    need(r['item'] == item and r['receipt_sha256'] == receipt_sha() and r['GT_input'] is False,
         'view cache identity changed')
    need(file.resolve().parent == (OUT/'tensors').resolve() and sha(file) == r['sha256'], 'cache SHA/path changed')
    data = load_file(str(file),device=str(device))
    need(set(data) == {'lq','latent','reference_velocity','reference_decode'}, 'cache schema changed')
    need(all(v.dtype == torch.float32 and torch.isfinite(v).all() for v in data.values()), 'invalid cache')
    need(data['lq'].shape == data['reference_decode'].shape == (1,3,512,512)
         and data['latent'].shape == data['reference_velocity'].shape, 'cache shapes differ')
    return data


def cache():
    import torch
    from safetensors.torch import save_file
    from flux2.sampling import batched_prc_img
    begun = time.monotonic(); b = bound(); local,device = base.setup(b)
    node = int(os.environ['FOUNDIR_FULLSTACK_NODE']); need(node in (1,3),'nodes1/3 only')
    rank = 2*((node-1)//2)+local
    model = base.load_model(b,device).eval().requires_grad_(False); ae,prompt = base.load_vae_prompt(b,device)
    generated, reused = 0,0
    for number,item in enumerate(b['views'][rank::4]):
        if cached(item,device) is not None:
            reused += 1
        else:
            lq = rgb(item,'lq',device)
            with torch.no_grad():
                latent = ae.encode(lq).float()
                data = {'lq':lq,'latent':latent,
                        'reference_velocity':base.core.native_velocity(model,latent,prompt,batched_prc_img),
                        'reference_decode':ae.decode(latent).float()}
            need(all(v.dtype == torch.float32 and torch.isfinite(v).all() for v in data.values()), 'invalid reference cache')
            file = OUT/'tensors'/(key(item)+f'_{time.time_ns()}.safetensors')
            save_file({k:v.cpu().contiguous() for k,v in data.items()},str(file))
            with file.open('rb') as handle: os.fsync(handle.fileno())
            write(OUT/'cache'/(key(item)+'.json'),{'item':item,'rank':rank,'GT_input':False,
                  'receipt_sha256':receipt_sha(),'tensor':str(file.relative_to(OUT)),'sha256':sha(file)})
            generated += 1; del data,lq,latent
        if (number+1)%64 == 0:
            print(f'BLUR_FULLGRID_CACHE rank={rank} complete={number+1}/3840 generated={generated} reused={reused}',flush=True)
    write(OUT/f'cache_done{rank}.json',{'rank':rank,'node':node,'exit_code':0,
          'keys':[key(i) for i in b['views'][rank::4]],'receipt_sha256':receipt_sha(),
          'seconds':time.monotonic()-begun,'generated':generated,'reused':reused})
    print(f'BLUR_FULLGRID_CACHE_END rank={rank} status=0',flush=True)


def cache_finish():
    begun = time.monotonic(); b = bound(); markers = {}; workers = []
    for rank in range(4):
        w = read(OUT/f'cache_done{rank}.json')
        need(w['rank'] == rank and w['exit_code'] == 0 and w['receipt_sha256'] == receipt_sha()
             and w['keys'] == [key(i) for i in b['views'][rank::4]], 'cache rank incomplete')
        workers.append(w)
    for item in b['views']:
        path = OUT/'cache'/(key(item)+'.json'); r = read(path)
        need(r['item'] == item and r['receipt_sha256'] == receipt_sha() and r['GT_input'] is False
             and (OUT/r['tensor']).is_file(), 'cache missing/changed')
        if key(item) in b['imports']: need(r.get('origin') == b['imports'][key(item)], 'import provenance changed')
        markers[key(item)] = sha(path)
    write(OUT/'cache_manifest.json',{'receipt_sha256':receipt_sha(),'markers':markers})
    value = {'status':'all15360_fullgrid_reference_caches_ready','training_sources':1024,
             'training_views':15360,'original4096_reused':True,'new_reference_tensors':11264,
             'tensors_SHA_verified_at_training_use':True,'all_tensor_SHA_rechecked_on_CPU':False,
             'workers':workers,'seconds':time.monotonic()-begun,'GPU_hours':0,'exit_code':0}
    write(ROOT/'reports/foundir_blur_fullgrid_v1_cache_2026-11-07.json',value)
    print('BLUR_FULLGRID_CACHE_VERIFIED '+json.dumps({k:v for k,v in value.items() if k!='workers'}),flush=True)
