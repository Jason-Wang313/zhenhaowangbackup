"""Four fixed, label-independent deployment-grid crops per existing training source."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from smoke import train_foundir_blur_exposure_2x2_v2 as previous
from smoke.existing_flux_replay_contract_v1 import read, sha, write_new as write

ROOT, base, need = previous.ROOT, previous.parent, previous.need
OUT = ROOT/'checkpoints/foundir_blur_multiview_v1'
SEED = 'blur_multiview_v1_20261107'
EVIDENCE = ROOT/'reports/foundir_blur_full_context_v1_result_2026-11-07.json'
RECIPE = {**previous.RECIPE, 'version': 4, 'start_step': 1280, 'train_steps': 1792,
          'new_updates': 512, 'views_per_source': 4, 'training_views': 4096,
          'final_exposures_per_source': 14, 'sample_order': 'original_source_shuffle; one_new_grid_view_per_source_per_epoch',
          'crop_selection': 'first4_SHA256(seed,record_id,x,y)_ordered_distinct_deployment_tiles; no_GT_or_scores',
          'input_normalization': 'official_CPU_float32_div127.5_sub1_before_device_transfer',
          'initialization': 'complete_step1280_model_Adam_four_rank_RNG_and_sampler',
          'migration': 'same_world4_state_preserved; intentional_new_data_views_not_bitwise_old_training',
          'evaluation': 'only_final1792_complete_official150; no_checkpoint_search'}
SOURCES = ['smoke/blur_multiview_data_v1.py', 'smoke/blur_multiview_state_v1.py',
           'smoke/train_foundir_blur_multiview_v1.py', 'smoke/run_foundir_fullstack_blur_multiview_v1.sh',
           'smoke/run_foundir_fullstack_blur_multiview_cache_v1.sh',
           'smoke/official_vlm_router_v4_contract.py', 'smoke/evaluate_foundir_official_vlm_router_v4.py']


def key(item):
    x, y = item['crop_xy']
    return f"train_{item['row']['record_id']}__v{item['view']}_x{x}_y{y}"


def views(rows):
    result = []
    for source, old in enumerate(rows):
        row = old['row']
        need(old['role'] == 'train' and row['split'] == 'train' and row['category'] == '01Blur'
             and (int(row['width']), int(row['height'])) == (1920, 1080), 'same training1080p only')
        grid = [(x, y) for y in (0, 384, 568) for x in (0, 384, 768, 1152, 1408)]
        def digest(xy):
            return hashlib.sha256(f"{SEED}:{row['record_id']}:{xy[0]}:{xy[1]}".encode()).hexdigest()
        for view, xy in enumerate(sorted(grid, key=digest)[:4]):
            result.append({'role': 'train', 'row': row, 'crop_xy': list(xy), 'view': view, 'source_index': source})
    need(len({key(i) for i in result}) == len(result), 'view cache keys collide')
    return result


def prepare():
    from concurrent.futures import ThreadPoolExecutor
    from PIL import Image
    from smoke.official_vlm_router_v4_contract import tiles
    begun = time.monotonic(); old = previous.bound()
    saved, marker = previous.state.load_step(previous.OUT, 1280, sha(previous.OUT/'receipt.json'))
    need(saved['step'] == 1280 and len(saved['optimizer']['state']) == 100, 'complete parent required')
    for rank in range(4):
        w = read(previous.OUT/f'worker{rank}.json')
        need(w['exit_code'] == 0 and w['step'] == 1280 and w['parameter_hash'] == saved['parameter_hash'], 'parent rank incomplete')
    items = views(old['rows'])
    need(len(items) == 4096 and len({i['row']['record_id'] for i in items}) == 1024, '4096views are1024sources')
    def check(item):
        row = item['row']
        coords = {(t['x'], t['y']) for t in tiles(row)}
        need(len(coords) == 15, 'deployment grid changed')
        for side in ('lq', 'gt'):
            file = ROOT/row[side+'_path']
            need(sha(file) == row[side+'_sha256'], 'training file changed')
            with Image.open(file) as image: need(image.size == (1920, 1080), 'training dimensions changed')
        for i in items[item['source_index']*4:item['source_index']*4+4]:
            need(tuple(i['crop_xy']) in coords, 'view outside deployment grid')
    with ThreadPoolExecutor(max_workers=8) as pool: list(pool.map(check, items[::4]))
    for name, digest in old['assets'].items(): need(sha(ROOT/name) == digest, 'asset changed')
    need(shutil.disk_usage(ROOT).free > 300*1024**3, 'need300GiB while preserving all assets')
    b = {k: old[k] for k in ('rows', 'assets', 'model', 'lora', 'prompt', 'metric_identity')}
    b.update(recipe=RECIPE, views=items, parent_receipt_sha256=sha(previous.OUT/'receipt.json'),
             parent_checkpoint=marker, evidence_sha256=sha(EVIDENCE),
             sources={n: sha(ROOT/n) for n in sorted(set(old['sources']) | set(SOURCES))})
    write(OUT/'receipt.json', b)
    result = {'status': 'destination_CPU_preflight_pass', 'training_sources': 1024, 'training_views': 4096,
              'new_updates': 512, 'final_step': 1792, 'parent_model_Adam_four_RNG_sampler_SHA_verified': True,
              'all2048_training_image_hashes_checked': True, 'same_world_state_resume': True,
              'equivalent_to_old_fixed_crop_continuation': False, 'official_pixels_decoded': False,
              'new_cache_root': str(OUT.relative_to(ROOT)), 'old_cache_modified': False,
              'receipt_sha256': sha(OUT/'receipt.json'), 'seconds': time.monotonic()-begun, 'GPU_hours': 0, 'exit_code': 0}
    write(ROOT/'reports/foundir_blur_multiview_v1_preflight_2026-11-07.json', result)
    print('BLUR_MULTIVIEW_PREFLIGHT_PASS '+json.dumps(result), flush=True)


def bound():
    b = read(OUT/'receipt.json'); old = previous.bound()
    need(b['recipe'] == RECIPE and b['rows'] == old['rows'] and b['views'] == views(old['rows'])
         and b['assets'] == old['assets'], 'multiview scope changed')
    need(b['parent_receipt_sha256'] == sha(previous.OUT/'receipt.json')
         and b['parent_checkpoint'] == read(previous.OUT/'steps/step_001280.json')
         and b['evidence_sha256'] == sha(EVIDENCE), 'parent/evidence changed')
    for n, h in b['sources'].items(): need(sha(ROOT/n) == h, 'bound source changed: '+n)
    return b


def rgb(item, side, device):
    from smoke.evaluate_foundir_official_vlm_router_v4 import tensor
    return tensor(base.parent.crop(item, side), device)


def cached(item, device):
    import torch
    from safetensors.torch import load_file
    marker = OUT/'cache'/(key(item)+'.json')
    if not marker.exists(): return None
    r = read(marker); file = OUT/r['tensor']
    need(r['item'] == item and r['receipt_sha256'] == sha(OUT/'receipt.json') and r['GT_input'] is False,
         'view cache identity changed')
    need(file.resolve().parent == (OUT/'tensors').resolve() and sha(file) == r['sha256'], 'view cache SHA/path changed')
    data = load_file(str(file), device=str(device))
    need(set(data) == {'lq', 'latent', 'reference_velocity', 'reference_decode'}, 'cache schema changed')
    need(all(t.dtype == torch.float32 and torch.isfinite(t).all() for t in data.values()), 'invalid cache')
    need(data['lq'].shape == data['reference_decode'].shape == (1, 3, 512, 512)
         and data['latent'].shape == data['reference_velocity'].shape, 'cache shapes differ')
    return data


def cache():
    import torch
    from safetensors.torch import save_file
    from flux2.sampling import batched_prc_img
    begun = time.monotonic(); b = bound(); local, device = base.setup(b)
    node = int(os.environ['FOUNDIR_FULLSTACK_NODE']); need(node in (1, 3), 'nodes1/3 only')
    rank = 2*((node-1)//2)+local
    model = base.load_model(b, device).eval().requires_grad_(False)
    ae, prompt = base.load_vae_prompt(b, device)
    for number, item in enumerate(b['views'][rank::4]):
        if cached(item, device) is not None: continue
        lq = rgb(item, 'lq', device)
        with torch.no_grad():
            latent = ae.encode(lq).float()
            velocity = base.core.native_velocity(model, latent, prompt, batched_prc_img)
            decoded = ae.decode(latent).float()
        data = {'lq': lq, 'latent': latent, 'reference_velocity': velocity, 'reference_decode': decoded}
        need(all(t.dtype == torch.float32 and torch.isfinite(t).all() for t in data.values()), 'invalid frozen cache')
        directory = OUT/'tensors'; directory.mkdir(parents=True, exist_ok=True)
        file = directory/(key(item)+f'_{time.time_ns()}.safetensors')
        save_file({k: v.cpu().contiguous() for k, v in data.items()}, str(file))
        with file.open('rb') as f: os.fsync(f.fileno())
        write(OUT/'cache'/(key(item)+'.json'), {'item': item, 'rank': rank, 'GT_input': False,
              'receipt_sha256': sha(OUT/'receipt.json'), 'tensor': str(file.relative_to(OUT)), 'sha256': sha(file)})
        if (number+1) % 32 == 0: print(f'BLUR_MULTIVIEW_CACHE rank={rank} complete={number+1}/1024', flush=True)
        del data, lq, latent, velocity, decoded
    write(OUT/f'cache_done{rank}.json', {'rank': rank, 'node': node, 'exit_code': 0,
          'keys': [key(i) for i in b['views'][rank::4]], 'receipt_sha256': sha(OUT/'receipt.json'),
          'seconds': time.monotonic()-begun})
    print(f'BLUR_MULTIVIEW_CACHE_END rank={rank} status=0', flush=True)


def cache_finish():
    start = time.monotonic(); b = bound(); markers = {}
    for rank in range(4):
        w = read(OUT/f'cache_done{rank}.json')
        need(w['rank'] == rank and w['exit_code'] == 0 and w['receipt_sha256'] == sha(OUT/'receipt.json')
             and w['keys'] == [key(i) for i in b['views'][rank::4]], 'cache rank incomplete')
    for item in b['views']:
        p = OUT/'cache'/(key(item)+'.json'); r = read(p)
        need(r['item'] == item and r['receipt_sha256'] == sha(OUT/'receipt.json') and r['GT_input'] is False
             and (OUT/r['tensor']).is_file(), 'cache missing/changed')
        markers[key(item)] = sha(p)
    value = {'receipt_sha256': sha(OUT/'receipt.json'), 'markers': markers}
    write(OUT/'cache_manifest.json', value)
    result = {'status': 'all4096_view_cache_metadata_verified', 'training_sources': 1024, 'training_views': 4096,
              'tensors_SHA_verified_at_training_use': True, 'all_tensor_SHA_rechecked_on_CPU': False,
              'seconds': time.monotonic()-start, 'GPU_hours': 0, 'exit_code': 0}
    write(ROOT/'reports/foundir_blur_multiview_v1_cache_2026-11-07.json', result)
    print('BLUR_MULTIVIEW_CACHE_VERIFIED '+json.dumps(result), flush=True)
