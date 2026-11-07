"""Immutable two-GPU execution binding; reuse all original fullgrid caches."""
from functools import lru_cache
import json
from pathlib import Path
import shutil
import time
from smoke import blur_fullgrid_data_v1 as original
from smoke import blur_fullgrid_state_v1 as parent_state
from smoke import blur_fullgrid_1x2_state_v1 as state
from smoke.existing_flux_replay_contract_v1 import read, sha, write_new as write

ROOT, base, need = original.ROOT, original.base, original.need
OUT = ROOT/'checkpoints/foundir_blur_fullgrid_1x2_v1'
CACHE_OUT = original.OUT
CONTEXT, geometry = original.CONTEXT, original.geometry
cached, key, rgb = original.cached, original.key, original.rgb
RECIPE = {**original.RECIPE, 'execution_version': 'resume_1x2_v1',
          'resume_step': 3687, 'remaining_updates': 5785, 'execution_world': 2,
          'execution_nodes': [3], 'parent_rng_ranks': [2, 3],
          'migration': 'model_Adam_history_sampler_preserved_world4_to2_not_bitwise',
          'training_fit_snapshots': list(state.SNAPSHOTS)}
SOURCES = ['smoke/blur_fullgrid_1x2_state_v1.py', 'smoke/blur_fullgrid_1x2_data_v1.py',
           'smoke/blur_fullgrid_1x2_snapshots_v1.py', 'smoke/train_foundir_blur_fullgrid_1x2_v1.py',
           'smoke/run_foundir_fullstack_blur_fullgrid_1x2_v1.sh',
           'smoke/evaluate_foundir_blur_fullgrid_1x2_official_v1.py',
           'smoke/run_foundir_fullstack_blur_fullgrid_1x2_official_v1.sh']


@lru_cache(maxsize=1)
def receipt_sha():
    return sha(OUT/'receipt.json')


def prepare():
    begin = time.monotonic(); old = original.bound()
    saved, marker = parent_state.latest(original.OUT, original.receipt_sha())
    need(saved is not None and saved['step'] == state.START, 'expected last committed parent3687 only')
    cache = read(original.OUT/'cache_manifest.json')
    need(cache['receipt_sha256'] == original.receipt_sha()
         and set(cache['markers']) == {key(v) for v in old['views']}, 'complete15360 existing cache required')
    for node in (1, 3):
        job = ROOT/f'checkpoints/foundir_fullstack_executor_3x2_h100_v2/jobs/round20_blur_fullgrid_train_v1_node{node}'
        need((job/'cancel.json').exists(), 'failed parent attempt must be fenced')
    ended = read(ROOT/'checkpoints/foundir_fullstack_executor_3x2_h100_v2/jobs/round20_blur_fullgrid_train_v1_node3/node3.ended.json')
    need(ended['status'] != 0, 'this migration requires failed parent attempt')
    need(shutil.disk_usage(ROOT).free > 3*1024**4, 'preserve old assets; require3TiB free')
    value = {**old, 'recipe': RECIPE, 'migration_parent_receipt_sha256': original.receipt_sha(),
             'migration_parent_checkpoint': marker, 'cache_manifest_sha256': sha(original.OUT/'cache_manifest.json'),
             'sources': {n: sha(ROOT/n) for n in sorted(set(old['sources']) | set(SOURCES))}}
    write(OUT/'receipt.json', value)
    result = {'status': 'CPU_preflight_pass', 'parent_step': state.START, 'target_step': state.END,
              'parent_SHA_model_Adam_four_RNG_sampler_verified': True, 'world_before': 4, 'world_after': 2,
              'RNG_parent_rank_mapping': list(state.PARENT_RNG_RANKS), 'bitwise_four_rank_continuation': False,
              'global_batch': 8, 'per_rank_samples': 4, 'remaining_steps': 5785,
              'new_cache_generation': False, 'cache_views_reused': 15360,
              'source_checkpoint': marker, 'GPU_started': False, 'official_Blur_pass': False,
              'seconds': time.monotonic()-begin, 'exit_code': 0, 'GPU_hours': 0}
    write(ROOT/'reports/foundir_blur_fullgrid_1x2_v1_preflight_2026-11-07.json', result)
    print('BLUR_FULLGRID_1X2_PREFLIGHT_PASS '+json.dumps(result), flush=True)


def bound():
    old = original.bound(); b = read(OUT/'receipt.json')
    need(b['recipe'] == RECIPE and b['rows'] == old['rows'] and b['views'] == old['views'], 'same schedule/data required')
    need(b['migration_parent_receipt_sha256'] == original.receipt_sha()
         and b['migration_parent_checkpoint'] == read(original.OUT/'steps/step_003687.json')
         and b['cache_manifest_sha256'] == sha(original.OUT/'cache_manifest.json'), 'migration/cache identity changed')
    for name, digest in b['sources'].items(): need(sha(ROOT/name) == digest, 'bound source changed: '+name)
    return b
