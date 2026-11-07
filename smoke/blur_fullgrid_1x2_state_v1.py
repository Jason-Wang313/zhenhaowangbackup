"""Resume the fixed fullgrid schedule on two GPUs; topology change is not bitwise."""
from pathlib import Path
from smoke import blur_fullgrid_state_v1 as parent
from smoke.existing_flux_replay_contract_v1 import read

START, END, WORLD = 3687, 9472, 2
SNAPSHOTS = (3712, 5632, 7552, 9472)
PARENT_RNG_RANKS = (2, 3)


def indices(step, rank, world=WORLD):
    if not START <= step < END or world != WORLD or rank not in range(WORLD):
        raise ValueError('fixed3687..9472, two ranks required')
    return parent.indices(step, rank, world)


def load_step(output, step, receipt_sha, world=WORLD):
    if not START <= step <= END or world != WORLD:
        raise ValueError('invalid two-rank continuation state')
    return parent.load_step(output, step, receipt_sha, world)


def latest(output, receipt_sha, world=WORLD):
    files = sorted((Path(output)/'steps').glob('step_*.json'))
    if not files:
        return None, None
    steps = [read(p)['step'] for p in files]
    if steps != list(range(START, START+len(steps))):
        raise ValueError('committed continuation gap; no fallback')
    return load_step(output, steps[-1], receipt_sha, world)


def migrate(model, optimizer, value, rank, device):
    if value['step'] != START or value['world'] != 4 or len(value['rank_rng']) != 4 or rank not in range(2):
        raise ValueError('requires complete parent3687/four-rank state')
    parent.restore_parameters_adam(model, optimizer, value)
    parent.original.restore_rng(value['rank_rng'][PARENT_RNG_RANKS[rank]], device)


restore = parent.restore
restore_parameters_adam = parent.restore_parameters_adam
