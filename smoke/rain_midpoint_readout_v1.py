"""Fixed call-25 endpoint, without GT, candidate selection or a changed schedule."""
from smoke.rain_prediction_gap_math_v1 import RolloutObserver

CALL_INDEX = 25


class PrefixComplete(Exception):
    """Intentional stop immediately after the 26th model call, before its Euler update."""


class PrefixObserver(RolloutObserver):
    def __init__(self, model):
        super().__init__(model, selected=(CALL_INDEX,))

    def __call__(self, **kwargs):
        result = super().__call__(**kwargs)
        if self.calls == CALL_INDEX + 1:
            raise PrefixComplete()
        return result


def run_prefix(model, sampler):
    observer = PrefixObserver(model)
    try:
        sampler(observer)
    except PrefixComplete:
        pass
    else:
        raise ValueError('sampler returned without the fixed prefix stop')
    if observer.calls != 26 or set(observer.snapshots) != {25}:
        raise ValueError('wrong prefix count')
    return observer.snapshots[25], observer.calls


def endpoint_latent(snapshot, device):
    import torch
    from flux2.sampling import scatter_ids
    x=snapshot['x'].to(device);pred=snapshot['guided'].to(device)
    if x.shape[0]!=1 or x.shape[1]!=2*pred.shape[1]:raise ValueError('unexpected native reference tokens')
    n=pred.shape[1]
    tokens=x[:,:n].float()-snapshot['t'].to(device).float().view(-1,1,1)*pred.float()
    return torch.cat(scatter_ids(tokens,snapshot['x_ids'][:,:n].to(device))).squeeze(2)


def decode_endpoint(snapshot, ae, device):
    import torch
    latent=endpoint_latent(snapshot,device)
    with torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
        return ae.decode(latent).float().clamp(-1,1)


def assigned_indices(rank, world=8):
    if world!=8 or rank not in range(world):raise ValueError('four nodes by two GPUs required')
    return list(range(64032+rank,64132,world))


def validate_rows(rows):
    selected=rows[64032:64132]
    if len(selected)!=100 or len(rows)!=64132:
        raise ValueError('need the original full100 development set')
    if any(r['usage']!='development' or r['split']!='validation' or r['category']!='10Rain' for r in selected):
        raise ValueError('wrong development scope')
    if len({r['record_id'] for r in selected})!=100 or len({r['source_group'] for r in selected})!=100:
        raise ValueError('duplicate development source/record')
    for key in ('record_id','source_group','gt_sha256'):
        if {r[key] for r in selected}&{r[key] for r in rows[:64032]}:
            raise ValueError('development overlaps training')
    return selected


def paired_summary(rows):
    import math
    arms=('fixed_prefix26','matched_full50','original_one_step_teacher')
    if len(rows)!=100 or {r['index'] for r in rows}!=set(range(64032,64132)):
        raise ValueError('missing/duplicate development samples')
    if any(set(r['metrics'])!=set(arms) for r in rows):raise ValueError('arm set changed')
    for r in rows:
        for arm in arms:
            if set(r['metrics'][arm])!={'psnr','lpips'} or not all(math.isfinite(v) for v in r['metrics'][arm].values()):
                raise ValueError('invalid metrics')
    means={a:{k:sum(r['metrics'][a][k] for r in rows)/100 for k in ('psnr','lpips')} for a in arms}
    candidate=means['fixed_prefix26'];baseline=means['matched_full50']
    return {'count':100,'means':means,
        'delta_vs_full50':{k:candidate[k]-baseline[k] for k in ('psnr','lpips')},
        'quality_gate_pass':candidate['psnr']>=34.2 and candidate['lpips']<=.08,
        'per_image_both_nonworse':sum(r['metrics']['fixed_prefix26']['psnr']>=r['metrics']['matched_full50']['psnr']
                                     and r['metrics']['fixed_prefix26']['lpips']<=r['metrics']['matched_full50']['lpips'] for r in rows),
        'per_image_psnr_worse':sum(r['metrics']['fixed_prefix26']['psnr']<r['metrics']['matched_full50']['psnr'] for r in rows),
        'per_image_lpips_worse':sum(r['metrics']['fixed_prefix26']['lpips']>r['metrics']['matched_full50']['lpips'] for r in rows)}
