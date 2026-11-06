"""Fixed two-by-two weight intervention; no training or per-image selection."""
import math

STEPS=(800,1000)
ARMS=tuple(f'p{p}_r{r}' for p in STEPS for r in STEPS)
INDICES=tuple(range(12800,12808))


def run_grid(activate,make_prefix,readout):
    states={};outputs={}
    for prefix in STEPS:
        activate(prefix);state=make_prefix();states[prefix]=state
        for endpoint in STEPS:
            activate(endpoint);outputs[f'p{prefix}_r{endpoint}']=readout(state)
    return states,outputs


def copy_adapter(parameters,state):
    import torch
    if set(parameters)!=set(state) or not parameters:raise ValueError('adapter registration mismatch')
    for k,p in parameters.items():
        v=state[k]
        if p.shape!=v.shape or p.dtype!=torch.float32 or v.dtype!=torch.float32 or not torch.isfinite(v).all():
            raise ValueError('invalid FP32 adapter: '+k)
    with torch.no_grad():
        for k,p in parameters.items():p.copy_(state[k].to(p.device))


def assigned(local):
    if local not in (0,1):raise ValueError('one node, two local ranks required')
    return list(INDICES[local::2])


def summarize(records):
    if len(records)!=8 or {r['index'] for r in records}!=set(INDICES):raise ValueError('missing/duplicate fixed training crops')
    for r in records:
        if set(r['metrics'])!=set(ARMS) or not r['training_fit_only']:raise ValueError('wrong arms/scope')
        if any(set(s)!= {'psnr','lpips'} or not all(math.isfinite(v) for v in s.values()) for s in r['metrics'].values()):raise ValueError('invalid scores')
    means={arm:{k:math.fsum(r['metrics'][arm][k] for r in records)/8 for k in ('psnr','lpips')} for arm in ARMS}
    pairs={'readout_change_fixed800_prefix':('p800_r1000','p800_r800'),
           'prefix_change_fixed800_readout':('p1000_r800','p800_r800'),
           'prefix_change_fixed1000_readout':('p1000_r1000','p800_r1000'),
           'readout_change_fixed1000_prefix':('p1000_r1000','p1000_r800'),
           'whole_update':('p1000_r1000','p800_r800')}
    deltas={name:{k:means[a][k]-means[b][k] for k in ('psnr','lpips')} for name,(a,b) in pairs.items()}
    parity=all(r['parity'][str(s)]['max_uint8_difference']==0 and
               r['parity'][str(s)]['max_metric_difference']<=1e-5 for r in records for s in STEPS)
    fixed=deltas['readout_change_fixed800_prefix'];drift=deltas['prefix_change_fixed1000_readout']
    signature=parity and fixed['psnr']>0 and fixed['lpips']<=0 and drift['psnr']<0
    return {'means':means,'conditional_deltas':deltas,'matched_parent_parity':parity,
            'prefix_cancellation_signature_in_this_eight_crop_probe':signature,
            'interaction':{k:means['p1000_r1000'][k]-means['p800_r1000'][k]-means['p1000_r800'][k]+means['p800_r800'][k] for k in ('psnr','lpips')},
            'automatic_training':False,'official_quality_claim':False,
            'scope':'eight fixed historical training probe crops; no generalization or statistical significance claim'}
