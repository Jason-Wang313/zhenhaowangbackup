"""Read-only observation of the unchanged native CFG sampler. GT path is diagnostic only."""


class RolloutObserver:
    def __init__(self, model, selected=(0, 10, 25, 40, 49)):
        self.model = model
        self.selected = set(selected)
        self.calls = 0
        self.snapshots = {}

    def __call__(self, **kwargs):
        result = self.model(**kwargs)
        index = self.calls
        self.calls += 1
        if index in self.selected:
            x = kwargs['x']
            if x.shape[0] != 2 or x.shape[1] % 2:
                raise ValueError('expected one image, duplicated CFG, equal reference/target tokens')
            n = x.shape[1] // 2
            uncond, cond = result[:, :n].chunk(2)
            # Exactly the BF16 CFG=1 expression in denoise_cfg, not just pred_cond.
            guided = uncond + 1.0 * (cond - uncond)
            self.snapshots[index] = {
                'x': x[:1].detach().cpu().clone(),
                'x_ids': kwargs['x_ids'][:1].detach().cpu().clone(),
                't': kwargs['timesteps'][:1].detach().cpu().clone(),
                'guided': guided.detach().cpu().clone(),
                'conditional': cond.detach().cpu().clone(),
            }
        return result  # Never replace, cast, or update sampler input/output.


def gt_path_prediction(model, clean, reference, epsilon, t, context, ids, pack):
    import torch
    from lowlight_restoration.flow import build_reference_flow_batch
    from lowlight_restoration.rain_flow_adaptation import endpoint_estimate
    flow = build_reference_flow_batch(clean.float(), reference.float(), epsilon.float(), t.float(), pack, 10)
    with torch.autocast(clean.device.type, dtype=torch.bfloat16, enabled=clean.is_cuda):
        pred = model(x=flow.model_tokens.to(torch.bfloat16), x_ids=flow.model_ids,
                     timesteps=flow.timesteps, ctx=context, ctx_ids=ids, guidance=None)[:, :flow.target_token_count]
    estimate = endpoint_estimate(flow.model_tokens[:, :flow.target_token_count], pred, t, clean.shape)
    return estimate, {'flow_mse': float((pred.float()-flow.target_velocity.float()).square().mean()),
                      'latent_mse': float((estimate-clean).square().mean()),
                      'gt_latent_l1': float((estimate-clean).abs().mean())}


def validate_indices(rows, start=64000, end=64032):
    selected = rows[start:end]
    if len(selected) != 32 or any(r['usage'] != 'train_probe' or r['split'] != 'train' for r in selected):
        raise ValueError('only the original 32 training probes are allowed')
    if len({r['record_id'] for r in selected}) != 32 or len({r['source_group'] for r in selected}) != 32:
        raise ValueError('training probes must be unique records and sources')
    protected = [r for r in rows if r['usage'] == 'development']
    for key in ('record_id', 'source_group', 'gt_sha256'):
        if {r[key] for r in selected} & {r[key] for r in protected}:
            raise ValueError('training probes overlap development')
    return selected


def assigned_indices(rank, world=8):
    if world != 8 or rank not in range(world):
        raise ValueError('four nodes by two GPUs required')
    return list(range(64000+rank, 64032, world))
