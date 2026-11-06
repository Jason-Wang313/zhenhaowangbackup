"""LQ-preserving FLUX correction; no GT argument and no new RGB network."""
import random
import torch


def native_velocity(model, latent, prompt, pack):
    if latent.ndim != 4 or latent.shape[0] != 1 or latent.dtype != torch.float32:
        raise ValueError('one original FP32 LQ latent required')
    tokens, ids = pack(latent.to(torch.bfloat16))
    refs = ids.clone()
    refs[..., 0] = 10
    if prompt['cfg_context'].shape[0] != 2 or prompt['cfg_ids'].shape[0] != 2:
        raise ValueError('native unconditional/conditional pair required')
    with torch.autocast(latent.device.type, dtype=torch.bfloat16, enabled=latent.is_cuda):
        velocity = model(x=torch.cat((tokens, tokens), 1).repeat(2, 1, 1),
                         x_ids=torch.cat((ids, refs), 1).repeat(2, 1, 1),
                         timesteps=torch.ones(2, device=latent.device, dtype=torch.bfloat16),
                         ctx=prompt['cfg_context'], ctx_ids=prompt['cfg_ids'], guidance=None)
    velocity = velocity[:, :tokens.shape[1]]
    if velocity.shape != (2, latent.shape[2]*latent.shape[3], latent.shape[1]):
        raise ValueError('native velocity layout changed')
    # Preserve the actual native CFG=1 BF16 arithmetic, including its rounding.
    unconditional, conditional = velocity.chunk(2)
    result = unconditional + 1.0 * (conditional - unconditional)
    return result.float().transpose(1, 2).reshape_as(latent)


def corrected_latent(latent, velocity, reference):
    if any(t.dtype != torch.float32 or t.shape != latent.shape for t in (latent, velocity, reference)):
        raise ValueError('matching FP32 latent and velocities required')
    if latent.requires_grad or reference.requires_grad:
        raise ValueError('LQ and frozen reference must be detached')
    return latent - (velocity - reference)


def corrected_rgb(lq, decoded, reference_decoded):
    if lq.shape != decoded.shape or decoded.shape != reference_decoded.shape:
        raise ValueError('RGB shape mismatch')
    if lq.requires_grad or reference_decoded.requires_grad:
        raise ValueError('LQ and reference decode must be detached')
    # Subtract decodes first: exact zero correction preserves the original LQ.
    return (lq.float() + (decoded.float() - reference_decoded.float())).clamp(-1, 1)


def uint8_output(rgb):
    return ((rgb.detach().float().clamp(-1, 1) + 1) * 127.5).round().byte()


def sample_indices(step, rank, *, count=128, batch=8, world=2, seed=60419):
    if step < 0 or not 0 <= rank < world or batch % world or count % batch:
        raise ValueError('invalid global sampling geometry')
    result = []
    for micro in range(batch // world):
        position = step * batch + micro * world + rank
        epoch, offset = divmod(position, count)
        order = list(range(count))
        random.Random(seed + epoch).shuffle(order)
        result.append(order[offset])
    return result


def objective(prediction, gt, lpips):
    rgb = (prediction + 1) * .5
    target = (gt + 1) * .5
    mse = (rgb - target).square().mean()
    perceptual = lpips(rgb, target).mean()
    return 10.0 * mse + .1 * perceptual, {'mse': mse, 'lpips': perceptual}
