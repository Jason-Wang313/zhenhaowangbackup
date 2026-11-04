"""Matched CE repair of the existing router; no change to any FLUX expert.

Only singleton routing is covered. Top-1 is NOT a proposed mixed-degradation
policy or a claim that the predicted expert is correct. No label enters routing.
"""
import torch
from .frozen_expert_router import FrozenExpertRouter

ARMS = ('image_only', 'image_plus_vlm')


def singleton_weights(logits):
    if logits.ndim != 2 or logits.shape[1] != 5 or not torch.isfinite(logits).all():
        raise ValueError('finite five-expert scores required')
    return torch.nn.functional.one_hot(logits.detach().argmax(-1), 5).to(logits)


def train_batch(model, optimizer, image, vlm, labels, *, arm):
    if arm not in ARMS or any(x.device.type != 'cpu' for x in (image, vlm, labels)):
        raise ValueError('two declared CPU router arms only')
    optimizer.zero_grad(set_to_none=True)
    logits, _, _, _ = model(image, vlm, vlm_enabled=arm == 'image_plus_vlm')
    loss = torch.nn.functional.cross_entropy(logits, labels)
    if not torch.isfinite(loss):
        raise ValueError('nonfinite classification loss')
    loss.backward()
    norms = {name: float(torch.nn.utils.clip_grad_norm_(head.parameters(), float('inf'), error_if_nonfinite=True))
             for name, head in [('image', model.image), ('vlm', model.vlm)]}
    optimizer.step()
    if any(not torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError('nonfinite router parameter')
    return {'ce': float(loss.detach()), 'batch_correct': int((logits.detach().argmax(-1) == labels).sum()),
            'batch_count': len(labels), 'gradient_norms': norms}


def infer(model, image, vlm, *, arm):
    if arm not in ARMS:
        raise ValueError('unknown router arm')
    with torch.no_grad():
        logits, soft, image_logits, delta = model(image, vlm, vlm_enabled=arm == 'image_plus_vlm')
        return {'logits': logits, 'soft_weights_diagnostic_only': soft,
                'hard_weights': singleton_weights(logits), 'image_logits': image_logits,
                'vlm_delta': delta if arm == 'image_plus_vlm' else torch.zeros_like(delta)}
