"""Read-only normalization/gradient/update telemetry for controlled LoRA probes."""
import math

import torch


class OptimizerTrace:
    def __init__(self, model):
        self.active = False
        self.normalizations = {}
        self.handles = []
        for name, module in model.named_modules():
            if 'vae_encoder' in name or 'text_encoder' in name:
                continue
            if module.__class__.__name__ == 'RMSNorm' or isinstance(module, torch.nn.GroupNorm):
                self.handles.append(module.register_forward_pre_hook(self._hook(name)))

    def _hook(self, name):
        def hook(module, args):
            if not self.active:
                return
            with torch.no_grad():
                x = args[0].detach().float()
                if isinstance(module, torch.nn.GroupNorm):
                    scale = x.reshape(x.shape[0], module.num_groups, -1).var(dim=-1, unbiased=False)
                else:
                    scale = x.square().mean(dim=-1)
                denom = (scale + module.eps).sqrt()
                self.normalizations[name] = torch.stack([denom.min(), denom.max(), x.abs().max()])
        return hook

    def start(self):
        self.normalizations = {}
        self.active = True

    def stop_forward(self):
        self.active = False

    def collect_gradients(self, named_params):
        groups = {}
        for name, p in named_params:
            if p.grad is None:
                raise RuntimeError(f'missing gradient: {name}')
            key = name.split('.lora_')[0]
            square = p.grad.detach().float().square().sum()
            groups[key] = groups.get(key, 0) + square
        names = list(groups)
        norms = torch.stack([groups[n].sqrt() for n in names]).cpu().tolist()
        norm_names = list(self.normalizations)
        norms_in = torch.stack([self.normalizations[n] for n in norm_names]).cpu().tolist() if norm_names else []
        return dict(module_gradient_norms=dict(zip(names,norms)),
                    normalization_inputs={n:dict(denominator_min=v[0],denominator_max=v[1],input_absmax=v[2]) for n,v in zip(norm_names,norms_in)})

    def close(self):
        for handle in self.handles:
            handle.remove()


def update_statistics(params, before):
    with torch.no_grad():
        sums = torch.stack([torch.stack([(p.detach()-b).float().square().sum(), b.float().square().sum()]) for p,b in zip(params,before)]).sum(0).cpu().tolist()
    update, magnitude = math.sqrt(sums[0]), math.sqrt(sums[1])
    return dict(update_l2=update,parameter_l2=magnitude,relative_update_l2=update/magnitude if magnitude else None)


def trace_step(step):
    return step == 1 or step % 20 == 0 or step >= 160
