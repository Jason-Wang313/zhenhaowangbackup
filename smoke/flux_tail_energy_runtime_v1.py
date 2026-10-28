"""Real frozen FLUX-tail cache construction and target-free learned guidance."""
from __future__ import annotations
import math
import os
import torch

from smoke.flux_energy_qualification_contract_v1 import path
from smoke.qualify_foundir_flux_energy_v1 import restore, deployed_proposal
from lowlight_restoration.flux_tail_diagnostic import Tail, continuous_blend, checkpoint_blocks
from lowlight_restoration.flux_tail_budget import mixed_directions, choose_trial
from lowlight_restoration.flux_tail_learned_energy import predicted_metrics


def measured_metrics(image, target, perceptual):
    with torch.no_grad():
        mse = ((image-target)/2).square().mean().clamp_min(1e-12)
        values = dict(psnr=float(-10*mse.log10()), lpips=float(perceptual(image, target).mean()))
    if not all(math.isfinite(v) for v in values.values()): raise RuntimeError('nonfinite measured metric')
    return values


class FrozenFlow:
    def __init__(self, receipt, device):
        from safetensors.torch import load_file
        from flux2.util import load_flow_model, load_ae
        from lowlight_restoration.lora import inject_lora, load_lora_checkpoint
        self.device = device
        os.environ['KLEIN_4B_BASE_MODEL_PATH'] = str(path(receipt['base_spec']['flow']))
        os.environ['AE_MODEL_PATH'] = str(path(receipt['base_spec']['ae']))
        self.model = load_flow_model('flux.2-klein-base-4b', device=device)
        names = inject_lora(self.model, ['double_blocks.*.img_attn.qkv', 'double_blocks.*.img_attn.proj',
                                      'single_blocks.*.linear1', 'single_blocks.*.linear2'], 32, 32, 0.)
        if len(names) != 50: raise RuntimeError('FLUX LoRA layout changed')
        load_lora_checkpoint(self.model, path(receipt['plan']['baseline_checkpoint']))
        self.model.eval().requires_grad_(False)
        if next(self.model.parameters()).dtype != torch.bfloat16: raise RuntimeError('requires historical BF16 model')
        self.ae = load_ae('flux.2-klein-base-4b', device=device).eval().requires_grad_(False)
        self.context = load_file(str(path(receipt['parent_cache'])/'prompt_noise.safetensors'), device=str(device))
        self.calls = dict(flux_forward=0, flux_block_forward=0, vae_decode=0, objective_backward=0,
                          energy_forward=0, historical_flux_forward=0, historical_vae_decode=0, vae_encode=0)
        checkpoint_blocks(self.model, self.calls)

    def session(self, observation_uint8, seed, expected_proposal=None):
        from flux2.sampling import batched_prc_img, get_schedule
        observation = observation_uint8[None].to(self.device).float()/127.5-1
        with torch.no_grad():
            raw = restore(self.model, self.ae, observation, self.context['cfg_context'], self.context['cfg_ids'], seed)
            self.calls['historical_flux_forward'] += 50
            self.calls['historical_vae_decode'] += 1
            self.calls['vae_encode'] += 2
            uint8, proposal = deployed_proposal(observation, raw)
            if expected_proposal is not None and not torch.equal(uint8[0].cpu(), expected_proposal.cpu()):
                raise RuntimeError('cached historical baseline changed')
            latent = self.ae.encode(observation).to(torch.bfloat16)
            reference, ref_ids = batched_prc_img(latent)
            ref_ids = ref_ids.clone(); ref_ids[..., 0] = 10
            noise = torch.randn(latent.shape, generator=torch.Generator(device=self.device).manual_seed(seed),
                                device=self.device, dtype=torch.bfloat16)
            tokens, ids = batched_prc_img(noise)
            tail = Tail(self.model, self.ae, ids, reference, ref_ids,
                        self.context['cfg_context'], self.context['cfg_ids'], get_schedule(50, tokens.shape[1]))
            tail.calls = self.calls
            final, states = tail.rollout(tokens, 0, capture=True)
            _, reproduced = deployed_proposal(observation, tail.decode(final))
            if not torch.equal(reproduced, proposal): raise RuntimeError('tail baseline differs from historical entry')
            initial = states[48].float().detach()
        return FlowSession(tail, observation, proposal, initial)

    def assert_frozen(self):
        if any(p.requires_grad or p.grad is not None for m in (self.model, self.ae) for p in m.parameters()):
            raise RuntimeError('FLUX/VAE must remain frozen without weight gradients')


class FlowSession:
    """Deliberately contains no target/GT field."""
    def __init__(self, tail, observation, proposal, initial):
        self.tail, self.observation, self.proposal, self.initial = tail, observation, proposal, initial
        self.timestep = tail.schedule[48]

    def render(self, state):
        raw = self.tail.image(state, 48, 'rollout_full')
        continuous = continuous_blend(self.observation, raw)
        uint8, quantized = deployed_proposal(self.observation, raw)
        return continuous, uint8, quantized

    def gradients(self, state, objective):
        value = state.detach().clone().requires_grad_(True)
        raw = self.tail.image(value, 48, 'rollout_full')
        image = continuous_blend(self.observation, raw)
        scores = objective(image)
        gm = torch.autograd.grad(scores[:, 0].sum(), value, retain_graph=True)[0]
        gp = torch.autograd.grad(scores[:, 1].sum(), value)[0]
        self.tail.calls['objective_backward'] += 2
        return gm, gp


def cache_training_frames(session, target, perceptual, spec, seed):
    frames, quantized, kinds = [], [], []
    def append(rendered, label):
        continuous, uint8, _ = rendered
        frames.append(continuous[0].detach().cpu().contiguous())
        quantized.append(uint8[0].detach().cpu().contiguous())
        kinds.append(label)
    with torch.no_grad():
        current_render = session.render(session.initial)
        append(current_render, 'fp32_control')
        original_uint8 = ((session.proposal+1)*127.5).round().clamp(0,255).to(torch.uint8)
        append((session.proposal, original_uint8, session.proposal), 'historical_proposal')
        baseline = measured_metrics(session.proposal, target, perceptual)
        current = measured_metrics(current_render[2], target, perceptual)
        scales = [float(((current_render[0]-target)/2).square().mean().clamp_min(1e-8)),
                  float(perceptual(current_render[0], target).mean().clamp_min(1e-6))]
        generator = torch.Generator(device=session.initial.device).manual_seed((seed+17) % 2**63)
        direction = torch.randn(session.initial.shape, generator=generator, device=session.initial.device)
        direction /= direction.square().mean().sqrt().clamp_min(1e-12)
        for amplitude in spec['cache_random_rms_amplitudes']:
            for sign in (-1, 1):
                append(session.render(session.initial+sign*amplitude*direction), f'random_{sign}_{amplitude}')
    def objective(image):
        return torch.stack((((image-target)/2).square().flatten(1).mean(1)/scales[0],
                            perceptual(image, target).reshape(1, -1).mean(1)/scales[1]), 1)
    state = session.initial.clone(); history = []
    for step in range(1, spec['cache_teacher_max_updates']+1):
        gm, gp = session.gradients(state, objective)
        if not bool(gm.isfinite().all() and gp.isfinite().all()):
            raise RuntimeError('nonfinite cache teacher gradient')
        if min(float(g.square().mean().sqrt()) for g in (gm, gp)) <= 1e-12:
            history.append(dict(step=step, trials=[], chosen_trial=None, stop_reason='negligible_teacher_direction'))
            break
        directions, gradient_info = mixed_directions(gm, gp, spec['direction_weights'])
        trials, rendered = [], []
        with torch.no_grad():
            for weight, direction in directions.items():
                if direction is None: continue
                for amplitude in spec['rms_latent_steps']:
                    image = session.render(state-amplitude*direction)
                    rendered.append(image)
                    trials.append(dict(lpips_weight=weight, amplitude=amplitude,
                                       metrics=measured_metrics(image[2], target, perceptual)))
        chosen, decisions = choose_trial(trials, current, baseline, 'psnr_budget', spec['psnr_gain_target'])
        history.append(dict(step=step, gradient=gradient_info, trials=trials, chosen_trial=chosen, decision=decisions))
        if trials:
            worst = max(range(len(trials)), key=lambda i: trials[i]['metrics']['lpips'])
            append(rendered[worst], f'local_high_lpips_step{step}')
        if chosen is None: break
        item = trials[chosen]
        state = (state-item['amplitude']*directions[item['lpips_weight']]).detach()
        current = item['metrics']; append(rendered[chosen], f'teacher_step{step}')
        del rendered, directions, gm, gp
    return dict(frames_continuous=torch.stack(frames), frames_quantized=torch.stack(quantized)), kinds, history


def learned_guidance(session, energy, descriptor, spec):
    """No GT argument or GT-derived score/scale/direction enters this function.

    Predicted safety is a hypothesis, NOT a no-harm guarantee. True metrics are
    computed by the caller only AFTER this function has finished selection.
    """
    timestep = torch.tensor([[session.timestep]], device=descriptor.device, dtype=torch.float32)
    def objective(image):
        session.tail.calls['energy_forward'] += 1
        return energy.score(session.observation, session.proposal, image, descriptor, timestep)
    def score(image):
        with torch.no_grad(): return predicted_metrics(objective(image))
    baseline = score(session.proposal)
    with torch.no_grad(): rendered = session.render(session.initial)
    control_uint8 = rendered[1].detach().cpu()
    current = score(rendered[2]); state = session.initial.clone()
    history, milestones = [], {}
    stopped = None; accepted = 0
    for step in range(1, spec['guidance_max_updates']+1):
        if stopped is None:
            gm, gp = session.gradients(state, objective)
            if not bool(gm.isfinite().all() and gp.isfinite().all()):
                raise RuntimeError('nonfinite learned latent derivative')
            if min(float(g.square().mean().sqrt()) for g in (gm, gp)) <= 1e-12:
                stopped = 'negligible_learned_direction'
                history.append(dict(step=step, current_before=current, current_after=current,
                    trials=[], chosen_trial=None, accepted_updates=accepted, stop_reason=stopped))
            else:
                directions, gradient_info = mixed_directions(gm, gp, spec['direction_weights'])
                trials, images = [], []
                with torch.no_grad():
                    for weight, direction in directions.items():
                        if direction is None: continue
                        for amplitude in spec['rms_latent_steps']:
                            image = session.render(state-amplitude*direction)
                            images.append(image)
                            trials.append(dict(lpips_weight=weight, amplitude=amplitude, metrics=score(image[2])))
                before = current
                chosen, decisions = choose_trial(trials, before, baseline, 'psnr_budget', spec['psnr_gain_target'])
                if chosen is None:
                    stopped = 'no_predicted_eligible_trial'
                else:
                    item = trials[chosen]; current = item['metrics']; rendered = images[chosen]
                    state = (state-item['amplitude']*directions[item['lpips_weight']]).detach(); accepted += 1
                history.append(dict(step=step, current_before=before, current_after=current,
                    gradient=gradient_info, trials=trials, decision=decisions, chosen_trial=chosen,
                    accepted_updates=accepted, stop_reason=stopped))
                del directions, images
            del gm, gp
        if step in spec['milestones']:
            milestones[step] = rendered[1].detach().cpu().clone()
    return dict(milestones=milestones, control_uint8=control_uint8, history=history,
                predicted_baseline=baseline, accepted_updates=accepted, stop_reason=stopped,
                gt_used_for_guidance=False)
