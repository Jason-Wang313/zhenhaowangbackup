import copy
import inspect
import json
from pathlib import Path
import tempfile
import unittest
import torch

from lowlight_restoration.flux_tail_learned_energy import TailQualityEnergy,quality_targets
from lowlight_restoration.flux_tail_relative_energy_v3 import (
    relative_loss,parameter_gradient_audit,pair_plan,generation_for_step,
    edit_stats,trust_limits,within_trust,bounded_one_step)
from smoke.flux_tail_checkpoint_v2 import rng_state,restore_rng
from smoke.flux_tail_relative_checkpoint_v3 import commit,load
from smoke.qualify_foundir_flux_tail_energy_v1 import parameter_hash
from smoke import run_foundir_flux_tail_relative_energy_v3 as runner


WEIGHTS=dict(relative=1.,absolute=.1,gradient=1.,direction=1.,relative_beta=.05)


def perceptual(a,b):return ((a-b).square()+.05*(a-b).pow(4)).mean((1,2,3),keepdim=True)


class RelativeEnergyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_true_scores_and_gradients_have_zero_loss(self):
        y=torch.randn(2,3,8,8)*.1;labels=quality_targets(y,torch.zeros_like(y),perceptual)
        loss,stats,_=relative_loss(labels,labels,WEIGHTS)
        self.assertLess(abs(float(loss)),1e-5)
        self.assertGreater(float(stats['mse_cosine']),.999)

    def test_relative_component_cancels_shared_image_difficulty(self):
        y=torch.randn(2,3,8,8)*.1;s,g=quality_targets(y,torch.zeros_like(y),perceptual)
        _,_,a=relative_loss((s+17,g),(s,g),WEIGHTS)
        self.assertLess(float(a['relative']),1e-8)
        self.assertGreater(float(a['absolute']),10.)

    def test_reversed_direction_penalized_even_with_perfect_scores(self):
        y=torch.randn(2,3,8,8)*.1;s,g=quality_targets(y,torch.zeros_like(y),perceptual)
        loss,stats,_=relative_loss((s,tuple(-v for v in g)),(s,g),WEIGHTS)
        self.assertGreater(float(loss),5.9)
        self.assertLess(float(stats['mse_cosine']),-.999)

    def test_zero_prediction_direction_backward_is_finite(self):
        s=torch.tensor([[-9.,-3.],[-8.,-2.]],requires_grad=True)
        gradients=tuple(torch.zeros(2,3,8,8,requires_grad=True) for _ in range(2))
        teachers=tuple(torch.ones_like(v)*.01 for v in gradients)
        loss,_,_=relative_loss((s,gradients),(s.detach(),teachers),WEIGHTS)
        loss.backward()
        self.assertTrue(all(bool(v.grad.isfinite().all()) for v in gradients))
        self.assertTrue(all(bool(v.grad.abs().sum()>0) for v in gradients))

    def test_zero_target_direction_is_not_invented(self):
        s=torch.tensor([[-9.,-3.],[-9.,-3.]],requires_grad=True)
        gradients=tuple(torch.zeros(2,3,8,8,requires_grad=True) for _ in range(2))
        loss,_,parts=relative_loss((s,gradients),(s.detach(),tuple(v.detach() for v in gradients)),WEIGHTS)
        self.assertEqual(float(parts['direction']),0.)
        loss.backward()
        self.assertTrue(all(bool(v.grad.isfinite().all()) for v in gradients))

    def test_real_model_second_order_and_component_audit(self):
        torch.manual_seed(3);model=TailQualityEnergy(width=4,condition_dim=8)
        x=torch.rand(2,3,16,16)*.4;p=x*.9;y=p+.01*torch.randn_like(p)
        labels=quality_targets(y,x*.8,perceptual)
        prediction=model(x,p,y,torch.zeros(2,5),torch.full((2,1),.2))
        loss,stats,parts=relative_loss(prediction,labels,WEIGHTS)
        audit=parameter_gradient_audit(model,parts,WEIGHTS)
        self.assertEqual(set(audit),{'relative','absolute','gradient','direction'})
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        loss.backward()
        self.assertTrue(all(p.grad is not None and bool(p.grad.isfinite().all()) for p in model.parameters()))
        self.assertTrue(all(torch.isfinite(v) for v in stats.values()))

    def test_four_eight_gpu_global_pair_stream_matches(self):
        for step in (1,4,5,50,51,200):
            streams=[]
            for world in (4,8):
                slots=sum([runner.old_contract.training_slots(step,rank,world) for rank in range(world)],[])
                self.assertEqual(len(slots),32)
                streams.append([pair_plan(i,73,[20]*24,[8]*24,[[3,5,7]]*24,[1]*24) for i in sorted(slots)])
            self.assertEqual(streams[0],streams[1])
        for i in range(40):
            _,pair=pair_plan(i,73,[20]*24,[8]*24,[[3,5,7]]*24,[1]*24)
            self.assertEqual(any(v[0]=='mining' for v in pair),i%4 in (1,2))

    def test_refresh_schedule_has_no_future_checkpoint_leak(self):
        self.assertEqual([generation_for_step(i,[4]) for i in (1,4,5,8)],[0,0,4,4])
        self.assertEqual([generation_for_step(i,[50,100,150]) for i in (50,51,100,101,151)],[0,50,50,100,150])

    def test_channel_edit_uses_rgb_zero_one_units(self):
        p=torch.zeros(1,3,8,8);y=p+.04
        v=edit_stats(y,p)
        self.assertAlmostEqual(v['pixel_rms'],.02,places=7)
        self.assertAlmostEqual(v['channel_mean_max'],.02,places=7)
        self.assertFalse(within_trust(y,p,dict(pixel_rms=.1,channel_mean_max=.01))[0])

    def test_training_derived_limits_and_empty_failure(self):
        s=runner.spec();rows=[dict(pixel_rms=.001,channel_mean_max=.0001),dict(pixel_rms=.01,channel_mean_max=.001)]
        limits=trust_limits(rows,s)
        self.assertGreater(limits['pixel_rms'],.01)
        self.assertEqual(limits['channel_mean_max'],s['trust_quantization_floor_0_1'])
        with self.assertRaises(ValueError):trust_limits([],s)

    def test_bounded_selection_rejects_huge_edit_and_preserves_exact_proposal(self):
        class Session:
            def __init__(self):
                self.tail=type('Tail',(),{'calls':{'energy_forward':0}})()
                self.initial=torch.full((1,3,8,8),.2)
                self.observation=self.initial.clone();self.proposal=self.initial.clone();self.timestep=.05
            def render(self,state):
                u=((state+1)*127.5).round().clamp(0,255).to(torch.uint8)
                return state,u,u.float()/127.5-1
            def gradients(self,state,objective):
                leaf=state.detach().requires_grad_(True);s=objective(leaf)
                return tuple(torch.autograd.grad(s[:,i].sum(),leaf,retain_graph=i==0)[0] for i in range(2))
        class Energy:
            def score(self,x,p,y,c,t):
                cost=y.square().flatten(1).mean(1)
                return torch.stack(((cost/4).log(),cost.log()),1)
        s=json.loads((runner.ROOT/'configs/foundir_flux_tail_energy_qualification_v1.json').read_text())
        result=bounded_one_step(Session(),Energy(),torch.zeros(1,5),s,dict(pixel_rms=0.,channel_mean_max=0.))
        self.assertIsNotNone(result['raw_choice']);self.assertIsNone(result['bounded_choice'])
        self.assertTrue(torch.equal(result['outputs']['bounded_step1'],result['outputs']['original']))
        self.assertFalse(result['gt_used_for_selection'])
        self.assertNotIn('target',inspect.signature(bounded_one_step).parameters)

    def test_scope_train_only_preserves_original_quality_goals(self):
        s=runner.spec()
        self.assertEqual(s['train_indices'],list(range(24)))
        self.assertEqual(s['probe_indices'],list(range(8)))
        self.assertEqual(s['modes']['trainfit']['steps'],200)
        self.assertEqual(s['crop'],512)
        self.assertFalse(s['read_calibration_or_evaluation_pixels'])
        self.assertEqual(s['psnr_gain_target'],.3)
        self.assertAlmostEqual(s['lpips_relative_reduction_target'],.09928107050377874)

    def test_labeled_frames_keep_real_scores_and_derivatives(self):
        candidates=torch.randn(3,3,8,8)*.1;target=torch.zeros(1,3,8,8)
        cached=runner.labeled_frames(candidates,target,perceptual,'cpu')
        for i in range(3):
            score,grad=quality_targets(candidates[i:i+1],target,perceptual)
            self.assertTrue(torch.equal(cached['scores'][i],score[0]))
            self.assertTrue(torch.equal(cached['gm'][i],grad[0][0]))
            self.assertTrue(torch.equal(cached['gp'][i],grad[1][0]))

    def test_real_cached_pair_loader_and_eval_role_rejection(self):
        from safetensors.torch import save_file
        with tempfile.TemporaryDirectory(prefix='tail-relative-pair-') as tmp:
            root=Path(tmp);bases=root/'base';mined=root/'mine';bases.mkdir();mined.mkdir()
            image=torch.full((3,8,8),128,dtype=torch.uint8)
            values=dict(candidates=torch.randn(8,3,8,8)*.1,scores=torch.randn(8,2),
                        gm=torch.randn(8,3,8,8),gp=torch.randn(8,3,8,8))
            save_file(values|dict(observation=image,proposal=image.clone()),str(bases/'x.safetensors'))
            current=torch.full_like(image,131).float()/127.5-1
            scores,gradients=quality_targets(current[None],torch.zeros(1,3,8,8),perceptual)
            save_file(values|dict(proposal_normalized=current,proposal_scores=scores[0],
                                  proposal_gm=gradients[0][0],proposal_gp=gradients[1][0]),
                      str(mined/'x.safetensors'))
            levels=json.dumps(dict(blur=1,noise=3,rain=1,haze=1,low_light=1))
            r=dict(mode='qualification',spec=runner.spec(),selected=[dict(role='train',row=dict(split='train',qwen_levels_json=levels)) for _ in range(24)])
            bm=[dict(file='x.safetensors',frames=8,proposal_index=1,teacher_indices=[3,5,7],timestep=.2) for _ in range(24)]
            nm=[dict(file='x.safetensors',frames=8) for _ in range(24)]
            metadata=(bases,mined,bm,nm)
            for slot in range(4):
                inputs,labels=runner.load_pair(r,metadata,slot,'cpu')
                self.assertEqual(inputs[2].shape,(2,3,8,8));self.assertEqual(labels[0].shape,(2,2))
                self.assertTrue(torch.equal(inputs[1],current[None].repeat(2,1,1,1)))
                if slot%4 in (0,1):
                    anchor=[i for i in range(2) if torch.equal(inputs[2][i],current)]
                    self.assertEqual(len(anchor),1)
                    self.assertTrue(torch.equal(labels[0][anchor[0]],scores[0]))
                    self.assertTrue(torch.equal(labels[1][0][anchor[0]],gradients[0][0]))
                    self.assertTrue(torch.equal(labels[1][1][anchor[0]],gradients[1][0]))
            for row in r['selected']:row['role']='engineering_evaluation'
            with self.assertRaises(RuntimeError):runner.load_pair(r,metadata,0,'cpu')

    def test_checkpoint_exact_resume_and_world_mismatch_rejected(self):
        with tempfile.TemporaryDirectory(prefix='flux-relative-resume-') as directory:
            torch.manual_seed(61);model=torch.nn.Linear(3,2);optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4)
            def update(m,opt):
                opt.zero_grad();m(torch.rand(2,3)).square().mean().backward();opt.step()
            update(model,optimizer)
            receipt=dict(protocol_sha256='bound',world_size=4,mode='qualification',
                spec=dict(learning_rate=1e-4,modes={'qualification':{'steps':8}}))
            rank=dict(rng=rng_state('cpu'),parameter_sha256=parameter_hash(model),peak_gib=0.,elapsed=1.)
            rec=commit(directory,model,optimizer,1,receipt,[rank]*4,[{'step':1}],1.)
            update(model,optimizer);expected=copy.deepcopy(model.state_dict())
            state,_=load(directory,receipt)
            restored=torch.nn.Linear(3,2);opt=torch.optim.AdamW(restored.parameters(),lr=1e-4)
            restored.load_state_dict(state['model']);opt.load_state_dict(state['optimizer'])
            restore_rng(state['rank_states'][0]['rng'],'cpu');update(restored,opt)
            self.assertTrue(all(torch.equal(v,restored.state_dict()[k]) for k,v in expected.items()))
            with self.assertRaises(RuntimeError):load(directory,receipt|{'world_size':8})
            (Path(directory)/rec['file']).write_bytes(b'corrupted')
            with self.assertRaises(RuntimeError):load(directory,receipt)


if __name__=='__main__':unittest.main()
