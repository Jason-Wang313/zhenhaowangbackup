import copy
import inspect
import json
from pathlib import Path
import sys
import unittest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from lowlight_restoration.flux_tail_learned_energy import TailQualityEnergy,quality_targets,learning_loss,predicted_metrics
from smoke import flux_tail_energy_contract_v1 as contract
from smoke.flux_tail_energy_runtime_v1 import learned_guidance
from smoke.qualify_foundir_flux_tail_energy_v1 import pair_choice,aggregate


def perceptual(a,b):
    return ((a-b).square()+.05*(a-b).pow(4)).mean((1,2,3),keepdim=True)


class LearnedEnergyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def setUp(self): self.spec=json.loads(contract.CONFIG.read_text())

    def test_actual_model_second_order_training_and_nonzero_input_gradient(self):
        torch.manual_seed(9)
        model=TailQualityEnergy(width=4,condition_dim=8)
        x=torch.rand(2,3,16,16)*.5
        p=x*.9; y=p+.01*torch.randn_like(p);gt=x*.8
        labels=quality_targets(y,gt,perceptual)
        out=model(x,p,y,torch.rand(2,5),torch.full((2,1),.05))
        self.assertEqual(out[0].shape,(2,2))
        self.assertTrue(all(bool(g.isfinite().all()) and g.abs().max()>0 for g in out[1]))
        loss,diagnostics=learning_loss(out,labels)
        loss.backward()
        self.assertTrue(all(p.grad is not None and bool(p.grad.isfinite().all()) for p in model.parameters()))
        self.assertTrue(all(bool(v.isfinite()) for v in diagnostics.values()))
        self.assertTrue(all(p.grad is None for p in (x,y,gt)))

    def test_perfect_scores_without_correct_gradient_are_penalized(self):
        y=torch.randn(2,3,8,8)*.1;target=torch.zeros_like(y)
        scores,gradients=quality_targets(y,target,perceptual)
        good,_=learning_loss((scores,gradients),(scores,gradients))
        wrong,_=learning_loss((scores,tuple(-g for g in gradients)),(scores,gradients))
        self.assertAlmostEqual(float(good),0.,places=6)
        self.assertGreater(float(wrong),3.9)

    def test_log_metric_units_and_gradient_direction(self):
        y=torch.full((1,3,8,8),.02);target=torch.zeros_like(y)
        scores,gradients=quality_targets(y,target,perceptual)
        m=predicted_metrics(scores)
        self.assertAlmostEqual(m['psnr'],40.,places=4)
        self.assertTrue(bool((gradients[0]>0).all()))
        self.assertTrue(bool((gradients[1]>0).all()))
        with self.assertRaises(RuntimeError): predicted_metrics(torch.full((1,2),float('nan')))

    def test_zero_teacher_gradient_is_not_explosively_normalized(self):
        scores=torch.tensor([[-27.,-18.],[-27.,-18.]])
        zero=torch.zeros(2,3,8,8)
        small=torch.full_like(zero,1e-5)
        loss,_=learning_loss((scores,(small,small)),(scores,(zero,zero)))
        self.assertLess(float(loss),1e-8)

    def test_global_slots_and_sampling_are_world_size_independent(self):
        for step in (1,2,25):
            results=[]
            for world in (4,8):
                slots=sum([contract.training_slots(step,r,world) for r in range(world)],[])
                self.assertEqual(sorted(slots),list(range((step-1)*32,step*32)))
                results.append([pair_choice(s,self.spec['seed'],[37]*24) for s in sorted(slots)])
                for indices in (range(32),range(24,32)):
                    shards=[contract.assigned(indices,r,world) for r in range(world)]
                    self.assertEqual(sorted(sum(shards,[])),list(indices))
            self.assertEqual(results[0],results[1])
        with self.assertRaises(ValueError): contract.training_slots(1,0,3)

    def synthetic_rows(self,count=340):
        levels=json.dumps(dict(blur=1,noise=3,rain=1,haze=1,low_light=1))
        return [dict(record_id=str(i),source_group=f's{i}',gt_sha256=f'g{i}',lq_sha256=f'l{i}',
                     split='train',category='05Noise',width=512,height=512,qwen_levels_json=levels) for i in range(count)]

    def test_new_sources_and_calibration_are_isolated(self):
        rows=self.synthetic_rows();old=rows[:32]
        selected,counts,roles=contract.partition(rows,old,self.spec['selection_seed'])
        self.assertEqual(counts,dict(calibration=128,engineering_evaluation=128,train=52))
        self.assertEqual(len(selected),32)
        self.assertEqual(sum(r['role']=='train' for r in selected),24)
        self.assertFalse({r['row']['source_group'] for r in selected}&{r['source_group'] for r in old})
        self.assertNotIn('calibration',{r['role'] for r in selected})
        bad=copy.deepcopy(rows)
        a=next(r for r in bad if roles.get(r['source_group'])=='train')
        b=next(r for r in bad if roles.get(r['source_group'])=='engineering_evaluation')
        a['gt_sha256']=b['gt_sha256']
        with self.assertRaises(RuntimeError): contract.partition(bad,old,self.spec['selection_seed'])

    def test_protected_source_or_gt_cannot_enter_training(self):
        rows=self.synthetic_rows()
        for field in ('source_group','gt_sha256'):
            protected=dict(rows[40],record_id='protected',split='internal_test',category='04Haze',
                           source_group='protected_source',gt_sha256='protected_gt')
            protected[field]=rows[40][field]
            with self.assertRaises(RuntimeError): contract.partition(rows+[protected],rows[:32],self.spec['selection_seed'])

    def test_scope_cannot_silently_scale_or_reuse_qualification(self):
        contract.validate_spec(self.spec)
        for k,v in [('qualification_may_seed_formal',True),('auto_formal',True),('training_steps',10000),
                    ('crop',256),('guidance_max_updates',4),('cache_teacher_max_updates',4),('psnr_gain_target',.1)]:
            with self.assertRaises(RuntimeError): contract.validate_spec(self.spec|{k:v})

    def test_target_free_guidance_and_zero_gradient_safe_stop(self):
        self.assertEqual(list(inspect.signature(learned_guidance).parameters),['session','energy','descriptor','spec'])
        class Tail:
            def __init__(self): self.calls={'energy_forward':0}
        class Session:
            def __init__(self):
                self.tail=Tail();self.initial=torch.full((1,3,8,8),.2)
                self.observation=self.initial.clone();self.proposal=self.initial.clone();self.timestep=.05
            def render(self,state):
                u=((state+1)*127.5).round().clamp(0,255).to(torch.uint8)
                return state,u,u.float()/127.5-1
            def gradients(self,state,objective):
                leaf=state.detach().requires_grad_(True);score=objective(leaf)
                return tuple(torch.autograd.grad(score[:,i].sum(),leaf,retain_graph=i==0)[0] for i in range(2))
        class Constant:
            def score(self,x,p,y,d,t):
                z=y.flatten(1).sum(1)*0
                return torch.stack((z-9,z-3),1)
        result=learned_guidance(Session(),Constant(),torch.zeros(1,5),self.spec)
        self.assertFalse(result['gt_used_for_guidance'])
        self.assertEqual(result['accepted_updates'],0)
        self.assertEqual(result['stop_reason'],'negligible_learned_direction')
        for value in result['milestones'].values():
            self.assertTrue(torch.equal(value,result['control_uint8']))

    def test_nonzero_learned_guidance_runs_the_full_predicted_policy(self):
        class Session:
            def __init__(self):
                self.tail=type('Tail',(),{'calls':{'energy_forward':0}})()
                self.initial=torch.full((1,3,8,8),.2)
                self.observation=self.initial.clone();self.proposal=self.initial.clone();self.timestep=.05
            def render(self,state):
                u=((state+1)*127.5).round().clamp(0,255).to(torch.uint8)
                return state,u,u.float()/127.5-1
            def gradients(self,state,objective):
                leaf=state.detach().requires_grad_(True);score=objective(leaf)
                return tuple(torch.autograd.grad(score[:,i].sum(),leaf,retain_graph=i==0)[0] for i in range(2))
        class Energy:
            def score(self,x,p,y,d,t):
                cost=y.square().flatten(1).mean(1).clamp_min(1e-12)
                return torch.stack(((cost/4).log(),cost.log()),1)
        result=learned_guidance(Session(),Energy(),torch.zeros(1,5),self.spec)
        self.assertFalse(result['gt_used_for_guidance'])
        self.assertEqual(result['accepted_updates'],16)
        self.assertEqual(len(result['history']),16)
        self.assertTrue(all(len(h['trials'])==25 for h in result['history']))
        self.assertFalse(torch.equal(result['milestones'][16],result['control_uint8']))

    def test_real_safetensor_pair_loader_preserves_quantization_and_role(self):
        import tempfile
        from safetensors.torch import save_file
        from smoke.qualify_foundir_flux_tail_energy_v1 import load_pair
        with tempfile.TemporaryDirectory(prefix='tail-energy-pair-test-') as tmp:
            root=Path(tmp)
            image=torch.full((3,8,8),128,dtype=torch.uint8)
            save_file(dict(observation=image,proposal=image.clone(),target=image.clone(),
                frames_continuous=torch.stack([torch.full((3,8,8),.01*i) for i in range(6)]),
                frames_quantized=torch.stack([torch.full((3,8,8),120+i,dtype=torch.uint8) for i in range(6)])),str(root/'test.safetensors'))
            metadata=[dict(file='test.safetensors',frame_kinds=list(range(6)),timestep=.05,
                identity=dict(role='train',row=self.synthetic_rows(1)[0])) for _ in range(24)]
            receipt=dict(spec=self.spec)
            *images,condition,timestep=load_pair(root,receipt,metadata,0,'cpu')
            self.assertTrue(all(v.shape==(2,3,8,8) and bool(v.isfinite().all()) for v in images))
            self.assertEqual(condition.shape,(2,5));self.assertEqual(timestep.shape,(2,1))
            index,_=pair_choice(0,self.spec['seed'],[6]*24)
            metadata[index]['identity']['role']='engineering_evaluation'
            with self.assertRaises(RuntimeError):load_pair(root,receipt,metadata,0,'cpu')

    def test_evaluation_aggregate_preserves_failure_and_rejects_gt_guidance(self):
        receipt=dict(spec=self.spec,world_size=4,config_sha256='c',source_sha256={'s':'h'},selected=[None]*24)
        rows=[]
        names=['historical_bf16','fp32_control',*[f'energy_step{k}' for k in (1,2,4,8,16)]]
        for i in range(24,32):
            identity=dict(role='engineering_evaluation',index=i)
            receipt['selected'].append(identity)
            rows.append(dict(index=i,identity=identity,rank=(i-24)%4,
                metrics={k:dict(psnr=40. if k=='historical_bf16' else 39.,lpips=.1) for k in names},
                config_sha256='c',source_sha256={'s':'h'},gt_used_for_guidance=False,
                gt_loaded_after_guidance=True,baseline_parity_pass=True))
        result=aggregate(rows,receipt)
        self.assertFalse(result['means']['energy_step16']['joint_target'])
        self.assertTrue(result['engineering_pass'])
        self.assertFalse(result['generalization_success_established'])
        for key,value in [('gt_used_for_guidance',True),('gt_loaded_after_guidance',False),('rank',9)]:
            bad=copy.deepcopy(rows);bad[0][key]=value
            with self.assertRaises(RuntimeError): aggregate(bad,receipt)


if __name__=='__main__':unittest.main()
